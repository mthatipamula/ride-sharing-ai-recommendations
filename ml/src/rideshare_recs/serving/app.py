"""Recommendation service (FastAPI).

    ARTIFACTS_DIR=ml/artifacts REDIS_URL=redis://localhost:6379/0 uv run uvicorn rideshare_recs.serving.app:app

Without REDIS_URL the service uses an in-memory feature store and publishes the snapshot itself at startup from
DATA_DIR, so it runs with no other process.
"""
from __future__ import annotations

import logging
import os
import time
import uuid
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from prometheus_client import Counter, Gauge, Histogram, make_asgi_app
from pydantic import BaseModel, Field, model_validator

from ..simulator import RIDE_MODES
from .experiments import Experiment
from .feature_store import open_store
from .publish import default_data_dir, publish
from .recommender import Context, Recommender, Timer, latest_artifact_dir

log = logging.getLogger("rideshare_recs")
ML_DIR = Path(__file__).resolve().parents[3]

LATENCY_BUCKETS = (0.002, 0.005, 0.01, 0.015, 0.02, 0.03, 0.05, 0.075, 0.1, 0.25, 0.5, 1.0)
REQUEST_LATENCY = Histogram("recs_request_latency_seconds", "End-to-end request latency", ["endpoint"],
                            buckets=LATENCY_BUCKETS)
STAGE_LATENCY = Histogram("recs_stage_latency_seconds", "Latency per pipeline stage", ["stage"],
                          buckets=LATENCY_BUCKETS)
REQUESTS = Counter("recs_requests_total", "Recommendation responses", ["endpoint", "variant", "path"])
FEEDBACK = Counter("recs_ride_completed_total", "Completed rides reported", ["variant", "suggested_rank"])
MODEL_INFO = Gauge("recs_model_info", "Loaded model version", ["version"])


class Origin(BaseModel):
    place_id: int | None = Field(None, description="Pickup place; or give x_km / y_km for the nearest place")
    x_km: float | None = None
    y_km: float | None = None

    @model_validator(mode="after")
    def one_of(self):
        if self.place_id is None and (self.x_km is None or self.y_km is None):
            raise ValueError("give origin.place_id or both origin.x_km and origin.y_km")
        return self


class RecommendationRequest(BaseModel):
    rider_id: int = Field(ge=0)
    origin: Origin
    at: datetime | None = Field(None, description="Local time; defaults to now")
    limit: int = Field(5, ge=1, le=10)
    include_ride_options: bool = True


class RideOptionsRequest(BaseModel):
    rider_id: int = Field(ge=0)
    origin: Origin
    destination_place_id: int
    at: datetime | None = None
    group_size: int = Field(1, ge=1, le=6)
    luggage: bool | None = None


class RideCompleted(BaseModel):
    rider_id: int = Field(ge=0)
    destination_place_id: int
    mode: str | None = None
    request_id: str | None = Field(None, description="The recommendation request that led to this ride, if any")

    @model_validator(mode="after")
    def known_mode(self):
        if self.mode is not None and self.mode not in RIDE_MODES:
            raise ValueError(f"mode must be one of {RIDE_MODES}")
        return self


def _artifacts_root() -> Path:
    return Path(os.environ.get("ARTIFACTS_DIR", ML_DIR / "artifacts"))


@asynccontextmanager
async def lifespan(app: FastAPI):
    store = open_store(os.environ.get("REDIS_URL"))
    if store.global_features() is None:
        info = publish(store, default_data_dir())          # in-memory store, or an empty Redis
        log.warning("feature snapshot was empty; published %s", info)
    rec = Recommender(latest_artifact_dir(_artifacts_root()), store)
    MODEL_INFO.labels(rec.version).set(1)
    app.state.rec, app.state.store, app.state.experiment = rec, store, Experiment.from_env()
    app.state.recent = {}                                    # request_id -> what was shown (bounded)
    yield


app = FastAPI(title="Rider recommendations", version="0.2.0", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=os.environ.get("CORS_ORIGINS", "http://localhost:5173").split(","),
                   allow_methods=["GET", "POST"], allow_headers=["*"])
app.mount("/metrics", make_asgi_app())


@app.middleware("http")
async def timing(request: Request, call_next):
    start = time.perf_counter()
    response: Response = await call_next(request)
    route = request.scope.get("route")
    REQUEST_LATENCY.labels(getattr(route, "path", "unmatched")).observe(time.perf_counter() - start)
    if hasattr(request.app.state, "rec"):
        response.headers["X-Model-Version"] = request.app.state.rec.version
    return response


def _context(rec: Recommender, rider_id: int, origin: Origin, at: datetime | None) -> Context:
    if origin.place_id is not None:
        if origin.place_id not in rec.place_by_id.index:
            raise HTTPException(404, f"unknown place {origin.place_id}")
        place = origin.place_id
    else:
        place = rec.nearest_place(origin.x_km, origin.y_km)
    when = at or datetime.now()
    if when.tzinfo is not None:
        when = when.astimezone().replace(tzinfo=None)      # models were trained on local wall-clock time
    return Context(rider_id, place, when)


def _observe(timer: Timer) -> None:
    for stage, ms in timer.stages.items():
        STAGE_LATENCY.labels(stage).observe(ms / 1000)


@app.get("/health/live")
def live():
    return {"status": "ok"}


@app.get("/health/ready")
def ready(request: Request):
    store_ok = request.app.state.store.ping()
    if not store_ok:
        raise HTTPException(503, "feature store unreachable")
    return {"status": "ready", "model_version": request.app.state.rec.version,
            "feature_snapshot": request.app.state.rec.global_features().version}


@app.post("/v1/recommendations")
def recommendations(body: RecommendationRequest, request: Request):
    rec: Recommender = request.app.state.rec
    ctx = _context(rec, body.rider_id, body.origin, body.at)
    variant = request.app.state.experiment.assign(body.rider_id)
    timer = Timer()
    result = rec.destinations(ctx, variant, body.limit, timer)
    ride = None
    if body.include_ride_options and result["items"]:
        ride = rec.ride_options(ctx, result["items"][0]["place_id"])
        STAGE_LATENCY.labels("ride_mode").observe(ride["timings_ms"]["ride_mode"] / 1000)
    _observe(timer)

    request_id = uuid.uuid4().hex
    shown = [i["place_id"] for i in result["items"]]
    recent = request.app.state.recent
    recent[request_id] = {"rider_id": body.rider_id, "variant": variant, "shown": shown}
    if len(recent) > 50_000:
        recent.pop(next(iter(recent)))
    try:
        request.app.state.store.log_event("impressions", {"request_id": request_id, "rider_id": body.rider_id,
                                                          "variant": variant, "path": result["path"],
                                                          "shown": shown, "model_version": rec.version})
    except Exception:
        log.exception("impression logging failed")      # never fail the request for analytics
    REQUESTS.labels("recommendations", variant, result["path"]).inc()

    origin = rec.place_by_id.loc[ctx.origin]
    return {
        "request_id": request_id,
        "rider_id": body.rider_id,
        "origin": {"place_id": ctx.origin, "name": origin["name"], "neighborhood": origin.neighborhood},
        "at": ctx.when.isoformat(),
        "experiment": {"name": request.app.state.experiment.name, "variant": variant},
        "path": result["path"],
        "destinations": result["items"],
        "ride_options": ride,
        "timings_ms": result["timings_ms"],
        "model_version": rec.version,
    }


@app.post("/v1/ride-options")
def ride_options(body: RideOptionsRequest, request: Request):
    rec: Recommender = request.app.state.rec
    if body.destination_place_id not in rec.place_by_id.index:
        raise HTTPException(404, f"unknown place {body.destination_place_id}")
    ctx = _context(rec, body.rider_id, body.origin, body.at)
    if ctx.origin == body.destination_place_id:
        raise HTTPException(422, "destination is the pickup place")
    out = rec.ride_options(ctx, body.destination_place_id, body.group_size, body.luggage)
    STAGE_LATENCY.labels("ride_mode").observe(out["timings_ms"]["ride_mode"] / 1000)
    REQUESTS.labels("ride_options", "none", "model" if out["personalized"] else "cold_start").inc()
    return out


@app.post("/v1/events/ride-completed", status_code=202)
def ride_completed(body: RideCompleted, request: Request):
    """Updates the rider's real-time features (recent trips, ride-mode counts) and logs the experiment outcome."""
    store = request.app.state.store
    store.record_ride(body.rider_id, body.destination_place_id, body.mode)
    shown = request.app.state.recent.get(body.request_id or "")
    rank = shown["shown"].index(body.destination_place_id) + 1 if shown and body.destination_place_id in shown["shown"] else None
    variant = shown["variant"] if shown else request.app.state.experiment.assign(body.rider_id)
    FEEDBACK.labels(variant, str(rank) if rank else "not_suggested").inc()
    try:
        store.log_event("outcomes", {"request_id": body.request_id, "rider_id": body.rider_id, "variant": variant,
                                     "destination": body.destination_place_id, "suggested_rank": rank,
                                     "mode": body.mode})
    except Exception:
        log.exception("outcome logging failed")
    return {"status": "accepted", "suggested_rank": rank, "variant": variant}


@app.get("/v1/riders/{rider_id}")
def rider(rider_id: int, request: Request):
    rec: Recommender = request.app.state.rec
    f = request.app.state.store.rider(rider_id)
    if f is None:
        raise HTTPException(404, "unknown rider")
    name = lambda p: rec.place_by_id.loc[p, "name"]  # noqa: E731
    top = sorted(f.visits.items(), key=lambda kv: -kv[1][0])[:5]
    return {"rider_id": rider_id, "segment": f.persona,
            "home": {"place_id": f.home, "name": name(f.home)}, "work": {"place_id": f.work, "name": name(f.work)},
            "recent_trips": [{"place_id": p, "name": name(p)} for p in reversed(f.hist)],
            "ride_modes": f.mode_counts,
            "top_places": [{"place_id": p, "name": name(p), "visits": t} for p, (t, _) in top],
            "experiment_variant": request.app.state.experiment.assign(rider_id)}


@app.get("/v1/places")
def places(request: Request, category: str | None = None):
    df = request.app.state.rec.places
    if category:
        df = df[df.category == category]
    return [{"place_id": int(r.place_id), "name": r.name, "category": r.category, "neighborhood": r.neighborhood,
             "x_km": round(float(r.x), 2), "y_km": round(float(r.y), 2)} for r in df.itertuples()]
