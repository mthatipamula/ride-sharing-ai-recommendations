"""Online inference: one rider, one moment -> ranked destinations and ride options.

    features (1 Redis round trip) -> two-tower query embedding . cached place embeddings -> top 50
    -> listwise ranker on the same features as training -> top 5 -> ride-mode model for the chosen trip

Every stage has a fallback, so the app always gets an answer: an unknown rider gets places popular at this hour
near them, and if the feature store is down the service answers from popularity instead of erroring.
"""
from __future__ import annotations

import dataclasses
import json
import time
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from ..features import HIST_LEN, RANKER_FEATURES, candidate_features_online, hour_bucket, place_arrays
from ..models import ranker as rk
from ..models import ride_mode as rm
from ..models import two_tower as tt
from ..simulator import MODE_CAPACITY, PERSONAS, RIDE_MODES, standard_fare, surge_for
from .feature_store import FeatureStore, GlobalFeatures, RiderFeatures

PICKUP_MINUTES = {"wait_save": 12, "standard": 5, "xl": 8, "lux": 7}
MODE_LABEL = {"wait_save": "Wait & Save", "standard": "Standard", "xl": "XL", "lux": "Lux"}


@dataclasses.dataclass
class Context:
    rider_id: int
    origin: int
    when: datetime

    @property
    def hour(self) -> int:
        return self.when.hour

    @property
    def dow(self) -> int:
        return self.when.weekday()


class Timer:
    def __init__(self):
        self.stages: dict[str, float] = {}

    @contextmanager
    def stage(self, name: str):
        start = time.perf_counter()
        try:
            yield
        finally:
            self.stages[name] = round((time.perf_counter() - start) * 1000, 3)


def latest_artifact_dir(root: Path) -> Path:
    return root / (root / "LATEST").read_text().strip()


class Recommender:
    def __init__(self, artifact_dir: Path, store: FeatureStore):
        torch.set_num_threads(1)              # one request per core beats intra-op threading for tiny batches
        self.dir = artifact_dir
        self.store = store
        self.config = json.loads((artifact_dir / "serving_config.json").read_text())
        self.version = self.config["version"]
        self.k = int(self.config["candidates"])
        self.sim_start = pd.Timestamp(self.config["sim_start"])
        self.mode_mix = np.array([self.config["ride_mode_global_share"][m] for m in RIDE_MODES])

        self.places = pd.read_parquet(artifact_dir / "places.parquet")
        self.place_by_id = self.places.set_index("place_id")
        self.categories, xy = place_arrays(self.places)
        self.xy_km = xy * 20.0
        self.two_tower = tt.load(artifact_dir / "two_tower.pt")
        self.ranker = rk.load(artifact_dir / "ranker.pt")
        self.ride_mode = rm.load(artifact_dir / "ride_mode.pt")
        with torch.no_grad():
            self.item_emb = self.two_tower.encode_items()
        self._global: GlobalFeatures | None = None
        self._global_at = 0.0

    # ---------- features ----------
    def global_features(self) -> GlobalFeatures:
        if self._global is None or time.monotonic() - self._global_at > 60:
            g = self.store.global_features()
            if g is None:
                raise RuntimeError("feature snapshot not published")
            self._global, self._global_at = g, time.monotonic()
        return self._global

    def nearest_place(self, x_km: float, y_km: float) -> int:
        d = np.hypot(self.xy_km[1:, 0] - x_km, self.xy_km[1:, 1] - y_km)
        return int(np.argmin(d) + 1)

    def day_index(self, when: datetime) -> int:
        ts = pd.Timestamp(when)
        ts = ts.tz_convert(None) if ts.tzinfo else ts
        return int((ts.normalize() - self.sim_start).days)

    def _hist(self, f: RiderFeatures) -> np.ndarray:
        hist = np.zeros(HIST_LEN, dtype=np.int64)
        if f.hist:
            hist[HIST_LEN - len(f.hist[-HIST_LEN:]):] = f.hist[-HIST_LEN:]
        return hist

    def _popularity(self, g: GlobalFeatures) -> np.ndarray:
        if getattr(self, "_pop_version", None) != g.version:
            pop = np.zeros(len(self.categories), dtype=np.float32)
            for p, c in g.popularity.items():
                pop[p] = c
            self._pop, self._pop_version = pop, g.version
        return self._pop

    def features(self, ctx: Context, f: RiderFeatures, g: GlobalFeatures, cands: np.ndarray,
                 tt_scores: np.ndarray) -> np.ndarray:
        return candidate_features_online(
            hour=ctx.hour, weekend=int(ctx.dow >= 5), day=self.day_index(ctx.when), origin=ctx.origin,
            hist=self._hist(f), candidates=cands, tt_scores=tt_scores, visits=f.visits,
            visits_hbucket=f.visits_hbucket, visits_daytype=f.visits_daytype, home=f.home, work=f.work,
            popularity=self._popularity(g), category_hour_lift=g.category_hour_lift,
            place_category=self.categories, place_xy_km=self.xy_km)

    # ---------- destinations ----------
    def destinations(self, ctx: Context, variant: str, n: int = 5, timer: Timer | None = None) -> dict:
        timer = timer or Timer()
        with timer.stage("features"):
            try:
                f = self.store.rider(ctx.rider_id)
                g = self.global_features()
            except Exception:
                f, g = None, None
        if g is None:
            return self._popular_now(ctx, n, None, path="fallback_no_features", timer=timer)
        if f is None or ctx.rider_id >= self.two_tower.cfg.n_riders:
            return self._popular_now(ctx, n, g, path="cold_start", timer=timer)

        if variant == "control":
            with timer.stage("heuristic"):
                ids, scores = self._frequency_by_time(ctx, f, g, n)
                x = self.features(ctx, f, g, ids, np.zeros(len(ids), np.float32))
            path = "heuristic"
        else:
            with timer.stage("retrieval"), torch.no_grad():
                t = lambda v: torch.as_tensor([v])  # noqa: E731
                batch = {"rider": t(ctx.rider_id), "origin": t(ctx.origin), "hour": t(ctx.hour), "dow": t(ctx.dow),
                         "persona": t(PERSONAS.index(f.persona)), "hist": torch.as_tensor(self._hist(f))[None]}
                top = torch.topk(self.two_tower.scores(batch, self.item_emb), self.k, dim=1)
                cands, tt_scores = top.indices.numpy()[0], top.values.numpy()[0]
            with timer.stage("ranking"):
                x = self.features(ctx, f, g, cands, tt_scores)
                rscores = rk.score_matrix(self.ranker, x)
                order = np.argsort(-rscores)[:n]
                ids, scores, x = cands[order], rscores[order], x[order]
            path = "model"
        with timer.stage("explain"):
            weekend = int(ctx.dow >= 5)
            items = [self._item(int(p), float(s), ctx, {**dict(zip(RANKER_FEATURES, row.tolist())), "weekend": weekend})
                     for p, s, row in zip(ids, scores, x)]
        return {"items": items, "path": path, "timings_ms": timer.stages}

    def _frequency_by_time(self, ctx: Context, f: RiderFeatures, g: GlobalFeatures, n: int):
        """Control arm: the strongest heuristic from offline evaluation (rider's most-visited at this time of day)."""
        hb = int(hour_bucket(ctx.hour))
        score = np.zeros(len(self.categories))
        pop = np.zeros(len(self.categories))
        for p, c in g.popularity.items():
            pop[p] = c
        score += 1e-4 * pop / max(pop.max(), 1)
        for p, (t, _) in f.visits.items():
            score[p] += 0.1 * t
        for (p, b), c in f.visits_hbucket.items():
            if b == hb:
                score[p] += c
        score[0] = score[ctx.origin] = -np.inf
        ids = np.argsort(-score)[:n]
        return ids, score[ids]

    def _popular_now(self, ctx: Context, n: int, g: GlobalFeatures | None, path: str, timer: Timer) -> dict:
        """No personal features: places popular at this hour, discounted by distance from the rider."""
        with timer.stage("fallback"):
            pop = np.zeros(len(self.categories))
            if g is not None:
                for p, c in g.popularity.items():
                    pop[p] = c
                pop *= g.category_hour_lift[self.categories, ctx.hour]
            else:
                pop[1:] = (self.places.category != "residential").to_numpy().astype(float)
            d = np.hypot(*(self.xy_km - self.xy_km[ctx.origin]).T)
            score = pop * np.exp(-d / 6.0)
            score[0] = score[ctx.origin] = -np.inf
            score[1:][self.places.category.to_numpy() == "residential"] = -np.inf
            ids = np.argsort(-score)[:n]
            items = [self._item(int(p), float(score[p]), ctx, None) for p in ids]
        return {"items": items, "path": path, "timings_ms": timer.stages}

    def _item(self, place_id: int, score: float, ctx: Context, feats: dict | None) -> dict:
        p = self.place_by_id.loc[place_id]
        dist = float(np.hypot(*(self.xy_km[place_id] - self.xy_km[ctx.origin])) * 1.3 + 0.3)
        fare = float(standard_fare(np.array([dist]), surge_for(np.array([ctx.hour]), np.array([ctx.dow])))[0])
        return {"place_id": place_id, "name": p["name"], "category": p.category, "neighborhood": p.neighborhood,
                "distance_km": round(dist, 1), "standard_fare": fare, "score": round(score, 4),
                "reasons": reasons(feats, p.category, ctx.hour) if feats else ["Popular around here right now"]}

    # ---------- ride options ----------
    def ride_options(self, ctx: Context, dest: int, group_size: int = 1, luggage: bool | None = None,
                     timer: Timer | None = None) -> dict:
        timer = timer or Timer()
        with timer.stage("features"):
            try:
                f = self.store.rider(ctx.rider_id)
            except Exception:
                f = None
        cats = self.place_by_id.category
        is_airport = int("airport" in (cats.loc[ctx.origin], cats.loc[dest]))
        luggage = bool(is_airport) if luggage is None else luggage
        dist = float(np.hypot(*(self.xy_km[dest] - self.xy_km[ctx.origin])) * 1.3 + 0.3)
        surge = float(surge_for(np.array([ctx.hour]), np.array([ctx.dow]))[0])
        fare = float(standard_fare(np.array([dist]), np.array([surge]))[0])
        prices = {m: round(fare * mult, 2) for m, mult in zip(RIDE_MODES, (0.78, 1.0, 1.55, 2.2))}

        counts = np.array([(f.mode_counts if f else {}).get(m, 0) for m in RIDE_MODES], dtype=float)
        shares = (counts + rm.PRIOR_STRENGTH * self.mode_mix) / (counts.sum() + rm.PRIOR_STRENGTH)
        row = {"distance_km": dist, "surge": surge, "group_size": group_size, "luggage": int(luggage),
               "is_airport": is_airport, "rider_id": ctx.rider_id if f else 0,
               "persona": f.persona if f else "commuter", "hour": ctx.hour, "dow": ctx.dow,
               "log_prior_trips": np.log1p(counts.sum()),
               **{f"price_{m}": prices[m] for m in RIDE_MODES}, **{f"share_{m}": shares[j] for j, m in enumerate(RIDE_MODES)}}
        with timer.stage("ride_mode"):
            probs = rm.predict_proba(self.ride_mode, rm.to_batch(pd.DataFrame([row]), {p: i for i, p in enumerate(PERSONAS)}))[0]
        options = []
        for j, m in enumerate(RIDE_MODES):
            feasible = group_size <= MODE_CAPACITY[j]
            options.append({"mode": m, "label": MODE_LABEL[m], "price": prices[m], "seats": int(MODE_CAPACITY[j]),
                            "pickup_minutes": round(PICKUP_MINUTES[m] * (1 + 0.5 * (surge - 1))),
                            "available": bool(feasible), "probability": round(float(probs[j]) if feasible else 0.0, 4)})
        options.sort(key=lambda o: (-o["available"], -o["probability"]))
        return {"origin": ctx.origin, "destination": dest, "distance_km": round(dist, 1), "surge": surge,
                "recommended": options[0]["mode"], "options": options, "personalized": f is not None,
                "timings_ms": timer.stages}


def reasons(feats: dict, category: str, hour: int) -> list[str]:
    """Plain-language reasons from the same features the ranker used (the LLM step rewrites these later)."""
    out = []
    when = ["mornings", "midday", "evenings", "late nights"][int(hour_bucket(hour))]
    if feats.get("is_home"):
        out.append("Home")
    elif feats.get("is_work"):
        out.append("Campus" if category == "university" else "Work")
    if feats.get("visits_hbucket", 0) >= 2:
        out.append(f"You often go here in the {when}" if when != "midday" else "You often go here around midday")
    elif feats.get("visits_daytype", 0) >= 2:
        out.append("A regular spot for you on " + ("weekends" if feats.get("weekend") else "weekdays"))
    if feats.get("hist_steps_back", 99) == 0:
        out.append("Your most recent destination")
    elif feats.get("hist_count", 0) >= 2:
        out.append("In your recent trips")
    if not out and feats.get("category_hour_lift", 0) > 1.5:
        out.append(f"Popular {category.replace('_', ' ')} spot at this hour")
    return out or ["Recommended for you"]
