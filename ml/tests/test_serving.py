"""Service tests against a tiny freshly trained model and the in-memory feature store."""
import sys

import pytest
from fastapi.testclient import TestClient

from rideshare_recs import train
from rideshare_recs.serving.experiments import Experiment


@pytest.fixture(scope="module")
def client(tmp_path_factory, monkeypatch_module):
    root = tmp_path_factory.mktemp("recs")
    monkeypatch_module.setattr(sys, "argv", ["rideshare-train", "--riders", "300", "--epochs", "1",
                                             "--data-dir", str(root / "data"), "--artifacts", str(root / "artifacts")])
    train.main()
    monkeypatch_module.setenv("ARTIFACTS_DIR", str(root / "artifacts"))
    monkeypatch_module.setenv("DATA_DIR", str(root / "data" / "riders-300-seed-7"))
    monkeypatch_module.delenv("REDIS_URL", raising=False)
    from rideshare_recs.serving.app import app

    with TestClient(app) as c:
        yield c


@pytest.fixture(scope="module")
def monkeypatch_module():
    mp = pytest.MonkeyPatch()
    yield mp
    mp.undo()


def rider_in(c, variant: str) -> dict:
    for rider_id in range(300):
        r = c.get(f"/v1/riders/{rider_id}").json()
        if r["experiment_variant"] == variant:
            return r
    raise AssertionError(f"no {variant} rider")


@pytest.mark.parametrize("variant,path", [("treatment", "model"), ("control", "heuristic")])
def test_recommendations_follow_the_experiment_arm(client, variant, path):
    rider = rider_in(client, variant)
    r = client.post("/v1/recommendations", json={"rider_id": rider["rider_id"],
                                                 "origin": {"place_id": rider["home"]["place_id"]},
                                                 "at": "2026-10-20T08:00:00"})
    assert r.status_code == 200
    body = r.json()
    assert body["experiment"]["variant"] == variant and body["path"] == path
    ids = [d["place_id"] for d in body["destinations"]]
    assert len(ids) == 5 and len(set(ids)) == 5
    assert rider["home"]["place_id"] not in ids                  # never suggest where you already are
    assert body["ride_options"]["destination"] == ids[0]
    assert r.headers["X-Model-Version"] == body["model_version"]


def test_ride_completed_updates_real_time_features(client):
    rider = rider_in(client, "treatment")
    rec = client.post("/v1/recommendations", json={"rider_id": rider["rider_id"],
                                                   "origin": {"place_id": rider["home"]["place_id"]}}).json()
    dest = rec["destinations"][2]["place_id"]
    out = client.post("/v1/events/ride-completed", json={"rider_id": rider["rider_id"], "destination_place_id": dest,
                                                         "mode": "xl", "request_id": rec["request_id"]})
    assert out.status_code == 202 and out.json()["suggested_rank"] == 3
    after = client.get(f"/v1/riders/{rider['rider_id']}").json()
    assert after["recent_trips"][0]["place_id"] == dest
    assert after["ride_modes"]["xl"] == rider["ride_modes"].get("xl", 0) + 1


def test_unknown_rider_gets_cold_start(client):
    body = client.post("/v1/recommendations", json={"rider_id": 10_000_000, "origin": {"x_km": 10, "y_km": 10}}).json()
    assert body["path"] == "cold_start" and len(body["destinations"]) == 5
    assert all(d["category"] != "residential" for d in body["destinations"])


def test_ride_options_never_offer_a_car_too_small(client):
    rider = rider_in(client, "control")
    places = client.get("/v1/places", params={"category": "airport"}).json()
    body = client.post("/v1/ride-options", json={"rider_id": rider["rider_id"],
                                                 "origin": {"place_id": rider["home"]["place_id"]},
                                                 "destination_place_id": places[0]["place_id"],
                                                 "group_size": 5}).json()
    assert body["recommended"] == "xl"
    unavailable = {o["mode"] for o in body["options"] if not o["available"]}
    assert unavailable == {"wait_save", "standard", "lux"}
    assert sum(o["probability"] for o in body["options"]) == pytest.approx(1.0, abs=1e-3)


def test_validation_errors(client):
    assert client.post("/v1/recommendations", json={"rider_id": 1, "origin": {}}).status_code == 422
    assert client.post("/v1/events/ride-completed",
                       json={"rider_id": 1, "destination_place_id": 5, "mode": "helicopter"}).status_code == 422


def test_metrics_and_health(client):
    assert client.get("/health/ready").json()["status"] == "ready"
    text = client.get("/metrics").text
    assert "recs_request_latency_seconds_bucket" in text and "recs_stage_latency_seconds" in text


def test_assignment_is_sticky_and_balanced():
    exp = Experiment("test_exp", 0.5)
    assert all(exp.assign(r) == exp.assign(r) for r in range(100))
    share = sum(exp.assign(r) == "treatment" for r in range(20_000)) / 20_000
    assert 0.48 < share < 0.52
    other = Experiment("another_exp", 0.5)
    agree = sum(exp.assign(r) == other.assign(r) for r in range(20_000)) / 20_000
    assert 0.47 < agree < 0.53                                  # experiments are independent
