import numpy as np
import pytest

from rideshare_recs.evaluation import rank_of_truth, ranking_metrics
from rideshare_recs.features import HIST_LEN, RANKER_FEATURES, Snapshot, candidate_features, destination_examples
from rideshare_recs.simulator import MODE_CAPACITY, RIDE_MODES, SimConfig, simulate


@pytest.fixture(scope="module")
def data():
    return simulate(SimConfig(n_riders=150, n_homes=40, days=14, seed=1))


def test_simulation_is_deterministic_and_consistent(data):
    places, riders, trips = data
    again = simulate(SimConfig(n_riders=150, n_homes=40, days=14, seed=1))[2]
    assert trips.equals(again)
    assert places.place_id.min() == 1                      # 0 is reserved for padding
    assert (trips.origin_place_id != trips.dest_place_id).all()
    cap = dict(zip(RIDE_MODES, MODE_CAPACITY))
    assert (trips.group_size <= trips["mode"].map(cap)).all()   # nobody books a car too small for the group


def test_history_only_contains_earlier_trips(data):
    _, riders, trips = data
    ex = destination_examples(trips, riders)
    for rider_id, g in ex.groupby("rider_id"):
        dests = g.sort_values("ts").dest.to_numpy()
        hist = np.stack(g.sort_values("ts")["hist"].to_numpy())
        for i in range(len(g)):
            expected = dests[max(0, i - HIST_LEN):i]
            assert list(hist[i][HIST_LEN - len(expected):]) == list(expected)
            assert (hist[i][:HIST_LEN - len(expected)] == 0).all()


def test_snapshot_ignores_trips_after_cutoff(data):
    places, riders, trips = data
    snap = Snapshot.build(trips, places, cutoff_day=7)
    assert snap.rider_place.last_day.max() < 7
    assert snap.rider_place.visits_total.sum() == (trips.day < 7).sum()

    ex = destination_examples(trips, riders)
    q = ex[ex.day >= 7].head(5).reset_index(drop=True)
    cands = np.tile(places.place_id.to_numpy()[:20], (len(q), 1))
    frame = candidate_features(q, cands, np.zeros_like(cands, dtype=np.float32), snap, places, riders)
    assert len(frame) == 5 * 20
    assert frame[RANKER_FEATURES].notna().all().all()


def test_rank_of_truth_counts_ties_half():
    scores = np.array([[0.9, 0.5, 0.5, 0.1]])
    assert rank_of_truth(scores, np.array([0]))[0] == 1
    assert rank_of_truth(scores, np.array([1]))[0] == 2.5
    m = ranking_metrics(np.array([1.0, 3.0, np.inf]), ks=(1, 3))
    assert m["recall@1"] == pytest.approx(1 / 3)
    assert m["recall@3"] == pytest.approx(2 / 3)
    assert m["mrr"] == pytest.approx((1 + 1 / 3) / 3)


def test_online_fast_path_matches_offline_features(data):
    """Training/serving parity: the NumPy features the service computes per request equal the pandas features
    the ranker was trained on, for the same rider, moment and candidates."""
    from rideshare_recs.features import candidate_features_online, place_arrays
    from rideshare_recs.serving.feature_store import InMemoryFeatureStore
    from rideshare_recs.serving.publish import build_payload

    places, riders, trips = data
    cutoff = 7
    snap = Snapshot.build(trips, places, cutoff_day=cutoff)
    store = InMemoryFeatureStore()
    store.publish(**build_payload(places, riders, trips, cutoff))
    g = store.global_features()
    cats, xy = place_arrays(places)
    pop = np.zeros(len(cats), dtype=np.float32)
    for p, c in g.popularity.items():
        pop[p] = c

    ex = destination_examples(trips, riders)
    q = ex[ex.day >= cutoff].sample(25, random_state=0).reset_index(drop=True)
    rng = np.random.default_rng(0)
    cands = np.stack([rng.choice(places.place_id.to_numpy(), 30, replace=False) for _ in range(len(q))])
    scores = rng.normal(size=cands.shape).astype(np.float32)
    offline = candidate_features(q, cands, scores, snap, places, riders)[RANKER_FEATURES].to_numpy(np.float32)

    for i, row in q.iterrows():
        f = store.rider(int(row.rider_id))
        online = candidate_features_online(
            hour=int(row.hour), weekend=int(row.weekend), day=int(row.day), origin=int(row.origin),
            hist=row["hist"], candidates=cands[i], tt_scores=scores[i], visits=f.visits,
            visits_hbucket=f.visits_hbucket, visits_daytype=f.visits_daytype, home=f.home, work=f.work,
            popularity=pop, category_hour_lift=g.category_hour_lift, place_category=cats, place_xy_km=xy * 20.0)
        np.testing.assert_allclose(online, offline[i * 30:(i + 1) * 30], rtol=1e-4, atol=1e-3)
