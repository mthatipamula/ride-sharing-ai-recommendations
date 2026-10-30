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
