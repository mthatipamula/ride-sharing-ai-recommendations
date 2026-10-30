"""Training examples and point-in-time features shared by training and serving.

Splits are by time, never random: weeks 1-6 train the retrieval model, week 7 trains the ranker on candidates the
retrieval model has not seen labels for, week 8 is the untouched test week. Every feature for a query is computed
from data strictly before that query (the trip history) or before the start of its week (the daily batch
snapshot), so nothing leaks from the future.
"""
from __future__ import annotations

import dataclasses

import numpy as np
import pandas as pd

from .simulator import CATEGORIES, PERSONAS

HIST_LEN = 10
TRAIN_END_DAY = 42      # days [0, 42) train retrieval
VAL_END_DAY = 49        # days [42, 49) train the ranker; [49, ...) test


def hour_bucket(hour: np.ndarray | int) -> np.ndarray:
    """0 morning (5-10), 1 midday (11-15), 2 evening (16-20), 3 night (21-4)."""
    h = np.asarray(hour)
    return np.select([(h >= 5) & (h <= 10), (h >= 11) & (h <= 15), (h >= 16) & (h <= 20)], [0, 1, 2], 3)


def split_of(day: np.ndarray) -> np.ndarray:
    return np.select([day < TRAIN_END_DAY, day < VAL_END_DAY], ["train", "val"], "test")


def destination_examples(trips: pd.DataFrame, riders: pd.DataFrame) -> pd.DataFrame:
    """One row per trip: the context when the rider opened the app, the last HIST_LEN destinations before it, and
    the destination they actually chose (the label)."""
    t = trips.sort_values(["rider_id", "ts"], kind="stable").reset_index(drop=True)
    hist = np.zeros((len(t), HIST_LEN), dtype=np.int64)
    dest = t.dest_place_id.to_numpy()
    starts = np.flatnonzero(np.r_[True, t.rider_id.to_numpy()[1:] != t.rider_id.to_numpy()[:-1]])
    ends = np.r_[starts[1:], len(t)]
    for s, e in zip(starts, ends):
        for i in range(s + 1, e):
            prev = dest[max(s, i - HIST_LEN):i]
            hist[i, HIST_LEN - len(prev):] = prev            # oldest -> newest, left-padded with 0

    persona_ix = {p: i for i, p in enumerate(PERSONAS)}
    rd = riders.set_index("rider_id")
    ex = pd.DataFrame({
        "trip_id": t.trip_id, "rider_id": t.rider_id, "ts": t.ts, "day": t.day, "hour": t.hour, "dow": t.dow,
        "origin": t.origin_place_id, "dest": t.dest_place_id,
        "persona": rd.persona.loc[t.rider_id].map(persona_ix).to_numpy(),
    })
    ex["hbucket"] = hour_bucket(ex.hour.to_numpy())
    ex["weekend"] = (ex.dow >= 5).astype(np.int8)
    ex["split"] = split_of(ex.day.to_numpy())
    ex["hist"] = list(hist)
    return ex


def place_arrays(places: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Category index and normalised coordinates per place id (row 0 is padding)."""
    cat_ix = {c: i for i, c in enumerate(CATEGORIES)}
    n = int(places.place_id.max()) + 1
    cats = np.zeros(n, dtype=np.int64)
    xy = np.zeros((n, 2), dtype=np.float32)
    cats[places.place_id] = places.category.map(cat_ix)
    xy[places.place_id] = places[["x", "y"]].to_numpy() / 20.0
    return cats, xy


@dataclasses.dataclass
class Snapshot:
    """Batch features as of a cutoff day: what a nightly pipeline would publish to the online feature store."""
    cutoff_day: int
    rider_place: pd.DataFrame        # rider_id, place_id, visits_total, last_day
    rider_place_hbucket: pd.DataFrame
    rider_place_daytype: pd.DataFrame
    popularity: pd.Series            # place_id -> trips
    category_hour_lift: np.ndarray   # (n_categories, 24): share of trips to category at hour / overall share

    @classmethod
    def build(cls, trips: pd.DataFrame, places: pd.DataFrame, cutoff_day: int) -> "Snapshot":
        h = trips[trips.day < cutoff_day].rename(columns={"dest_place_id": "place_id"})
        h = h.assign(hbucket=hour_bucket(h.hour.to_numpy()), weekend=(h.dow >= 5).astype(np.int8))
        rp = h.groupby(["rider_id", "place_id"]).agg(visits_total=("trip_id", "size"), last_day=("day", "max"))
        rph = h.groupby(["rider_id", "place_id", "hbucket"]).size().rename("visits_hbucket")
        rpw = h.groupby(["rider_id", "place_id", "weekend"]).size().rename("visits_daytype")
        popularity = h.groupby("place_id").size().rename("popularity")

        cats, _ = place_arrays(places)
        c = cats[h.place_id.to_numpy()]
        counts = np.zeros((len(CATEGORIES), 24))
        np.add.at(counts, (c, h.hour.to_numpy()), 1)
        share_at_hour = counts / np.maximum(counts.sum(0, keepdims=True), 1)
        overall = counts.sum(1, keepdims=True) / max(counts.sum(), 1)
        lift = np.where(overall > 0, share_at_hour / np.maximum(overall, 1e-9), 0.0)
        return cls(cutoff_day, rp.reset_index(), rph.reset_index(), rpw.reset_index(), popularity, lift)


RANKER_FEATURES = [
    "tt_score", "tt_rank", "visits_total", "visits_hbucket", "visits_daytype", "days_since_visit",
    "hist_count", "hist_steps_back", "distance_km", "popularity", "category", "category_hour_lift",
    "is_home", "is_work",
]


def candidate_features(queries: pd.DataFrame, candidates: np.ndarray, tt_scores: np.ndarray, snap: Snapshot,
                       places: pd.DataFrame, riders: pd.DataFrame) -> pd.DataFrame:
    """Long frame (query x candidate) of ranker features. The same function serves one online request, so the
    features cannot drift between training and serving."""
    q, k = candidates.shape
    rep = lambda col: np.repeat(queries[col].to_numpy(), k)  # noqa: E731
    df = pd.DataFrame({
        "qid": np.repeat(np.arange(q), k), "rider_id": rep("rider_id"), "place_id": candidates.ravel(),
        "hbucket": rep("hbucket"), "weekend": rep("weekend"), "day": rep("day"), "hour": rep("hour"),
        "tt_score": tt_scores.ravel(), "tt_rank": np.tile(np.arange(1, k + 1), q),
    })
    df = df.merge(snap.rider_place, on=["rider_id", "place_id"], how="left")
    df = df.merge(snap.rider_place_hbucket, on=["rider_id", "place_id", "hbucket"], how="left")
    df = df.merge(snap.rider_place_daytype, on=["rider_id", "place_id", "weekend"], how="left")
    for col in ("visits_total", "visits_hbucket", "visits_daytype"):
        df[col] = df[col].fillna(0).astype(np.float32)
    df["days_since_visit"] = np.minimum((df.day - df.last_day).fillna(99), 99).astype(np.float32)

    hist = np.stack(queries["hist"].to_numpy())                                  # (q, HIST_LEN)
    match = hist[:, None, :] == candidates[:, :, None]                        # (q, k, HIST_LEN)
    df["hist_count"] = match.sum(-1).ravel().astype(np.float32)
    newest = np.where(match.any(-1), np.argmax(match[:, :, ::-1], axis=-1), HIST_LEN)
    df["hist_steps_back"] = newest.ravel().astype(np.float32)               # 0 = the last place they went

    xy = places.set_index("place_id")[["x", "y"]]
    o = xy.loc[np.repeat(queries.origin.to_numpy(), k)].to_numpy()
    c = xy.loc[df.place_id].to_numpy()
    df["distance_km"] = (np.hypot(*(o - c).T) * 1.3 + 0.3).astype(np.float32)
    df["popularity"] = snap.popularity.reindex(df.place_id).fillna(0).to_numpy(np.float32)
    cats, _ = place_arrays(places)
    df["category"] = cats[df.place_id.to_numpy()]
    df["category_hour_lift"] = snap.category_hour_lift[df.category.to_numpy(), df.hour.to_numpy()].astype(np.float32)
    rd = riders.set_index("rider_id")
    df["is_home"] = (rd.home_place_id.loc[df.rider_id].to_numpy() == df.place_id.to_numpy()).astype(np.int8)
    df["is_work"] = (rd.work_place_id.loc[df.rider_id].to_numpy() == df.place_id.to_numpy()).astype(np.int8)
    return df
