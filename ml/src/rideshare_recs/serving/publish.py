"""Nightly batch job: compute the feature snapshot as of a cutoff day and publish it to the online store.

    uv run rideshare-publish-features --redis-url redis://localhost:6379/0

Defaults to the start of the test week (day 49), so the service recommends for trips the models never saw.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import pandas as pd

from ..features import HIST_LEN, VAL_END_DAY, Snapshot
from ..simulator import SimConfig, load_or_simulate
from .feature_store import FeatureStore, encode_snap, open_store

ML_DIR = Path(__file__).resolve().parents[3]


def default_data_dir() -> Path:
    return Path(os.environ.get("DATA_DIR", ML_DIR / "data" / "riders-5000-seed-7"))


def build_payload(places: pd.DataFrame, riders: pd.DataFrame, trips: pd.DataFrame, cutoff_day: int) -> dict:
    snap = Snapshot.build(trips, places, cutoff_day)
    before = trips[trips.day < cutoff_day].sort_values("ts", kind="stable")

    profiles = {int(r.rider_id): {"persona": r.persona, "home": int(r.home_place_id), "work": int(r.work_place_id)}
                for r in riders.itertuples()}
    hists = {int(k): g.dest_place_id.tail(HIST_LEN).astype(int).tolist() for k, g in before.groupby("rider_id")}
    modes = {int(k): {m: int(n) for m, n in g["mode"].value_counts().items()} for k, g in before.groupby("rider_id")}

    visits: dict[int, dict] = {}
    for r in snap.rider_place.itertuples():
        visits.setdefault(int(r.rider_id), {})[int(r.place_id)] = (int(r.visits_total), int(r.last_day))
    hb: dict[int, dict] = {}
    for r in snap.rider_place_hbucket.itertuples():
        hb.setdefault(int(r.rider_id), {})[(int(r.place_id), int(r.hbucket))] = int(r.visits_hbucket)
    wk: dict[int, dict] = {}
    for r in snap.rider_place_daytype.itertuples():
        wk.setdefault(int(r.rider_id), {})[(int(r.place_id), int(r.weekend))] = int(r.visits_daytype)
    snaps = {rid: encode_snap(visits.get(rid, {}), hb.get(rid, {}), wk.get(rid, {})) for rid in profiles}

    meta = {
        "cutoff_day": cutoff_day,
        "version": time.strftime(f"snapshot-day{cutoff_day}-%Y%m%d-%H%M%S"),
        "popularity": json.dumps({int(k): int(v) for k, v in snap.popularity.items()}),
        "category_hour_lift": json.dumps(snap.category_hour_lift.round(4).tolist()),
    }
    return {"profiles": profiles, "hists": hists, "modes": modes, "snaps": snaps, "meta": meta}


def publish(store: FeatureStore, data_dir: Path, cutoff_day: int = VAL_END_DAY) -> dict:
    riders_n = int(data_dir.name.split("-")[1]) if data_dir.name.startswith("riders-") else SimConfig().n_riders
    places, riders, trips = load_or_simulate(data_dir, SimConfig(n_riders=riders_n, n_homes=max(100, riders_n // 5)))
    payload = build_payload(places, riders, trips, cutoff_day)
    store.publish(**payload)
    return {"riders": len(payload["profiles"]), "cutoff_day": cutoff_day, "version": payload["meta"]["version"]}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--redis-url", default=os.environ.get("REDIS_URL", "redis://localhost:6379/0"))
    ap.add_argument("--data-dir", type=Path, default=default_data_dir())
    ap.add_argument("--cutoff-day", type=int, default=VAL_END_DAY)
    args = ap.parse_args()
    start = time.time()
    info = publish(open_store(args.redis_url), args.data_dir, args.cutoff_day)
    print(f"published features for {info['riders']:,} riders as of day {info['cutoff_day']} "
          f"({info['version']}) in {time.time() - start:.1f}s")


if __name__ == "__main__":
    main()
