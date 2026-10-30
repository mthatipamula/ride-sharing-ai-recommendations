"""Data drift report: is live traffic still shaped like the data the models were trained on?

    uv run rideshare-drift                      # training weeks vs the test week
    uv run rideshare-drift --simulate-shift     # same, with a demand shock injected, to show the alarm firing

Population Stability Index per feature: under 0.1 stable, 0.1-0.25 watch, above 0.25 investigate and retrain.
The service exposes the latest report as a Prometheus gauge, so Grafana shows it next to latency and A/B results.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from .features import TRAIN_END_DAY, VAL_END_DAY
from .serving.publish import default_data_dir
from .serving.recommender import latest_artifact_dir
from .simulator import load_or_simulate

ML_DIR = Path(__file__).resolve().parents[2]
WATCH, ALERT = 0.1, 0.25
NUMERIC = {"distance_km": 10, "surge": 8, "price_standard": 10}
CATEGORICAL = ["hour", "dow", "group_size", "mode", "dest_category"]


def psi(expected: np.ndarray, actual: np.ndarray, eps: float = 1e-4) -> float:
    """Both are counts over the same bins."""
    e = np.maximum(expected / max(expected.sum(), 1), eps)
    a = np.maximum(actual / max(actual.sum(), 1), eps)
    return float(((a - e) * np.log(a / e)).sum())


def psi_numeric(base: pd.Series, cur: pd.Series, bins: int) -> float:
    edges = np.unique(np.quantile(base, np.linspace(0, 1, bins + 1)))
    edges[0], edges[-1] = -np.inf, np.inf
    return psi(np.histogram(base, edges)[0].astype(float), np.histogram(cur, edges)[0].astype(float))


def psi_categorical(base: pd.Series, cur: pd.Series) -> float:
    keys = sorted(set(base.unique()) | set(cur.unique()), key=str)
    return psi(base.value_counts().reindex(keys).fillna(0).to_numpy(float),
               cur.value_counts().reindex(keys).fillna(0).to_numpy(float))


def status(value: float) -> str:
    return "alert" if value > ALERT else "watch" if value > WATCH else "stable"


def inject_shift(cur: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """A demand shock: fares surge, and half of Standard riders move to XL (a large event in town)."""
    cur = cur.copy()
    cur["surge"] = (cur.surge * 1.5).clip(upper=3.0)
    cur["price_standard"] = cur.price_standard * 1.5
    moved = (cur["mode"] == "standard") & (rng.random(len(cur)) < 0.5)
    cur.loc[moved, "mode"] = "xl"
    return cur


def report(trips: pd.DataFrame, places: pd.DataFrame, simulate_shift: bool = False, seed: int = 7) -> dict:
    t = trips.assign(dest_category=trips.dest_place_id.map(places.set_index("place_id").category))
    base, cur = t[t.day < TRAIN_END_DAY], t[t.day >= VAL_END_DAY]
    if simulate_shift:
        cur = inject_shift(cur, np.random.default_rng(seed))
    values = {f: psi_numeric(base[f], cur[f], bins) for f, bins in NUMERIC.items()}
    values |= {f: psi_categorical(base[f], cur[f]) for f in CATEGORICAL}
    features = {f: {"psi": round(v, 4), "status": status(v)} for f, v in sorted(values.items(), key=lambda kv: -kv[1])}
    return {"baseline": f"days 0-{TRAIN_END_DAY - 1}", "current": f"days {VAL_END_DAY}+", "simulated_shift": simulate_shift,
            "baseline_trips": len(base), "current_trips": len(cur), "features": features,
            "alerts": [f for f, v in features.items() if v["status"] == "alert"]}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--artifacts", type=Path, default=Path(os.environ.get("ARTIFACTS_DIR", ML_DIR / "artifacts")))
    ap.add_argument("--data-dir", type=Path, default=default_data_dir())
    ap.add_argument("--simulate-shift", action="store_true")
    args = ap.parse_args()
    places, _, trips = load_or_simulate(args.data_dir)
    r = report(trips, places, args.simulate_shift)
    print(f"{'feature':<16}{'PSI':>8}  status")
    for f, v in r["features"].items():
        print(f"{f:<16}{v['psi']:>8.4f}  {v['status']}")
    print(f"\nalerts: {r['alerts'] or 'none'}")
    if not args.simulate_shift:                       # a drill must never overwrite the real report
        out = latest_artifact_dir(args.artifacts) / "drift_report.json"
        out.write_text(json.dumps(r, indent=2))
        print(f"wrote {out}")


if __name__ == "__main__":
    main()
