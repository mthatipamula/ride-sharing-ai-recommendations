"""A/B experiment readout: two-stage model (treatment) vs the frequency-by-time heuristic (control).

    uv run rideshare-abtest

Replays the test week as if the experiment had been live: riders are split by the same hash the service uses, each
arm ranks every trip's destinations, and the outcome is whether the rider's actual destination was the **first
suggestion** (primary) or among the **top 3 shortcuts** (secondary).

The statistics are the part that's easy to get wrong:
- **Randomised by rider, measured per trip.** Trips from one rider are correlated, so a naive two-proportion z-test
  over trips understates the variance. The readout uses the **delta method** for the per-trip ratio clustered by
  rider, and reports the naive version alongside to show the gap.
- **Sample-ratio mismatch** check before reading any metric.
- **Power**: minimum detectable effect for this sample, and riders needed for a 1% relative lift.
- **A/A calibration**: hundreds of random control-vs-control splits should reject at ~5%; if they don't, the test is
  miscalibrated.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from . import evaluation as ev
from .features import VAL_END_DAY, Snapshot, candidate_features, destination_examples
from .models import ranker as rk
from .models import two_tower as tt
from .serving.experiments import Experiment
from .serving.publish import default_data_dir
from .serving.recommender import latest_artifact_dir
from .simulator import SimConfig, load_or_simulate

ML_DIR = Path(__file__).resolve().parents[2]
Z_975, Z_80 = stats.norm.ppf(0.975), stats.norm.ppf(0.80)


def ratio_stats(successes: np.ndarray, trials: np.ndarray) -> tuple[float, float]:
    """Pooled per-trip rate and its variance, clustered by rider (delta method)."""
    n = len(trials)
    mean_s, mean_n = successes.mean(), trials.mean()
    r = mean_s / mean_n
    var = (np.var(successes, ddof=1) - 2 * r * np.cov(successes, trials, ddof=1)[0, 1]
           + r ** 2 * np.var(trials, ddof=1)) / (n * mean_n ** 2)
    return float(r), float(var)


def compare(per_rider: pd.DataFrame, metric: str) -> dict:
    """Treatment vs control on a per-trip rate, analysed at the rider level."""
    out = {}
    arms = {a: per_rider[per_rider.variant == a] for a in ("control", "treatment")}
    (rc, vc), (rt, vt) = (ratio_stats(arms[a][metric].to_numpy(), arms[a].trips.to_numpy()) for a in arms)
    diff, se = rt - rc, np.sqrt(vc + vt)
    out["control_rate"], out["treatment_rate"] = rc, rt
    out["absolute_lift"], out["relative_lift"] = diff, diff / rc
    out["ci95_absolute"] = [diff - Z_975 * se, diff + Z_975 * se]
    out["ci95_relative"] = [(diff - Z_975 * se) / rc, (diff + Z_975 * se) / rc]
    out["z"] = diff / se
    out["p_value"] = float(2 * stats.norm.sf(abs(diff / se)))

    # naive: treat every trip as independent
    nc, nt = arms["control"].trips.sum(), arms["treatment"].trips.sum()
    naive_se = np.sqrt(rc * (1 - rc) / nc + rt * (1 - rt) / nt)
    out["naive_trip_level_se"], out["clustered_se"] = float(naive_se), float(se)
    out["design_effect"] = float((se / naive_se) ** 2)

    # power for this sample size
    out["mde_absolute"] = float((Z_975 + Z_80) * se)
    out["mde_relative"] = float(out["mde_absolute"] / rc)
    riders_now = len(per_rider)
    out["riders_for_1pct_relative_lift"] = int(np.ceil(riders_now * (out["mde_absolute"] / (0.01 * rc)) ** 2))
    return out


def aa_false_positive_rate(control: pd.DataFrame, metric: str, runs: int, seed: int) -> float:
    rng = np.random.default_rng(seed)
    rejections = 0
    for _ in range(runs):
        side = rng.random(len(control)) < 0.5
        (ra, va), (rb, vb) = (ratio_stats(control[m][metric].to_numpy(), control[m].trips.to_numpy())
                              for m in (side, ~side))
        rejections += abs(rb - ra) / np.sqrt(va + vb) > Z_975
    return rejections / runs


def run(artifact_dir: Path, data_dir: Path, experiment: Experiment, aa_runs: int = 500, seed: int = 7) -> dict:
    riders_n = int(data_dir.name.split("-")[1])
    places, riders, trips = load_or_simulate(data_dir, SimConfig(n_riders=riders_n, n_homes=max(100, riders_n // 5)))
    ex = destination_examples(trips, riders)
    test = ex[(ex.split == "test") & (ex.origin != ex.dest)].reset_index(drop=True)
    snap = Snapshot.build(trips, places, VAL_END_DAY)
    truth = test.dest.to_numpy()
    k = json.loads((artifact_dir / "serving_config.json").read_text())["candidates"]

    model, ranker = tt.load(artifact_dir / "two_tower.pt"), rk.load(artifact_dir / "ranker.pt")
    cands, scores, _ = tt.retrieve(model, test, k)
    frame = candidate_features(test, cands, scores, snap, places, riders)
    treatment_rank = ev.final_ranks(cands, rk.rerank(ranker, frame, k), truth)
    control_rank = ev.rank_of_truth(ev.baseline_scores(test, snap, len(riders), int(places.place_id.max()) + 1,
                                                       "frequency_by_time"), truth)

    test["variant"] = [experiment.assign(int(r)) for r in test.rider_id]
    test["rank"] = np.where(test.variant == "treatment", treatment_rank, control_rank)
    test["first_suggestion"] = (test["rank"] <= 1).astype(int)
    test["top3"] = (test["rank"] <= 3).astype(int)
    per_rider = test.groupby(["rider_id", "variant"]).agg(trips=("trip_id", "size"),
                                                          first_suggestion=("first_suggestion", "sum"),
                                                          top3=("top3", "sum")).reset_index()

    counts = per_rider.variant.value_counts().reindex(["control", "treatment"]).fillna(0).to_numpy()
    expected = np.array([1 - experiment.treatment_share, experiment.treatment_share]) * counts.sum()
    srm_p = float(stats.chisquare(counts, expected).pvalue)
    control = per_rider[per_rider.variant == "control"].reset_index(drop=True)
    return {
        "experiment": experiment.name,
        "riders": {"control": int(counts[0]), "treatment": int(counts[1])},
        "trips": {a: int(per_rider[per_rider.variant == a].trips.sum()) for a in ("control", "treatment")},
        "sample_ratio_mismatch_p": srm_p,
        "srm_ok": srm_p > 0.001,
        "primary_first_suggestion": compare(per_rider, "first_suggestion"),
        "secondary_top3": compare(per_rider, "top3"),
        "aa_false_positive_rate": aa_false_positive_rate(control, "first_suggestion", aa_runs, seed),
        "aa_runs": aa_runs,
    }


def _pct(x: float) -> str:
    return f"{100 * x:+.2f}%"


def report(r: dict) -> str:
    lines = [f"Experiment {r['experiment']}",
             f"  riders  control {r['riders']['control']:,}  treatment {r['riders']['treatment']:,}   "
             f"trips  control {r['trips']['control']:,}  treatment {r['trips']['treatment']:,}",
             f"  sample ratio mismatch p = {r['sample_ratio_mismatch_p']:.3f}  "
             f"({'ok' if r['srm_ok'] else 'MISMATCH: do not read the metrics'})"]
    for name, key in (("Primary: destination was the first suggestion", "primary_first_suggestion"),
                      ("Secondary: destination in the top 3", "secondary_top3")):
        m = r[key]
        lines += [f"\n{name}",
                  f"  control {m['control_rate']:.4f}  treatment {m['treatment_rate']:.4f}  "
                  f"lift {_pct(m['relative_lift'])}  (95% CI {_pct(m['ci95_relative'][0])} to "
                  f"{_pct(m['ci95_relative'][1])})  p = {m['p_value']:.2g}",
                  f"  rider-clustered SE {m['clustered_se']:.5f} vs naive trip-level SE {m['naive_trip_level_se']:.5f}"
                  f"  (design effect {m['design_effect']:.2f})",
                  f"  minimum detectable effect at 80% power: {_pct(m['mde_relative'])} relative; "
                  f"riders for a 1% relative lift: {m['riders_for_1pct_relative_lift']:,}"]
    lines.append(f"\nA/A calibration: {r['aa_runs']} random control/control splits rejected at "
                 f"{100 * r['aa_false_positive_rate']:.1f}% (expected ~5%)")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--artifacts", type=Path, default=Path(os.environ.get("ARTIFACTS_DIR", ML_DIR / "artifacts")))
    ap.add_argument("--data-dir", type=Path, default=default_data_dir())
    ap.add_argument("--aa-runs", type=int, default=500)
    args = ap.parse_args()
    artifact_dir = latest_artifact_dir(args.artifacts)
    result = run(artifact_dir, args.data_dir, Experiment.from_env(), args.aa_runs)
    (artifact_dir / "ab_test.json").write_text(json.dumps(result, indent=2))
    print(report(result))
    print(f"\nwrote {artifact_dir / 'ab_test.json'}")


if __name__ == "__main__":
    main()
