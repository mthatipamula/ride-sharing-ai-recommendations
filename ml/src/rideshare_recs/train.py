"""End-to-end offline pipeline: simulate -> features -> two-tower -> ranker -> ride-mode -> evaluate -> registry.

    uv run rideshare-train                 # full run (about 5-10 minutes on a laptop CPU)
    uv run rideshare-train --riders 800    # quick smoke run

Writes a versioned artifact directory (models, place embeddings, metrics, model card) and points
artifacts/LATEST at it, which is what the serving layer loads.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from . import evaluation as ev
from .features import (RANKER_FEATURES, VAL_END_DAY, TRAIN_END_DAY, Snapshot, candidate_features,
                       destination_examples, place_arrays, split_of)
from .models import ranker as rk
from .models import ride_mode as rm
from .models import two_tower as tt
from .simulator import CATEGORIES, MODE_CAPACITY, PERSONAS, RIDE_MODES, SimConfig, load_or_simulate

ML_DIR = Path(__file__).resolve().parents[2]
K = 50   # candidates passed from retrieval to the ranker


def destination_pipeline(places, riders, trips, args, log) -> tuple[dict, tt.TwoTower, object]:
    ex = destination_examples(trips, riders)
    ex = ex[ex.origin != ex.dest].reset_index(drop=True)
    train_ex, val_ex, test_ex = (ex[ex.split == s].reset_index(drop=True) for s in ("train", "val", "test"))
    log(f"destination examples: train {len(train_ex):,}  val {len(val_ex):,}  test {len(test_ex):,}")

    cats, xy = place_arrays(places)
    cfg = tt.TwoTowerConfig(n_riders=len(riders), n_places=len(cats), n_categories=len(CATEGORIES),
                            n_personas=len(PERSONAS))
    model, tt_history = tt.train_two_tower(tt.TwoTower(cfg, cats, xy), train_ex, val_ex, epochs=args.epochs,
                                           seed=args.seed, log=log)

    # stage 2: ranker trained on week 7 with a week-6 snapshot, evaluated on week 8 with a week-7 snapshot
    val_c, val_s, _ = tt.retrieve(model, val_ex, K)
    train_frame = candidate_features(val_ex, val_c, val_s, Snapshot.build(trips, places, TRAIN_END_DAY), places, riders)
    labels = (train_frame.place_id.to_numpy() == np.repeat(val_ex.dest.to_numpy(), K)).astype(int)
    ranker, ranker_history = rk.train_ranker(train_frame, labels, K, seed=args.seed, log=log)

    test_snap = Snapshot.build(trips, places, VAL_END_DAY)
    test_c, test_s, tt_ranks = tt.retrieve(model, test_ex, K)
    test_frame = candidate_features(test_ex, test_c, test_s, test_snap, places, riders)
    truth = test_ex.dest.to_numpy()

    results = {}
    for kind in ("popularity", "frequency", "frequency_by_time"):
        scores = ev.baseline_scores(test_ex, test_snap, len(riders), len(cats), kind)
        results[f"baseline_{kind}"] = ev.ranking_metrics(ev.rank_of_truth(scores, truth))
    results["two_tower"] = ev.ranking_metrics(tt_ranks)
    ranker_scores = rk.rerank(ranker, test_frame, K)
    final = ev.final_ranks(test_c, ranker_scores, truth)
    results["two_tower+ranker"] = ev.ranking_metrics(final)
    results["retrieval_recall@50"] = float(np.isfinite(ev.final_ranks(test_c, test_s, truth)).mean())

    # where the gain comes from: by rider history length
    hist_len = (np.stack(test_ex["hist"].to_numpy()) > 0).sum(1)
    freq = ev.rank_of_truth(ev.baseline_scores(test_ex, test_snap, len(riders), len(cats), "frequency_by_time"), truth)
    segments = {}
    for name, m in {"history<5": hist_len < 5, "history>=5": hist_len >= 5}.items():
        if m.any():
            segments[name] = {"queries": int(m.sum()), "model_recall@1": float((final[m] <= 1).mean()),
                              "frequency_by_time_recall@1": float((freq[m] <= 1).mean())}
    hit = test_c == truth[:, None]
    truth_pos = np.where(hit.any(1), hit.argmax(1), -1)
    return {"destination": results, "destination_segments": segments, "two_tower_training": tt_history,
            "ranker_training": ranker_history,
            "ranker_feature_importance": rk.permutation_importance(ranker, test_frame, K, truth_pos)}, model, ranker


def ride_mode_pipeline(riders, trips, args, log) -> tuple[dict, rm.RideModeModel]:
    persona_ix = {p: i for i, p in enumerate(PERSONAS)}
    df = trips.merge(riders[["rider_id", "persona"]], on="rider_id")
    train_mix = df[df.day < TRAIN_END_DAY]["mode"].value_counts(normalize=True).reindex(RIDE_MODES).to_numpy()
    df = rm.add_prior_mode_shares(df, train_mix)
    split = split_of(df.day.to_numpy())
    train, val, test = (df[split == s].reset_index(drop=True) for s in ("train", "val", "test"))
    dense = rm.dense_matrix(train)
    model = rm.RideModeModel(rm.RideModeConfig(n_riders=len(riders), n_personas=len(PERSONAS)),
                             dense.mean(0), dense.std(0) + 1e-6)
    model, history = rm.train_ride_mode(model, rm.to_batch(train, persona_ix), rm.to_batch(val, persona_ix),
                                        seed=args.seed, log=log)

    test_b = rm.to_batch(test, persona_ix)
    labels = test_b["mode"].numpy()
    feasible = test.group_size.to_numpy()[:, None] <= MODE_CAPACITY[None, :]

    # baselines: global mode shares, and each rider's own history (smoothed), both restricted to feasible modes
    hist = pd.concat([train, val])
    global_share = hist["mode"].value_counts(normalize=True).reindex(RIDE_MODES).to_numpy()
    counts = pd.crosstab(hist.rider_id, hist["mode"]).reindex(index=range(len(riders)), columns=RIDE_MODES).fillna(0)
    personal = counts.to_numpy()[test.rider_id.to_numpy()] + 2.0 * global_share
    results = {}
    for name, p in {"baseline_global_share": np.tile(global_share, (len(test), 1)),
                    "baseline_rider_history": personal,
                    "dlrm": rm.predict_proba(model, test_b)}.items():
        p = np.where(feasible, p, 0.0)
        results[name] = ev.classification_metrics(p / p.sum(1, keepdims=True), labels)
    return {"ride_mode": results, "ride_mode_training": history,
            "ride_mode_global_share": dict(zip(RIDE_MODES, train_mix.round(6).tolist()))}, model


def _model_card(metrics: dict, version: str) -> str:
    d, r = metrics["destination"], metrics["ride_mode"]
    rows = "\n".join(f"| {k} | {v['recall@1']:.3f} | {v['recall@3']:.3f} | {v['recall@10']:.3f} | {v['ndcg@10']:.3f} "
                     f"| {v['mrr']:.3f} |" for k, v in d.items() if isinstance(v, dict))
    mode_rows = "\n".join(f"| {k} | {v['accuracy']:.3f} | {v['log_loss']:.3f} | {v['macro_auc']:.3f} | {v['ece']:.3f} |"
                          for k, v in r.items())
    return f"""# Model card: {version}

Synthetic data only (see `simulator.py`). Time-based split: weeks 1-6 train retrieval, week 7 trains the ranker,
week 8 is the test week.

## "Where to?" destination recommendations (test week)

| Model | Recall@1 | Recall@3 | Recall@10 | NDCG@10 | MRR |
|---|---|---|---|---|---|
{rows}

Retrieval recall@{K} (ceiling for the ranker): {d['retrieval_recall@50']:.3f}

## Ride-mode conversion (test week)

| Model | Accuracy | Log loss | Macro AUC | ECE |
|---|---|---|---|---|
{mode_rows}

## Ranker feature importance (drop in top-1 accuracy when the feature is shuffled)

{json.dumps(metrics['ranker_feature_importance'], indent=2)}
"""


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--riders", type=int, default=5000)
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--data-dir", type=Path, default=ML_DIR / "data")
    ap.add_argument("--artifacts", type=Path, default=ML_DIR / "artifacts")
    args = ap.parse_args()
    torch.set_num_threads(max(1, torch.get_num_threads()))
    start = time.time()
    log = lambda msg: print(f"[{time.time() - start:6.1f}s] {msg}", flush=True)  # noqa: E731

    cfg = SimConfig(n_riders=args.riders, n_homes=max(100, args.riders // 5), seed=args.seed)
    data_dir = args.data_dir / f"riders-{args.riders}-seed-{args.seed}"
    places, riders, trips = load_or_simulate(data_dir, cfg)
    log(f"data: {len(places):,} places, {len(riders):,} riders, {len(trips):,} trips")

    dest_metrics, two_tower, ranker = destination_pipeline(places, riders, trips, args, log)
    mode_metrics, ride_mode = ride_mode_pipeline(riders, trips, args, log)
    metrics = {**dest_metrics, **mode_metrics, "data": {"places": len(places), "riders": len(riders),
                                                        "trips": len(trips), "seed": args.seed}}

    version = time.strftime("v%Y%m%d-%H%M%S")
    out = args.artifacts / version
    out.mkdir(parents=True)
    tt.save(two_tower, out / "two_tower.pt")
    np.save(out / "place_embeddings.npy", two_tower.encode_items().detach().numpy())
    rk.save(ranker, out / "ranker.pt")
    rm.save(ride_mode, out / "ride_mode.pt")
    places.to_parquet(out / "places.parquet", index=False)
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))
    (out / "model_card.md").write_text(_model_card(metrics, version))
    (out / "ranker_features.json").write_text(json.dumps(RANKER_FEATURES))
    (out / "serving_config.json").write_text(json.dumps({
        "version": version, "candidates": K, "sim_start": cfg.start,
        "ride_mode_global_share": metrics["ride_mode_global_share"],
    }, indent=2))
    (args.artifacts / "LATEST").write_text(version)
    log(f"wrote {out}")
    print(_model_card(metrics, version))


if __name__ == "__main__":
    main()
