"""Offline metrics and the baselines every model has to beat."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import log_loss, roc_auc_score

from .features import Snapshot


def rank_of_truth(scores: np.ndarray, truth: np.ndarray) -> np.ndarray:
    """1-based rank of the true item; ties count half so a constant scorer gets no free credit."""
    t = scores[np.arange(len(truth)), truth][:, None]
    return 1.0 + (scores > t).sum(1) + 0.5 * ((scores == t).sum(1) - 1)


def ranking_metrics(ranks: np.ndarray, ks=(1, 3, 10)) -> dict[str, float]:
    """ranks: 1-based, np.inf when the true item was never retrieved."""
    out = {f"recall@{k}": float(np.mean(ranks <= k)) for k in ks}
    out["ndcg@10"] = float(np.mean(np.where(ranks <= 10, 1.0 / np.log2(ranks + 1.0), 0.0)))
    out["mrr"] = float(np.mean(np.where(np.isfinite(ranks), 1.0 / ranks, 0.0)))
    return out


def baseline_scores(queries, snap: Snapshot, n_riders: int, n_places: int, kind: str) -> np.ndarray:
    """Scores over every place for: global popularity, the rider's most-visited places, or most-visited at this
    time of day. The personal-frequency baselines are strong; a model that can't beat them shouldn't ship."""
    pop = np.zeros(n_places, dtype=np.float32)
    pop[snap.popularity.index.to_numpy()] = snap.popularity.to_numpy()
    pop /= pop.max()
    if kind == "popularity":
        scores = np.tile(pop, (len(queries), 1))
    else:
        rp = np.zeros((n_riders, n_places), dtype=np.float32)
        np.add.at(rp, (snap.rider_place.rider_id, snap.rider_place.place_id), snap.rider_place.visits_total)
        scores = rp[queries.rider_id.to_numpy()] + 1e-3 * pop
        if kind == "frequency_by_time":
            rph = np.zeros((4, n_riders, n_places), dtype=np.float32)
            t = snap.rider_place_hbucket
            np.add.at(rph, (t.hbucket, t.rider_id, t.place_id), t.visits_hbucket)
            scores = rph[queries.hbucket.to_numpy(), queries.rider_id.to_numpy()] + 0.1 * scores
    scores[:, 0] = -np.inf
    scores[np.arange(len(queries)), queries.origin.to_numpy()] = -np.inf
    return scores


def expected_calibration_error(probs: np.ndarray, labels: np.ndarray, bins: int = 15) -> float:
    conf, pred = probs.max(1), probs.argmax(1)
    edges = np.linspace(0, 1, bins + 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            ece += m.mean() * abs((pred[m] == labels[m]).mean() - conf[m].mean())
    return float(ece)


def classification_metrics(probs: np.ndarray, labels: np.ndarray) -> dict[str, float]:
    probs = np.clip(probs, 1e-7, 1.0)
    probs = probs / probs.sum(1, keepdims=True)
    return {
        "accuracy": float((probs.argmax(1) == labels).mean()),
        "log_loss": float(log_loss(labels, probs, labels=np.arange(probs.shape[1]))),
        "macro_auc": float(roc_auc_score(labels, probs, multi_class="ovr", average="macro")),
        "ece": expected_calibration_error(probs, labels),
    }
