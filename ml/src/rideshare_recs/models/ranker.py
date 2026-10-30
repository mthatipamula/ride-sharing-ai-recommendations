"""Second stage: a listwise neural ranker re-orders the two-tower's top candidates using personal, temporal and
geographic features the retrieval model can't afford to compute for every place.

Each query is a list of K candidates with exactly one positive (the place the rider went); the loss is softmax
cross-entropy over the list (ListNet with a one-hot target), which optimises the top of the ranking directly.
"""
from __future__ import annotations

import copy
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn

from ..features import RANKER_FEATURES
from ..simulator import CATEGORIES

NUMERIC = [f for f in RANKER_FEATURES if f != "category"]
LOG_SCALED = {"visits_total", "visits_hbucket", "visits_daytype", "popularity", "distance_km", "tt_rank"}


@dataclass
class RankerConfig:
    n_features: int = len(NUMERIC)
    n_categories: int = len(CATEGORIES)
    hidden: int = 128


class ListwiseRanker(nn.Module):
    def __init__(self, cfg: RankerConfig, mean: np.ndarray, std: np.ndarray):
        super().__init__()
        self.cfg = cfg
        self.category_emb = nn.Embedding(cfg.n_categories, 8)
        self.mlp = nn.Sequential(nn.Linear(cfg.n_features + 8, cfg.hidden), nn.ReLU(),
                                 nn.Linear(cfg.hidden, cfg.hidden // 2), nn.ReLU(), nn.Linear(cfg.hidden // 2, 1))
        self.register_buffer("mean", torch.as_tensor(mean, dtype=torch.float32))
        self.register_buffer("std", torch.as_tensor(std, dtype=torch.float32))

    def forward(self, x: torch.Tensor, category: torch.Tensor) -> torch.Tensor:
        """x: (..., n_features) raw numeric features, category: (...) -> scores (...)."""
        z = torch.cat([(x - self.mean) / self.std, self.category_emb(category)], dim=-1)
        return self.mlp(z).squeeze(-1)


def tensors(frame: pd.DataFrame, k: int) -> tuple[torch.Tensor, torch.Tensor]:
    """(n_queries, k, n_features) and (n_queries, k) from the long candidate frame."""
    x = frame[NUMERIC].to_numpy(np.float32).copy()
    for j, f in enumerate(NUMERIC):
        if f in LOG_SCALED:
            x[:, j] = np.log1p(x[:, j])
    return (torch.as_tensor(x).view(-1, k, len(NUMERIC)),
            torch.as_tensor(frame.category.to_numpy()).view(-1, k))


def train_ranker(frame: pd.DataFrame, labels: np.ndarray, k: int, epochs: int = 12, batch_size: int = 512,
                 lr: float = 2e-3, seed: int = 7, log=print) -> tuple[ListwiseRanker, list[dict]]:
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    x, cat = tensors(frame, k)
    y = torch.as_tensor(labels.reshape(-1, k))
    keep = y.sum(1) > 0                        # queries whose destination wasn't retrieved carry no signal
    x, cat, target = x[keep], cat[keep], y[keep].argmax(1)

    n = len(target)
    order = torch.as_tensor(rng.permutation(n))
    hold, fit = order[: n // 10], order[n // 10:]          # 10% of queries for early stopping
    flat = x[fit].reshape(-1, x.shape[-1])
    model = ListwiseRanker(RankerConfig(), flat.mean(0).numpy(), flat.std(0).numpy() + 1e-6)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    best, best_loss, history = None, float("inf"), []
    for epoch in range(1, epochs + 1):
        model.train()
        perm = fit[torch.as_tensor(rng.permutation(len(fit)))]
        for i in range(0, len(perm), batch_size):
            idx = perm[i:i + batch_size]
            loss = F.cross_entropy(model(x[idx], cat[idx]), target[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            s = model(x[hold], cat[hold])
            hold_loss = F.cross_entropy(s, target[hold]).item()
            hold_top1 = (s.argmax(1) == target[hold]).float().mean().item()
        history.append({"epoch": epoch, "holdout_loss": hold_loss, "holdout_top1": hold_top1})
        log(f"  ranker epoch {epoch}: holdout loss {hold_loss:.4f}  top-1 {hold_top1:.3f}")
        if hold_loss < best_loss:
            best_loss, best = hold_loss, copy.deepcopy(model.state_dict())
    model.load_state_dict(best)
    return model.eval(), history


@torch.no_grad()
def rerank(model: ListwiseRanker, frame: pd.DataFrame, k: int) -> np.ndarray:
    """(n_queries, k) ranker scores aligned with the candidate matrix the frame was built from."""
    x, cat = tensors(frame, k)
    return model.eval()(x, cat).numpy()


@torch.no_grad()
def permutation_importance(model: ListwiseRanker, frame: pd.DataFrame, k: int, truth_pos: np.ndarray,
                           seed: int = 7) -> dict[str, float]:
    """Drop in top-1 accuracy when one feature is shuffled across queries (only queries with a retrieved positive)."""
    rng = np.random.default_rng(seed)
    x, cat = tensors(frame, k)
    ok = truth_pos >= 0
    x, cat, truth = x[ok], cat[ok], torch.as_tensor(truth_pos[ok])
    base = (model(x, cat).argmax(1) == truth).float().mean().item()
    out = {}
    for j, f in enumerate(NUMERIC + ["category"]):
        xs, cs = x.clone(), cat.clone()
        perm = torch.as_tensor(rng.permutation(len(xs)))
        if f == "category":
            cs = cs[perm]
        else:
            xs[..., j] = xs[perm][..., j]
        out[f] = round(base - (model(xs, cs).argmax(1) == truth).float().mean().item(), 4)
    return dict(sorted(out.items(), key=lambda kv: -kv[1]))


def save(model: ListwiseRanker, path) -> None:
    torch.save({"config": asdict(model.cfg), "state_dict": model.state_dict()}, path)


def load(path) -> ListwiseRanker:
    blob = torch.load(path, weights_only=False)
    sd = blob["state_dict"]
    model = ListwiseRanker(RankerConfig(**blob["config"]), sd["mean"].numpy(), sd["std"].numpy())
    model.load_state_dict(sd)
    return model.eval()
