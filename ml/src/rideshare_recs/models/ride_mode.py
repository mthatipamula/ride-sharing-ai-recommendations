"""Ride-mode conversion model: which of Wait & Save / Standard / XL / Lux will this rider pick for this trip?

DLRM-style: sparse features (rider, segment, hour, day) become embeddings, dense features (distance, surge, group
size, luggage, airport, the four quoted prices) go through a bottom MLP, then pairwise dot products between all the
vectors feed a top MLP that scores each mode. Modes that can't seat the group are masked, never just down-ranked.
"""
from __future__ import annotations

import copy
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn

from ..simulator import MODE_CAPACITY, RIDE_MODES

DENSE = (["log_distance", "surge", "group_size", "luggage", "is_airport"] + [f"log_price_{m}" for m in RIDE_MODES]
         + [f"share_{m}" for m in RIDE_MODES] + ["log_prior_trips"])
PRIOR_STRENGTH = 2.0   # pseudo-trips of the global mix blended into each rider's own mode shares


def add_prior_mode_shares(trips: pd.DataFrame, global_share: np.ndarray) -> pd.DataFrame:
    """Each rider's ride-mode mix over the trips strictly before this one, smoothed toward the global mix. Online,
    this is a per-rider counter in the feature store, incremented when a ride completes."""
    t = trips.sort_values(["rider_id", "ts"], kind="stable")
    onehot = pd.get_dummies(t["mode"]).reindex(columns=RIDE_MODES, fill_value=0).astype(np.float32)
    prior = onehot.groupby(t.rider_id.to_numpy()).cumsum() - onehot
    n = prior.sum(1).to_numpy()
    shares = (prior.to_numpy() + PRIOR_STRENGTH * global_share) / (n + PRIOR_STRENGTH)[:, None]
    out = t.copy()
    for j, m in enumerate(RIDE_MODES):
        out[f"share_{m}"] = shares[:, j]
    out["log_prior_trips"] = np.log1p(n)
    return out.sort_index()


@dataclass
class RideModeConfig:
    n_riders: int
    n_personas: int
    dim: int = 16
    n_modes: int = len(RIDE_MODES)


class RideModeModel(nn.Module):
    def __init__(self, cfg: RideModeConfig, dense_mean: np.ndarray, dense_std: np.ndarray):
        super().__init__()
        self.cfg = cfg
        d = cfg.dim
        self.sparse = nn.ModuleList([nn.Embedding(cfg.n_riders, d), nn.Embedding(cfg.n_personas, d),
                                     nn.Embedding(24, d), nn.Embedding(7, d)])
        self.bottom = nn.Sequential(nn.Linear(len(DENSE), 64), nn.ReLU(), nn.Linear(64, d))
        n_vec = len(self.sparse) + 1
        n_pairs = n_vec * (n_vec - 1) // 2
        self.top = nn.Sequential(nn.Linear(d + n_pairs, 64), nn.ReLU(), nn.Linear(64, cfg.n_modes))
        self.register_buffer("dense_mean", torch.as_tensor(dense_mean, dtype=torch.float32))
        self.register_buffer("dense_std", torch.as_tensor(dense_std, dtype=torch.float32))
        self.register_buffer("capacity", torch.as_tensor(MODE_CAPACITY, dtype=torch.float32))

    def forward(self, b: dict[str, torch.Tensor]) -> torch.Tensor:
        dense = self.bottom((b["dense"] - self.dense_mean) / self.dense_std)
        vecs = torch.stack([dense] + [emb(b[k]) for emb, k in zip(self.sparse, ("rider", "persona", "hour", "dow"))], 1)
        inter = vecs @ vecs.transpose(1, 2)
        i, j = torch.triu_indices(vecs.shape[1], vecs.shape[1], offset=1)
        logits = self.top(torch.cat([dense, inter[:, i, j]], dim=1))
        group = b["dense"][:, DENSE.index("group_size")].unsqueeze(1)
        return logits.masked_fill(group > self.capacity, float("-inf"))


def dense_matrix(df: pd.DataFrame) -> np.ndarray:
    cols = [np.log1p(df.distance_km), df.surge, df.group_size, df.luggage, df.is_airport]
    cols += [np.log(df[f"price_{m}"]) for m in RIDE_MODES]
    cols += [df[f"share_{m}"] for m in RIDE_MODES] + [df.log_prior_trips]
    return np.stack([np.asarray(c, dtype=np.float32) for c in cols], axis=1)


def to_batch(df: pd.DataFrame, persona_ix: dict[str, int]) -> dict[str, torch.Tensor]:
    b = {
        "dense": torch.as_tensor(dense_matrix(df)),
        "rider": torch.as_tensor(df.rider_id.to_numpy().copy()),
        "persona": torch.as_tensor(df.persona.map(persona_ix).to_numpy().copy()),
        "hour": torch.as_tensor(df.hour.to_numpy().copy()),
        "dow": torch.as_tensor(df.dow.to_numpy().copy()),
    }
    if "mode" in df:
        b["mode"] = torch.as_tensor(df["mode"].map({m: i for i, m in enumerate(RIDE_MODES)}).to_numpy())
    return b


@torch.no_grad()
def predict_proba(model: RideModeModel, b: dict[str, torch.Tensor]) -> np.ndarray:
    model.eval()
    return torch.softmax(model(b), dim=1).numpy()


def train_ride_mode(model: RideModeModel, train_b, val_b, epochs: int = 8, batch_size: int = 2048,
                    lr: float = 3e-3, seed: int = 7, log=print) -> tuple[RideModeModel, list[dict]]:
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    n = len(train_b["mode"])
    best, best_loss, history = None, float("inf"), []
    for epoch in range(1, epochs + 1):
        model.train()
        order = torch.as_tensor(rng.permutation(n))
        for i in range(0, n, batch_size):
            idx = order[i:i + batch_size]
            mb = {k: v[idx] for k, v in train_b.items()}
            loss = F.cross_entropy(model(mb), mb["mode"])
            opt.zero_grad()
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            val_loss = F.cross_entropy(model(val_b), val_b["mode"]).item()
        history.append({"epoch": epoch, "val_logloss": val_loss})
        log(f"  ride-mode epoch {epoch}: val log loss {val_loss:.4f}")
        if val_loss < best_loss:
            best_loss, best = val_loss, copy.deepcopy(model.state_dict())
    model.load_state_dict(best)
    return model, history


def save(model: RideModeModel, path) -> None:
    torch.save({"config": asdict(model.cfg), "state_dict": model.state_dict()}, path)


def load(path) -> RideModeModel:
    blob = torch.load(path, weights_only=False)
    sd = blob["state_dict"]
    model = RideModeModel(RideModeConfig(**blob["config"]), sd["dense_mean"].numpy(), sd["dense_std"].numpy())
    model.load_state_dict(sd)
    return model.eval()
