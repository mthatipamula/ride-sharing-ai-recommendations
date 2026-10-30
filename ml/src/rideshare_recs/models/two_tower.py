"""Two-tower retrieval for "Where to?".

Query tower: rider + origin + time context + a one-layer transformer over the last HIST_LEN destinations
(SASRec-style sequence signal). Item tower: place id + category + location. Both are L2-normalised, so serving is a
dot product against precomputed place embeddings (an ANN index such as ScaNN at real catalogue sizes).

The catalogue here is ~1.2k places, so training uses the exact softmax over every place. At millions of items you'd
switch to in-batch negatives with logQ correction; the towers stay the same.
"""
from __future__ import annotations

import copy
import math
from dataclasses import asdict, dataclass

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from ..features import HIST_LEN


@dataclass
class TwoTowerConfig:
    n_riders: int
    n_places: int          # including padding row 0
    n_categories: int
    n_personas: int
    dim: int = 64
    hist_len: int = HIST_LEN


class TwoTower(nn.Module):
    def __init__(self, cfg: TwoTowerConfig, place_category: np.ndarray, place_xy: np.ndarray):
        super().__init__()
        self.cfg = cfg
        d = cfg.dim
        self.place_emb = nn.Embedding(cfg.n_places, d, padding_idx=0)   # shared by both towers
        self.rider_emb = nn.Embedding(cfg.n_riders, d)
        self.hour_emb = nn.Embedding(24, 16)
        self.dow_emb = nn.Embedding(7, 8)
        self.persona_emb = nn.Embedding(cfg.n_personas, 8)
        self.pos_emb = nn.Embedding(cfg.hist_len, d)
        layer = nn.TransformerEncoderLayer(d, nhead=4, dim_feedforward=2 * d, dropout=0.1, batch_first=True)
        self.history = nn.TransformerEncoder(layer, num_layers=1, enable_nested_tensor=False)
        self.query_mlp = nn.Sequential(nn.Linear(3 * d + 32, 128), nn.ReLU(), nn.Linear(128, d))

        self.category_emb = nn.Embedding(cfg.n_categories, 8)
        self.item_mlp = nn.Sequential(nn.Linear(d + 8 + 2, 128), nn.ReLU(), nn.Linear(128, d))
        self.log_scale = nn.Parameter(torch.tensor(math.log(20.0)))      # learned inverse temperature
        self.register_buffer("place_category", torch.as_tensor(place_category, dtype=torch.long))
        self.register_buffer("place_xy", torch.as_tensor(place_xy, dtype=torch.float32))

    def encode_items(self) -> torch.Tensor:
        ids = torch.arange(self.cfg.n_places, device=self.place_xy.device)
        x = torch.cat([self.place_emb(ids), self.category_emb(self.place_category), self.place_xy], dim=-1)
        return F.normalize(self.item_mlp(x), dim=-1)

    def encode_queries(self, rider, origin, hour, dow, persona, hist) -> torch.Tensor:
        pad = hist == 0
        pad[:, -1] = False                       # riders with no history still attend to one (zero) slot
        positions = torch.arange(hist.shape[1], device=hist.device)
        h = self.history(self.place_emb(hist) + self.pos_emb(positions), src_key_padding_mask=pad)
        keep = (~pad).unsqueeze(-1).float()
        pooled = (h * keep).sum(1) / keep.sum(1)
        x = torch.cat([self.rider_emb(rider), self.place_emb(origin), pooled,
                       self.hour_emb(hour), self.dow_emb(dow), self.persona_emb(persona)], dim=-1)
        return F.normalize(self.query_mlp(x), dim=-1)

    def scores(self, batch: dict[str, torch.Tensor], items: torch.Tensor | None = None) -> torch.Tensor:
        """(B, n_places) logits; padding and the origin itself are masked out."""
        items = self.encode_items() if items is None else items
        q = self.encode_queries(batch["rider"], batch["origin"], batch["hour"], batch["dow"], batch["persona"],
                                batch["hist"])
        logits = q @ items.T * self.log_scale.exp().clamp(max=100.0)
        logits[:, 0] = float("-inf")
        return logits.scatter(1, batch["origin"].unsqueeze(1), float("-inf"))


def tensorize(ex) -> dict[str, torch.Tensor]:
    """Columns of an examples frame as tensors, built once; batches are index slices of these."""
    out = {name: torch.as_tensor(ex[col].to_numpy().copy())
           for name, col in (("rider", "rider_id"), ("origin", "origin"), ("hour", "hour"), ("dow", "dow"),
                             ("persona", "persona"), ("dest", "dest"))}
    out["hist"] = torch.as_tensor(np.stack(ex["hist"].to_numpy()))
    return out


def take(t: dict[str, torch.Tensor], idx) -> dict[str, torch.Tensor]:
    return {k: v[idx] for k, v in t.items()}


@torch.no_grad()
def retrieve(model: TwoTower, ex, k: int = 50, batch_size: int = 4096) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Top-k place ids and scores per query, plus the full-catalogue rank of the true destination."""
    from ..evaluation import rank_of_truth

    model.eval()
    items = model.encode_items()
    t = tensorize(ex)
    ids, top_scores, ranks = [], [], []
    for i in range(0, len(ex), batch_size):
        b = take(t, slice(i, i + batch_size))
        s = model.scores(b, items)
        top = torch.topk(s, k, dim=1)
        ids.append(top.indices.numpy())
        top_scores.append(top.values.numpy())
        ranks.append(rank_of_truth(s.numpy(), b["dest"].numpy()))
    return np.concatenate(ids), np.concatenate(top_scores), np.concatenate(ranks)


def train_two_tower(model: TwoTower, train_ex, val_ex, epochs: int = 6, batch_size: int = 1024, lr: float = 2e-3,
                    seed: int = 7, log=print) -> tuple[TwoTower, list[dict]]:
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    best, best_recall, history = None, -1.0, []
    train_t = tensorize(train_ex)
    for epoch in range(1, epochs + 1):
        model.train()
        order = torch.as_tensor(rng.permutation(len(train_ex)))
        total = 0.0
        for i in range(0, len(order), batch_size):
            b = take(train_t, order[i:i + batch_size])
            loss = F.cross_entropy(model.scores(b), b["dest"])
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item() * len(b["dest"])
        recall = float((retrieve(model, val_ex, k=10)[2] <= 10).mean())
        history.append({"epoch": epoch, "train_loss": total / len(order), "val_recall@10": recall})
        log(f"  two-tower epoch {epoch}: loss {total / len(order):.3f}  val recall@10 {recall:.3f}")
        if recall > best_recall:
            best_recall, best = recall, copy.deepcopy(model.state_dict())
    model.load_state_dict(best)
    return model, history


def save(model: TwoTower, path) -> None:
    torch.save({"config": asdict(model.cfg), "state_dict": model.state_dict()}, path)


def load(path) -> TwoTower:
    blob = torch.load(path, weights_only=False)
    sd = blob["state_dict"]
    model = TwoTower(TwoTowerConfig(**blob["config"]), sd["place_category"].numpy(), sd["place_xy"].numpy())
    model.load_state_dict(sd)
    return model.eval()
