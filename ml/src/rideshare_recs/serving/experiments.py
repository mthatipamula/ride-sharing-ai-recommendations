"""A/B assignment: deterministic, sticky and independent per experiment.

The unit is the rider, not the request: a rider sees one arm for the whole experiment, so their behaviour
isn't a mix of both. Hashing (experiment, rider) means no assignment table, the same answer on every pod, and a new
experiment reshuffles riders instead of reusing the last experiment's split.
"""
from __future__ import annotations

import dataclasses
import hashlib
import os

BUCKETS = 10_000


@dataclasses.dataclass(frozen=True)
class Experiment:
    name: str = "where_to_two_stage_v1"
    treatment_share: float = 0.5        # control: frequency-by-time heuristic, treatment: two-tower + ranker

    @classmethod
    def from_env(cls) -> "Experiment":
        return cls(os.environ.get("EXPERIMENT_NAME", cls.name),
                   float(os.environ.get("EXPERIMENT_TREATMENT_SHARE", cls.treatment_share)))

    def bucket(self, rider_id: int) -> int:
        digest = hashlib.sha256(f"{self.name}:{rider_id}".encode()).digest()
        return int.from_bytes(digest[:8], "big") % BUCKETS

    def assign(self, rider_id: int) -> str:
        return "treatment" if self.bucket(rider_id) < self.treatment_share * BUCKETS else "control"
