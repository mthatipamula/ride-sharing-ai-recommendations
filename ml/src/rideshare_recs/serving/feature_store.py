"""Online feature store: everything one recommendation request needs about a rider, in one round trip.

Two kinds of features, refreshed on different clocks:
- batch snapshot (nightly job, `publish.py`): visits per place overall / by time of day / weekday vs weekend, last
  visit day, global popularity and category-by-hour lift
- real-time (on every completed ride, `record_ride`): the last 10 destinations and ride-mode counts

Redis layout (Valkey locally, Memorystore on GCP):
    rider:{id}           HASH  persona, home, work                 profile
    rider:{id}:hist      LIST  last 10 destination ids, newest first
    rider:{id}:modes     HASH  mode -> rides
    rider:{id}:snap      HASH  t:{place} total, l:{place} last day, h:{place}:{bucket}, w:{place}:{weekend}
    snapshot:meta        HASH  cutoff_day, version, popularity (JSON), category_hour_lift (JSON)
    events:impressions   STREAM  what each request showed (experiment logging)
    events:outcomes      STREAM  rides completed, linked to the request that suggested them
"""
from __future__ import annotations

import dataclasses
import json
import time
from collections import defaultdict
from typing import Protocol

import numpy as np

from ..features import HIST_LEN

STREAM_MAXLEN = 100_000


@dataclasses.dataclass
class RiderFeatures:
    rider_id: int
    persona: str
    home: int
    work: int
    hist: list[int]                                   # oldest -> newest, at most HIST_LEN
    mode_counts: dict[str, int]
    visits: dict[int, tuple[int, int]]                # place -> (total, last_day)
    visits_hbucket: dict[tuple[int, int], int]        # (place, hour bucket) -> visits
    visits_daytype: dict[tuple[int, int], int]        # (place, weekend) -> visits


@dataclasses.dataclass
class GlobalFeatures:
    cutoff_day: int
    version: str
    popularity: dict[int, int]
    category_hour_lift: np.ndarray


class FeatureStore(Protocol):
    def ping(self) -> bool: ...
    def rider(self, rider_id: int) -> RiderFeatures | None: ...
    def global_features(self) -> GlobalFeatures | None: ...
    def record_ride(self, rider_id: int, dest: int, mode: str | None) -> None: ...
    def log_event(self, stream: str, event: dict) -> None: ...
    def publish(self, profiles: dict, hists: dict, modes: dict, snaps: dict, meta: dict) -> None: ...


def encode_snap(visits, hbucket, daytype) -> dict[str, int]:
    out: dict[str, int] = {}
    for place, (total, last_day) in visits.items():
        out[f"t:{place}"] = total
        out[f"l:{place}"] = last_day
    out.update({f"h:{p}:{b}": n for (p, b), n in hbucket.items()})
    out.update({f"w:{p}:{w}": n for (p, w), n in daytype.items()})
    return out


def decode_snap(raw: dict) -> tuple[dict, dict, dict]:
    totals, last, hb, wk = {}, {}, {}, {}
    for k, v in raw.items():
        k = k.decode() if isinstance(k, bytes) else k
        parts = k.split(":")
        n = int(v)
        if parts[0] == "t":
            totals[int(parts[1])] = n
        elif parts[0] == "l":
            last[int(parts[1])] = n
        elif parts[0] == "h":
            hb[(int(parts[1]), int(parts[2]))] = n
        elif parts[0] == "w":
            wk[(int(parts[1]), int(parts[2]))] = n
    return {p: (t, last.get(p, -99)) for p, t in totals.items()}, hb, wk


def _decode_meta(meta: dict) -> GlobalFeatures | None:
    if not meta:
        return None
    meta = {(k.decode() if isinstance(k, bytes) else k): (v.decode() if isinstance(v, bytes) else v)
            for k, v in meta.items()}
    return GlobalFeatures(int(meta["cutoff_day"]), meta["version"],
                          {int(k): v for k, v in json.loads(meta["popularity"]).items()},
                          np.asarray(json.loads(meta["category_hour_lift"])))


class RedisFeatureStore:
    def __init__(self, url: str):
        import redis

        self.r = redis.Redis.from_url(url, socket_timeout=0.05, socket_connect_timeout=0.5)

    def ping(self) -> bool:
        try:
            return bool(self.r.ping())
        except Exception:
            return False

    def rider(self, rider_id: int) -> RiderFeatures | None:
        p = self.r.pipeline(transaction=False)
        p.hgetall(f"rider:{rider_id}")
        p.lrange(f"rider:{rider_id}:hist", 0, HIST_LEN - 1)
        p.hgetall(f"rider:{rider_id}:modes")
        p.hgetall(f"rider:{rider_id}:snap")
        profile, hist, modes, snap = p.execute()
        if not profile:
            return None
        profile = {k.decode(): v.decode() for k, v in profile.items()}
        visits, hb, wk = decode_snap(snap)
        return RiderFeatures(rider_id, profile["persona"], int(profile["home"]), int(profile["work"]),
                             [int(x) for x in reversed(hist)], {k.decode(): int(v) for k, v in modes.items()},
                             visits, hb, wk)

    def global_features(self) -> GlobalFeatures | None:
        return _decode_meta(self.r.hgetall("snapshot:meta"))

    def record_ride(self, rider_id: int, dest: int, mode: str | None) -> None:
        p = self.r.pipeline(transaction=True)
        p.lpush(f"rider:{rider_id}:hist", dest)
        p.ltrim(f"rider:{rider_id}:hist", 0, HIST_LEN - 1)
        if mode:
            p.hincrby(f"rider:{rider_id}:modes", mode, 1)
        p.execute()

    def log_event(self, stream: str, event: dict) -> None:
        self.r.xadd(f"events:{stream}", {k: json.dumps(v) for k, v in event.items()}, maxlen=STREAM_MAXLEN,
                    approximate=True)

    def publish(self, profiles, hists, modes, snaps, meta) -> None:
        p = self.r.pipeline(transaction=False)
        for i, rider_id in enumerate(profiles):
            p.delete(f"rider:{rider_id}:hist", f"rider:{rider_id}:modes", f"rider:{rider_id}:snap")
            p.hset(f"rider:{rider_id}", mapping=profiles[rider_id])
            if hists.get(rider_id):
                p.rpush(f"rider:{rider_id}:hist", *reversed(hists[rider_id]))
            if modes.get(rider_id):
                p.hset(f"rider:{rider_id}:modes", mapping=modes[rider_id])
            if snaps.get(rider_id):
                p.hset(f"rider:{rider_id}:snap", mapping=snaps[rider_id])
            if i % 500 == 499:
                p.execute()
        p.hset("snapshot:meta", mapping=meta)   # written last: readers never see a half-published snapshot's meta
        p.execute()


class InMemoryFeatureStore:
    """Same contract as Redis, for tests and for running the service without Docker."""

    def __init__(self):
        self.profiles: dict[int, dict] = {}
        self.hists: dict[int, list[int]] = defaultdict(list)       # oldest -> newest
        self.modes: dict[int, dict[str, int]] = defaultdict(dict)
        self.snaps: dict[int, dict[str, int]] = {}
        self.meta: dict = {}
        self.events: dict[str, list[dict]] = defaultdict(list)

    def ping(self) -> bool:
        return True

    def rider(self, rider_id: int) -> RiderFeatures | None:
        p = self.profiles.get(rider_id)
        if p is None:
            return None
        visits, hb, wk = decode_snap(self.snaps.get(rider_id, {}))
        return RiderFeatures(rider_id, p["persona"], int(p["home"]), int(p["work"]), list(self.hists[rider_id]),
                             dict(self.modes[rider_id]), visits, hb, wk)

    def global_features(self) -> GlobalFeatures | None:
        return _decode_meta(self.meta)

    def record_ride(self, rider_id: int, dest: int, mode: str | None) -> None:
        self.hists[rider_id] = (self.hists[rider_id] + [dest])[-HIST_LEN:]
        if mode:
            self.modes[rider_id][mode] = self.modes[rider_id].get(mode, 0) + 1

    def log_event(self, stream: str, event: dict) -> None:
        self.events[stream] = (self.events[stream] + [{**event, "logged_at": time.time()}])[-STREAM_MAXLEN:]

    def publish(self, profiles, hists, modes, snaps, meta) -> None:
        self.profiles = dict(profiles)
        self.hists = defaultdict(list, {k: list(v) for k, v in hists.items()})
        self.modes = defaultdict(dict, {k: dict(v) for k, v in modes.items()})
        self.snaps = dict(snaps)
        self.meta = dict(meta)


def open_store(url: str | None) -> FeatureStore:
    return RedisFeatureStore(url) if url else InMemoryFeatureStore()

