"""Replay the test week through the gateway, as riders would use the app: ask for suggestions, then complete the
trip they actually took. This drives the live A/B panels in Grafana from real request and outcome events.

    uv run rideshare-replay --url http://localhost:8090 --rate 20 --trips 3000
"""
from __future__ import annotations

import argparse
import asyncio
import os
import time

import httpx

from ..features import VAL_END_DAY
from ..simulator import load_or_simulate
from .publish import default_data_dir


async def _one(client: httpx.AsyncClient, url: str, t, stats: dict) -> None:
    try:
        r = await client.post(f"{url}/v1/recommendations", json={
            "rider_id": int(t.rider_id), "origin": {"place_id": int(t.origin_place_id)},
            "at": t.ts.isoformat(), "include_ride_options": False})
        r.raise_for_status()
        done = await client.post(f"{url}/v1/events/ride-completed", json={
            "rider_id": int(t.rider_id), "destination_place_id": int(t.dest_place_id), "mode": t.mode,
            "request_id": r.json()["request_id"]})
        done.raise_for_status()
        rank, variant = done.json()["suggested_rank"], done.json()["variant"]
        arm = stats.setdefault(variant, {"trips": 0, "first": 0, "top5": 0})
        arm["trips"] += 1
        arm["first"] += rank == 1
        arm["top5"] += rank is not None
    except httpx.HTTPError:
        stats["errors"] = stats.get("errors", 0) + 1


async def replay(url: str, rate: float, n: int) -> dict:
    _, _, trips = load_or_simulate(default_data_dir())
    week = trips[(trips.day >= VAL_END_DAY) & (trips.origin_place_id != trips.dest_place_id)]
    week = week.sample(min(n, len(week)), random_state=7).sort_values("ts")     # across the week, not just Monday
    stats: dict = {}
    tasks = []
    async with httpx.AsyncClient(timeout=5.0) as client:
        start = time.monotonic()
        for i, t in enumerate(week.itertuples()):
            await asyncio.sleep(max(0.0, start + i / rate - time.monotonic()))
            tasks.append(asyncio.create_task(_one(client, url, t, stats)))
        await asyncio.gather(*tasks)
    return stats


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", default=os.environ.get("GATEWAY_URL", "http://localhost:8090"))
    ap.add_argument("--rate", type=float, default=20.0, help="trips per second")
    ap.add_argument("--trips", type=int, default=3000)
    args = ap.parse_args()
    stats = asyncio.run(replay(args.url.rstrip("/"), args.rate, args.trips))
    for arm in ("control", "treatment"):
        a = stats.get(arm)
        if a:
            print(f"{arm:<10} trips {a['trips']:>6,}  first suggestion {a['first'] / a['trips']:.3f}  "
                  f"in top 5 {a['top5'] / a['trips']:.3f}")
    print(f"errors: {stats.get('errors', 0)}")


if __name__ == "__main__":
    main()
