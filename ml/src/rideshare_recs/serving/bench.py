"""Latency benchmark against a running service.

    uv run rideshare-bench --url http://localhost:8080 --requests 2000 --concurrency 16

Random riders, pickup points and times of day, so the feature store and both experiment arms are all exercised.
Reports client-side p50 / p95 / p99 and throughput; the service's own per-stage histograms are on /metrics.
"""
from __future__ import annotations

import argparse
import asyncio
import random
import time

import httpx
import numpy as np


async def _worker(client: httpx.AsyncClient, url: str, n: int, riders: int, latencies: list, errors: list, rng):
    for _ in range(n):
        body = {"rider_id": rng.randrange(riders),
                "origin": {"x_km": round(rng.uniform(1, 19), 2), "y_km": round(rng.uniform(1, 19), 2)},
                "at": f"2026-10-{rng.randint(19, 25):02d}T{rng.randint(6, 23):02d}:{rng.randint(0, 59):02d}:00"}
        start = time.perf_counter()
        try:
            r = await client.post(f"{url}/v1/recommendations", json=body)
            r.raise_for_status()
            latencies.append(time.perf_counter() - start)
        except httpx.HTTPError as e:
            errors.append(str(e))


async def bench(url: str, requests: int, concurrency: int, riders: int, seed: int) -> dict:
    rng = random.Random(seed)
    latencies: list[float] = []
    errors: list[str] = []
    async with httpx.AsyncClient(timeout=5.0, limits=httpx.Limits(max_connections=concurrency)) as client:
        for _ in range(min(50, requests)):                       # warm-up, not measured
            await client.post(f"{url}/v1/recommendations", json={"rider_id": 1, "origin": {"x_km": 10, "y_km": 10}})
        start = time.perf_counter()
        per = requests // concurrency
        await asyncio.gather(*(_worker(client, url, per, riders, latencies, errors, random.Random(rng.random()))
                               for _ in range(concurrency)))
        elapsed = time.perf_counter() - start
    ms = np.array(latencies) * 1000
    return {"requests": len(latencies), "errors": len(errors), "concurrency": concurrency,
            "throughput_rps": round(len(latencies) / elapsed, 1),
            "p50_ms": round(float(np.percentile(ms, 50)), 2), "p95_ms": round(float(np.percentile(ms, 95)), 2),
            "p99_ms": round(float(np.percentile(ms, 99)), 2)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", default="http://localhost:8080")
    ap.add_argument("--requests", type=int, default=2000)
    ap.add_argument("--concurrency", type=int, default=16)
    ap.add_argument("--riders", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()
    print(asyncio.run(bench(args.url.rstrip("/"), args.requests, args.concurrency, args.riders, args.seed)))


if __name__ == "__main__":
    main()
