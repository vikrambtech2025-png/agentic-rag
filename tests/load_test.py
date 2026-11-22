"""Simple load test: N concurrent queries against a running server."""
from __future__ import annotations

import argparse
import asyncio
import time

import httpx


async def one(client, url, q):
    t0 = time.perf_counter()
    try:
        r = await client.post(url, json={"message": q}, timeout=60)
        return r.status_code, time.perf_counter() - t0
    except Exception as e:
        return 0, time.perf_counter() - t0


async def main(n, url, q):
    async with httpx.AsyncClient() as client:
        tasks = [one(client, url, q) for _ in range(n)]
        results = await asyncio.gather(*tasks)
    statuses = {}
    for code, dt in results:
        statuses[code] = statuses.get(code, 0) + 1
    dts = [dt for _, dt in results]
    print("n=", n, "statuses=", statuses)
    print("p50=", sorted(dts)[len(dts)//2], "p95=", sorted(dts)[int(len(dts)*0.95)], "max=", max(dts))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--url", default="http://127.0.0.1:8000")
    ap.add_argument("--q", default="what is rag")
    args = ap.parse_args()
    asyncio.run(main(args.n, args.url, args.q))
