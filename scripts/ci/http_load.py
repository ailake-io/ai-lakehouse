#!/usr/bin/env python3
"""Small dependency-free HTTP load and multi-writer test for ailake serve."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import pathlib
import statistics
import time
import urllib.error
import urllib.request
import uuid


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, int((len(ordered) * fraction + 0.999999) - 1)))
    return ordered[index]


def request(base_url: str, method: str, path: str, payload: dict | None = None) -> tuple[int, float, str]:
    body = None
    headers = {"Accept": "application/json"}
    if payload is not None:
        body = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"
    started = time.perf_counter()
    try:
        req = urllib.request.Request(base_url + path, data=body, headers=headers, method=method)
        with urllib.request.urlopen(req, timeout=30) as response:
            text = response.read().decode("utf-8")
            return response.status, (time.perf_counter() - started) * 1000.0, text
    except urllib.error.HTTPError as error:
        return error.code, (time.perf_counter() - started) * 1000.0, error.read().decode("utf-8")


def vector(index: int, dim: int) -> list[float]:
    values = [0.0] * dim
    values[index % dim] = 1.0
    values[(index * 7 + 3) % dim] = 0.25
    return values


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:7700")
    parser.add_argument("--requests", type=int, default=100)
    parser.add_argument("--concurrency", type=int, default=16)
    parser.add_argument("--write-workers", type=int, default=4)
    parser.add_argument("--write-batch-size", type=int, default=4)
    parser.add_argument("--dim", type=int, default=32)
    parser.add_argument("--max-p95-ms", type=float, default=500.0)
    parser.add_argument("--search-ready-timeout", type=float, default=30.0)
    parser.add_argument("--output", type=pathlib.Path, default=pathlib.Path("perf/http-load.json"))
    args = parser.parse_args()

    base_url = args.base_url.rstrip("/")
    status, _, _ = request(base_url, "GET", "/healthz")
    if status != 200:
        raise SystemExit(f"healthz failed with HTTP {status}")

    def write_one(worker: int) -> tuple[int, float, str]:
        embeddings = [
            vector(worker * args.write_batch_size + row, args.dim)
            for row in range(args.write_batch_size)
        ]
        payload = {
            "texts": [f"ci-load-{worker}-{row}" for row in range(args.write_batch_size)],
            "embeddings": embeddings,
            "batch_id": f"ci-load-{worker}-{uuid.uuid4()}",
        }
        return request(base_url, "POST", "/write", payload)

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.write_workers) as pool:
        writes = list(pool.map(write_one, range(args.write_workers)))
    write_errors = [result for result in writes if result[0] != 200]
    if write_errors:
        raise SystemExit(f"multi-writer phase failed: {write_errors[:3]}")

    def search_one(index: int) -> tuple[int, float, str]:
        return request(
            base_url,
            "POST",
            "/search",
            {"query": vector(index, args.dim), "top_k": 5, "pruning_threshold": 1.0},
        )

    # A write returns after the snapshot is committed, but index publication
    # and catalog visibility can briefly lag behind that commit. Probe the
    # actual search endpoint before starting concurrent load so the test
    # measures serving rather than the indexing visibility window.
    ready_deadline = time.monotonic() + args.search_ready_timeout
    ready_status = None
    ready_body = ""
    while time.monotonic() < ready_deadline:
        ready_status, _, ready_body = search_one(0)
        if ready_status == 200:
            break
        time.sleep(0.5)
    else:
        excerpt = ready_body.replace("\n", " ")[:500]
        raise SystemExit(
            "search endpoint did not become ready within "
            f"{args.search_ready_timeout:.1f}s (HTTP {ready_status}): {excerpt}"
        )

    started = time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        searches = list(pool.map(search_one, range(args.requests)))
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    search_errors = [result for result in searches if result[0] != 200]
    latencies = [result[1] for result in searches]
    p50 = percentile(latencies, 0.50)
    p95 = percentile(latencies, 0.95)
    p99 = percentile(latencies, 0.99)
    result = {
        "requests": args.requests,
        "concurrency": args.concurrency,
        "writers": args.write_workers,
        "write_statuses": [status for status, _, _ in writes],
        "search_errors": len(search_errors),
        "elapsed_ms": elapsed_ms,
        "qps": args.requests / max(elapsed_ms / 1000.0, 0.001),
        "latency_ms": {
            "p50": p50,
            "p95": p95,
            "p99": p99,
            "mean": statistics.mean(latencies),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    if search_errors:
        raise SystemExit(f"{len(search_errors)} search requests failed")
    if p95 > args.max_p95_ms:
        raise SystemExit(f"search p95 {p95:.2f} ms exceeds {args.max_p95_ms:.2f} ms")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
