#!/usr/bin/env python3
"""Small dependency-free HTTP load and multi-writer test for ailake serve."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import pathlib
import statistics
import threading
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


def process_rss_bytes(pid: int) -> int:
    status = pathlib.Path(f"/proc/{pid}/status").read_text(encoding="utf-8")
    for line in status.splitlines():
        if line.startswith("VmRSS:"):
            return int(line.split()[1]) * 1024
    raise ValueError(f"VmRSS is missing for process {pid}")


def sample_process_rss(pid: int, stop: threading.Event, samples: list[int]) -> None:
    while not stop.is_set():
        try:
            samples.append(process_rss_bytes(pid))
        except (OSError, ValueError):
            pass
        stop.wait(0.05)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:7700")
    parser.add_argument("--requests", type=int, default=100)
    parser.add_argument("--concurrency", type=int, default=16)
    parser.add_argument("--write-workers", type=int, default=4)
    parser.add_argument("--write-rounds", type=int, default=1)
    parser.add_argument("--write-batch-size", type=int, default=4)
    parser.add_argument("--dim", type=int, default=32)
    parser.add_argument("--server-pid", type=int)
    parser.add_argument("--max-p95-ms", type=float, default=500.0)
    parser.add_argument("--search-ready-timeout", type=float, default=30.0)
    parser.add_argument("--output", type=pathlib.Path, default=pathlib.Path("perf/http-load.json"))
    args = parser.parse_args()
    positive_values = (
        args.requests,
        args.concurrency,
        args.write_workers,
        args.write_rounds,
        args.write_batch_size,
        args.dim,
    )
    if min(positive_values) < 1:
        parser.error(
            "requests, concurrency, write workers/rounds/batch size and dim "
            "must be positive"
        )

    base_url = args.base_url.rstrip("/")
    status, _, _ = request(base_url, "GET", "/healthz")
    if status != 200:
        raise SystemExit(f"healthz failed with HTTP {status}")

    rss_samples: list[int] = []
    rss_stop = threading.Event()
    rss_monitor = None
    if args.server_pid is not None:
        try:
            rss_samples.append(process_rss_bytes(args.server_pid))
        except (OSError, ValueError) as error:
            raise SystemExit(f"cannot read RSS for server PID {args.server_pid}: {error}")
        rss_monitor = threading.Thread(
            target=sample_process_rss,
            args=(args.server_pid, rss_stop, rss_samples),
            daemon=True,
        )
        rss_monitor.start()

    def write_one(worker: int) -> list[tuple[int, float, str]]:
        results = []
        for round_index in range(args.write_rounds):
            first_row = (
                worker * args.write_rounds * args.write_batch_size
                + round_index * args.write_batch_size
            )
            embeddings = [
                vector(first_row + row, args.dim) for row in range(args.write_batch_size)
            ]
            payload = {
                "texts": [
                    f"ci-load-{worker}-{round_index}-{row}"
                    for row in range(args.write_batch_size)
                ],
                "embeddings": embeddings,
                "batch_id": f"ci-load-{worker}-{round_index}-{uuid.uuid4()}",
            }
            results.append(request(base_url, "POST", "/write", payload))
        return results

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.write_workers) as pool:
        writes = [
            result
            for worker_results in pool.map(write_one, range(args.write_workers))
            for result in worker_results
        ]
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
    rss_stop.set()
    if rss_monitor is not None:
        rss_monitor.join(timeout=1)
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
        "write_rounds": args.write_rounds,
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
    if rss_samples:
        result["server_rss_bytes"] = {
            "start": rss_samples[0],
            "peak": max(rss_samples),
            "end": rss_samples[-1],
            "samples": len(rss_samples),
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
