#!/usr/bin/env python3
"""Deep Lake SIFT-1M benchmark.

Limitations
-----------
Deep Lake's approximate nearest-neighbor search (Deep Memory) requires a paid
Activeloop plan and a cloud dataset. This script measures:
  - Write throughput to a local Deep Lake dataset.
  - Exact brute-force kNN search (free, in-memory) on a configurable subset.

For a fair ANN comparison with pgvector / LanceDB / AI-Lake, use --limit 10000
(exact search on 10k vectors) as a write-throughput reference and acknowledge
that recall@k = 1.0 by design (exact search).

Usage
-----
    pip install deeplake numpy
    python3 scripts/deeplake_bench.py --dataset-dir data/sift1m [--limit 10000]
"""

import argparse
import json
import pathlib
import shutil
import struct
import tempfile
import time

import numpy as np


# ── Dataset helpers ───────────────────────────────────────────────────────────

def read_fvecs(path: pathlib.Path) -> np.ndarray:
    vecs = []
    with open(path, "rb") as f:
        while chunk := f.read(4):
            dim = struct.unpack("<i", chunk)[0]
            vecs.append(np.frombuffer(f.read(dim * 4), dtype=np.float32))
    return np.array(vecs, dtype=np.float32)


def read_ivecs(path: pathlib.Path) -> np.ndarray:
    vecs = []
    with open(path, "rb") as f:
        while chunk := f.read(4):
            dim = struct.unpack("<i", chunk)[0]
            vecs.append(np.frombuffer(f.read(dim * 4), dtype=np.int32).astype(np.uint32))
    return np.array(vecs, dtype=np.uint32)


def recall_at_k(result_ids, ground_truth, k: int) -> float:
    found = set(result_ids[:k])
    truth = set(int(x) for x in ground_truth[:k])
    return len(found & truth) / len(truth) if truth else 0.0


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset-dir", required=True, type=pathlib.Path)
    parser.add_argument("--limit", type=int, default=10_000,
                        help="Base set size (default 10k; exact search does not scale to 1M)")
    parser.add_argument("--top-k", type=int, default=10)
    args = parser.parse_args()

    try:
        import deeplake
    except ImportError:
        print("ERROR: deeplake not installed. Run: pip install deeplake")
        raise SystemExit(1)

    ds_dir = args.dataset_dir
    print(f"Loading SIFT-1M (limit={args.limit}) …")
    base    = read_fvecs(ds_dir / "sift_base.fvecs")[: args.limit]
    queries = read_fvecs(ds_dir / "sift_query.fvecs")
    gt      = read_ivecs(ds_dir / "sift_groundtruth.ivecs")
    dim     = base.shape[1]
    print(f"  base: {len(base)}  queries: {len(queries)}  dim: {dim}")

    tmpdir  = tempfile.mkdtemp()
    dl_path = str(pathlib.Path(tmpdir) / "sift1m_bench")

    # ── Write phase ───────────────────────────────────────────────────────────
    print(f"\nDeep Lake write phase ({len(base)} vectors) …")
    t_write = time.perf_counter()

    dl_ds = deeplake.dataset(dl_path, overwrite=True)
    with dl_ds:
        dl_ds.create_tensor("embedding", dtype="float32")

    BATCH = 1_000
    with dl_ds:
        for i in range(0, len(base), BATCH):
            chunk = base[i : i + BATCH]
            dl_ds.embedding.extend(chunk)
            print(f"\r  {min(i + BATCH, len(base))}/{len(base)} vectors …", end="", flush=True)
    print()

    write_elapsed = time.perf_counter() - t_write
    write_vps = len(base) / write_elapsed
    print(f"  write: {write_elapsed:.1f}s  {write_vps:.0f} vec/s")

    # ── Search (exact brute-force, in numpy) ──────────────────────────────────
    print(f"\nDeep Lake search phase (exact kNN, top_k={args.top_k}) …")
    print("  NOTE: Deep Lake free tier = exact search only.")
    print("  ANN (Deep Memory) requires a paid Activeloop plan.")

    # numpy() returns shape (N, dim) — load once into RAM
    base_np = np.array(dl_ds.embedding.numpy(aslist=False))

    num_q      = len(queries)
    latencies  = []
    recall_sum = 0.0
    t_wall     = time.perf_counter()

    for qi, q in enumerate(queries):
        t0    = time.perf_counter()
        dists = np.sum((base_np - q) ** 2, axis=1)
        top_idx = np.argpartition(dists, args.top_k)[: args.top_k]
        top_idx = top_idx[np.argsort(dists[top_idx])]
        latencies.append((time.perf_counter() - t0) * 1e6)  # µs
        recall_sum += recall_at_k(top_idx.tolist(), gt[qi].tolist(), args.top_k)
        if (qi + 1) % 1_000 == 0:
            print(f"\r  {qi+1}/{num_q} queries …", end="", flush=True)
    print(f"\r  {num_q}/{num_q} queries done")

    wall_s = time.perf_counter() - t_wall
    latencies.sort()
    mean_ms = sum(latencies) / len(latencies) / 1_000
    p50_ms  = latencies[len(latencies) // 2] / 1_000
    p95_ms  = latencies[int(len(latencies) * 0.95)] / 1_000
    p99_ms  = latencies[int(len(latencies) * 0.99)] / 1_000
    qps     = num_q / wall_s
    recall  = recall_sum / num_q

    print(f"\nDeep Lake (exact kNN on {len(base)} vectors) — SIFT-1M")
    print("=" * 58)
    print(f"  Search type   : exact brute-force (free tier)")
    print(f"  Base size     : {len(base)} (not full 1M)")
    print()
    print(f"Write phase")
    print(f"  Wall time     : {write_elapsed:.1f} s")
    print(f"  Throughput    : {write_vps:.0f} vec/s")
    print()
    print(f"Search phase  (top_k={args.top_k})")
    print(f"  Recall@{args.top_k}     : {recall:.4f}  (≈1.0 expected for exact search)")
    print(f"  QPS           : {qps:.0f}")
    print(f"  Latency mean  : {mean_ms:.3f} ms")
    print(f"  Latency p50   : {p50_ms:.3f} ms")
    print(f"  Latency p95   : {p95_ms:.3f} ms")
    print(f"  Latency p99   : {p99_ms:.3f} ms")
    print()

    result = {
        "engine":        f"Deep Lake exact kNN ({len(base)} vecs)",
        "write_vps":     round(write_vps, 1),
        "recall":        round(recall, 4),
        "qps":           round(qps, 1),
        "mean_ms":       round(mean_ms, 3),
        "p50_ms":        round(p50_ms, 3),
        "p95_ms":        round(p95_ms, 3),
        "p99_ms":        round(p99_ms, 3),
        "note":          "Exact search on subset only — ANN requires Deep Memory (paid plan)",
    }
    print(f"JSON output:\n{json.dumps(result, indent=2)}")

    shutil.rmtree(tmpdir, ignore_errors=True)


if __name__ == "__main__":
    main()
