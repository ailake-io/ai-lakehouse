#!/usr/bin/env python3
"""
Verifies the ailake Python SDK: TableWriter.write_batch → commit → search → assemble_context.

Build with:
    cd ailake-py && maturin develop --release

Usage:
    python tests/compat/check_ailake_py.py
"""

import sys
import math
import tempfile
import pathlib

try:
    import ailake
except ImportError as e:
    print(f"SKIP: ailake not installed — {e}")
    print("      Build with: cd ailake-py && maturin develop --release")
    sys.exit(0)

DIM = 8
N = 20


def make_embedding(i: int) -> list:
    v = [float(i * DIM + j + 1) for j in range(DIM)]
    norm = math.sqrt(sum(x * x for x in v))
    return [x / norm for x in v]


with tempfile.TemporaryDirectory() as tmp:
    path = str(pathlib.Path(tmp) / "sdk_test")

    # ── Write ──────────────────────────────────────────────────────────────────
    writer = ailake.TableWriter(path, vector_column="embedding", dim=DIM, metric="cosine")
    texts = [f"document_{i}" for i in range(N)]
    embeddings = [make_embedding(i) for i in range(N)]
    writer.write_batch(texts, embeddings)
    snap_id = writer.commit()

    assert snap_id >= 0, f"FAIL: commit returned {snap_id}"
    print(f"PASS (write): {N} rows committed, snapshot_id={snap_id}")

    # ── Search ─────────────────────────────────────────────────────────────────
    query_idx = 10
    query = make_embedding(query_idx)
    results = ailake.search(path, query, top_k=5)

    assert isinstance(results, list), f"FAIL: search returned {type(results)}"
    assert len(results) > 0, "FAIL: search returned empty results"

    top = min(results, key=lambda r: r["distance"])
    assert top["row_id"] == query_idx, (
        f"FAIL: nearest row_id={top['row_id']}, expected {query_idx}"
    )
    assert top["distance"] < 1e-4, (
        f"FAIL: self-distance={top['distance']}, expected ~0"
    )
    print(f"PASS (search): top-1 row_id={top['row_id']} distance={top['distance']:.6f}")
    print(f"      results={[(r['row_id'], round(r['distance'], 4)) for r in results]}")

    # ── assemble_context ───────────────────────────────────────────────────────
    chunks = [
        {
            "document_id": "doc-1",
            "chunk_index": i,
            "chunk_text": f"This is chunk number {i} of the test document.",
            "distance": 0.05 * i,
        }
        for i in range(3)
    ]
    ctx = ailake.assemble_context(chunks, max_tokens=1024)
    assert len(ctx) > 0, "FAIL: assemble_context returned empty string"
    print(f"PASS (assemble_context): {len(ctx)} chars generated")

print()
print("PASS: ailake Python SDK — write, search, and assemble_context all functional.")
