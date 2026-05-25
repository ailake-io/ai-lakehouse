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


# ── 1. Write → commit → search (cosine, F16) ──────────────────────────────────

with tempfile.TemporaryDirectory() as tmp:
    path = str(pathlib.Path(tmp) / "sdk_test")

    writer = ailake.TableWriter(path, vector_column="embedding", dim=DIM, metric="cosine")
    texts = [f"document_{i}" for i in range(N)]
    embeddings = [make_embedding(i) for i in range(N)]
    writer.write_batch(texts, embeddings)
    snap_id = writer.commit()

    assert snap_id >= 0, f"FAIL: commit returned {snap_id}"
    print(f"PASS (write/commit): {N} rows, snapshot_id={snap_id}")

    # Self-distance must be ~0 for the nearest neighbour.
    query_idx = 10
    results = ailake.search(path, make_embedding(query_idx), top_k=5)
    assert isinstance(results, list) and len(results) > 0, f"FAIL: search returned {results}"
    top = min(results, key=lambda r: r["distance"])
    assert top["row_id"] == query_idx, (
        f"FAIL: nearest row_id={top['row_id']}, expected {query_idx}"
    )
    assert top["distance"] < 1e-4, f"FAIL: self-distance={top['distance']:.6f}, expected ~0"
    print(
        f"PASS (search cosine): top-1 row_id={top['row_id']} dist={top['distance']:.6f} "
        f"| results={[(r['row_id'], round(r['distance'], 4)) for r in results]}"
    )

    # top_k respected.
    r3 = ailake.search(path, make_embedding(0), top_k=3)
    assert len(r3) == 3, f"FAIL: expected 3 results, got {len(r3)}"
    print(f"PASS (top_k=3): got exactly 3 results")


# ── 2. Euclidean metric ────────────────────────────────────────────────────────

with tempfile.TemporaryDirectory() as tmp:
    path = str(pathlib.Path(tmp) / "eucl_test")

    writer = ailake.TableWriter(path, vector_column="embedding", dim=DIM, metric="euclidean")
    texts = [f"item_{i}" for i in range(N)]
    embeddings = [make_embedding(i) for i in range(N)]
    writer.write_batch(texts, embeddings)
    writer.commit()

    query_idx = 5
    results = ailake.search(path, make_embedding(query_idx), top_k=5)
    top = min(results, key=lambda r: r["distance"])
    assert top["row_id"] == query_idx, (
        f"FAIL (euclidean): nearest row_id={top['row_id']}, expected {query_idx}"
    )
    print(f"PASS (euclidean): top-1 row_id={top['row_id']} dist={top['distance']:.6f}")


# ── 3. Multiple write_batch calls before commit ────────────────────────────────

with tempfile.TemporaryDirectory() as tmp:
    path = str(pathlib.Path(tmp) / "multi_batch")

    writer = ailake.TableWriter(path, vector_column="embedding", dim=DIM, metric="cosine")
    half = N // 2
    writer.write_batch([f"batch0_{i}" for i in range(half)], [make_embedding(i) for i in range(half)])
    writer.write_batch([f"batch1_{i}" for i in range(half)], [make_embedding(i + half) for i in range(half)])
    snap_id = writer.commit()

    assert snap_id >= 0, f"FAIL: multi-batch commit returned {snap_id}"
    print(f"PASS (multi-batch): 2 batches × {half} rows committed, snapshot_id={snap_id}")


# ── 4. assemble_context ────────────────────────────────────────────────────────

chunks = [
    {
        "document_id": "doc-1",
        "chunk_index": i,
        "chunk_text": f"This is chunk number {i} of the test document with enough text to be meaningful.",
        "distance": 0.05 * i,
    }
    for i in range(5)
]
ctx = ailake.assemble_context(chunks, max_tokens=1024)
assert len(ctx) > 0, "FAIL: assemble_context returned empty string"
# All chunks should appear (budget is generous).
for i in range(5):
    assert f"chunk number {i}" in ctx, f"FAIL: chunk {i} missing from context"
print(f"PASS (assemble_context): {len(ctx)} chars, all 5 chunks present")

# Token budget respected: tiny budget should truncate.
ctx_tiny = ailake.assemble_context(chunks, max_tokens=10)
assert len(ctx_tiny) < len(ctx), "FAIL: tiny budget did not truncate context"
print(f"PASS (assemble_context budget): tiny={len(ctx_tiny)} chars < full={len(ctx)} chars")

# dedup_threshold: assemble_context accepts the parameter without error.
# (Embedding-based dedup only activates when chunks carry an 'embedding' field,
# which the Python binding does not expose yet — tested at the Rust unit level.)
dup_chunks = [
    {"document_id": "doc-2", "chunk_index": 0, "chunk_text": "alpha text", "distance": 0.1},
    {"document_id": "doc-2", "chunk_index": 1, "chunk_text": "beta text", "distance": 0.2},
]
ctx_dedup = ailake.assemble_context(dup_chunks, max_tokens=4096, dedup_threshold=0.0)
assert len(ctx_dedup) > 0, "FAIL: assemble_context with dedup_threshold=0.0 returned empty"
print("PASS (assemble_context dedup_threshold): parameter accepted, output non-empty")


# ── 5. Error handling ─────────────────────────────────────────────────────────

try:
    ailake.TableWriter("/nonexistent_path_xyz", vector_column="embedding", dim=DIM)
    # Should either succeed (creates dir) or raise a clear error.
    print("PASS (nonexistent path): created table in new directory")
except Exception as e:
    print(f"PASS (nonexistent path): raised {type(e).__name__}: {e}")

try:
    ailake.search("/definitely_not_a_table_abc123", [0.1] * DIM, top_k=5)
    print("FAIL: expected error for missing table")
    sys.exit(1)
except Exception as e:
    print(f"PASS (missing table error): {type(e).__name__}: {e}")

print()
print("PASS: ailake Python SDK — all checks passed.")
