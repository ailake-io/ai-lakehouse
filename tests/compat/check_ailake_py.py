# SPDX-License-Identifier: MIT OR Apache-2.0
#!/usr/bin/env python3
"""
Verifies the ailake Python SDK: legacy API (backward compat) + fluent API.

Build with:
    cd ailake-py && maturin develop --release

Usage:
    python tests/compat/check_ailake_py.py
"""

import asyncio
import math
import sys
import tempfile
import pathlib

try:
    import ailake
except ImportError as e:
    print(f"SKIP: ailake not installed — {e}")
    print("      Build with: cd ailake-py && maturin develop --release")
    sys.exit(0)

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False

DIM = 8
N = 20


def make_embedding(i: int) -> list:
    v = [float(i * DIM + j + 1) for j in range(DIM)]
    norm = math.sqrt(sum(x * x for x in v))
    return [x / norm for x in v]


# ── 1. Legacy TableWriter API (backward compat) ───────────────────────────────

with tempfile.TemporaryDirectory() as tmp:
    path = str(pathlib.Path(tmp) / "sdk_test")

    writer = ailake.TableWriter(path, vector_column="embedding", dim=DIM, metric="cosine")
    texts = [f"document_{i}" for i in range(N)]
    embeddings = [make_embedding(i) for i in range(N)]
    writer.write_batch(texts, embeddings)
    snap_id = writer.commit()

    assert snap_id >= 0, f"FAIL: commit returned {snap_id}"
    print(f"PASS (legacy TableWriter write/commit): {N} rows, snapshot_id={snap_id}")

    # search() returns SearchQuery — materialise with .to_list()
    query_idx = 10
    results = ailake.search(path, make_embedding(query_idx), top_k=5).to_list()
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

    # top_k respected
    r3 = ailake.search(path, make_embedding(0), top_k=3).to_list()
    assert len(r3) == 3, f"FAIL: expected 3 results, got {len(r3)}"
    print(f"PASS (top_k=3): got exactly 3 results")


# ── 2. Euclidean metric ────────────────────────────────────────────────────────

with tempfile.TemporaryDirectory() as tmp:
    path = str(pathlib.Path(tmp) / "eucl_test")

    writer = ailake.TableWriter(path, vector_column="embedding", dim=DIM, metric="euclidean")
    writer.write_batch([f"item_{i}" for i in range(N)], [make_embedding(i) for i in range(N)])
    writer.commit()

    query_idx = 5
    results = ailake.search(path, make_embedding(query_idx), top_k=5).to_list()
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


# ── 4. Fluent API — open_table + Table.insert + Table.search ──────────────────

with tempfile.TemporaryDirectory() as tmp:
    path = str(pathlib.Path(tmp) / "fluent_test")

    table = ailake.open_table(path, dim=DIM, metric="cosine")
    assert repr(table).startswith("Table("), f"FAIL: bad repr {repr(table)}"
    assert "_repr_html_" in dir(table), "FAIL: no _repr_html_ on Table"
    html = table._repr_html_()
    assert "AI-Lake Table" in html, "FAIL: _repr_html_ missing header"
    print(f"PASS (Table repr + html): repr={repr(table)!r}")

    # chainable insert
    table.insert([f"doc_{i}" for i in range(N)], [make_embedding(i) for i in range(N)])
    snap = table.commit()
    assert snap >= 0, f"FAIL: fluent commit returned {snap}"
    print(f"PASS (open_table / insert / commit): snapshot_id={snap}")

    # fluent search chain
    q = ailake.search(path, make_embedding(5))
    assert repr(q) == "SearchQuery(top_k=10, pending)", f"FAIL: pending repr={repr(q)!r}"
    assert "_repr_html_" in dir(q), "FAIL: no _repr_html_ on SearchQuery"

    results = q.limit(3).to_list()
    assert len(results) == 3, f"FAIL: limit(3) returned {len(results)}"
    assert repr(q) == "SearchQuery(3 results, top_k=3)", f"FAIL: executed repr={repr(q)!r}"
    html_q = q._repr_html_()
    assert "row_id" in html_q, "FAIL: _repr_html_ missing row_id column"
    print(f"PASS (fluent SearchQuery chain + repr + html): {len(results)} results")

    # Table.search
    sq = table.search(make_embedding(0), top_k=2)
    r2 = sq.to_list()
    assert len(r2) == 2, f"FAIL: Table.search(top_k=2) returned {len(r2)}"
    print(f"PASS (Table.search): {len(r2)} results")

    if HAS_PANDAS:
        df = ailake.search(path, make_embedding(0), top_k=5).to_pandas()
        assert list(df.columns) == ["row_id", "distance", "file"], f"FAIL: columns={list(df.columns)}"
        assert len(df) == 5, f"FAIL: DataFrame has {len(df)} rows"
        print(f"PASS (to_pandas): shape={df.shape}")
    else:
        print("SKIP (to_pandas): pandas not installed")

    # context manager
    with ailake.open_table(path, dim=DIM) as t2:
        assert isinstance(t2, ailake.Table), "FAIL: context manager did not return Table"
    print("PASS (context manager)")


# ── 5. Async API ──────────────────────────────────────────────────────────────

async def _async_checks(path: str) -> None:
    table = ailake.open_table(path, dim=DIM, metric="cosine")
    await table.insert_async([f"a{i}" for i in range(N)], [make_embedding(i) for i in range(N)])
    snap = await table.commit_async()
    assert snap >= 0, f"FAIL: async commit returned {snap}"

    # fluent async chain
    results = await ailake.search(path, make_embedding(0), top_k=3).to_list_async()
    assert isinstance(results, list) and len(results) == 3, f"FAIL: async to_list_async={results}"

    # parallel
    r1, r2 = await asyncio.gather(
        table.search(make_embedding(0)).to_list_async(),
        table.search(make_embedding(1)).to_list_async(),
    )
    assert len(r1) > 0 and len(r2) > 0, "FAIL: parallel async search"
    assert r1[0]["row_id"] != r2[0]["row_id"], "FAIL: parallel searches returned same top-1"

    if HAS_PANDAS:
        df = await ailake.search(path, make_embedding(0), top_k=4).to_pandas_async()
        assert len(df) == 4, f"FAIL: async to_pandas_async rows={len(df)}"

with tempfile.TemporaryDirectory() as tmp:
    path = str(pathlib.Path(tmp) / "async_test")
    asyncio.run(_async_checks(path))
    print("PASS (async API): insert_async, commit_async, to_list_async, parallel gather")


# ── 6. assemble_context ───────────────────────────────────────────────────────

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
for i in range(5):
    assert f"chunk number {i}" in ctx, f"FAIL: chunk {i} missing from context"
print(f"PASS (assemble_context): {len(ctx)} chars, all 5 chunks present")

ctx_tiny = ailake.assemble_context(chunks, max_tokens=10)
assert len(ctx_tiny) < len(ctx), "FAIL: tiny budget did not truncate context"
print(f"PASS (assemble_context budget): tiny={len(ctx_tiny)} chars < full={len(ctx)} chars")

dup_chunks = [
    {"document_id": "doc-2", "chunk_index": 0, "chunk_text": "alpha text", "distance": 0.1},
    {"document_id": "doc-2", "chunk_index": 1, "chunk_text": "beta text", "distance": 0.2},
]
ctx_dedup = ailake.assemble_context(dup_chunks, max_tokens=4096, dedup_threshold=0.0)
assert len(ctx_dedup) > 0, "FAIL: assemble_context with dedup_threshold=0.0 returned empty"
print("PASS (assemble_context dedup_threshold): parameter accepted, output non-empty")


# ── 7. Error handling ─────────────────────────────────────────────────────────

try:
    ailake.TableWriter("/nonexistent_path_xyz", vector_column="embedding", dim=DIM)
    print("PASS (nonexistent path): created table in new directory")
except Exception as e:
    print(f"PASS (nonexistent path): raised {type(e).__name__}: {e}")

try:
    ailake.search("/definitely_not_a_table_abc123", [0.1] * DIM, top_k=5).to_list()
    print("FAIL: expected error for missing table")
    sys.exit(1)
except Exception as e:
    print(f"PASS (missing table error): {type(e).__name__}: {e}")


print()
print("PASS: ailake Python SDK — all checks passed.")
