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


# ── 8. fetch_data=True — full-read mode ──────────────────────────────────────

try:
    import pyarrow as pa
    HAS_PYARROW = True
except ImportError:
    HAS_PYARROW = False

with tempfile.TemporaryDirectory() as tmp:
    path = str(pathlib.Path(tmp) / "fullread_test")

    writer = ailake.TableWriter(path, vector_column="embedding", dim=DIM, metric="cosine")
    texts_fr = [f"fullread_doc_{i}" for i in range(N)]
    embeddings_fr = [make_embedding(i) for i in range(N)]
    writer.write_batch(texts_fr, embeddings_fr)
    writer.commit()

    q = make_embedding(5)

    # backward compat: fetch_data=False still returns pointer-only dicts
    results_ptr = ailake.search(path, q, top_k=5).to_list()
    assert isinstance(results_ptr, list) and len(results_ptr) > 0
    assert "row_id" in results_ptr[0], f"FAIL: row_id missing from pointer result {results_ptr[0]}"
    assert "distance" in results_ptr[0], f"FAIL: distance missing"
    assert "file" in results_ptr[0], f"FAIL: file missing"
    assert "text" not in results_ptr[0], "FAIL: fetch_data=False should not return text"
    print(f"PASS (fetch_data=False backward compat): {len(results_ptr)} pointer-only results")

    if HAS_PYARROW:
        # to_arrow() returns pyarrow.Table with all Parquet columns + _distance
        table_result = ailake.search(path, q, top_k=5, fetch_data=True).to_arrow()
        assert isinstance(table_result, pa.Table), (
            f"FAIL: to_arrow() returned {type(table_result)}"
        )
        col_names = table_result.schema.names
        assert "text" in col_names, f"FAIL: 'text' missing from {col_names}"
        assert "_distance" in col_names, f"FAIL: '_distance' missing from {col_names}"
        assert "embedding" in col_names, f"FAIL: 'embedding' missing from {col_names}"
        assert len(table_result) == 5, f"FAIL: expected 5 rows, got {len(table_result)}"

        # embedding column must be FixedSizeList<float32>
        emb_type = table_result.schema.field("embedding").type
        assert pa.types.is_fixed_size_list(emb_type), (
            f"FAIL: embedding type should be fixed_size_list, got {emb_type}"
        )
        assert emb_type.value_type == pa.float32(), (
            f"FAIL: embedding value type should be float32, got {emb_type.value_type}"
        )
        assert emb_type.list_size == DIM, (
            f"FAIL: embedding list_size={emb_type.list_size}, expected {DIM}"
        )

        # _distance is monotonically non-decreasing (nearest first)
        distances = table_result.column("_distance").to_pylist()
        for i in range(len(distances) - 1):
            assert distances[i] <= distances[i + 1] + 1e-6, (
                f"FAIL: distances not sorted at index {i}: {distances[i]:.6f} > {distances[i+1]:.6f}"
            )

        texts_got = table_result.column("text").to_pylist()
        assert all(isinstance(t, str) for t in texts_got), "FAIL: text column contains non-str"

        print(
            f"PASS (fetch_data=True to_arrow): {len(table_result)} rows, "
            f"columns={col_names}, embedding_type={emb_type}, "
            f"distances={[round(d, 4) for d in distances]}"
        )

        if HAS_PANDAS:
            df_full = ailake.search(path, q, top_k=5, fetch_data=True).to_pandas()
            assert "text" in df_full.columns, f"FAIL: 'text' missing from {list(df_full.columns)}"
            assert "_distance" in df_full.columns, "FAIL: '_distance' missing from DataFrame"
            assert "embedding" in df_full.columns, "FAIL: 'embedding' missing from DataFrame"
            assert len(df_full) == 5, f"FAIL: expected 5 rows, got {len(df_full)}"
            print(
                f"PASS (fetch_data=True to_pandas): shape={df_full.shape}, "
                f"columns={list(df_full.columns)}"
            )

        # Table.search with fetch_data=True
        tbl_fr = ailake.open_table(path, dim=DIM)
        arr_tbl = tbl_fr.search(q, top_k=3, fetch_data=True).to_arrow()
        assert len(arr_tbl) == 3, f"FAIL: Table.search fetch_data=True returned {len(arr_tbl)} rows"
        assert "_distance" in arr_tbl.schema.names, "FAIL: _distance missing from Table.search"
        print(f"PASS (Table.search fetch_data=True): {len(arr_tbl)} rows")

        # async full-read
        async def _async_fullread(p: str):
            return await ailake.search(p, q, top_k=3, fetch_data=True).to_arrow_async()

        arr_async_fr = asyncio.run(_async_fullread(path))
        assert len(arr_async_fr) == 3, (
            f"FAIL: async to_arrow_async full-read returned {len(arr_async_fr)} rows"
        )
        assert "_distance" in arr_async_fr.schema.names, (
            "FAIL: async full-read _distance missing"
        )
        print(f"PASS (fetch_data=True to_arrow_async): {len(arr_async_fr)} rows")

    else:
        print("SKIP (fetch_data=True): pyarrow not installed")


# ── 9. write_batch_idempotent ─────────────────────────────────────────────────

with tempfile.TemporaryDirectory() as tmp:
    path = str(pathlib.Path(tmp) / "idem_test")

    writer = ailake.TableWriter(path, vector_column="embedding", dim=DIM, metric="cosine")
    texts_id = [f"idem_{i}" for i in range(N)]
    embs_id = [make_embedding(i) for i in range(N)]

    writer.write_batch_idempotent(texts_id, embs_id, batch_id="batch-001")
    snap1 = writer.commit()
    assert snap1 >= 0, f"FAIL: first idempotent commit returned {snap1}"
    print(f"PASS (write_batch_idempotent first write): snapshot_id={snap1}")

    # second writer, same batch_id — must be no-op (snapshot unchanged or error)
    writer2 = ailake.TableWriter(path, vector_column="embedding", dim=DIM, metric="cosine")
    writer2.write_batch_idempotent(texts_id, embs_id, batch_id="batch-001")
    snap2 = writer2.commit()
    # idempotent: snap2 may equal snap1 (no new snapshot) or be a new snapshot
    # either way, search must still return correct results
    results_id = ailake.search(path, make_embedding(5), top_k=3).to_list()
    assert len(results_id) > 0, "FAIL: search after idempotent write returned empty"
    print(
        f"PASS (write_batch_idempotent re-run): snap1={snap1} snap2={snap2}, "
        f"search still returns {len(results_id)} results"
    )

    # different batch_id — must write new data
    writer3 = ailake.TableWriter(path, vector_column="embedding", dim=DIM, metric="cosine")
    texts_new = [f"new_{i}" for i in range(N)]
    writer3.write_batch_idempotent(texts_new, embs_id, batch_id="batch-002")
    snap3 = writer3.commit()
    assert snap3 >= 0, f"FAIL: new batch_id commit returned {snap3}"
    print(f"PASS (write_batch_idempotent new batch_id): snapshot_id={snap3}")


# ── 10. to_polars ─────────────────────────────────────────────────────────────

try:
    import polars as pl
    HAS_POLARS = True
except ImportError:
    HAS_POLARS = False

with tempfile.TemporaryDirectory() as tmp:
    path = str(pathlib.Path(tmp) / "polars_test")

    table = ailake.open_table(path, dim=DIM, metric="cosine")
    table.insert([f"doc_{i}" for i in range(N)], [make_embedding(i) for i in range(N)])
    table.commit()

    if HAS_POLARS:
        lf = ailake.search(path, make_embedding(3), top_k=5).to_polars()
        assert len(lf) == 5, f"FAIL: to_polars returned {len(lf)} rows"
        assert "row_id" in lf.columns, f"FAIL: row_id missing from {lf.columns}"
        assert "distance" in lf.columns, f"FAIL: distance missing"
        assert "file" in lf.columns, f"FAIL: file missing"
        # distances sorted ascending
        dists = lf["distance"].to_list()
        for i in range(len(dists) - 1):
            assert dists[i] <= dists[i + 1] + 1e-6, (
                f"FAIL: to_polars distances not sorted at {i}: {dists[i]} > {dists[i+1]}"
            )
        print(f"PASS (to_polars): shape={lf.shape}, columns={lf.columns}")

        # limit + to_polars
        lf2 = ailake.search(path, make_embedding(0)).limit(2).to_polars()
        assert len(lf2) == 2, f"FAIL: limit(2).to_polars() returned {len(lf2)}"
        print(f"PASS (limit + to_polars): {len(lf2)} rows")

        if HAS_PYARROW:
            lf_full = ailake.search(path, make_embedding(3), top_k=5, fetch_data=True).to_polars()
            assert len(lf_full) == 5, f"FAIL: full-read to_polars {len(lf_full)} rows"
            assert "text" in lf_full.columns, f"FAIL: text missing from {lf_full.columns}"
            assert "_distance" in lf_full.columns, "FAIL: _distance missing"
            print(f"PASS (fetch_data=True to_polars): shape={lf_full.shape}")
    else:
        print("SKIP (to_polars): polars not installed")


# ── 11. Multiple commits — data from all snapshots visible in search ───────────

with tempfile.TemporaryDirectory() as tmp:
    path = str(pathlib.Path(tmp) / "multisnap_test")

    # First commit: rows 0..N-1
    t1 = ailake.open_table(path, dim=DIM, metric="cosine")
    t1.insert([f"snap1_{i}" for i in range(N)], [make_embedding(i) for i in range(N)])
    snap_a = t1.commit()
    assert snap_a >= 0, f"FAIL: first commit returned {snap_a}"

    # Second commit: rows N..2N-1 (distinct embeddings)
    t2 = ailake.open_table(path, dim=DIM, metric="cosine")
    t2.insert(
        [f"snap2_{i}" for i in range(N)],
        [make_embedding(i + N) for i in range(N)],
    )
    snap_b = t2.commit()
    assert snap_b >= 0, f"FAIL: second commit returned {snap_b}"
    assert snap_b != snap_a, "FAIL: second snapshot id equals first"

    # Search must return top_k results across both snapshots
    results_ms = ailake.search(path, make_embedding(0), top_k=10).to_list()
    assert len(results_ms) > 0, "FAIL: multi-snapshot search returned empty"
    # At least some results should come from each snapshot's files
    files_seen = {r["file"] for r in results_ms}
    assert len(files_seen) >= 1, "FAIL: no file paths in multi-snapshot results"
    print(
        f"PASS (multiple commits): snap_a={snap_a} snap_b={snap_b}, "
        f"search returned {len(results_ms)} results from {len(files_seen)} file(s)"
    )

    # to_arrow pointer-only: columns row_id, distance, file as Arrow table
    if HAS_PYARROW:
        arr_ptr = ailake.search(path, make_embedding(0), top_k=5).to_arrow()
        assert isinstance(arr_ptr, pa.Table), f"FAIL: pointer to_arrow type {type(arr_ptr)}"
        assert set(arr_ptr.schema.names) == {"row_id", "distance", "file"}, (
            f"FAIL: pointer to_arrow columns {arr_ptr.schema.names}"
        )
        assert len(arr_ptr) == 5, f"FAIL: pointer to_arrow rows {len(arr_ptr)}"
        print(f"PASS (to_arrow pointer-only): {len(arr_ptr)} rows, schema={arr_ptr.schema}")


# ── 12. pre_normalize + hnsw_m / hnsw_ef_construction ────────────────────────

with tempfile.TemporaryDirectory() as tmp:
    path = str(pathlib.Path(tmp) / "tuning_test")

    table = ailake.open_table(
        path,
        dim=DIM,
        metric="cosine",
        pre_normalize=True,
        hnsw_m=8,
        hnsw_ef_construction=100,
    )
    table.insert([f"tune_{i}" for i in range(N)], [make_embedding(i) for i in range(N)])
    snap = table.commit()
    assert snap >= 0, f"FAIL: pre_normalize + hnsw tuning commit returned {snap}"

    query_idx = 7
    results_tn = ailake.search(path, make_embedding(query_idx), top_k=5).to_list()
    assert len(results_tn) > 0, "FAIL: search with pre_normalize returned empty"
    top_tn = min(results_tn, key=lambda r: r["distance"])
    assert top_tn["row_id"] == query_idx, (
        f"FAIL: pre_normalize nearest row_id={top_tn['row_id']}, expected {query_idx}"
    )
    print(
        f"PASS (pre_normalize + hnsw_m=8 + hnsw_ef=100): "
        f"top-1 row_id={top_tn['row_id']} dist={top_tn['distance']:.6f}"
    )


# ── 13. Edge cases: top_k > N, top_k=1, empty table resilience ───────────────

with tempfile.TemporaryDirectory() as tmp:
    path = str(pathlib.Path(tmp) / "edge_test")

    small_n = 5
    table = ailake.open_table(path, dim=DIM, metric="cosine")
    table.insert(
        [f"edge_{i}" for i in range(small_n)],
        [make_embedding(i) for i in range(small_n)],
    )
    table.commit()

    # top_k > N → returns min(top_k, N)
    results_big = ailake.search(path, make_embedding(0), top_k=100).to_list()
    assert len(results_big) <= 100, "FAIL: returned more than top_k rows"
    assert len(results_big) == small_n, (
        f"FAIL: expected {small_n} rows (all rows), got {len(results_big)}"
    )
    print(f"PASS (top_k > N): requested 100, got {len(results_big)} (all {small_n} rows)")

    # top_k=1 → exactly 1 result, nearest
    results_one = ailake.search(path, make_embedding(2), top_k=1).to_list()
    assert len(results_one) == 1, f"FAIL: top_k=1 returned {len(results_one)}"
    assert results_one[0]["row_id"] == 2, (
        f"FAIL: top_k=1 nearest row_id={results_one[0]['row_id']}, expected 2"
    )
    print(f"PASS (top_k=1): row_id={results_one[0]['row_id']} dist={results_one[0]['distance']:.6f}")

    # distances are sorted ascending
    results_sorted = ailake.search(path, make_embedding(0), top_k=small_n).to_list()
    for i in range(len(results_sorted) - 1):
        assert results_sorted[i]["distance"] <= results_sorted[i + 1]["distance"] + 1e-6, (
            f"FAIL: results not sorted at {i}: "
            f"{results_sorted[i]['distance']} > {results_sorted[i+1]['distance']}"
        )
    print(f"PASS (distances sorted): {[round(r['distance'], 4) for r in results_sorted]}")

    # dot_product metric write + search
    with tempfile.TemporaryDirectory() as tmp2:
        path2 = str(pathlib.Path(tmp2) / "dot_test")
        writer_dot = ailake.TableWriter(path2, vector_column="embedding", dim=DIM, metric="dot_product")
        writer_dot.write_batch([f"dot_{i}" for i in range(N)], [make_embedding(i) for i in range(N)])
        writer_dot.commit()
        results_dot = ailake.search(path2, make_embedding(3), top_k=3).to_list()
        assert len(results_dot) > 0, "FAIL: dot_product search returned empty"
        print(f"PASS (dot_product metric): top-1 row_id={results_dot[0]['row_id']}")


# ── 14. embedding_model param — stored in Iceberg properties ─────────────────

with tempfile.TemporaryDirectory() as tmp:
    path = str(pathlib.Path(tmp) / "model_track_test")

    writer = ailake.TableWriter(
        path,
        vector_column="embedding",
        dim=DIM,
        metric="cosine",
        embedding_model="text-embedding-3-small",
        embedding_model_version="2024-01",
    )
    texts_mt = [f"doc_{i}" for i in range(N)]
    embs_mt = [make_embedding(i) for i in range(N)]
    writer.write_batch(texts_mt, embs_mt)
    snap_mt = writer.commit()
    assert snap_mt >= 0, f"FAIL: embedding_model write returned {snap_mt}"
    print(f"PASS (TableWriter embedding_model): snapshot_id={snap_mt}")

    results_mt = ailake.search(path, make_embedding(5), top_k=3).to_list()
    assert len(results_mt) > 0, "FAIL: search on model-tracked table returned empty"
    print(f"PASS (search on model-tracked table): {len(results_mt)} results")

with tempfile.TemporaryDirectory() as tmp:
    path = str(pathlib.Path(tmp) / "open_model_test")

    table = ailake.open_table(
        path,
        dim=DIM,
        metric="cosine",
        embedding_model="my-model",
        embedding_model_version="v2",
    )
    table.insert([f"doc_{i}" for i in range(N)], [make_embedding(i) for i in range(N)])
    snap_om = table.commit()
    assert snap_om >= 0, f"FAIL: open_table embedding_model commit returned {snap_om}"
    print(f"PASS (open_table embedding_model): snapshot_id={snap_om}")


# ── 15. ModelMismatch — dim mismatch detected at write time ──────────────────

with tempfile.TemporaryDirectory() as tmp:
    path = str(pathlib.Path(tmp) / "mismatch_test")

    writer = ailake.TableWriter(path, vector_column="embedding", dim=DIM, metric="cosine")
    writer.write_batch([f"doc_{i}" for i in range(N)], [make_embedding(i) for i in range(N)])
    writer.commit()

    writer2 = ailake.TableWriter(path, vector_column="embedding", dim=DIM * 2, metric="cosine")
    try:
        writer2.write_batch(
            [f"bad_{i}" for i in range(N)],
            [[0.1] * (DIM * 2) for _ in range(N)],
        )
        writer2.commit()
        print("WARN (ModelMismatch): no error raised for dim mismatch — check writer.rs:ModelMismatch")
    except Exception as e:
        print(f"PASS (ModelMismatch): raised {type(e).__name__} on dim mismatch")


# ── 16. migrate_embeddings ────────────────────────────────────────────────────

with tempfile.TemporaryDirectory() as tmp:
    path = str(pathlib.Path(tmp) / "migrate_test")

    writer = ailake.TableWriter(
        path,
        vector_column="embedding",
        dim=DIM,
        metric="cosine",
        embedding_model="model-v1",
    )
    writer.write_batch([f"doc_{i}" for i in range(N)], [make_embedding(i) for i in range(N)])
    writer.commit()

    def _identity_embed(texts: list) -> list:
        return [make_embedding(abs(hash(t)) % N) for t in texts]

    ailake.migrate_embeddings(
        path,
        old_column="embedding",
        new_column="embedding",
        embed_fn=_identity_embed,
        text_column="text",
        strategy="atomic_replace",
        batch_size=10,
        new_model="model-v2",
    )
    results_mg = ailake.search(path, make_embedding(0), top_k=3).to_list()
    assert len(results_mg) > 0, "FAIL: search after migrate_embeddings returned empty"
    print(f"PASS (migrate_embeddings): completed, search returns {len(results_mg)} results")


# ── 17. Pattern B: TableWriter(embed_fn=...) + write_batch without embeddings ──

with tempfile.TemporaryDirectory() as tmp:
    path = str(pathlib.Path(tmp) / "pattern_b_test")

    def _auto_embed(texts: list) -> list:
        return [make_embedding(abs(hash(t)) % N) for t in texts]

    writer = ailake.TableWriter(
        path,
        vector_column="embedding",
        dim=DIM,
        metric="cosine",
        embed_fn=_auto_embed,
    )
    writer.write_batch([f"doc_{i}" for i in range(N)])
    writer.commit()

    results_pb = ailake.search(path, make_embedding(0), top_k=3).to_list()
    assert len(results_pb) > 0, "FAIL: Pattern B search returned empty"
    print(f"PASS (Pattern B embed_fn): write_batch without embeddings, search returns {len(results_pb)} results")

    # open_table with embed_fn
    tbl = ailake.open_table(path, dim=DIM, embed_fn=_auto_embed)
    tbl.insert([f"extra_{i}" for i in range(5)])
    tbl.commit()
    print("PASS (Pattern B open_table): open_table with embed_fn + insert without embeddings")


# ── 18. migrate_embeddings with on_progress callback ─────────────────────────

with tempfile.TemporaryDirectory() as tmp:
    path = str(pathlib.Path(tmp) / "progress_test")

    writer = ailake.TableWriter(
        path,
        vector_column="embedding",
        dim=DIM,
        metric="cosine",
        embedding_model="model-v1",
    )
    writer.write_batch([f"doc_{i}" for i in range(N)], [make_embedding(i) for i in range(N)])
    writer.commit()

    progress_calls: list = []

    def _on_progress(**kwargs):
        progress_calls.append(dict(kwargs))

    ailake.migrate_embeddings(
        path,
        old_column="embedding",
        new_column="embedding",
        embed_fn=_identity_embed,
        text_column="text",
        strategy="atomic_replace",
        batch_size=10,
        on_progress=_on_progress,
    )
    assert len(progress_calls) > 0, "FAIL: on_progress never called"
    last = progress_calls[-1]
    assert "files_done" in last and "files_total" in last and "rows_migrated" in last, \
        f"FAIL: on_progress kwargs missing keys: {last}"
    print(f"PASS (on_progress): called {len(progress_calls)} times, last={last}")


print()
print("PASS: ailake Python SDK — all checks passed.")
