# SPDX-License-Identifier: MIT OR Apache-2.0
# Copyright (c) 2026 Thiago Egon Lange
"""Real end-to-end tests for the Fase 15 capability-parity fixes.

No mocks: every test builds a real table via TableWriter/Table and exercises
the actual native (PyO3) code path. Run against a `maturin develop` build:

    maturin develop --release
    pytest ailake-py/tests/
"""
from __future__ import annotations

import random

import pytest

import ailake


def _rand_vec(dim: int) -> list[float]:
    return [random.random() for _ in range(dim)]


def _embed_fn(texts: list[str]) -> list[list[float]]:
    random.seed(hash(tuple(texts)) & 0xFFFFFFFF)
    return [_rand_vec(4) for _ in texts]


# ── Finding #3: now_ns()/TimestampNs → real Timestamp column ───────────────

def test_timestamp_ns_column_is_decay_memories_compatible(tmp_path):
    path = str(tmp_path / "t1")
    w = ailake.TableWriter(path, dim=4)
    texts = [f"doc {i}" for i in range(6)]
    ts = ailake.TimestampNs(ailake.now_ns())
    w.write_batch(texts, [_rand_vec(4) for _ in texts], extra_columns={
        "last_accessed_at": [ts] * 6,
    })
    w.commit()
    # Previously: build_batch_with_extra had no Timestamp branch, so a plain
    # int from now_ns() became Int64 — decay_memories() then raised
    # PyValueError("last_accessed_at must be Timestamp(...) or Utf8").
    updated = ailake.decay_memories(path)
    assert updated == 1


# ── Finding #1/#12: assemble_context dedup + config/output fields ──────────

def test_assemble_context_dedups_by_embedding(tmp_path):
    chunks = [
        {"document_id": "d1", "chunk_index": 0, "chunk_text": "hello world", "embedding": [1.0, 0.0]},
        {"document_id": "d1", "chunk_index": 1, "chunk_text": "hello world dup", "embedding": [1.0, 0.0001]},
        {"document_id": "d1", "chunk_index": 2, "chunk_text": "totally different", "embedding": [0.0, 1.0]},
    ]
    ctx = ailake.assemble_context(chunks, dedup_threshold=0.05)
    assert isinstance(ctx, dict)
    assert set(ctx.keys()) == {"text", "chunk_count", "token_estimate"}
    # Previously: embedding was hardcoded to None, so dedup was a permanent no-op.
    assert ctx["chunk_count"] == 2


def test_assemble_context_without_embedding_no_dedup():
    chunks = [
        {"document_id": "d1", "chunk_index": 0, "chunk_text": "a"},
        {"document_id": "d1", "chunk_index": 1, "chunk_text": "b"},
    ]
    ctx = ailake.assemble_context(chunks)
    assert ctx["chunk_count"] == 2


def test_assemble_context_respects_max_chunks_per_document():
    chunks = [
        {"document_id": "d1", "chunk_index": i, "chunk_text": f"chunk {i}"}
        for i in range(15)
    ]
    ctx = ailake.assemble_context(chunks, max_chunks_per_document=3)
    assert ctx["chunk_count"] <= 3


# ── Finding #2: search(fetch_data=True) parity with pointer-only search ────

def test_search_fetch_data_honors_hybrid_and_ef_search(tmp_path):
    path = str(tmp_path / "t2")
    w = ailake.TableWriter(path, dim=4, bm25_text_column="text")
    w.write_batch(
        ["rust async programming", "python sync code", "golang concurrency"],
        [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]],
    )
    w.commit()

    q = ailake.search(
        path, [1.0, 0.0, 0.0, 0.0], top_k=3, fetch_data=True,
        hybrid_text="rust", text_column="text", ef_search=200,
    )
    table = q.to_arrow()
    assert table.num_rows > 0

    # Same SearchQuery object: to_list() must stay pointer-only regardless.
    rows = q.to_list()
    assert set(rows[0].keys()) == {"row_id", "distance", "file"}


def test_search_with_data_and_scan_are_same_capability(tmp_path):
    path = str(tmp_path / "t2b")
    w = ailake.TableWriter(path, dim=4)
    w.write_batch(["a", "b"], [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]])
    w.commit()
    raw = ailake.scan(path, [1.0, 0.0, 0.0, 0.0], top_k=2)
    assert isinstance(raw, (bytes, bytearray))
    assert ailake.scan is ailake.search_with_data


# ── Finding #7: compact() native binding (no CLI subprocess) ───────────────

def test_compact_native_merges_files(tmp_path):
    path = str(tmp_path / "t3")
    for _ in range(2):
        w = ailake.TableWriter(path, dim=4)
        w.write_batch([f"t{i}" for i in range(3)], [_rand_vec(4) for _ in range(3)])
        w.commit()

    result = ailake.compact(path, min_files=2)
    assert result["ok"] is True
    assert result["files_compacted"] == 1
    assert isinstance(result["output_path"], str)

    # Nothing left to compact — must not spuriously report ok+0 as if the
    # native path silently failed (the old subprocess design's failure mode).
    result2 = ailake.compact(path, min_files=2)
    assert result2 == {"ok": True, "files_compacted": 0, "output_path": None}


# ── Finding #11: estimate() ─────────────────────────────────────────────────

def test_estimate_returns_six_precision_modes():
    rows = ailake.estimate(rows=1_000_000, dim=1536)
    assert len(rows) == 6
    modes = {r["mode"] for r in rows}
    assert "F32 (baseline)" in modes
    assert "PQ-only (pq_only=True)" in modes
    f32 = next(r for r in rows if r["mode"] == "F32 (baseline)")
    f16 = next(r for r in rows if r["mode"] == "F16 (default)")
    assert f32["vectors_bytes"] > f16["vectors_bytes"]
    assert f32["vectors_bytes"] == 1_000_000 * 1536 * 4


# ── Finding #5: add_vector_column / backfill_vector_column exported ────────

def test_add_and_backfill_vector_column_reachable(tmp_path):
    path = str(tmp_path / "t4")
    w = ailake.TableWriter(path, dim=4)
    w.write_batch(["a", "b"], [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]],
                  extra_columns={"chunk_text": ["a", "b"]})
    w.commit()

    schema_id = ailake.add_vector_column(path, "image_embedding", 2)
    assert schema_id >= 0

    ailake.backfill_vector_column(path, "image_embedding", _embed_fn, text_column="chunk_text")


# ── Finding #6: write_batch_ivf_pq / _deferred ──────────────────────────────

def test_write_batch_ivf_pq(tmp_path):
    path = str(tmp_path / "t5")
    w = ailake.TableWriter(path, dim=4)
    w.write_batch_ivf_pq([f"v{i}" for i in range(20)], [_rand_vec(4) for _ in range(20)])
    w.commit()


def test_write_batch_ivf_pq_deferred(tmp_path):
    path = str(tmp_path / "t5b")
    w = ailake.TableWriter(path, dim=4)
    w.write_batch_ivf_pq_deferred([f"v{i}" for i in range(20)], [_rand_vec(4) for _ in range(20)])
    w.commit()


# ── Finding #8/#9: write_batch_multi extra_columns + per-column VectorColSpec ──

def test_write_batch_multi_extra_columns_and_per_column_spec(tmp_path):
    path = str(tmp_path / "t6")
    w = ailake.TableWriter(path, dim=4)
    spec_text = ailake.VectorColSpec("embedding", 4, metric="cosine")
    spec_img = ailake.VectorColSpec(
        "image_embedding", 2, metric="cosine", modality="image",
        precision="f32", hnsw_m=8,
    )
    assert spec_img.precision == "f32"
    assert spec_img.hnsw_m == 8

    w.write_batch_multi(
        ["m1", "m2"],
        [(spec_text, [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]]),
         (spec_img, [[0.5, 0.5], [0.2, 0.8]])],
        extra_columns={"media_uri": ["s3://a", "s3://b"]},
    )
    w.commit()

    results = ailake.search_multimodal(
        path,
        [("embedding", [1.0, 0.0, 0.0, 0.0], 0.7), ("image_embedding", [0.5, 0.5], 0.3)],
        top_k=2,
    )
    assert len(results) == 2


# ── Finding #13: Table fluent API — write_batch_idempotent / write_batch_multi ──

def test_table_write_batch_idempotent(tmp_path):
    path = str(tmp_path / "t7")
    t = ailake.open_table(path, dim=4)
    t.write_batch_idempotent(["x1"], [[1.0, 0.0, 0.0, 0.0]], batch_id="batch-1")
    snap1 = t.commit()

    # Re-running the same batch_id must be a no-op (idempotent).
    t2 = ailake.open_table(path, dim=4)
    t2.write_batch_idempotent(["x1"], [[1.0, 0.0, 0.0, 0.0]], batch_id="batch-1")
    snap2 = t2.commit()
    assert snap1 is not None
    assert snap2 == 0 or snap2 == snap1


def test_table_write_batch_multi(tmp_path):
    path = str(tmp_path / "t7b")
    t = ailake.open_table(path, dim=4)
    t.write_batch_multi(["m1"], [(ailake.VectorColSpec("embedding", 4), [[1.0, 0.0, 0.0, 0.0]])])
    t.commit()


# ── Finding #4: Agent — real typed columns, decay_memories compatible ──────

def test_agent_uses_real_columns_and_is_decay_compatible(tmp_path):
    path = str(tmp_path / "t8")
    agent = ailake.Agent(path, embed_fn=_embed_fn, agent_id="agent-x")
    mem_id = agent.remember("user likes concise answers", importance=0.9)
    call_id = agent.log_tool_call(
        "web_search", {"q": "rust"}, {"hits": 3}, outcome="success", latency_ms=120,
    )
    agent.commit()

    results = agent.recall(_embed_fn(["concise"])[0], top_k=5)
    assert len(results) == 2

    types = {r["type"] for r in results}
    assert types == {"memory", "tool_call"}

    mem_entry = next(r for r in results if r["type"] == "memory")
    assert mem_entry["mem_id"] == mem_id

    tool_entry = next(r for r in results if r["type"] == "tool_call")
    assert tool_entry["call_id"] == call_id
    assert tool_entry["tool_name"] == "web_search"
    assert tool_entry["outcome"] == "success"
    assert tool_entry["latency_ms"] == 120

    # Previously: Agent packed metadata as a JSON prefix on `text` — no real
    # last_accessed_at column existed, so decay_memories() silently matched
    # zero files and always returned 0.
    updated = ailake.decay_memories(path)
    assert updated >= 1


def test_agent_assemble_context_returns_string(tmp_path):
    path = str(tmp_path / "t8b")
    agent = ailake.Agent(path, embed_fn=_embed_fn, agent_id="agent-y")
    agent.remember("some memory", importance=0.5)
    agent.commit()

    ctx = agent.assemble_context(_embed_fn(["some memory"])[0])
    assert isinstance(ctx, str)


# ── top_k cap + non-finite embedding rejection (post-Fase-15 safety fixes) ──
#
# search()/write_batch() call ailake_query::scanner/writer directly (not through
# ailake-jni's C-ABI, which had its own top_k cap) — these two behaviors used to
# only be enforced for Spark/Trino/Flink, not for this binding. No prior test
# coverage existed for either.

def test_search_rejects_top_k_over_cap(tmp_path):
    path = str(tmp_path / "t9")
    w = ailake.TableWriter(path, dim=4)
    w.write_batch(["a"], [[1.0, 0.0, 0.0, 0.0]])
    w.commit()

    # ailake.search() builds a lazy SearchQuery — the Rust call (and the top_k
    # validation) only happens when the query is materialized.
    with pytest.raises(ValueError, match="top_k"):
        ailake.search(path, [1.0, 0.0, 0.0, 0.0], top_k=200_000).to_list()


def test_write_batch_rejects_non_finite_embedding(tmp_path):
    path = str(tmp_path / "t10")
    w = ailake.TableWriter(path, dim=4)
    with pytest.raises(ValueError, match="non-finite"):
        w.write_batch(["a"], [[1.0, float("nan"), 0.0, 0.0]])


# ── New: create_table (Fase 15+) ──────────────────────────────────────────────

def test_create_empty_table_search_returns_zero_rows(tmp_path):
    path = str(tmp_path / "empty_table")
    ailake.create_table(path, dim=4)
    results = ailake.search(path, [1.0, 0.0, 0.0, 0.0], top_k=10).to_list()
    assert len(results) == 0


def test_create_table_duplicate_raises(tmp_path):
    path = str(tmp_path / "dup_table")
    ailake.create_table(path, dim=4)
    with pytest.raises((ValueError, RuntimeError)):
        ailake.create_table(path, dim=4)


def test_create_table_with_custom_params(tmp_path):
    path = str(tmp_path / "custom_params")
    ailake.create_table(
        path, dim=768, vector_column="my_vec",
        metric="euclidean", precision="f32",
        format_version=2,
    )
    results = ailake.search(path, [1.0] * 768, top_k=5).to_list()
    assert len(results) == 0


def test_create_table_then_write_then_search(tmp_path):
    path = str(tmp_path / "crt_write_search")
    ailake.create_table(path, dim=4)
    w = ailake.TableWriter(path, dim=4)
    w.write_batch(["hello", "world"], [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]])
    w.commit()
    results = ailake.search(path, [1.0, 0.0, 0.0, 0.0], top_k=10).to_list()
    assert len(results) == 2
