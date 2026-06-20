# SPDX-License-Identifier: MIT OR Apache-2.0
#!/usr/bin/env python3
"""
Validates the ailake-jni C-ABI:
  ailake_write_batch_json + ailake_search_json + ailake_search_multimodal_json +
  ailake_delete_where_json + ailake_evolve_schema_json.
This is the common interface used by all JVM connectors (Flink, Spark, Trino) via JNA.

Library location (in order):
  1. AILAKE_NATIVE_LIB env var — explicit path to the .so
  2. AILAKE_LIB_PATH env var — directory containing libailake_jni.so
  3. LD_LIBRARY_PATH (standard dynamic linker search)

Optional side-effect:
  If AILAKE_SPARK_TRINO_FIXTURE is set, writes a second fixture with table="table"
  (the hardcoded table name used by AilakeNative.search in Spark/Trino bridges).

Usage:
    AILAKE_NATIVE_LIB=target/release/libailake_jni.so \\
      python3 tests/compat/check_jni_cabi.py

    AILAKE_NATIVE_LIB=target/release/libailake_jni.so \\
    AILAKE_SPARK_TRINO_FIXTURE=$(pwd)/spark-trino-fixture \\
      python3 tests/compat/check_jni_cabi.py
"""

import sys
import os
import json
import ctypes
import math
import tempfile
import pathlib

DIM = 8
N = 20


# ── Load library ───────────────────────────────────────────────────────────────

def _load_lib():
    explicit = os.environ.get("AILAKE_NATIVE_LIB")
    if explicit:
        return ctypes.CDLL(explicit)
    lib_dir = os.environ.get("AILAKE_LIB_PATH")
    if lib_dir:
        for name in ("libailake_jni.so", "ailake_jni.dll", "libailake_jni.dylib"):
            candidate = pathlib.Path(lib_dir) / name
            if candidate.exists():
                return ctypes.CDLL(str(candidate))
    try:
        return ctypes.CDLL("libailake_jni.so")
    except OSError:
        return None


lib = _load_lib()
if lib is None:
    print("SKIP: libailake_jni.so not found.")
    print("      Build with: cargo build --release -p ailake-jni")
    print("      Then: AILAKE_NATIVE_LIB=target/release/libailake_jni.so python3 ...")
    sys.exit(0)

# Wire up C-ABI signatures
lib.ailake_version.argtypes = []
lib.ailake_version.restype = ctypes.c_char_p

lib.ailake_write_batch_json.argtypes = [ctypes.c_char_p]
lib.ailake_write_batch_json.restype = ctypes.c_void_p

lib.ailake_search_json.argtypes = [ctypes.c_char_p]
lib.ailake_search_json.restype = ctypes.c_void_p

lib.ailake_search_multimodal_json.argtypes = [ctypes.c_char_p]
lib.ailake_search_multimodal_json.restype = ctypes.c_void_p

lib.ailake_delete_where_json.argtypes = [ctypes.c_char_p]
lib.ailake_delete_where_json.restype = ctypes.c_void_p

lib.ailake_evolve_schema_json.argtypes = [ctypes.c_char_p]
lib.ailake_evolve_schema_json.restype = ctypes.c_void_p

lib.ailake_search_text_json.argtypes = [ctypes.c_char_p]
lib.ailake_search_text_json.restype = ctypes.c_void_p

lib.ailake_free_string.argtypes = [ctypes.c_void_p]
lib.ailake_free_string.restype = None

print(f"ailake-jni version: {lib.ailake_version().decode()}")


# ── Helpers ────────────────────────────────────────────────────────────────────

def _call(fn, req: dict) -> dict:
    ptr = fn(json.dumps(req).encode())
    try:
        return json.loads(ctypes.string_at(ptr).decode())
    finally:
        lib.ailake_free_string(ptr)


def make_embedding(i: int) -> list:
    v = [float(i * DIM + j + 1) for j in range(DIM)]
    norm = math.sqrt(sum(x * x for x in v))
    return [x / norm for x in v]


def write_fixture(warehouse: str, namespace: str, table: str) -> None:
    embeddings = [make_embedding(i) for i in range(N)]
    resp = _call(lib.ailake_write_batch_json, {
        "warehouse": warehouse,
        "namespace": namespace,
        "table": table,
        "vec_col": "embedding",
        "dim": DIM,
        "metric": "cosine",
        "precision": "f16",
        "ids": list(range(N)),
        "embeddings": embeddings,
    })
    assert resp.get("ok"), f"write_batch failed: {resp}"


def search_fixture(warehouse: str, namespace: str, table: str, query_idx: int) -> list:
    resp = _call(lib.ailake_search_json, {
        "warehouse": warehouse,
        "namespace": namespace,
        "table": table,
        "vec_col": "embedding",
        "dim": DIM,
        "query": make_embedding(query_idx),
        "top_k": 5,
        "ef_search": 50,
    })
    assert resp.get("ok"), f"search failed: {resp}"
    return resp["results"]


# ── Write + search in temp dir ─────────────────────────────────────────────────

with tempfile.TemporaryDirectory() as tmp:
    write_fixture(tmp, "default", "cabi_test")
    print(f"PASS (write): {N} rows written to temp warehouse")

    query_idx = 7
    results = search_fixture(tmp, "default", "cabi_test", query_idx)
    assert len(results) > 0, "FAIL: search returned empty results"
    best = min(results, key=lambda r: r["distance"])
    assert best["row_id"] == query_idx, (
        f"FAIL: nearest row_id={best['row_id']}, expected {query_idx}"
    )
    print(f"PASS (search): top-1 row_id={best['row_id']} distance={best['distance']:.6f}")
    print(f"      results={[(r['row_id'], round(r['distance'], 4)) for r in results]}")

# ── Write with embedding_model field ──────────────────────────────────────────

with tempfile.TemporaryDirectory() as tmp:
    embeddings = [make_embedding(i) for i in range(N)]
    resp = _call(lib.ailake_write_batch_json, {
        "warehouse": tmp,
        "namespace": "default",
        "table": "model_test",
        "vec_col": "embedding",
        "dim": DIM,
        "metric": "cosine",
        "precision": "f16",
        "embedding_model": "test-model@v2",
        "ids": list(range(N)),
        "embeddings": embeddings,
    })
    assert resp.get("ok"), f"FAIL: write with embedding_model failed: {resp}"
    print(f"PASS (embedding_model write): snapshot_id={resp['snapshot_id']}")

# ── ailake_search_multimodal_json — single-column RRF search ─────────────────

with tempfile.TemporaryDirectory() as tmp:
    write_fixture(tmp, "default", "mm_test")

    resp = _call(lib.ailake_search_multimodal_json, {
        "warehouse": tmp,
        "namespace": "default",
        "table": "mm_test",
        "queries": [
            {"col": "embedding", "query": make_embedding(3), "weight": 1.0, "dim": 0}
        ],
        "top_k": 5,
    })
    assert resp.get("ok"), f"FAIL: search_multimodal failed: {resp}"
    results = resp["results"]
    assert len(results) > 0, "FAIL: search_multimodal returned empty results"
    best = max(results, key=lambda r: r["rrf_score"])
    assert best["row_id"] == 3, (
        f"FAIL: multimodal top-1 row_id={best['row_id']}, expected 3"
    )
    assert best["rrf_score"] > 0, f"FAIL: rrf_score={best['rrf_score']}, expected > 0"
    print(f"PASS (search_multimodal): top-1 row_id={best['row_id']} rrf_score={best['rrf_score']:.6f}")

# ── ailake_delete_where_json ──────────────────────────────────────────────────

with tempfile.TemporaryDirectory() as tmp:
    write_fixture(tmp, "default", "del_test")

    resp = _call(lib.ailake_delete_where_json, {
        "warehouse": tmp,
        "namespace": "default",
        "table": "del_test",
        "column": "id",
        "values": ["0", "1", "2"],
    })
    assert resp.get("ok"), f"FAIL: delete_where failed: {resp}"
    print("PASS (delete_where): 3 rows marked deleted via equality delete")

    # Empty values list must be a no-op (ok=true, no file written)
    resp_noop = _call(lib.ailake_delete_where_json, {
        "warehouse": tmp,
        "namespace": "default",
        "table": "del_test",
        "column": "id",
        "values": [],
    })
    assert resp_noop.get("ok"), f"FAIL: delete_where empty noop: {resp_noop}"
    print("PASS (delete_where noop): empty values list is no-op")

# ── ailake_evolve_schema_json ─────────────────────────────────────────────────

with tempfile.TemporaryDirectory() as tmp:
    write_fixture(tmp, "default", "evo_test")

    resp = _call(lib.ailake_evolve_schema_json, {
        "warehouse": tmp,
        "namespace": "default",
        "table": "evo_test",
        "add_columns": [{"name": "source", "type": "string"}],
        "rename_columns": [],
    })
    assert resp.get("ok"), f"FAIL: evolve_schema failed: {resp}"
    assert "new_schema_id" in resp, f"FAIL: evolve_schema missing new_schema_id: {resp}"
    print(f"PASS (evolve_schema add_column): new_schema_id={resp['new_schema_id']}")

    # Empty add+rename must be a no-op
    resp_noop = _call(lib.ailake_evolve_schema_json, {
        "warehouse": tmp,
        "namespace": "default",
        "table": "evo_test",
        "add_columns": [],
        "rename_columns": [],
    })
    assert resp_noop.get("ok"), f"FAIL: evolve_schema empty noop: {resp_noop}"
    print("PASS (evolve_schema noop): empty add/rename is no-op")

# ── Optional: write Spark/Trino fixture (table="table") ───────────────────────

# ── ailake_write_batch_json with fts_columns (Phase T) ───────────────────────

with tempfile.TemporaryDirectory() as tmp:
    texts_fts = [
        "rust programming ownership memory safety",
        "python machine learning numpy pandas",
        "rust async tokio concurrency futures",
        "database sql query optimization btree",
    ]
    embeddings_fts = [make_embedding(i) for i in range(len(texts_fts))]

    resp = _call(lib.ailake_write_batch_json, {
        "warehouse": tmp,
        "namespace": "default",
        "table": "fts_test",
        "vec_col": "embedding",
        "dim": DIM,
        "metric": "cosine",
        "precision": "f16",
        "ids": list(range(len(texts_fts))),
        "embeddings": embeddings_fts,
        "fts_columns": ["text"],
        "fts_tokenizer": "default",
    })
    assert resp.get("ok"), f"FAIL: write with fts_columns failed: {resp}"
    print(f"PASS (fts write): snapshot_id={resp['snapshot_id']}")

    # search_text_json — Tantivy fast path when AILK_FTS blob present
    resp_txt = _call(lib.ailake_search_text_json, {
        "warehouse": tmp,
        "namespace": "default",
        "table": "fts_test",
        "query_text": "rust",
        "text_columns": ["text"],
        "top_k": 5,
    })
    assert resp_txt.get("ok"), f"FAIL: search_text_json failed: {resp_txt}"
    results = resp_txt.get("results", [])
    assert len(results) > 0, f"FAIL: search_text_json returned 0 results for 'rust'"
    hit_ids = {r["row_id"] for r in results}
    assert hit_ids & {0, 2}, \
        f"FAIL: expected rust rows (0 or 2) in results, got row_ids={sorted(hit_ids)}"
    print(f"PASS (fts search_text_json): {len(results)} hit(s) for 'rust', row_ids={sorted(hit_ids)}")

    # Empty fts_columns — write succeeds, no FTS blob embedded
    resp_nofts = _call(lib.ailake_write_batch_json, {
        "warehouse": tmp,
        "namespace": "default",
        "table": "fts_nofts",
        "vec_col": "embedding",
        "dim": DIM,
        "metric": "cosine",
        "precision": "f16",
        "ids": list(range(len(texts_fts))),
        "embeddings": embeddings_fts,
        "fts_columns": [],
    })
    assert resp_nofts.get("ok"), f"FAIL: write with empty fts_columns failed: {resp_nofts}"
    print(f"PASS (fts empty fts_columns no-op): snapshot_id={resp_nofts['snapshot_id']}")

# ── Optional: write Spark/Trino fixture (table="table") ───────────────────────

spark_trino_fixture = os.environ.get("AILAKE_SPARK_TRINO_FIXTURE")
if spark_trino_fixture:
    pathlib.Path(spark_trino_fixture).mkdir(parents=True, exist_ok=True)
    write_fixture(spark_trino_fixture, "default", "table")
    print(f"PASS (spark/trino fixture): written to {spark_trino_fixture}/default/table")

print()
print("PASS: ailake-jni C-ABI (write + search + search_multimodal + delete_where + evolve_schema + fts) — JNA bridge interface validated.")
