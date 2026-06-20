# SPDX-License-Identifier: MIT OR Apache-2.0
# Copyright (c) 2026 Thiago Egon Lange
#!/usr/bin/env python3
"""
Writes a fixture AI-Lake table for the DuckDB extension integration tests.

Uses the ailake-jni C-ABI (libailake_jni.so) to write the table at:
  {output_dir}/default/table/   (namespace=default, table=table)

This matches the hardcoded namespace/table that ailake_search() in the DuckDB
extension passes when given only a warehouse path as first argument.

Environment (in priority order):
  AILAKE_LIB          — full path to libailake_jni.so  (used by ci-duckdb.yml)
  AILAKE_NATIVE_LIB   — alias for AILAKE_LIB
  AILAKE_LIB_PATH     — directory containing libailake_jni.so
  LD_LIBRARY_PATH     — standard dynamic linker search (fallback)

Usage:
    python tests/fixtures/write_fixture.py [output_dir]
    output_dir defaults to ./compat-fixture
"""

import sys
import os
import json
import ctypes
import math
import struct
import pathlib

DIM = 128
N = 1000

# ── Load library ───────────────────────────────────────────────────────────────

def _load_lib():
    for env in ("AILAKE_LIB", "AILAKE_NATIVE_LIB"):
        explicit = os.environ.get(env)
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
    print("FAIL: libailake_jni.so not found.")
    print("      Build with: cargo build --release -p ailake-jni")
    print("      Then set AILAKE_LIB=target/release/libailake_jni.so")
    sys.exit(1)

lib.ailake_write_batch_json.argtypes = [ctypes.c_char_p]
lib.ailake_write_batch_json.restype = ctypes.c_void_p
lib.ailake_search_text_json.argtypes = [ctypes.c_char_p]
lib.ailake_search_text_json.restype = ctypes.c_void_p
lib.ailake_free_string.argtypes = [ctypes.c_void_p]
lib.ailake_free_string.restype = None


def _call_write(req: dict) -> dict:
    ptr = lib.ailake_write_batch_json(json.dumps(req).encode())
    try:
        return json.loads(ctypes.string_at(ptr).decode())
    finally:
        lib.ailake_free_string(ptr)


def make_embedding(i: int) -> list:
    v = [float((i * DIM + j + 1) % 97) for j in range(DIM)]
    norm = math.sqrt(sum(x * x for x in v))
    return [x / norm for x in v]


# ── Write fixture ──────────────────────────────────────────────────────────────

out_dir = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else "./compat-fixture")
out_dir.mkdir(parents=True, exist_ok=True)

warehouse = str(out_dir.resolve())
namespace = "default"
table = "table"

print(f"writing {N} rows (dim={DIM}) to {warehouse}/{namespace}/{table}/")

embeddings = [make_embedding(i) for i in range(N)]
ids = list(range(N))

resp = _call_write({
    "warehouse": warehouse,
    "namespace": namespace,
    "table": table,
    "vec_col": "embedding",
    "dim": DIM,
    "metric": "cosine",
    "precision": "f16",
    "embedding_model": "fixture-model@v1",
    "ids": ids,
    "embeddings": embeddings,
})

assert resp.get("ok"), f"write_batch failed: {resp}"
print(f"committed: snapshot_id={resp['snapshot_id']}")

# ── Phase T: FTS fixture (table="fts_table") — for Go + DuckDB FTS integration ──

FTS_TEXTS = [
    "rust programming ownership memory safety systems",
    "python machine learning data science numpy pandas",
    "rust async tokio concurrency futures channels",
    "database sql query optimization index btree",
    "vector search approximate nearest neighbor hnsw embeddings",
    "distributed computing apache spark hadoop mapreduce",
    "rust cargo crates dependencies ecosystem toolchain",
    "deep learning neural network transformer attention mechanism",
]
fts_embeddings = [make_embedding(i) for i in range(len(FTS_TEXTS))]

def _call_search_text(req: dict) -> dict:
    ptr = lib.ailake_search_text_json(json.dumps(req).encode())
    try:
        return json.loads(ctypes.string_at(ptr).decode())
    finally:
        lib.ailake_free_string(ptr)

# Use a separate warehouse for the FTS table so its data files don't collide with
# the main table's data/part-00000.parquet (both write relative to warehouse root).
fts_warehouse = str((out_dir / "fts").resolve())
pathlib.Path(fts_warehouse).mkdir(parents=True, exist_ok=True)

resp_fts = _call_write({
    "warehouse": fts_warehouse,
    "namespace": namespace,
    "table": "fts_table",
    "vec_col": "embedding",
    "dim": DIM,
    "metric": "cosine",
    "precision": "f16",
    "ids": list(range(len(FTS_TEXTS))),
    "embeddings": fts_embeddings,
    "fts_columns": ["text"],
    "fts_tokenizer": "default",
    "columns": {"text": FTS_TEXTS},
})
assert resp_fts.get("ok"), f"FTS write_batch failed: {resp_fts}"
print(f"fts_table committed: snapshot_id={resp_fts['snapshot_id']}  rows={len(FTS_TEXTS)}  fts_col=text")

# Smoke-check FTS search so the fixture is guaranteed searchable
resp_txt = _call_search_text({
    "warehouse": fts_warehouse,
    "namespace": namespace,
    "table": "fts_table",
    "query_text": "rust",
    "text_columns": ["text"],
    "top_k": 5,
})
assert resp_txt.get("ok"), f"FTS smoke-search failed: {resp_txt}"
fts_hits = resp_txt.get("results", [])
assert any(r["row_id"] in (0, 2, 6) for r in fts_hits), \
    f"FTS smoke-search: expected rust rows (0,2,6), got {[r['row_id'] for r in fts_hits]}"
print(f"fts_table smoke-search: {len(fts_hits)} hit(s) for 'rust'  row_ids={[r['row_id'] for r in fts_hits]}")

(out_dir / "fixture_fts_rows.txt").write_text(str(len(FTS_TEXTS)))

# ── fixture_query.bin — query vector for test_search.py ───────────────────────

query_vec = make_embedding(0)
query_bytes = struct.pack(f"<{DIM}f", *query_vec)
query_file = out_dir / "fixture_query.bin"
query_file.write_bytes(query_bytes)
print(f"saved query vector ({DIM}-dim f32) → {query_file}")

# ── fixture_files.txt / fixture_rows.txt — for check_duckdb.py ───────────────

data_dir = out_dir / namespace / table / "data"
parquet_files = sorted(data_dir.glob("*.parquet")) if data_dir.exists() else []

(out_dir / "fixture_files.txt").write_text(
    "\n".join(str(p.resolve()) for p in parquet_files)
)
(out_dir / "fixture_rows.txt").write_text(str(N))

print(f"found {len(parquet_files)} parquet file(s) under {data_dir}")
print(f"fixture ready at {out_dir}")
