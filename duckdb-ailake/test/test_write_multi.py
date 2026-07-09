# SPDX-License-Identifier: MIT OR Apache-2.0
# Copyright (c) 2026 Thiago Egon Lange
"""
DuckDB ailake extension — ailake_write_batch_multi function tests (Phase 8 multimodal).

Prerequisites:
  1. Build libailake_jni.so:  cargo build --release -p ailake-jni
  2. Build DuckDB extension:  cmake --build duckdb-ailake/build

Usage:
  AILAKE_LIB=./target/release/libailake_jni.so \
  AILAKE_EXT=./duckdb-ailake/build/ailake.duckdb_extension \
  AILAKE_TMPDIR=/tmp/ailake_duck_write_multi \
  python duckdb-ailake/test/test_write_multi.py
"""
import os
import sys
import pathlib
import tempfile
import shutil
import ctypes

_old_flags = sys.getdlopenflags()
sys.setdlopenflags(_old_flags | os.RTLD_GLOBAL)
import duckdb
sys.setdlopenflags(_old_flags)

EXT_PATH = os.environ.get("AILAKE_EXT", "./duckdb-ailake/build/ailake.duckdb_extension")
LIB_PATH = os.environ.get("AILAKE_LIB", "./target/release/libailake_jni.so")
TMP_DIR  = os.environ.get("AILAKE_TMPDIR", "")


def require(cond, msg):
    if not cond:
        print(f"FAIL: {msg}")
        sys.exit(1)


def setup_connection():
    conn = duckdb.connect(config={
        "allow_unsigned_extensions": True,
        "allow_extensions_metadata_mismatch": True,
    })
    ctypes.CDLL(LIB_PATH, ctypes.RTLD_GLOBAL)
    conn.execute(f"LOAD '{EXT_PATH}'")
    return conn


def make_table_dir(suffix=""):
    if TMP_DIR:
        d = pathlib.Path(TMP_DIR) / f"test_write_multi{suffix}"
        shutil.rmtree(d, ignore_errors=True)
        d.mkdir(parents=True, exist_ok=True)
        return str(d), None
    tmp = tempfile.mkdtemp()
    return tmp, tmp


def vector_columns_literal():
    # Field order must match the registered STRUCT type:
    # col, dim, embeddings, metric, precision, modality.
    return """[
        {'col': 'embedding', 'dim': 4, 'embeddings': [[1.0,0.0,0.0,0.0],[0.0,1.0,0.0,0.0],[0.0,0.0,1.0,0.0]]::FLOAT[][], 'metric': 'cosine', 'precision': 'f16', 'modality': ''},
        {'col': 'image_embedding', 'dim': 2, 'embeddings': [[1.0,0.0],[0.0,1.0],[0.5,0.5]]::FLOAT[][], 'metric': 'cosine', 'precision': 'f16', 'modality': ''}
    ]"""


def test_write_batch_multi_returns_snapshot_id():
    conn = setup_connection()
    table_dir, cleanup = make_table_dir("_basic")
    try:
        row = conn.execute(f"""
            SELECT ailake_write_batch_multi(
                '{table_dir}',
                [0, 1, 2]::BIGINT[],
                {vector_columns_literal()}
            )
        """).fetchone()
        require(row is not None, "write_batch_multi returned NULL row")
        require(row[0] > 0, f"write_batch_multi returned {row[0]}, expected a positive snapshot_id")
        print(f"PASS test_write_batch_multi_returns_snapshot_id: snapshot_id={row[0]}")
    finally:
        if cleanup:
            shutil.rmtree(cleanup, ignore_errors=True)
        conn.close()


def test_write_batch_multi_searchable_via_multimodal():
    conn = setup_connection()
    table_dir, cleanup = make_table_dir("_search")
    try:
        conn.execute(f"""
            SELECT ailake_write_batch_multi(
                '{table_dir}',
                [0, 1, 2]::BIGINT[],
                {vector_columns_literal()}
            )
        """)
        rows = conn.execute(f"""
            SELECT * FROM ailake_search_multimodal(
                '{table_dir}',
                [{{'col': 'embedding', 'query': [1.0,0.0,0.0,0.0]::FLOAT[], 'weight': 0.7}},
                 {{'col': 'image_embedding', 'query': [1.0,0.0]::FLOAT[], 'weight': 0.3}}],
                3
            ) ORDER BY rrf_score DESC
        """).fetchall()
        require(len(rows) == 3, f"expected 3 results, got {len(rows)}")
        require(rows[0][0] == 0, f"expected row_id 0 to rank first (exact match on both columns), got {rows[0][0]}")
        print(f"PASS test_write_batch_multi_searchable_via_multimodal: top result={rows[0]}")
    finally:
        if cleanup:
            shutil.rmtree(cleanup, ignore_errors=True)
        conn.close()


def test_write_batch_multi_rejects_mismatched_lengths():
    conn = setup_connection()
    table_dir, cleanup = make_table_dir("_mismatch")
    try:
        row = conn.execute(f"""
            SELECT ailake_write_batch_multi(
                '{table_dir}',
                [0, 1, 2]::BIGINT[],
                [{{'col': 'embedding', 'dim': 4, 'embeddings': [[1.0,0.0,0.0,0.0]]::FLOAT[][], 'metric': 'cosine', 'precision': 'f16', 'modality': ''}}]
            )
        """).fetchone()
        require(row is not None, "mismatched-length call returned NULL row")
        require(row[0] == -1, f"expected -1 for ids/embeddings length mismatch, got {row[0]}")
        print(f"PASS test_write_batch_multi_rejects_mismatched_lengths: result={row[0]}")
    finally:
        if cleanup:
            shutil.rmtree(cleanup, ignore_errors=True)
        conn.close()


if __name__ == "__main__":
    test_write_batch_multi_returns_snapshot_id()
    test_write_batch_multi_searchable_via_multimodal()
    test_write_batch_multi_rejects_mismatched_lengths()
    print()
    print("PASS: ailake_write_batch_multi — all tests passed")
