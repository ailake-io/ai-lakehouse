# SPDX-License-Identifier: MIT OR Apache-2.0
# Copyright (c) 2026 Thiago Egon Lange
"""
DuckDB ailake extension — ailake_compact function tests.

Prerequisites:
  1. Build libailake_jni.so:  cargo build --release -p ailake-jni
  2. Build DuckDB extension:  cmake --build duckdb-ailake/build

Usage:
  AILAKE_LIB=./target/release/libailake_jni.so \
  AILAKE_EXT=./duckdb-ailake/build/ailake.duckdb_extension \
  AILAKE_TMPDIR=/tmp/ailake_duck_compact \
  python duckdb-ailake/test/test_compact.py
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
        d = pathlib.Path(TMP_DIR) / f"test_compact{suffix}"
        shutil.rmtree(d, ignore_errors=True)
        d.mkdir(parents=True, exist_ok=True)
        return str(d), None
    tmp = tempfile.mkdtemp()
    return tmp, tmp


def write_small_batch(conn, table_path, start_id, n=2, dim=4):
    ids = list(range(start_id, start_id + n))
    embs = [[float(i + j) for j in range(dim)] for i in ids]
    conn.execute(f"""
        SELECT ailake_write_batch(
            '{table_path}',
            {ids}::BIGINT[],
            {embs}::FLOAT[][],
            'embedding', 'cosine', 'f16'
        )
    """)


def test_compact_nothing_eligible_returns_zero():
    conn = setup_connection()
    table_dir, cleanup = make_table_dir("_noop")
    try:
        write_small_batch(conn, table_dir, 0)
        # Default min_files=4, only 1 file written — nothing eligible.
        row = conn.execute(f"SELECT ailake_compact('{table_dir}')").fetchone()
        require(row is not None, "compact returned NULL row")
        require(row[0] == 0, f"expected 0 files compacted, got {row[0]}")
        print(f"PASS test_compact_nothing_eligible_returns_zero: files_compacted={row[0]}")
    finally:
        if cleanup:
            shutil.rmtree(cleanup, ignore_errors=True)
        conn.close()


def test_compact_merges_small_files_and_preserves_rows():
    conn = setup_connection()
    table_dir, cleanup = make_table_dir("_merge")
    try:
        write_small_batch(conn, table_dir, 0)
        write_small_batch(conn, table_dir, 2)

        row = conn.execute(f"SELECT ailake_compact('{table_dir}', min_files := 2)").fetchone()
        require(row is not None, "compact returned NULL row")
        require(row[0] == 1, f"expected 1 file compacted, got {row[0]}")

        results = conn.execute(f"""
            SELECT * FROM ailake_search(
                '{table_dir}', [0.0, 1.0, 2.0, 3.0]::FLOAT[], 10
            )
        """).fetchall()
        require(len(results) == 4, f"expected all 4 rows searchable after compact, got {len(results)}")
        print(f"PASS test_compact_merges_small_files_and_preserves_rows: files_compacted={row[0]}, rows={len(results)}")
    finally:
        if cleanup:
            shutil.rmtree(cleanup, ignore_errors=True)
        conn.close()


def test_compact_missing_table_raises():
    conn = setup_connection()
    table_dir, cleanup = make_table_dir("_missing")
    try:
        # Table was never created — load_table fails inside the native call
        # (ok:false). Previously folded into a silent -1 return; now raised as
        # a clear SQL error so the actual reason reaches the caller.
        try:
            conn.execute(f"SELECT ailake_compact('{table_dir}')").fetchone()
            require(False, "expected an exception for a nonexistent table, got a result")
        except duckdb.Error as e:
            require("ailake_compact failed" in str(e), f"unexpected error message: {e}")
            print("PASS test_compact_missing_table_raises")
    finally:
        if cleanup:
            shutil.rmtree(cleanup, ignore_errors=True)
        conn.close()


if __name__ == "__main__":
    test_compact_nothing_eligible_returns_zero()
    test_compact_merges_small_files_and_preserves_rows()
    test_compact_missing_table_raises()
    print()
    print("PASS: ailake_compact — all tests passed")
