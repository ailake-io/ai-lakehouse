# SPDX-License-Identifier: MIT OR Apache-2.0
# Copyright (c) 2026 Thiago Egon Lange
"""
DuckDB ailake extension — ailake_create_table function tests.

Verifies that:
  1. Creating an empty table succeeds.
  2. Searching on an empty table returns 0 rows (no error).
  3. Creating the same table again raises an error.
  4. Custom vector_column/metric/precision are honored.
  5. create → insert → search round-trips correctly.

Prerequisites:
  1. Build libailake_jni.so:  cargo build --release -p ailake-jni
  2. Build DuckDB extension:  cmake --build duckdb-ailake/build

Usage:
  AILAKE_LIB=./target/release/libailake_jni.so \
  AILAKE_EXT=./duckdb-ailake/build/ailake.duckdb_extension \
  AILAKE_TMPDIR=/tmp/ailake_duck_create_table \
  python duckdb-ailake/test/test_create_table.py
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


def make_table_dir(name):
    if TMP_DIR:
        d = pathlib.Path(TMP_DIR) / name
        shutil.rmtree(d, ignore_errors=True)
        d.mkdir(parents=True, exist_ok=True)
        return str(d), None
    tmp = tempfile.mkdtemp()
    return str(pathlib.Path(tmp) / name), tmp


def float_list_sql(dim):
    return "[" + ",".join(["0.1"] * dim) + "]::FLOAT[]"


def test_create_empty_table_search_returns_zero_rows():
    conn = setup_connection()
    table_dir, cleanup = make_table_dir("empty_table")
    try:
        # Signature: (table_path, dim, vector_column, metric, precision, format_version).
        conn.execute(f"SELECT ailake_create_table('{table_dir}', 1536, 'embedding', 'cosine', 'f16', 2)")
        rows = conn.execute(f"SELECT * FROM ailake_search('{table_dir}', {float_list_sql(1536)}, 10)").fetchall()
        require(len(rows) == 0, f"expected 0 results from empty table, got {len(rows)}")
        print("PASS test_create_empty_table_search_returns_zero_rows")
    finally:
        if cleanup:
            shutil.rmtree(cleanup, ignore_errors=True)
        conn.close()


def test_create_duplicate_raises():
    conn = setup_connection()
    table_dir, cleanup = make_table_dir("dup_table")
    try:
        conn.execute(f"SELECT ailake_create_table('{table_dir}', 1536, 'embedding', 'cosine', 'f16', 2)")
        try:
            conn.execute(f"SELECT ailake_create_table('{table_dir}', 1536, 'embedding', 'cosine', 'f16', 2)")
            require(False, "expected an exception creating a duplicate table, got a result")
        except Exception as e:
            require(
                "already exists" in str(e) or "TableAlreadyExists" in str(e) or "Conflict" in str(e),
                f"unexpected error message: {e}",
            )
            print("PASS test_create_duplicate_raises")
    finally:
        if cleanup:
            shutil.rmtree(cleanup, ignore_errors=True)
        conn.close()


def test_create_with_custom_params():
    conn = setup_connection()
    table_dir, cleanup = make_table_dir("custom_table")
    try:
        conn.execute(f"SELECT ailake_create_table('{table_dir}', 768, 'my_vec', 'euclidean', 'f32', 2)")
        rows = conn.execute(
            f"SELECT * FROM ailake_search('{table_dir}', {float_list_sql(768)}, 5, vec_col='my_vec')"
        ).fetchall()
        require(len(rows) == 0, f"expected 0 results from empty table, got {len(rows)}")
        print("PASS test_create_with_custom_params")
    finally:
        if cleanup:
            shutil.rmtree(cleanup, ignore_errors=True)
        conn.close()


def test_create_then_insert_then_search():
    conn = setup_connection()
    table_dir, cleanup = make_table_dir("insert_table")
    try:
        conn.execute(f"SELECT ailake_create_table('{table_dir}', 3, 'embedding', 'cosine', 'f16', 2)")
        # Arity 3 (table_path, ids, embeddings) — defaults (embedding/cosine/f16) match
        # what the table was just created with, so no need to repeat them.
        conn.execute(f"""
            SELECT ailake_write_batch(
                '{table_dir}',
                [1, 2]::BIGINT[],
                [[1.0,0.0,0.0],[0.0,1.0,0.0]]::FLOAT[][]
            )
        """)
        rows = conn.execute(f"SELECT * FROM ailake_search('{table_dir}', [1.0,0.0,0.0]::FLOAT[], 5)").fetchall()
        require(len(rows) == 2, f"expected 2 rows after insert, got {len(rows)}")
        print("PASS test_create_then_insert_then_search")
    finally:
        if cleanup:
            shutil.rmtree(cleanup, ignore_errors=True)
        conn.close()


if __name__ == "__main__":
    test_create_empty_table_search_returns_zero_rows()
    test_create_duplicate_raises()
    test_create_with_custom_params()
    test_create_then_insert_then_search()
    print()
    print("PASS: ailake_create_table — all tests passed")
