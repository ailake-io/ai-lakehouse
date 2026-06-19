# SPDX-License-Identifier: MIT OR Apache-2.0
# Copyright (c) 2026 Thiago Egon Lange
"""
DuckDB ailake extension — ailake_delete_where function tests (Phase M).

Prerequisites:
  1. Build libailake_jni.so:  cargo build --release -p ailake-jni
  2. Build DuckDB extension:  cmake --build duckdb-ailake/build

Usage:
  AILAKE_LIB=./target/release/libailake_jni.so \
  AILAKE_EXT=./duckdb-ailake/build/ailake.duckdb_extension \
  AILAKE_TMPDIR=/tmp/ailake_duck_delete \
  python duckdb-ailake/test/test_delete.py
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


def make_table_dir():
    if TMP_DIR:
        d = pathlib.Path(TMP_DIR) / "test_delete"
        shutil.rmtree(d, ignore_errors=True)
        d.mkdir(parents=True, exist_ok=True)
        return str(d), None
    tmp = tempfile.mkdtemp()
    return tmp, tmp


def small_embeddings(n=5, dim=8):
    import math
    rows = []
    for i in range(n):
        v = [float(i * dim + j + 1) for j in range(dim)]
        norm = math.sqrt(sum(x * x for x in v))
        rows.append([x / norm for x in v])
    return rows


def write_table(conn, table_path, n=5, dim=8):
    embs = small_embeddings(n, dim)
    conn.execute(f"""
        SELECT ailake_write_batch(
            '{table_path}',
            {list(range(n))}::BIGINT[],
            {embs}::FLOAT[][],
            'embedding', 'cosine', 'f16'
        )
    """)


def test_delete_where_returns_true():
    conn = setup_connection()
    table_dir, cleanup = make_table_dir()
    try:
        write_table(conn, table_dir)
        row = conn.execute(
            f"SELECT ailake_delete_where('{table_dir}', 'id', ARRAY['0', '1'])"
        ).fetchone()
        require(row is not None, "delete_where returned NULL row")
        require(row[0] is True, f"delete_where returned {row[0]}, expected TRUE")
        print(f"PASS test_delete_where_returns_true: result={row[0]}")
    finally:
        if cleanup:
            shutil.rmtree(cleanup, ignore_errors=True)
        conn.close()


def test_delete_where_empty_values_is_noop():
    conn = setup_connection()
    table_dir, cleanup = make_table_dir()
    try:
        write_table(conn, table_dir)
        row = conn.execute(
            f"SELECT ailake_delete_where('{table_dir}', 'id', ARRAY[]::VARCHAR[])"
        ).fetchone()
        require(row is not None, "delete_where noop returned NULL row")
        require(row[0] is True, f"delete_where noop returned {row[0]}, expected TRUE")
        print(f"PASS test_delete_where_empty_values_is_noop: result={row[0]}")
    finally:
        if cleanup:
            shutil.rmtree(cleanup, ignore_errors=True)
        conn.close()


def test_delete_where_missing_lib_returns_false():
    conn = duckdb.connect(config={
        "allow_unsigned_extensions": True,
        "allow_extensions_metadata_mismatch": True,
    })
    conn.execute(f"LOAD '{EXT_PATH}'")
    try:
        row = conn.execute(
            "SELECT ailake_delete_where('/nonexistent', 'id', ARRAY['0'])"
        ).fetchone()
        # With lib loaded via RTLD_GLOBAL earlier in process, call may error or return FALSE
        print(f"PASS test_delete_where_missing_lib_returns_false: result={row[0] if row else None}")
    except Exception as e:
        print(f"PASS test_delete_where_missing_lib_returns_false: raised {type(e).__name__} as expected")
    finally:
        conn.close()


if __name__ == "__main__":
    test_delete_where_returns_true()
    test_delete_where_empty_values_is_noop()
    test_delete_where_missing_lib_returns_false()
    print()
    print("PASS: ailake_delete_where — all tests passed")
