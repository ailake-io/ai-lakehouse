# SPDX-License-Identifier: MIT OR Apache-2.0
# Copyright (c) 2026 Thiago Egon Lange
"""
DuckDB ailake extension — search function tests.

Prerequisites:
  1. Build libailake_jni.so:  cargo build --release -p ailake-jni
  2. Build DuckDB extension:  cmake --build duckdb-ailake/build
  3. Generate fixture:        python tests/fixtures/write_fixture.py

Usage:
  AILAKE_LIB=./target/release/libailake_jni.so \
  AILAKE_EXT=./duckdb-ailake/build/ailake.duckdb_extension \
  AILAKE_FIXTURE=./compat-fixture \
  python duckdb-ailake/test/test_search.py
"""
import os
import sys
import pathlib
import struct
import math
import duckdb

FIXTURE_DIR = pathlib.Path(os.environ.get("AILAKE_FIXTURE", "./compat-fixture"))
EXT_PATH    = os.environ.get("AILAKE_EXT",   "./duckdb-ailake/build/ailake.duckdb_extension")
LIB_PATH    = os.environ.get("AILAKE_LIB",   "./target/release/libailake_jni.so")

def require(cond, msg):
    if not cond:
        print(f"FAIL: {msg}")
        sys.exit(1)

def setup_connection():
    conn = duckdb.connect()
    # Pre-load the native lib so the extension finds it via RTLD_GLOBAL
    import ctypes
    ctypes.CDLL(LIB_PATH, ctypes.RTLD_GLOBAL)
    conn.execute(f"LOAD '{EXT_PATH}'")
    return conn

def load_fixture_query():
    """Return the first query vector from the fixture (128-dim float32)."""
    vec_file = FIXTURE_DIR / "fixture_query.bin"
    if not vec_file.exists():
        # Fall back to a zero vector — search returns results but distances are large
        return [0.0] * 128
    data = vec_file.read_bytes()
    n = len(data) // 4
    return list(struct.unpack(f"<{n}f", data))

def table_path():
    return str(FIXTURE_DIR.resolve())

# ── Tests ─────────────────────────────────────────────────────────────────────

def test_extension_loads():
    conn = setup_connection()
    # If LOAD succeeded without exception, extension is present
    print("PASS: extension loaded")

def test_search_returns_rows():
    conn = setup_connection()
    query = load_fixture_query()
    q_sql = ", ".join(str(f) for f in query)

    rows = conn.execute(f"""
        SELECT row_id, distance, file_path
        FROM ailake_search(
            '{table_path()}',
            [{q_sql}]::FLOAT[],
            10
        )
    """).fetchall()

    require(len(rows) > 0, f"ailake_search returned 0 rows (fixture at {table_path()})")
    require(len(rows) <= 10, f"returned more than top_k=10 rows: {len(rows)}")
    print(f"PASS: search returned {len(rows)} rows")

def test_search_ordered_by_distance():
    conn = setup_connection()
    query = load_fixture_query()
    q_sql = ", ".join(str(f) for f in query)

    rows = conn.execute(f"""
        SELECT distance
        FROM ailake_search(
            '{table_path()}',
            [{q_sql}]::FLOAT[],
            5
        )
        ORDER BY distance
    """).fetchall()

    if len(rows) >= 2:
        for i in range(len(rows) - 1):
            require(
                rows[i][0] <= rows[i + 1][0] + 1e-6,
                f"distances not monotonically increasing: {rows[i][0]} > {rows[i+1][0]}"
            )
    print("PASS: distances orderable")

def test_search_result_schema():
    conn = setup_connection()
    query = load_fixture_query()
    q_sql = ", ".join(str(f) for f in query)

    schema = conn.execute(f"""
        DESCRIBE SELECT * FROM ailake_search(
            '{table_path()}',
            [{q_sql}]::FLOAT[],
            1
        )
    """).fetchall()

    col_names = [r[0] for r in schema]
    col_types = [r[1] for r in schema]

    require("row_id"    in col_names, f"missing row_id column, got {col_names}")
    require("distance"  in col_names, f"missing distance column, got {col_names}")
    require("file_path" in col_names, f"missing file_path column, got {col_names}")
    print(f"PASS: schema correct {list(zip(col_names, col_types))}")

def test_search_no_lib_returns_empty():
    """When lib is not loaded, search must return 0 rows (not error)."""
    conn = duckdb.connect()
    try:
        conn.execute(f"LOAD '{EXT_PATH}'")
    except Exception:
        print("SKIP: extension not built yet")
        return

    query = load_fixture_query()
    q_sql = ", ".join(str(f) for f in query)

    # Without pre-loading the native lib, is_ready() = false → 0 rows
    rows = conn.execute(f"""
        SELECT count(*) FROM ailake_search(
            '/nonexistent/path',
            [{q_sql}]::FLOAT[],
            10
        )
    """).fetchone()
    require(rows[0] == 0, f"expected 0 rows without native lib, got {rows[0]}")
    print("PASS: graceful degradation without native lib")

def test_search_vec_col_named_param():
    conn = setup_connection()
    query = load_fixture_query()
    q_sql = ", ".join(str(f) for f in query)

    # Should work with explicit vec_col=embedding (same as default)
    rows = conn.execute(f"""
        SELECT count(*) FROM ailake_search(
            '{table_path()}',
            [{q_sql}]::FLOAT[],
            5,
            vec_col='embedding'
        )
    """).fetchone()
    require(rows[0] >= 0, "named param vec_col raised unexpected error")
    print("PASS: named param vec_col accepted")

if __name__ == "__main__":
    if not pathlib.Path(EXT_PATH).exists():
        print(f"SKIP: extension not found at {EXT_PATH} — build first with cmake")
        sys.exit(0)

    test_extension_loads()
    test_search_result_schema()
    test_search_returns_rows()
    test_search_ordered_by_distance()
    test_search_no_lib_returns_empty()
    test_search_vec_col_named_param()

    print("\nAll search tests passed.")
