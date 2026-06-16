# SPDX-License-Identifier: MIT OR Apache-2.0
# Copyright (c) 2026 Thiago Egon Lange
"""
DuckDB ailake extension — ailake_search_multimodal() tests.

Uses the same single-column compat-fixture as test_search.py.
A single-column multimodal query is valid — it exercises the function's
registration, JSON envelope, RRF accumulation, and result schema.

Prerequisites:
  1. Build libailake_jni.so:  cargo build --release -p ailake-jni
  2. Build DuckDB extension:  cmake --build duckdb-ailake/build
  3. Generate fixture:        python tests/fixtures/write_fixture.py

Usage:
  AILAKE_LIB=./target/release/libailake_jni.so \
  AILAKE_EXT=./duckdb-ailake/build/ailake.duckdb_extension \
  AILAKE_FIXTURE=./compat-fixture \
  python duckdb-ailake/test/test_multimodal.py
"""
import os
import sys
import struct
import math
import ctypes
import pathlib

_old_flags = sys.getdlopenflags()
sys.setdlopenflags(_old_flags | os.RTLD_GLOBAL)
import duckdb
sys.setdlopenflags(_old_flags)

PASS = 0
FAIL = 0


def require(cond, msg):
    global PASS, FAIL
    if cond:
        print(f"  PASS  {msg}")
        PASS += 1
    else:
        print(f"  FAIL  {msg}")
        FAIL += 1


def setup_connection():
    lib_path = os.environ.get("AILAKE_LIB", "")
    ext_path = os.environ.get("AILAKE_EXT", "")
    if not lib_path or not ext_path:
        print("SKIP: AILAKE_LIB and AILAKE_EXT must be set")
        sys.exit(0)

    ctypes.CDLL(lib_path, ctypes.RTLD_GLOBAL)
    conn = duckdb.connect(config={
        "allow_unsigned_extensions": True,
        "allow_extensions_metadata_mismatch": True,
    })
    conn.execute(f"LOAD '{ext_path}'")
    return conn


def fixture_path():
    f = os.environ.get("AILAKE_FIXTURE", "./compat-fixture")
    return str(pathlib.Path(f).resolve())


def load_fixture_query():
    dim = 128
    query_file = pathlib.Path(fixture_path()) / "fixture_query.bin"
    if not query_file.exists():
        return [float(i % 13 + 1) / 13.0 for i in range(dim)]
    raw = query_file.read_bytes()
    floats = list(struct.unpack(f"<{dim}f", raw[:dim * 4]))
    norm = math.sqrt(sum(x * x for x in floats))
    return [x / norm for x in floats] if norm > 0 else floats


def test_extension_loads():
    conn = setup_connection()
    # Verify function is registered in duckdb_functions().
    rows = conn.execute(
        "SELECT count(*) FROM duckdb_functions() WHERE function_name = 'ailake_search_multimodal'"
    ).fetchone()
    require(rows[0] >= 1, "ailake_search_multimodal registered in duckdb_functions()")
    conn.close()


def test_multimodal_result_schema():
    conn = setup_connection()
    query_vec = load_fixture_query()
    table_path = fixture_path()

    sql = f"""
        SELECT * FROM ailake_search_multimodal(
            '{table_path}',
            [
                {{'col': 'embedding', 'query': {query_vec!r}::FLOAT[], 'weight': 1.0}}
            ],
            5
        ) LIMIT 1
    """
    try:
        result = conn.execute(sql)
        col_names = [d[0] for d in result.description]
        require("row_id"    in col_names, "result has row_id column")
        require("rrf_score" in col_names, "result has rrf_score column")
        require("file_path" in col_names, "result has file_path column")
    except Exception as e:
        require(False, f"query raised: {e}")
    conn.close()


def test_multimodal_returns_rows():
    conn = setup_connection()
    query_vec = load_fixture_query()
    table_path = fixture_path()

    sql = f"""
        SELECT row_id, rrf_score, file_path
        FROM ailake_search_multimodal(
            '{table_path}',
            [
                {{'col': 'embedding', 'query': {query_vec!r}::FLOAT[], 'weight': 1.0}}
            ],
            10
        )
        ORDER BY rrf_score DESC
    """
    try:
        rows = conn.execute(sql).fetchall()
        require(len(rows) > 0, f"ailake_search_multimodal returned rows (got {len(rows)})")
        require(len(rows) <= 10, f"returned ≤ top_k=10 rows (got {len(rows)})")
        if rows:
            require(rows[0][1] > 0.0, f"rrf_score > 0 (got {rows[0][1]})")
        # Ordering: rrf_score descending
        if len(rows) > 1:
            require(
                all(rows[i][1] >= rows[i+1][1] for i in range(len(rows)-1)),
                "rows sorted descending by rrf_score"
            )
    except Exception as e:
        require(False, f"query raised: {e}")
    conn.close()


def test_multimodal_no_lib_returns_empty():
    """When libailake_jni.so is not loaded, function returns 0 rows gracefully."""
    ext_path = os.environ.get("AILAKE_EXT", "")
    if not ext_path:
        return
    conn2 = duckdb.connect(config={
        "allow_unsigned_extensions": True,
        "allow_extensions_metadata_mismatch": True,
    })
    try:
        conn2.execute(f"LOAD '{ext_path}'")
        count = conn2.execute("""
            SELECT count(*) FROM ailake_search_multimodal(
                '/nonexistent/path',
                [{'col': 'embedding', 'query': [0.1, 0.2]::FLOAT[], 'weight': 1.0}],
                5
            )
        """).fetchone()[0]
        require(count == 0, f"returns 0 rows when native lib not present (got {count})")
    except Exception as e:
        require(False, f"raised instead of returning empty: {e}")
    conn2.close()


if __name__ == "__main__":
    print("── ailake_search_multimodal tests ──────────────────────────────────────")
    test_extension_loads()
    test_multimodal_result_schema()
    test_multimodal_returns_rows()
    test_multimodal_no_lib_returns_empty()
    print(f"\n{'PASS' if FAIL == 0 else 'FAIL'}  {PASS} passed, {FAIL} failed")
    sys.exit(0 if FAIL == 0 else 1)
