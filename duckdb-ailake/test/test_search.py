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
import ctypes
import tempfile

# Force _duckdb.so to load with RTLD_GLOBAL so DuckDB extensions can resolve
# its C++ typeinfo symbols (TableFunction → SimpleNamedParameterFunction).
# Python's default dlopen flags are RTLD_LOCAL, which hides symbols from
# subsequently loaded extensions at RTLD_NOW resolution time.
_old_flags = sys.getdlopenflags()
sys.setdlopenflags(_old_flags | os.RTLD_GLOBAL)
import duckdb
sys.setdlopenflags(_old_flags)

FIXTURE_DIR = pathlib.Path(os.environ.get("AILAKE_FIXTURE", "./compat-fixture"))
EXT_PATH    = os.environ.get("AILAKE_EXT",   "./duckdb-ailake/build/ailake.duckdb_extension")
LIB_PATH    = os.environ.get("AILAKE_LIB",   "./target/release/libailake_jni.so")

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

def test_search_nonexistent_table_raises():
    """Querying a table path that doesn't exist must raise a clear SQL error,
    not silently return 0 rows.

    AilakeLib is a process-wide singleton (`AilakeLib::get()`'s static instance);
    by the time this test runs, earlier tests in this same script have already
    triggered a real search, so is_ready() is already true here regardless of
    using a fresh connection — this test was previously (and incorrectly, given
    that) asserting on a code path that never actually ran. The real thing
    exercised was the *other* early-return: an ok:false backend response
    (I/O error: table root doesn't exist) used to be silently folded into an
    empty result, indistinguishable from a genuine zero-match search. It's now
    surfaced as an exception.
    """
    conn = duckdb.connect(config={
        "allow_unsigned_extensions": True,
        "allow_extensions_metadata_mismatch": True,
    })
    try:
        conn.execute(f"LOAD '{EXT_PATH}'")
    except Exception:
        print("SKIP: extension not built yet")
        return

    query = load_fixture_query()
    q_sql = ", ".join(str(f) for f in query)

    try:
        conn.execute(f"""
            SELECT count(*) FROM ailake_search(
                '/nonexistent/path',
                [{q_sql}]::FLOAT[],
                10
            )
        """).fetchone()
        require(False, "expected an exception for a nonexistent table path, got a result")
    except duckdb.Error as e:
        require("ailake_search failed" in str(e), f"unexpected error message: {e}")
        print("PASS: nonexistent table path raises a clear error")

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

def test_search_partition_filter_named_param():
    """partition_filter= is accepted as a named parameter without raising an error."""
    conn = setup_connection()
    query = load_fixture_query()
    q_sql = ", ".join(str(f) for f in query)

    # 'nonexistent-agent' matches no files → 0 results, but no exception.
    rows = conn.execute(f"""
        SELECT count(*) FROM ailake_search(
            '{table_path()}',
            [{q_sql}]::FLOAT[],
            10,
            partition_filter='nonexistent-agent'
        )
    """).fetchone()
    require(rows[0] >= 0, "partition_filter named param caused an unexpected error")
    print(f"PASS: partition_filter named param accepted (returned {rows[0]} rows)")

def test_search_hybrid_named_params():
    """hybrid_text, text_column, bm25_weight named params are accepted without error."""
    conn = setup_connection()
    query = load_fixture_query()
    q_sql = ", ".join(str(f) for f in query)

    rows = conn.execute(f"""
        SELECT count(*) FROM ailake_search(
            '{table_path()}',
            [{q_sql}]::FLOAT[],
            10,
            hybrid_text='vector search approximate nearest neighbor',
            text_column='chunk_text',
            bm25_weight=0.4
        )
    """).fetchone()
    require(rows[0] >= 0, "hybrid named params caused an unexpected error")
    print(f"PASS: hybrid named params accepted (returned {rows[0]} rows)")

def test_search_text_schema():
    """ailake_search_text returns correct schema: row_id, distance, file_path."""
    conn = setup_connection()

    schema = conn.execute(f"""
        DESCRIBE SELECT * FROM ailake_search_text(
            '{table_path()}',
            'vector search',
            5
        )
    """).fetchall()

    col_names = [r[0] for r in schema]
    require("row_id"    in col_names, f"ailake_search_text missing row_id, got {col_names}")
    require("distance"  in col_names, f"ailake_search_text missing distance, got {col_names}")
    require("file_path" in col_names, f"ailake_search_text missing file_path, got {col_names}")
    print(f"PASS: ailake_search_text schema correct {col_names}")

def test_search_text_no_lib_returns_empty():
    """When native lib not loaded, ailake_search_text returns 0 rows gracefully."""
    conn = duckdb.connect(config={
        "allow_unsigned_extensions": True,
        "allow_extensions_metadata_mismatch": True,
    })
    try:
        conn.execute(f"LOAD '{EXT_PATH}'")
    except Exception:
        print("SKIP: extension not built yet")
        return

    rows = conn.execute(f"""
        SELECT count(*) FROM ailake_search_text(
            '/nonexistent/path',
            'rust programming',
            10
        )
    """).fetchone()
    require(rows[0] == 0, f"expected 0 rows without native lib, got {rows[0]}")
    print("PASS: ailake_search_text graceful degradation without native lib")


def test_search_text_text_columns_named_param():
    """text_columns LIST(VARCHAR) named param accepted; returns correct schema."""
    conn = setup_connection()

    # text_columns := ['chunk_text', 'title'] must parse and bind without error.
    schema = conn.execute(f"""
        DESCRIBE SELECT * FROM ailake_search_text(
            '{table_path()}',
            'vector search',
            5,
            text_columns := ['chunk_text', 'title']
        )
    """).fetchall()

    col_names = [r[0] for r in schema]
    require("row_id"    in col_names, f"ailake_search_text missing row_id, got {col_names}")
    require("distance"  in col_names, f"ailake_search_text missing distance, got {col_names}")
    require("file_path" in col_names, f"ailake_search_text missing file_path, got {col_names}")
    print(f"PASS: ailake_search_text text_columns LIST named param accepted, schema={col_names}")


def test_search_text_legacy_text_column_param():
    """Legacy text_column VARCHAR named param still accepted (single-column fallback)."""
    conn = setup_connection()

    schema = conn.execute(f"""
        DESCRIBE SELECT * FROM ailake_search_text(
            '{table_path()}',
            'vector search',
            5,
            text_column := 'document_text'
        )
    """).fetchall()

    col_names = [r[0] for r in schema]
    require("row_id" in col_names, f"ailake_search_text schema wrong: {col_names}")
    print(f"PASS: ailake_search_text legacy text_column VARCHAR param accepted")


def test_search_namespace_named_param_isolation():
    """Regression: `ailake_search`'s request JSON hardcoded `"namespace":"default"`,
    so a table written under a non-default namespace (via `ailake_write_batch`'s
    `namespace` arg) could never be found by `ailake_search` no matter what
    `namespace :=` was passed — the named param didn't exist at all before this fix.

    Writes to <dir>/custom_ns/my_table, then confirms:
    - search WITH namespace:='custom_ns', table_name:='my_table' finds it
    - search WITHOUT namespace (defaults) does NOT find it (proves real isolation,
      not just "the param is accepted but ignored")
    """
    conn = setup_connection()
    table_dir = tempfile.mkdtemp(prefix="ailake_search_ns_")
    n, dim = 3, 4
    embs = [[float(i * dim + j) / (n * dim) for j in range(dim)] for i in range(n)]

    ids_sql = "[" + ", ".join(str(i) for i in range(n)) + "]::BIGINT[]"
    emb_sql = "[" + ", ".join(
        "[" + ", ".join(str(f) for f in row) + "]::FLOAT[]" for row in embs
    ) + "]"

    write_row = conn.execute(f"""
        SELECT ailake_write_batch(
            'file://{table_dir}',
            {ids_sql},
            {emb_sql},
            'embedding', 'cosine', 'f16',
            '', '', '', 2, '', '', -1, -1, false, false,
            'custom_ns', 'my_table'
        )
    """).fetchone()
    require(write_row is not None and write_row[0] != -1, "setup write_batch failed")

    q_sql = "[" + ", ".join(str(f) for f in embs[0]) + "]::FLOAT[]"

    matched = conn.execute(f"""
        SELECT row_id FROM ailake_search(
            'file://{table_dir}', {q_sql}, 5,
            namespace := 'custom_ns', table_name := 'my_table'
        )
    """).fetchall()
    require(len(matched) > 0, "search with correct namespace/table_name found 0 rows")

    unmatched = conn.execute(f"""
        SELECT row_id FROM ailake_search('file://{table_dir}', {q_sql}, 5)
    """).fetchall()
    require(
        len(unmatched) == 0,
        f"search with default namespace/table_name unexpectedly found {len(unmatched)} rows "
        "— namespace isolation broken"
    )
    print(f"PASS: ailake_search namespace named param — found {len(matched)} row(s) in "
          f"custom_ns/my_table, 0 rows under default namespace")


if __name__ == "__main__":
    if not pathlib.Path(EXT_PATH).exists():
        print(f"SKIP: extension not found at {EXT_PATH} — build first with cmake")
        sys.exit(0)

    test_extension_loads()
    test_search_result_schema()
    test_search_returns_rows()
    test_search_ordered_by_distance()
    test_search_nonexistent_table_raises()
    test_search_vec_col_named_param()
    test_search_partition_filter_named_param()
    test_search_hybrid_named_params()
    test_search_text_schema()
    test_search_text_no_lib_returns_empty()
    test_search_text_text_columns_named_param()
    test_search_text_legacy_text_column_param()
    test_search_namespace_named_param_isolation()

    print("\nAll search tests passed.")
