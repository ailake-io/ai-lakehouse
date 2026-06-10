# SPDX-License-Identifier: MIT OR Apache-2.0
# Copyright (c) 2026 Thiago Egon Lange
"""
DuckDB ailake extension — write_batch function tests.

Prerequisites:
  1. Build libailake_jni.so:  cargo build --release -p ailake-jni
  2. Build DuckDB extension:  cmake --build duckdb-ailake/build

Usage:
  AILAKE_LIB=./target/release/libailake_jni.so \
  AILAKE_EXT=./duckdb-ailake/build/ailake.duckdb_extension \
  AILAKE_TMPDIR=/tmp/ailake_duck_test \
  python duckdb-ailake/test/test_write.py
"""
import os
import sys
import pathlib
import tempfile
import shutil
import ctypes
import duckdb

EXT_PATH = os.environ.get("AILAKE_EXT", "./duckdb-ailake/build/ailake.duckdb_extension")
LIB_PATH = os.environ.get("AILAKE_LIB", "./target/release/libailake_jni.so")
TMP_DIR  = os.environ.get("AILAKE_TMPDIR", "")

def require(cond, msg):
    if not cond:
        print(f"FAIL: {msg}")
        sys.exit(1)

def setup_connection():
    conn = duckdb.connect(config={"allow_unsigned_extensions": True})
    ctypes.CDLL(LIB_PATH, ctypes.RTLD_GLOBAL)
    conn.execute(f"LOAD '{EXT_PATH}'")
    return conn

def make_table_dir():
    if TMP_DIR:
        p = pathlib.Path(TMP_DIR)
        p.mkdir(parents=True, exist_ok=True)
        return str(p)
    return tempfile.mkdtemp(prefix="ailake_duck_")

def small_embeddings(n=3, dim=8):
    """Return n embeddings of dimension dim (simple deterministic values)."""
    return [[float(i * dim + j) / (n * dim) for j in range(dim)] for i in range(n)]

# ── Tests ─────────────────────────────────────────────────────────────────────

def test_write_returns_snapshot_id():
    conn = setup_connection()
    table_dir = make_table_dir()
    n, dim = 3, 8
    embs = small_embeddings(n, dim)

    ids_sql = "[" + ", ".join(str(i) for i in range(n)) + "]::BIGINT[]"
    emb_sql = "[" + ", ".join(
        "[" + ", ".join(str(f) for f in row) + "]::FLOAT[]"
        for row in embs
    ) + "]"

    row = conn.execute(f"""
        SELECT ailake_write_batch(
            'file://{table_dir}',
            {ids_sql},
            {emb_sql}
        )
    """).fetchone()

    require(row is not None, "write_batch returned NULL row")
    snap_id = row[0]
    require(snap_id != -1, f"write_batch returned -1 (error); table_dir={table_dir}")
    print(f"PASS: write_batch returned snapshot_id={snap_id}")

def test_write_creates_parquet():
    conn = setup_connection()
    table_dir = make_table_dir()
    n, dim = 5, 4
    embs = small_embeddings(n, dim)

    ids_sql = "[" + ", ".join(str(i) for i in range(n)) + "]::BIGINT[]"
    emb_sql = "[" + ", ".join(
        "[" + ", ".join(str(f) for f in row) + "]::FLOAT[]"
        for row in embs
    ) + "]"

    conn.execute(f"""
        SELECT ailake_write_batch(
            'file://{table_dir}',
            {ids_sql},
            {emb_sql}
        )
    """)

    # Verify a Parquet file was written that DuckDB can read without the extension
    parquet_files = list(pathlib.Path(table_dir).rglob("*.parquet"))
    require(len(parquet_files) > 0, f"no .parquet files created under {table_dir}")

    plain_conn = duckdb.connect()
    rows = plain_conn.execute(
        f"SELECT count(*) FROM parquet_scan('{parquet_files[0]}')"
    ).fetchone()[0]
    require(rows == n, f"expected {n} rows in parquet, got {rows}")
    print(f"PASS: created {len(parquet_files)} parquet file(s) readable by plain DuckDB")

def test_write_full_signature():
    conn = setup_connection()
    table_dir = make_table_dir()
    n, dim = 2, 4
    embs = small_embeddings(n, dim)

    ids_sql = "[" + ", ".join(str(i) for i in range(n)) + "]::BIGINT[]"
    emb_sql = "[" + ", ".join(
        "[" + ", ".join(str(f) for f in row) + "]::FLOAT[]"
        for row in embs
    ) + "]"

    row = conn.execute(f"""
        SELECT ailake_write_batch(
            'file://{table_dir}',
            {ids_sql},
            {emb_sql},
            'embedding',
            'cosine',
            'f16'
        )
    """).fetchone()

    require(row is not None and row[0] != -1, "6-arg write_batch failed")
    print(f"PASS: 6-arg write_batch returned snapshot_id={row[0]}")

def test_write_then_search():
    conn = setup_connection()
    table_dir = make_table_dir()
    n, dim = 10, 8
    embs = small_embeddings(n, dim)

    ids_sql = "[" + ", ".join(str(i) for i in range(n)) + "]::BIGINT[]"
    emb_sql = "[" + ", ".join(
        "[" + ", ".join(str(f) for f in row) + "]::FLOAT[]"
        for row in embs
    ) + "]"

    conn.execute(f"""
        SELECT ailake_write_batch(
            'file://{table_dir}',
            {ids_sql},
            {emb_sql}
        )
    """)

    # Search with the first embedding as query
    q_sql = "[" + ", ".join(str(f) for f in embs[0]) + "]::FLOAT[]"
    rows = conn.execute(f"""
        SELECT row_id, distance
        FROM ailake_search(
            'file://{table_dir}',
            {q_sql},
            5
        )
        ORDER BY distance
    """).fetchall()

    require(len(rows) > 0, "search after write returned 0 rows")
    # Nearest neighbor of embs[0] should be itself (distance ≈ 0)
    nearest_id = rows[0][0]
    nearest_dist = rows[0][1]
    require(nearest_dist < 0.01, f"nearest distance should be ~0, got {nearest_dist}")
    require(nearest_id == 0, f"nearest id should be 0, got {nearest_id}")
    print(f"PASS: write→search roundtrip: nearest row_id={nearest_id} distance={nearest_dist:.6f}")

def test_write_empty_returns_minus_one():
    conn = setup_connection()
    row = conn.execute("""
        SELECT ailake_write_batch(
            'file:///tmp/nonexistent_table',
            []::BIGINT[],
            []::FLOAT[][]
        )
    """).fetchone()
    require(row is not None and row[0] == -1, f"expected -1 for empty write, got {row}")
    print("PASS: empty write returns -1")

if __name__ == "__main__":
    if not pathlib.Path(EXT_PATH).exists():
        print(f"SKIP: extension not found at {EXT_PATH} — build first with cmake")
        sys.exit(0)

    test_write_returns_snapshot_id()
    test_write_creates_parquet()
    test_write_full_signature()
    test_write_then_search()
    test_write_empty_returns_minus_one()

    print("\nAll write tests passed.")
