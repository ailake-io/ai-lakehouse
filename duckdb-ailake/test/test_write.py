# SPDX-License-Identifier: MIT OR Apache-2.0
# Copyright (c) 2026 Thiago Egon Lange
"""
DuckDB ailake extension — write_batch function tests.

Prerequisites:
  1. Build DuckDB extension (also builds ailake-jni as a static lib via corrosion):
       cmake --build duckdb-ailake/build

Usage:
  AILAKE_EXT=./duckdb-ailake/build/ailake.duckdb_extension \
  AILAKE_TMPDIR=/tmp/ailake_duck_test \
  python duckdb-ailake/test/test_write.py
"""
import os
import sys
import pathlib
import tempfile
import shutil

import duckdb

EXT_PATH = os.environ.get("AILAKE_EXT", "./duckdb-ailake/build/ailake.duckdb_extension")
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
    conn.execute(f"LOAD '{EXT_PATH}'")
    return conn

def make_table_dir():
    # Always create a fresh unique dir — reusing the same warehouse across
    # tests with different dims causes write_batch to return -1 (schema mismatch).
    base = pathlib.Path(TMP_DIR) if TMP_DIR else pathlib.Path(tempfile.gettempdir())
    base.mkdir(parents=True, exist_ok=True)
    return tempfile.mkdtemp(prefix="ailake_", dir=str(base))

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

def test_write_with_partition_by():
    """Arity-7: (table_path, ids, embeddings, vec_col, metric, precision, partition_by)."""
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
            'f16',
            'agent_id'
        )
    """).fetchone()
    require(row is not None, "arity-7 write_batch returned NULL")
    require(row[0] != -1, f"arity-7 write_batch returned -1 (error); table_dir={table_dir}")
    print(f"PASS: arity-7 write_batch (partition_by='agent_id') snap_id={row[0]}")

def test_write_with_partition_by_and_value():
    """Arity-8: (table_path, ids, embeddings, vec_col, metric, precision, partition_by, partition_value)."""
    conn = setup_connection()
    table_dir = make_table_dir()
    n, dim = 3, 4
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
            'f16',
            'agent_id',
            'agent-A'
        )
    """).fetchone()
    require(row is not None, "arity-8 write_batch returned NULL")
    require(row[0] != -1, f"arity-8 write_batch returned -1 (error); table_dir={table_dir}")
    print(f"PASS: arity-8 write_batch (partition_by + partition_value='agent-A') snap_id={row[0]}")

def test_write_with_fts_columns():
    """Arity-11: (table_path, ids, embeddings, vec_col, metric, precision,
    partition_by, partition_value, partition_fields_json, format_version, fts_columns_json)."""
    conn = setup_connection()
    table_dir = make_table_dir()
    n, dim = 3, 4
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
            'f16',
            '',
            '',
            '',
            2,
            '["chunk_text"]'
        )
    """).fetchone()
    require(row is not None, "arity-11 write_batch returned NULL")
    require(row[0] != -1, f"arity-11 write_batch (fts_columns) returned -1; table_dir={table_dir}")
    print(f"PASS: arity-11 write_batch (fts_columns_json) snap_id={row[0]}")


def test_write_with_fts_columns_and_tokenizer():
    """Arity-12: adds fts_tokenizer VARCHAR as the 12th argument."""
    conn = setup_connection()
    table_dir = make_table_dir()
    n, dim = 3, 4
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
            'f16',
            '',
            '',
            '',
            2,
            '["chunk_text"]',
            'default'
        )
    """).fetchone()
    require(row is not None, "arity-12 write_batch returned NULL")
    require(row[0] != -1, f"arity-12 write_batch (fts_tokenizer) returned -1; table_dir={table_dir}")
    print(f"PASS: arity-12 write_batch (fts_columns + fts_tokenizer) snap_id={row[0]}")


def test_write_with_namespace_and_table_name():
    """Arity-18: (... deferred, namespace, table_name).

    Regression: `ailake_write_batch`'s header documented `namespace`/`table_name`
    as overridable, but no registered arity included them and both were
    hardcoded to 'default'/'table' in AilakeWriteExecFull — every write landed
    at <warehouse>/default/table/ regardless of what the caller passed. Confirms
    the file actually lands under the requested namespace/table_name path, not
    under the old hardcoded default.
    """
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
            'embedding', 'cosine', 'f16',
            '', '', '', 2, '', '', -1, -1, false, false,
            'custom_ns', 'my_table'
        )
    """).fetchone()
    require(row is not None, "arity-18 write_batch returned NULL")
    require(row[0] != -1, f"arity-18 write_batch returned -1 (error); table_dir={table_dir}")

    expected_dir = pathlib.Path(table_dir) / "custom_ns" / "my_table"
    require(
        expected_dir.is_dir(),
        f"expected data under {expected_dir}, not found — namespace/table_name were ignored"
    )
    default_dir = pathlib.Path(table_dir) / "default" / "table"
    require(
        not default_dir.exists(),
        f"data leaked into hardcoded default path {default_dir} despite explicit namespace/table_name"
    )
    print(f"PASS: arity-18 write_batch (namespace='custom_ns', table_name='my_table') snap_id={row[0]}, "
          f"data correctly under {expected_dir}")


if __name__ == "__main__":
    if not pathlib.Path(EXT_PATH).exists():
        print(f"SKIP: extension not found at {EXT_PATH} — build first with cmake")
        sys.exit(0)

    test_write_returns_snapshot_id()
    test_write_creates_parquet()
    test_write_full_signature()
    test_write_then_search()
    test_write_empty_returns_minus_one()
    test_write_with_partition_by()
    test_write_with_partition_by_and_value()
    test_write_with_fts_columns()
    test_write_with_fts_columns_and_tokenizer()
    test_write_with_namespace_and_table_name()

    print("\nAll write tests passed.")
