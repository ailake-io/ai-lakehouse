# SPDX-License-Identifier: MIT OR Apache-2.0
# Copyright (c) 2026 Thiago Egon Lange
"""
DuckDB ailake extension — ailake_add_vector_column / ailake_backfill_vector_column /
ailake_migrate function tests.

ailake_write_batch (this extension's ingest path) writes only `id` + the vector
column — it has no passthrough for arbitrary text columns (unlike
ailake_write_batch_json's own "columns" field, which the JNI layer accepts but
this extension does not yet forward — a real gap, tracked separately from the
6-op parity fix these tests cover). That means a table written entirely
through DuckDB SQL never has a `chunk_text` column for
ailake_backfill_vector_column / ailake_migrate to read from, so those two
functions can only be exercised on the real error path here (missing source
text column) rather than a full embed round-trip. ailake_add_vector_column
itself needs no text column — it's tested end-to-end.

Prerequisites:
  1. Build DuckDB extension (also builds ailake-jni as a static lib via corrosion):
       cmake --build duckdb-ailake/build

Usage:
  AILAKE_EXT=./duckdb-ailake/build/ailake.duckdb_extension \
  AILAKE_TMPDIR=/tmp/ailake_duck_add_vector_column \
  python duckdb-ailake/test/test_add_vector_column.py
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


def make_table_dir(suffix=""):
    if TMP_DIR:
        d = pathlib.Path(TMP_DIR) / f"test_add_vector_column{suffix}"
        shutil.rmtree(d, ignore_errors=True)
        d.mkdir(parents=True, exist_ok=True)
        return str(d), None
    tmp = tempfile.mkdtemp()
    return tmp, tmp


def write_table(conn, table_path, n=3, dim=4):
    ids = list(range(n))
    embs = [[float(i + j) for j in range(dim)] for i in ids]
    conn.execute(f"""
        SELECT ailake_write_batch(
            '{table_path}',
            {ids}::BIGINT[],
            {embs}::FLOAT[][],
            'embedding', 'cosine', 'f16'
        )
    """)


def test_add_vector_column_returns_new_schema_id():
    conn = setup_connection()
    table_dir, cleanup = make_table_dir("_ok")
    try:
        write_table(conn, table_dir)
        schema_id = conn.execute(f"""
            SELECT ailake_add_vector_column(
                '{table_dir}', 'image_embedding', 512, metric := 'euclidean'
            )
        """).fetchone()[0]
        require(schema_id is not None and schema_id >= 0, f"expected schema_id >= 0, got {schema_id}")
        print(f"PASS test_add_vector_column_returns_new_schema_id: new_schema_id={schema_id}")
    finally:
        if cleanup:
            shutil.rmtree(cleanup, ignore_errors=True)
        conn.close()


def test_add_vector_column_missing_table_raises():
    conn = setup_connection()
    table_dir, cleanup = make_table_dir("_missing")
    try:
        try:
            conn.execute(
                f"SELECT ailake_add_vector_column('{table_dir}', 'v2', 4)"
            ).fetchone()
            require(False, "expected an exception for a nonexistent table, got a result")
        except duckdb.Error as e:
            require("ailake_add_vector_column failed" in str(e), f"unexpected error message: {e}")
            print("PASS test_add_vector_column_missing_table_raises")
    finally:
        if cleanup:
            shutil.rmtree(cleanup, ignore_errors=True)
        conn.close()


def test_backfill_vector_column_missing_text_column_raises():
    """Real error-path test — see module docstring for why this can't be a
    happy-path embed round-trip through pure SQL yet."""
    conn = setup_connection()
    table_dir, cleanup = make_table_dir("_backfill")
    try:
        write_table(conn, table_dir)
        conn.execute(f"SELECT ailake_add_vector_column('{table_dir}', 'v2', 2)")
        try:
            conn.execute(f"""
                SELECT ailake_backfill_vector_column(
                    '{table_dir}', 'v2', 'chunk_text',
                    'python3 -c "import sys,json; t=json.load(sys.stdin); print(json.dumps([[1.0,2.0] for _ in t]))"'
                )
            """).fetchone()
            require(False, "expected an exception for a missing chunk_text column, got a result")
        except duckdb.Error as e:
            require(
                "ailake_backfill_vector_column failed" in str(e),
                f"unexpected error message: {e}"
            )
            print("PASS test_backfill_vector_column_missing_text_column_raises")
    finally:
        if cleanup:
            shutil.rmtree(cleanup, ignore_errors=True)
        conn.close()


def test_migrate_missing_old_column_raises():
    """Real error-path test — see module docstring."""
    conn = setup_connection()
    table_dir, cleanup = make_table_dir("_migrate")
    try:
        write_table(conn, table_dir)
        try:
            conn.execute(f"""
                SELECT ailake_migrate(
                    '{table_dir}', 'nonexistent_col', 'embedding_v2',
                    'python3 -c "import sys,json; t=json.load(sys.stdin); print(json.dumps([[1.0,2.0] for _ in t]))"'
                )
            """).fetchone()
            require(False, "expected an exception for a missing old_column, got a result")
        except duckdb.Error as e:
            require("ailake_migrate failed" in str(e), f"unexpected error message: {e}")
            print("PASS test_migrate_missing_old_column_raises")
    finally:
        if cleanup:
            shutil.rmtree(cleanup, ignore_errors=True)
        conn.close()


if __name__ == "__main__":
    test_add_vector_column_returns_new_schema_id()
    test_add_vector_column_missing_table_raises()
    test_backfill_vector_column_missing_text_column_raises()
    test_migrate_missing_old_column_raises()
    print()
    print("PASS: ailake_add_vector_column / ailake_backfill_vector_column / ailake_migrate — all tests passed")
