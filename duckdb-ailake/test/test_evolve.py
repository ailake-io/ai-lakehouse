# SPDX-License-Identifier: MIT OR Apache-2.0
# Copyright (c) 2026 Thiago Egon Lange
"""
DuckDB ailake extension — ailake_evolve_schema function tests (Phase M).

Prerequisites:
  1. Build DuckDB extension (also builds ailake-jni as a static lib via corrosion):
       cmake --build duckdb-ailake/build

Usage:
  AILAKE_EXT=./duckdb-ailake/build/ailake.duckdb_extension \
  AILAKE_TMPDIR=/tmp/ailake_duck_evolve \
  python duckdb-ailake/test/test_evolve.py
"""
import os
import sys
import pathlib
import tempfile
import shutil
import json

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
        d = pathlib.Path(TMP_DIR) / f"test_evolve{suffix}"
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


def test_evolve_schema_add_column_returns_schema_id():
    conn = setup_connection()
    table_dir, cleanup = make_table_dir("_add")
    try:
        write_table(conn, table_dir)
        add_cols = json.dumps([{"name": "source", "type": "string"}])
        rename_cols = json.dumps([])
        row = conn.execute(
            f"SELECT ailake_evolve_schema('{table_dir}', '{add_cols}', '{rename_cols}')"
        ).fetchone()
        require(row is not None, "evolve_schema returned NULL row")
        require(row[0] >= 0, f"evolve_schema returned {row[0]}, expected >= 0 schema_id")
        print(f"PASS test_evolve_schema_add_column_returns_schema_id: new_schema_id={row[0]}")
    finally:
        if cleanup:
            shutil.rmtree(cleanup, ignore_errors=True)
        conn.close()


def test_evolve_schema_empty_is_noop():
    conn = setup_connection()
    table_dir, cleanup = make_table_dir("_noop")
    try:
        write_table(conn, table_dir)
        row = conn.execute(
            f"SELECT ailake_evolve_schema('{table_dir}', '[]', '[]')"
        ).fetchone()
        require(row is not None, "evolve_schema noop returned NULL row")
        # noop returns 0 (no schema change)
        require(row[0] == 0, f"evolve_schema noop returned {row[0]}, expected 0")
        print(f"PASS test_evolve_schema_empty_is_noop: result={row[0]}")
    finally:
        if cleanup:
            shutil.rmtree(cleanup, ignore_errors=True)
        conn.close()


def test_evolve_schema_rename_column():
    conn = setup_connection()
    table_dir, cleanup = make_table_dir("_rename")
    try:
        write_table(conn, table_dir)
        # First add a column so we can rename it
        add_cols = json.dumps([{"name": "old_name", "type": "string"}])
        conn.execute(
            f"SELECT ailake_evolve_schema('{table_dir}', '{add_cols}', '[]')"
        )
        rename_cols = json.dumps([{"from": "old_name", "to": "new_name"}])
        row = conn.execute(
            f"SELECT ailake_evolve_schema('{table_dir}', '[]', '{rename_cols}')"
        ).fetchone()
        require(row is not None, "evolve_schema rename returned NULL row")
        require(row[0] >= 0, f"evolve_schema rename returned {row[0]}, expected >= 0")
        print(f"PASS test_evolve_schema_rename_column: new_schema_id={row[0]}")
    finally:
        if cleanup:
            shutil.rmtree(cleanup, ignore_errors=True)
        conn.close()


if __name__ == "__main__":
    test_evolve_schema_add_column_returns_schema_id()
    test_evolve_schema_empty_is_noop()
    test_evolve_schema_rename_column()
    print()
    print("PASS: ailake_evolve_schema — all tests passed")
