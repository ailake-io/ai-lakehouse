# SPDX-License-Identifier: MIT OR Apache-2.0
# Copyright (c) 2026 Thiago Egon Lange
"""
DuckDB ailake extension — ailake_delete_rows function tests.

Prerequisites:
  1. Build DuckDB extension (also builds ailake-jni as a static lib via corrosion):
       cmake --build duckdb-ailake/build

Usage:
  AILAKE_EXT=./duckdb-ailake/build/ailake.duckdb_extension \
  AILAKE_TMPDIR=/tmp/ailake_duck_delete_rows \
  python duckdb-ailake/test/test_delete_rows.py
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
        d = pathlib.Path(TMP_DIR) / f"test_delete_rows{suffix}"
        shutil.rmtree(d, ignore_errors=True)
        d.mkdir(parents=True, exist_ok=True)
        return str(d), None
    tmp = tempfile.mkdtemp()
    return tmp, tmp


def write_table(conn, table_path, n=4, dim=4):
    # Deletion Vectors (what ailake_delete_rows writes) require Iceberg V3 —
    # format_version defaults to 2, so this must reach the arity-10 overload.
    ids = list(range(n))
    embs = [[float(i + j) for j in range(dim)] for i in ids]
    conn.execute(f"""
        SELECT ailake_write_batch(
            '{table_path}',
            {ids}::BIGINT[],
            {embs}::FLOAT[][],
            'embedding', 'cosine', 'f16', '', '', '[]', 3
        )
    """)


def test_delete_rows_masks_row_in_subsequent_search():
    conn = setup_connection()
    table_dir, cleanup = make_table_dir("_mask")
    try:
        write_table(conn, table_dir, n=4, dim=4)

        # Discover the real file_path via a full-recall search.
        rows = conn.execute(f"""
            SELECT row_id, file_path FROM ailake_search(
                '{table_dir}', [0.0, 1.0, 2.0, 3.0]::FLOAT[], 10
            )
        """).fetchall()
        require(len(rows) == 4, f"expected 4 rows before delete, got {len(rows)}")
        target_row_id = rows[0][0]
        file_path     = rows[0][1]

        ok = conn.execute(
            f"SELECT ailake_delete_rows('{table_dir}', '{file_path}', [{target_row_id}]::UINTEGER[])"
        ).fetchone()[0]
        require(ok is True, f"delete_rows returned {ok}, expected TRUE")

        rows_after = conn.execute(f"""
            SELECT row_id FROM ailake_search(
                '{table_dir}', [0.0, 1.0, 2.0, 3.0]::FLOAT[], 10
            )
        """).fetchall()
        require(
            len(rows_after) == 3,
            f"expected 3 rows after delete_rows, got {len(rows_after)}"
        )
        print(f"PASS test_delete_rows_masks_row_in_subsequent_search: {len(rows)} -> {len(rows_after)}")
    finally:
        if cleanup:
            shutil.rmtree(cleanup, ignore_errors=True)
        conn.close()


def test_delete_rows_missing_table_raises():
    conn = setup_connection()
    table_dir, cleanup = make_table_dir("_missing")
    try:
        try:
            conn.execute(
                f"SELECT ailake_delete_rows('{table_dir}', 'data/part-00000.parquet', [0]::UINTEGER[])"
            ).fetchone()
            require(False, "expected an exception for a nonexistent table, got a result")
        except duckdb.Error as e:
            require("ailake_delete_rows failed" in str(e), f"unexpected error message: {e}")
            print("PASS test_delete_rows_missing_table_raises")
    finally:
        if cleanup:
            shutil.rmtree(cleanup, ignore_errors=True)
        conn.close()


if __name__ == "__main__":
    test_delete_rows_masks_row_in_subsequent_search()
    test_delete_rows_missing_table_raises()
    print()
    print("PASS: ailake_delete_rows — all tests passed")
