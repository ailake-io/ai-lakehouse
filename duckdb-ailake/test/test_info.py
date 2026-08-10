# SPDX-License-Identifier: MIT OR Apache-2.0
# Copyright (c) 2026 Thiago Egon Lange
"""
DuckDB ailake extension — ailake_info function tests.

Prerequisites:
  1. Build DuckDB extension (also builds ailake-jni as a static lib via corrosion):
       cmake --build duckdb-ailake/build

Usage:
  AILAKE_EXT=./duckdb-ailake/build/ailake.duckdb_extension \
  AILAKE_TMPDIR=/tmp/ailake_duck_info \
  python duckdb-ailake/test/test_info.py
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
        d = pathlib.Path(TMP_DIR) / f"test_info{suffix}"
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


def test_info_reports_files_rows_and_index_status():
    conn = setup_connection()
    table_dir, cleanup = make_table_dir("_ok")
    try:
        write_table(conn, table_dir)
        row = conn.execute(f"SELECT * FROM ailake_info('{table_dir}')").fetchone()
        cols = [d[0] for d in conn.description]
        info = dict(zip(cols, row))

        require(info["table"] == "default.table", f"unexpected table: {info['table']}")
        require(info["vector_column"] == "embedding", f"unexpected vector_column: {info}")
        require(info["vector_dim"] == "4", f"unexpected vector_dim: {info}")
        require(info["files"] == 1, f"expected 1 file, got {info['files']}")
        require(info["indexed_files"] == 1, f"expected 1 indexed file, got {info}")
        require(info["failed_files"] == 0, f"expected 0 failed files, got {info}")
        require(info["foreign_files"] == 0, f"expected 0 foreign files, got {info}")
        require(info["foreign_file_paths"] == [], f"expected empty foreign_file_paths, got {info}")
        require(info["rows"] == 3, f"expected 3 rows, got {info}")
        require(info["size_bytes"] > 0, f"expected size_bytes > 0, got {info}")
        require(info["snapshot_id"] is not None, f"expected non-null snapshot_id, got {info}")
        print(f"PASS test_info_reports_files_rows_and_index_status: {info}")
    finally:
        if cleanup:
            shutil.rmtree(cleanup, ignore_errors=True)
        conn.close()


def test_info_missing_table_raises():
    conn = setup_connection()
    table_dir, cleanup = make_table_dir("_missing")
    try:
        try:
            conn.execute(f"SELECT * FROM ailake_info('{table_dir}')").fetchall()
            require(False, "expected an exception for a nonexistent table, got a result")
        except duckdb.Error as e:
            require("ailake_info failed" in str(e), f"unexpected error message: {e}")
            print("PASS test_info_missing_table_raises")
    finally:
        if cleanup:
            shutil.rmtree(cleanup, ignore_errors=True)
        conn.close()


if __name__ == "__main__":
    test_info_reports_files_rows_and_index_status()
    test_info_missing_table_raises()
    print()
    print("PASS: ailake_info — all tests passed")
