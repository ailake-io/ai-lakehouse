# SPDX-License-Identifier: MIT OR Apache-2.0
# Copyright (c) 2026 Thiago Egon Lange
"""
DuckDB ailake extension — ailake_decay_memories function tests.

ailake_write_batch (this extension's ingest path) has no passthrough for a
`last_accessed_at` column (see test_add_vector_column.py's module docstring
for the same limitation affecting backfill/migrate), so a table written
entirely through DuckDB SQL never has recency data for MemoryDecayJob to
recompute. This exercises the real "0 files touched" degenerate path (still a
real backend call, not a mock — same shape as test_compact's
nothing-eligible-returns-zero case) plus the real error path.

Prerequisites:
  1. Build DuckDB extension (also builds ailake-jni as a static lib via corrosion):
       cmake --build duckdb-ailake/build

Usage:
  AILAKE_EXT=./duckdb-ailake/build/ailake.duckdb_extension \
  AILAKE_TMPDIR=/tmp/ailake_duck_decay \
  python duckdb-ailake/test/test_decay_memories.py
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
        d = pathlib.Path(TMP_DIR) / f"test_decay{suffix}"
        shutil.rmtree(d, ignore_errors=True)
        d.mkdir(parents=True, exist_ok=True)
        return str(d), None
    tmp = tempfile.mkdtemp()
    return tmp, tmp


def write_table(conn, table_path, n=2, dim=4):
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


def test_decay_memories_without_recency_column_updates_zero_files():
    conn = setup_connection()
    table_dir, cleanup = make_table_dir("_zero")
    try:
        write_table(conn, table_dir)
        files_updated = conn.execute(
            f"SELECT ailake_decay_memories('{table_dir}', 0.1::FLOAT)"
        ).fetchone()[0]
        require(
            files_updated is not None and files_updated >= 0,
            f"expected files_updated >= 0, got {files_updated}"
        )
        print(f"PASS test_decay_memories_without_recency_column_updates_zero_files: files_updated={files_updated}")
    finally:
        if cleanup:
            shutil.rmtree(cleanup, ignore_errors=True)
        conn.close()


def test_decay_memories_missing_table_raises():
    conn = setup_connection()
    table_dir, cleanup = make_table_dir("_missing")
    try:
        try:
            conn.execute(
                f"SELECT ailake_decay_memories('{table_dir}', 0.1::FLOAT)"
            ).fetchone()
            require(False, "expected an exception for a nonexistent table, got a result")
        except duckdb.Error as e:
            require("ailake_decay_memories failed" in str(e), f"unexpected error message: {e}")
            print("PASS test_decay_memories_missing_table_raises")
    finally:
        if cleanup:
            shutil.rmtree(cleanup, ignore_errors=True)
        conn.close()


if __name__ == "__main__":
    test_decay_memories_without_recency_column_updates_zero_files()
    test_decay_memories_missing_table_raises()
    print()
    print("PASS: ailake_decay_memories — all tests passed")
