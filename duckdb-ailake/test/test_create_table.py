# SPDX-License-Identifier: MIT OR Apache-2.0
# Copyright (c) 2026 Thiago Egon Lange

"""Tests for ailake_create_table extension function.

Verifies that:
  1. Creating an empty table succeeds.
  2. Searching on an empty table returns 0 rows (no error).
  3. Creating the same table again raises an error.
"""

import os
import pathlib
import pytest
import duckdb


def _ext_path():
    env = os.environ.get("AILAKE_EXT")
    if env:
        return env
    here = pathlib.Path(__file__).resolve().parent
    candidates = [
        here.parent / "build" / "release" / "ailake.duckdb_extension",
        here.parent / "build" / "ailake.duckdb_extension",
    ]
    for c in candidates:
        if c.exists():
            return str(c)
    raise RuntimeError(f"ailake.duckdb_extension not found — set AILAKE_EXT env var or build first. Tried: {[str(x) for x in candidates]}")


@pytest.fixture
def db(tmp_path):
    conn = duckdb.connect(config={"allow_unsigned_extensions": True})
    conn.execute(f"LOAD '{_ext_path()}'")
    yield conn
    conn.close()


def make_table_path(tmp_path, name: str) -> str:
    return str(tmp_path / name)


def test_create_empty_table_search_returns_zero_rows(db, tmp_path):
    tbl = make_table_path(tmp_path, "empty_table")
    db.execute(f"SELECT ailake_create_table('{tbl}', 'embedding', 1536, 'cosine', 'f16', 2)")
    rows = db.execute(f"SELECT ailake_search('{tbl}', 'AAAA', 10)").fetchall()
    assert len(rows) == 0, f"expected 0 results from empty table, got {len(rows)}"


def test_create_duplicate_raises(db, tmp_path):
    tbl = make_table_path(tmp_path, "dup_table")
    db.execute(f"SELECT ailake_create_table('{tbl}', 'embedding', 1536, 'cosine', 'f16', 2)")
    with pytest.raises(Exception, match="already exists|TableAlreadyExists|Conflict"):
        db.execute(f"SELECT ailake_create_table('{tbl}', 'embedding', 1536, 'cosine', 'f16', 2)")


def test_create_with_custom_params(db, tmp_path):
    tbl = make_table_path(tmp_path, "custom_table")
    db.execute(
        f"SELECT ailake_create_table('{tbl}', 'my_vec', 768, 'euclidean', 'f32', 2)"
    )
    rows = db.execute(f"SELECT ailake_search('{tbl}', 'AAAA', 5)").fetchall()
    assert len(rows) == 0


def test_create_then_insert_then_search(db, tmp_path):
    tbl = make_table_path(tmp_path, "insert_table")
    db.execute(f"SELECT ailake_create_table('{tbl}', 'embedding', 3, 'cosine', 'f16', 2)")
    db.execute(
        f"SELECT ailake_write_batch('{tbl}', 'embedding', 3, 'cosine', 'f16', "
        "[1,2], [[1.0,0.0,0.0],[0.0,1.0,0.0]], 'default', 'insert_table')"
    )
    rows = db.execute(f"SELECT ailake_search('{tbl}', 'AAAA', 5)").fetchall()
    assert len(rows) == 2
