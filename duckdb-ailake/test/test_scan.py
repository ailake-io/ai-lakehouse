"""
Integration tests for ailake_scan() — full-row table function.

Requires:
  - libailake_jni.so: built via `cargo build --release -p ailake-jni`
  - ailake.duckdb_extension: built via cmake in duckdb-ailake/build/
  - duckdb Python package (same minor version as DUCKDB_VERSION in CMakeLists.txt)

Run:
  pytest duckdb-ailake/test/test_scan.py -v
"""

import ctypes
import io
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).parent.parent.parent
LIB_PATH  = REPO_ROOT / "target/release/libailake_jni.so"
EXT_PATH  = REPO_ROOT / "duckdb-ailake/build/ailake.duckdb_extension"

# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(scope="session")
def duckdb_conn():
    # Pre-load native lib into global symbol table (required for DuckDB extensions
    # that resolve symbols via RTLD_DEFAULT rather than a private handle).
    if LIB_PATH.exists():
        ctypes.CDLL(str(LIB_PATH), ctypes.RTLD_GLOBAL)

    # Force RTLD_GLOBAL when importing duckdb so extension symbols are visible.
    old_flags = sys.getdlopenflags()
    sys.setdlopenflags(old_flags | os.RTLD_GLOBAL)
    import duckdb
    sys.setdlopenflags(old_flags)

    conn = duckdb.connect()
    conn.execute("SET allow_unsigned_extensions = true")
    if EXT_PATH.exists():
        conn.execute(f"LOAD '{EXT_PATH}'")
    return conn


@pytest.fixture(scope="session")
def sample_table(duckdb_conn):
    """Write a small AI-Lake table and return its path."""
    import ailake  # noqa: F401 — must be importable

    dim = 8
    n   = 20
    rng = np.random.default_rng(42)
    vecs = rng.random((n, dim), dtype=np.float32)

    import pyarrow as pa
    schema = pa.schema([
        pa.field("id",   pa.int64()),
        pa.field("text", pa.string()),
    ])
    ids   = list(range(n))
    texts = [f"doc_{i}" for i in range(n)]
    batch = pa.record_batch({"id": ids, "text": texts}, schema=schema)

    tmpdir = tempfile.mkdtemp(prefix="ailake_scan_test_")
    writer = ailake.TableWriter(tmpdir)
    writer.write_batch(batch, embeddings=vecs)
    writer.commit()

    return tmpdir, dim, vecs


# ── Tests ──────────────────────────────────────────────────────────────────────

def test_ext_loads(duckdb_conn):
    if not EXT_PATH.exists():
        pytest.skip("Extension not built")
    result = duckdb_conn.execute("SELECT 1").fetchall()
    assert result == [(1,)]


def test_scan_returns_full_rows(duckdb_conn, sample_table):
    if not EXT_PATH.exists():
        pytest.skip("Extension not built")
    if not LIB_PATH.exists():
        pytest.skip("libailake_jni.so not built")

    path, dim, vecs = sample_table
    query = vecs[0].tolist()
    top_k = 5

    query_sql = ", ".join(str(f) for f in query)
    sql = f"""
        SELECT *
        FROM ailake_scan('{path}', [{query_sql}]::FLOAT[], {top_k})
        ORDER BY _distance
    """
    rows = duckdb_conn.execute(sql).fetchall()
    cols = [d[0] for d in duckdb_conn.description]

    assert len(rows) == top_k, f"Expected {top_k} rows, got {len(rows)}"
    assert "_distance" in cols, "_distance column missing"
    assert "id" in cols, "id column missing"
    assert "text" in cols, "text column missing"


def test_scan_distance_ordered(duckdb_conn, sample_table):
    if not EXT_PATH.exists():
        pytest.skip("Extension not built")
    if not LIB_PATH.exists():
        pytest.skip("libailake_jni.so not built")

    path, dim, vecs = sample_table
    query = vecs[0].tolist()
    query_sql = ", ".join(str(f) for f in query)
    sql = f"""
        SELECT _distance
        FROM ailake_scan('{path}', [{query_sql}]::FLOAT[], 10)
        ORDER BY _distance
    """
    distances = [r[0] for r in duckdb_conn.execute(sql).fetchall()]
    assert distances == sorted(distances), "Results not ordered by distance"
    assert distances[0] >= 0.0


def test_scan_vs_search_same_ids(duckdb_conn, sample_table):
    """ailake_scan and ailake_search must return the same top-k row ids."""
    if not EXT_PATH.exists():
        pytest.skip("Extension not built")
    if not LIB_PATH.exists():
        pytest.skip("libailake_jni.so not built")

    path, dim, vecs = sample_table
    query = vecs[3].tolist()
    query_sql = ", ".join(str(f) for f in query)
    top_k = 5

    scan_ids = set(
        r[0] for r in duckdb_conn.execute(
            f"SELECT id FROM ailake_scan('{path}', [{query_sql}]::FLOAT[], {top_k})"
        ).fetchall()
    )
    search_row_ids = set(
        r[0] for r in duckdb_conn.execute(
            f"SELECT row_id FROM ailake_search('{path}', [{query_sql}]::FLOAT[], {top_k})"
        ).fetchall()
    )

    # row_id is the 0-based index; id in our table equals row_id.
    assert scan_ids == search_row_ids, (
        f"Scan ids {scan_ids} != search row_ids {search_row_ids}"
    )


def test_scan_no_lib_graceful(duckdb_conn):
    """Without the native lib, ailake_scan returns zero rows (no crash)."""
    if not EXT_PATH.exists():
        pytest.skip("Extension not built")
    if LIB_PATH.exists():
        pytest.skip("Native lib present — graceful-degradation path not active")

    rows = duckdb_conn.execute(
        "SELECT * FROM ailake_scan('/nonexistent', [0.1, 0.2]::FLOAT[], 5)"
    ).fetchall()
    assert rows == []
