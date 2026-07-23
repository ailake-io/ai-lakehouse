# SPDX-License-Identifier: MIT OR Apache-2.0
# Copyright (c) 2026 Thiago Egon Lange
"""
DuckDB ailake extension — ailake_scan() full-row table function tests.

Prerequisites:
  1. Build DuckDB extension (also builds ailake-jni as a static lib via corrosion):
       cmake --build duckdb-ailake/build
  2. Generate fixture:  python tests/fixtures/write_fixture.py

Usage:
  AILAKE_EXT=./duckdb-ailake/build/ailake.duckdb_extension \
  AILAKE_FIXTURE=./compat-fixture \
  python duckdb-ailake/test/test_scan.py
"""
import math
import os
import pathlib
import struct
import sys

import duckdb

# ── Config from environment ────────────────────────────────────────────────────

REPO_ROOT = pathlib.Path(__file__).parent.parent.parent
EXT_PATH  = pathlib.Path(os.environ.get("AILAKE_EXT",  str(REPO_ROOT / "duckdb-ailake/build/ailake.duckdb_extension")))
FIXTURE   = pathlib.Path(os.environ.get("AILAKE_FIXTURE", str(REPO_ROOT / "compat-fixture")))

DIM = 128  # matches write_fixture.py

# ── Helpers ────────────────────────────────────────────────────────────────────

PASS = 0
FAIL = 0

def require(cond, msg):
    global PASS, FAIL
    if cond:
        print(f"  PASS  {msg}")
        PASS += 1
    else:
        print(f"  FAIL  {msg}")
        FAIL += 1

def setup_connection():
    conn = duckdb.connect(config={
        "allow_unsigned_extensions": True,
        "allow_extensions_metadata_mismatch": True,
    })
    if EXT_PATH.exists():
        conn.execute(f"LOAD '{EXT_PATH}'")
    return conn

def table_path():
    return str(FIXTURE.resolve())

def fixture_query():
    """Return a unit-norm query vector of the right dimension."""
    v = [math.sin(i * 0.1) for i in range(DIM)]
    norm = math.sqrt(sum(x * x for x in v)) or 1.0
    return [x / norm for x in v]

# ── Tests ──────────────────────────────────────────────────────────────────────

def test_extension_loads():
    print("\ntest_extension_loads")
    conn = setup_connection()
    rows = conn.execute("SELECT 1").fetchall()
    require(rows == [(1,)], "DuckDB connection works")


def test_scan_returns_full_rows():
    print("\ntest_scan_returns_full_rows")
    if not EXT_PATH.exists():
        print("  SKIP  extension not built")
        return
    if not FIXTURE.exists():
        print("  SKIP  fixture not generated")
        return

    conn = setup_connection()
    q    = fixture_query()
    q_sql = ", ".join(str(f) for f in q)
    top_k = 10

    rows = conn.execute(
        f"SELECT * FROM ailake_scan('{table_path()}', [{q_sql}]::FLOAT[], {top_k})"
    ).fetchall()
    col_names = [d[0] for d in conn.description]

    require(len(rows) > 0,          f"ailake_scan returned rows (got {len(rows)})")
    require(len(rows) <= top_k,     f"rows <= top_k={top_k} (got {len(rows)})")
    require("_distance" in col_names, f"_distance column present, got {col_names}")
    # Fixture has at least an 'id' or row-id-like integer column.
    int_cols = [c for c in col_names if c != "_distance"]
    require(len(int_cols) >= 1,     f"at least one non-distance column present")


def test_scan_distance_ordered():
    print("\ntest_scan_distance_ordered")
    if not (EXT_PATH.exists() and FIXTURE.exists()):
        print("  SKIP  prerequisites missing")
        return

    conn  = setup_connection()
    q     = fixture_query()
    q_sql = ", ".join(str(f) for f in q)

    distances = [
        r[0] for r in conn.execute(
            f"SELECT _distance FROM ailake_scan('{table_path()}', [{q_sql}]::FLOAT[], 10)"
            " ORDER BY _distance"
        ).fetchall()
    ]
    require(len(distances) > 0, "got distance rows")
    require(distances == sorted(distances), "distances are ascending")
    require(all(d >= 0.0 for d in distances), "all distances >= 0")


def test_scan_vs_search_consistency():
    """ailake_scan and ailake_search must agree on top-k distance ordering."""
    print("\ntest_scan_vs_search_consistency")
    if not (EXT_PATH.exists() and FIXTURE.exists()):
        print("  SKIP  prerequisites missing")
        return

    conn  = setup_connection()
    q     = fixture_query()
    q_sql = ", ".join(str(f) for f in q)
    top_k = 5

    scan_dists = [
        r[0] for r in conn.execute(
            f"SELECT _distance FROM ailake_scan('{table_path()}', [{q_sql}]::FLOAT[], {top_k})"
            " ORDER BY _distance"
        ).fetchall()
    ]
    search_dists = [
        r[0] for r in conn.execute(
            f"SELECT distance FROM ailake_search('{table_path()}', [{q_sql}]::FLOAT[], {top_k})"
            " ORDER BY distance"
        ).fetchall()
    ]

    require(len(scan_dists) == len(search_dists),
            f"same row count: scan={len(scan_dists)} search={len(search_dists)}")
    if scan_dists and search_dists:
        max_diff = max(abs(a - b) for a, b in zip(scan_dists, search_dists))
        require(max_diff < 1e-4, f"distances agree within 1e-4 (max_diff={max_diff:.2e})")


def test_scan_nonexistent_table_raises():
    """A genuine backend rejection (nonexistent table path) raises InvalidInputException,
    not a silent empty result — see README "Error handling"."""
    print("\ntest_scan_nonexistent_table_raises")
    if not EXT_PATH.exists():
        print("  SKIP  extension not built")
        return

    conn = setup_connection()
    try:
        conn.execute(
            "SELECT * FROM ailake_scan('/nonexistent', [0.1, 0.2]::FLOAT[], 5)"
        ).fetchall()
        require(False, "expected an exception for a nonexistent table path, got a result")
    except duckdb.Error as e:
        require("ailake_scan failed" in str(e) or "failed" in str(e), f"unexpected error message: {e}")


# ── Runner ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    test_extension_loads()
    test_scan_returns_full_rows()
    test_scan_distance_ordered()
    test_scan_vs_search_consistency()
    test_scan_nonexistent_table_raises()

    total = PASS + FAIL
    print(f"\n{PASS}/{total} tests passed")
    if FAIL:
        sys.exit(1)
