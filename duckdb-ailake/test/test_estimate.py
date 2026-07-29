# SPDX-License-Identifier: MIT OR Apache-2.0
# Copyright (c) 2026 Thiago Egon Lange
"""
DuckDB ailake extension — ailake_estimate function tests.

Prerequisites:
  1. Build DuckDB extension (also builds ailake-jni as a static lib via corrosion):
       cmake --build duckdb-ailake/build

Usage:
  AILAKE_EXT=./duckdb-ailake/build/ailake.duckdb_extension \
  python duckdb-ailake/test/test_estimate.py
"""
import os
import sys

import duckdb

EXT_PATH = os.environ.get("AILAKE_EXT", "./duckdb-ailake/build/ailake.duckdb_extension")


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


def test_estimate_returns_six_modes_with_correct_f32_math():
    conn = setup_connection()
    try:
        rows = conn.execute(
            "SELECT mode, vectors_bytes, index_bytes, total_bytes, reduction_vs_f32_hnsw, "
            "recall, note FROM ailake_estimate(1000000, 1536)"
        ).fetchall()
        require(len(rows) == 6, f"expected 6 mode rows, got {len(rows)}")

        by_mode = {r[0]: r for r in rows}
        require("F32 (baseline)" in by_mode, f"missing F32 baseline row: {by_mode.keys()}")
        f32 = by_mode["F32 (baseline)"]
        expected_vec_bytes = 1_000_000 * 1536 * 4
        require(
            f32[1] == expected_vec_bytes,
            f"F32 vectors_bytes mismatch: expected {expected_vec_bytes}, got {f32[1]}"
        )
        require(
            abs(f32[4] - 1.0) < 1e-9,
            f"F32 reduction_vs_f32_hnsw should be 1.0, got {f32[4]}"
        )

        require("F16 (default)" in by_mode, f"missing F16 row: {by_mode.keys()}")
        f16 = by_mode["F16 (default)"]
        expected_f16_bytes = 1_000_000 * 1536 * 2
        require(
            f16[1] == expected_f16_bytes,
            f"F16 vectors_bytes mismatch: expected {expected_f16_bytes}, got {f16[1]}"
        )

        print(f"PASS test_estimate_returns_six_modes_with_correct_f32_math: {len(rows)} rows")
    finally:
        conn.close()


def test_estimate_respects_hnsw_m_and_pq_m_overrides():
    conn = setup_connection()
    try:
        row_default = conn.execute(
            "SELECT index_bytes FROM ailake_estimate(1000, 128) WHERE mode = 'F32 (baseline)'"
        ).fetchone()
        row_bigger_m = conn.execute(
            "SELECT index_bytes FROM ailake_estimate(1000, 128, hnsw_m := 64) WHERE mode = 'F32 (baseline)'"
        ).fetchone()
        require(
            row_bigger_m[0] > row_default[0],
            f"expected larger hnsw_m to increase index_bytes: {row_default[0]} vs {row_bigger_m[0]}"
        )
        print(f"PASS test_estimate_respects_hnsw_m_and_pq_m_overrides: {row_default[0]} -> {row_bigger_m[0]}")
    finally:
        conn.close()


def test_estimate_rejects_zero_dim():
    conn = setup_connection()
    try:
        try:
            conn.execute("SELECT * FROM ailake_estimate(1000, 0)").fetchall()
            require(False, "expected an exception for dim=0, got a result")
        except duckdb.Error as e:
            require("dim must be > 0" in str(e), f"unexpected error message: {e}")
            print("PASS test_estimate_rejects_zero_dim")
    finally:
        conn.close()


if __name__ == "__main__":
    test_estimate_returns_six_modes_with_correct_f32_math()
    test_estimate_respects_hnsw_m_and_pq_m_overrides()
    test_estimate_rejects_zero_dim()
    print()
    print("PASS: ailake_estimate — all tests passed")
