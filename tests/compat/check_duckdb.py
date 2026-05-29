# SPDX-License-Identifier: MIT OR Apache-2.0
#!/usr/bin/env python3
"""
Verifies that AI-Lake Parquet files are readable by DuckDB's parquet_scan()
without any AI-Lake SDK.

Usage:
    python tests/compat/check_duckdb.py [fixture_dir]
    fixture_dir defaults to ./compat-fixture
"""

import sys
import pathlib
import duckdb

fixture_dir = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else "./compat-fixture")

files_txt = fixture_dir / "fixture_files.txt"
rows_txt = fixture_dir / "fixture_rows.txt"

if not files_txt.exists():
    print(f"FAIL: fixture_files.txt not found at {fixture_dir}. Run write_fixture first.")
    sys.exit(1)

parquet_paths = [p.strip() for p in files_txt.read_text().splitlines() if p.strip()]
expected_rows = int(rows_txt.read_text().strip())

print(f"fixture: {fixture_dir}")
print(f"expected: {expected_rows} rows across {len(parquet_paths)} file(s)")
print()

conn = duckdb.connect()

# Per-file checks
for path in parquet_paths:
    p = pathlib.Path(path)
    assert p.exists(), f"FAIL: parquet file not found: {path}"

    rows = conn.execute(f"SELECT count(*) FROM parquet_scan('{path}')").fetchone()[0]
    cols = [r[0] for r in conn.execute(f"DESCRIBE SELECT * FROM parquet_scan('{path}')").fetchall()]

    assert "id" in cols, f"FAIL: 'id' column missing in {path}"
    assert "text" in cols, f"FAIL: 'text' column missing in {path}"
    assert "embedding" in cols, f"FAIL: 'embedding' column missing in {path}"

    print(f"  {p.name}: {rows} rows, columns={cols}")

# Multi-file scan (glob)
data_dir = str(fixture_dir / "data" / "*.parquet")
total = conn.execute(f"SELECT count(*) FROM parquet_scan('{data_dir}')").fetchone()[0]
assert total == expected_rows, f"FAIL: total rows {total} != expected {expected_rows}"

# Verify id range
id_min, id_max = conn.execute(
    f"SELECT min(id), max(id) FROM parquet_scan('{data_dir}')"
).fetchone()
assert id_min == 0, f"FAIL: min id {id_min} != 0"
assert id_max == expected_rows - 1, f"FAIL: max id {id_max} != {expected_rows - 1}"

print()
print(f"PASS: DuckDB scanned {total} rows (id range [{id_min}, {id_max}])")
print("      AI-Lake Parquet files are fully compatible with DuckDB parquet_scan.")
