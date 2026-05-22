#!/usr/bin/env python3
"""
Verifies that AI-Lake Parquet files are readable by standard PyArrow
without any AI-Lake SDK. This is the core Parquet compatibility guarantee.

Usage:
    python tests/compat/check_pyarrow.py [fixture_dir]
    fixture_dir defaults to ./compat-fixture
"""

import sys
import pathlib
import pyarrow.parquet as pq
import pyarrow as pa

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

total_rows = 0
for path in parquet_paths:
    p = pathlib.Path(path)
    assert p.exists(), f"FAIL: parquet file not found: {path}"

    tbl = pq.read_table(path)
    rows = tbl.num_rows
    total_rows += rows

    # Schema checks
    schema = tbl.schema
    assert schema.get_field_index("id") >= 0, f"FAIL: 'id' column missing in {path}"
    assert schema.get_field_index("text") >= 0, f"FAIL: 'text' column missing in {path}"
    assert schema.get_field_index("embedding") >= 0, f"FAIL: 'embedding' column missing in {path}"

    id_col = tbl.column("id")
    assert pa.types.is_integer(id_col.type), f"FAIL: 'id' not integer in {path}"

    print(f"  {p.name}: {rows} rows, schema={[f.name for f in schema]}")

assert total_rows == expected_rows, (
    f"FAIL: total rows {total_rows} != expected {expected_rows}"
)

print()
print(f"PASS: PyArrow read {total_rows} rows across {len(parquet_paths)} AI-Lake file(s)")
print("      Standard Parquet readers are fully compatible with the AI-Lake format.")
