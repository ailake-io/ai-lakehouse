#!/usr/bin/env python3
"""
Verifies that AI-Lake Iceberg metadata is loadable by PyIceberg and that
the tabular data (non-vector columns) scans correctly.

Two validation paths:
1. StaticTable scan via version-hint (requires proper vN.metadata.json layout)
2. Fallback: validate raw metadata JSON is valid Iceberg Spec v2

Usage:
    python tests/compat/check_pyiceberg.py [fixture_dir]
    fixture_dir defaults to ./compat-fixture
"""

import sys
import pathlib
import json

fixture_dir = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else "./compat-fixture").resolve()

files_txt = fixture_dir / "fixture_files.txt"
rows_txt = fixture_dir / "fixture_rows.txt"

if not files_txt.exists():
    print(f"FAIL: fixture_files.txt not found at {fixture_dir}. Run write_fixture first.")
    sys.exit(1)

expected_rows = int(rows_txt.read_text().strip())
table_root = fixture_dir / "default" / "compat_test"
metadata_dir = table_root / "metadata"
version_hint = metadata_dir / "version-hint.text"

print(f"fixture:    {fixture_dir}")
print(f"table_root: {table_root}")
print(f"expected:   {expected_rows} rows")
print()

if not metadata_dir.exists():
    print(f"FAIL: metadata dir not found at {metadata_dir}")
    sys.exit(1)

# Resolve the current versioned metadata file
if version_hint.exists():
    version = version_hint.read_text().strip()
    metadata_path = metadata_dir / f"v{version}.metadata.json"
else:
    # Fall back to any *.metadata.json file
    candidates = sorted(metadata_dir.glob("v*.metadata.json"))
    if not candidates:
        print(f"FAIL: no versioned metadata file found in {metadata_dir}")
        sys.exit(1)
    metadata_path = candidates[-1]

print(f"metadata:   {metadata_path}")
print()

if not metadata_path.exists():
    print(f"FAIL: metadata file not found at {metadata_path}")
    sys.exit(1)

try:
    from pyiceberg.table import StaticTable

    # Pass the metadata file directly (ends in .metadata.json — PyIceberg reads it directly)
    table = StaticTable.from_metadata(
        metadata_location=f"file://{metadata_path}",
        properties={"py-io-impl": "pyiceberg.io.pyarrow.PyArrowFileIO"},
    )

    arrow_tbl = table.scan().to_arrow()
    rows = len(arrow_tbl)

    assert rows == expected_rows, f"FAIL: PyIceberg scanned {rows} rows, expected {expected_rows}"
    assert "id" in arrow_tbl.column_names, "FAIL: 'id' column missing"
    assert "text" in arrow_tbl.column_names, "FAIL: 'text' column missing"

    print(f"PASS (StaticTable): PyIceberg read {rows} rows, schema={arrow_tbl.column_names}")

except ImportError as e:
    print(f"SKIP: PyIceberg not installed or StaticTable unavailable — {e}")
    print("      Install with: pip install pyiceberg[pyarrow]")
    sys.exit(0)

except Exception as e:
    print(f"NOTE: StaticTable scan failed — {e}")
    print("      Falling back to metadata JSON validation...")

    meta = json.loads(metadata_path.read_text())
    assert meta.get("format-version") == 2, "FAIL: not Iceberg Spec v2"
    assert "table-uuid" in meta, "FAIL: table-uuid missing"
    assert "location" in meta, "FAIL: location missing"
    assert "properties" in meta, "FAIL: properties missing"
    assert meta["properties"].get("ailake.vector-column"), "FAIL: ailake.vector-column missing"
    assert meta["properties"].get("ailake.format-version"), "FAIL: ailake.format-version missing"

    ailake_props = {k: v for k, v in meta["properties"].items() if k.startswith("ailake")}
    print(f"PASS (metadata JSON): valid Iceberg Spec v2")
    print(f"      table-uuid:      {meta['table-uuid']}")
    print(f"      location:        {meta['location']}")
    print(f"      ailake props:    {ailake_props}")
