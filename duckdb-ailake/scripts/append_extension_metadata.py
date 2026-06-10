# SPDX-License-Identifier: MIT OR Apache-2.0
# Copyright (c) 2026 Thiago Egon Lange
"""
Post-build script: appends the 512-byte metadata block that DuckDB 1.0+
requires for every extension, regardless of signing.

Format (512 bytes appended after the shared library binary):
  bytes   0-255  : signature block (all zeros = unsigned extension)
  bytes 256-511  : JSON metadata, null-padded to 256 bytes

Usage:
  python append_extension_metadata.py <ext_path> <duckdb_version> <ext_version>
"""

import json
import sys

ext_path     = sys.argv[1]
duckdb_ver   = sys.argv[2]   # e.g. "v1.1.3"
ext_ver      = sys.argv[3]   # e.g. "0.0.16"

metadata_json = json.dumps(
    {
        "duckdb_version":    duckdb_ver,
        "extension_version": ext_ver,
        "build_type":        "Release",
    },
    separators=(",", ":"),
).encode("utf-8")

assert len(metadata_json) <= 256, (
    f"metadata JSON too large ({len(metadata_json)} bytes, max 256)"
)

signature_block  = b"\x00" * 256
json_block       = metadata_json + b"\x00" * (256 - len(metadata_json))
metadata_block   = signature_block + json_block   # 512 bytes total

with open(ext_path, "ab") as f:
    f.write(metadata_block)

print(
    f"appended {len(metadata_block)}-byte DuckDB extension metadata to {ext_path} "
    f"(duckdb={duckdb_ver} ext={ext_ver})"
)
