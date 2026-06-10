# SPDX-License-Identifier: MIT OR Apache-2.0
# Copyright (c) 2026 Thiago Egon Lange
"""
Post-build: appends the 512-byte DuckDB 1.0+ extension metadata block.

DuckDB reads the last 512 bytes of every extension binary and parses them as:
  bytes   0-255  : 8×32-byte metadata fields (null-padded strings)
  bytes 256-511  : signature block (256 zeros = unsigned extension)

DuckDB reads fields 0-7 sequentially, then calls std::reverse(), so the field
that ends up at index 0 (magic_value) is stored LAST in the file.

File layout (in the 256-byte metadata region):
  field 0 (bytes   0- 31): padding (unused)
  field 1 (bytes  32- 63): padding (unused)
  field 2 (bytes  64- 95): padding (unused)
  field 3 (bytes  96-127): abi_metadata  → "CPP"
  field 4 (bytes 128-159): extension_version
  field 5 (bytes 160-191): duckdb_version
  field 6 (bytes 192-223): platform      → e.g. "linux_amd64"
  field 7 (bytes 224-255): magic_value   → "4" + 31 null bytes  (MUST be last)

Usage:
  python append_extension_metadata.py <ext_path> <duckdb_version> <ext_version>
"""

import sys

FIELD_SIZE     = 32
NUM_FIELDS     = 8
METADATA_SIZE  = FIELD_SIZE * NUM_FIELDS   # 256
SIGNATURE_SIZE = 256
FOOTER_SIZE    = METADATA_SIZE + SIGNATURE_SIZE  # 512

# Magic value as defined in extension.hpp:
#   static constexpr const char *EXPECTED_MAGIC_VALUE = {"4\0\0\0..."};
MAGIC_VALUE = "4"


def pad_field(s: str) -> bytes:
    b = s.encode("utf-8")
    assert len(b) <= FIELD_SIZE, f"field too long ({len(b)} > {FIELD_SIZE}): {s!r}"
    return b + b"\x00" * (FIELD_SIZE - len(b))


def detect_platform() -> str:
    """Query the installed duckdb Python package for its platform string."""
    try:
        import duckdb as _ddb
        conn = _ddb.connect()
        row = conn.execute("SELECT platform FROM pragma_platform()").fetchone()
        conn.close()
        if row:
            return row[0]
    except Exception:
        pass
    # Fallback: derive from Python's platform detection (matches platform.hpp logic)
    import struct as _s
    import platform as _p
    os_name = "linux"
    arch = "amd64" if _s.calcsize("P") == 8 else "i686"
    if _p.system() == "Darwin":
        os_name = "osx"
    elif _p.system() == "Windows":
        os_name = "windows"
    if _p.machine().lower() in ("arm64", "aarch64"):
        arch = "arm64"
    return f"{os_name}_{arch}"


ext_path   = sys.argv[1]
duckdb_ver = sys.argv[2]   # e.g. "v1.1.3"
ext_ver    = sys.argv[3]   # e.g. "0.0.16"

platform = detect_platform()

# Build the 8 fields in file order (before DuckDB's reverse()):
fields = [b"\x00" * FIELD_SIZE] * NUM_FIELDS
fields[3] = pad_field("CPP")        # abi_metadata
fields[4] = pad_field(ext_ver)      # extension_version
fields[5] = pad_field(duckdb_ver)   # duckdb_version
fields[6] = pad_field(platform)     # platform
fields[7] = pad_field(MAGIC_VALUE)  # magic_value (LAST = index 0 after reverse)

metadata_region = b"".join(fields)
assert len(metadata_region) == METADATA_SIZE

signature_block = b"\x00" * SIGNATURE_SIZE  # zeros = unsigned extension
footer = metadata_region + signature_block
assert len(footer) == FOOTER_SIZE

with open(ext_path, "ab") as f:
    f.write(footer)

print(
    f"appended {FOOTER_SIZE}-byte DuckDB extension metadata: "
    f"magic={MAGIC_VALUE!r} platform={platform!r} "
    f"duckdb_version={duckdb_ver!r} ext_version={ext_ver!r} abi=CPP"
)
