# ICEBERG_COMPAT.md — Iceberg Compatibility Contract

## The guarantee

Any table written by the AI-Lake SDK MUST be readable by any Iceberg-compatible framework (PyIceberg, Spark iceberg-spark, Trino iceberg connector, DuckDB iceberg extension, Snowflake, AWS Athena) without modification, plugin, or error.

This document specifies exactly how that guarantee is maintained.

---

## Two compatibility layers

AI-Lake compatibility relies on **two specifications simultaneously**:

1. **Iceberg Spec v2** — for `metadata.json` and Avro manifests
2. **Apache Parquet Spec** — for the file-level extension (AI-Lake footer after `PAR1`)

Both specifications define mechanisms for application-specific extensions that unknown readers ignore safely. We use both.

---

## What we write — and why it stays compatible

### 1. `metadata/v{N}.metadata.json`

Written as strict Iceberg Spec v2. The only AI-Lake additions are inside the `properties` object, which Iceberg explicitly defines as a free-form string map for table configuration.

```json
{
  "format-version": 2,
  "table-uuid": "550e8400-e29b-41d4-a716-446655440000",
  "location": "s3://my-lake/my_table",
  "last-sequence-number": 3,
  "last-updated-ms": 1722470400000,
  "last-column-id": 12,
  "schemas": [...],
  "current-schema-id": 0,
  "partition-specs": [...],
  "sort-orders": [...],
  "snapshots": [...],
  "current-snapshot-id": 1234567890,
  "properties": {
    "ailake.format-version": "1",
    "ailake.vector-column": "embedding",
    "ailake.vector-dim": "1536",
    "ailake.vector-metric": "cosine",
    "ailake.vector-precision": "f16"
  }
}
```

**Why this is safe**: `properties` is a `Map<String, String>` with no reserved namespace in Iceberg Spec v2. All existing readers pass unknown keys through or ignore them.

### 2. `metadata/{SNAP_ID}-m0.avro` and `metadata/snap-{SNAP_ID}-1.avro`

Two Avro OCF files per snapshot:

- **Manifest file** (`{snap_id}-m0.avro`): one `manifest_entry` record per `DataFile`. Carries AI-Lake vector statistics in the `key_metadata` bytes field (JSON-encoded `AilakeEntryExt`).
- **Manifest list** (`snap-{snap_id}-1.avro`): one `manifest_file` record per manifest file, with row-count statistics.

Both files are written by `avro_raw.rs` — a custom Avro OCF serializer that embeds the schema JSON **verbatim** in the file header. This bypasses `apache-avro 0.16`, which strips unknown schema properties (like `"field-id"`) when serializing schema back to JSON, breaking PyIceberg's `avro_schema_to_iceberg` conversion.

Per-file AI-Lake metadata is encoded as JSON in the `key_metadata` bytes field:

```json
{
  "centroid_b64": "<base64-encoded f32 array>",
  "radius": 0.342,
  "hnsw_offset": 12582912,
  "hnsw_len": 4194304,
  "vector_column": "embedding",
  "vector_dim": 1536
}
```

**Why this is safe**: `key_metadata` is a standard `bytes` field in the Iceberg `data_file` Avro schema (Spec v2 §4.1.7, field-id 131). Iceberg readers that don't know AI-Lake pass this field through as opaque bytes or ignore it entirely. The centroid is base64-encoded as a compact f32 array (e.g. for dim=1536: 6144 bytes raw → ~8.2 KB base64).

### 3. `data/part-NNNNN.parquet`

**This is the critical part of the single-file design.**

The file is a **valid Parquet file** with the AI-Lake footer appended after the Parquet `PAR1` end marker. Per the Parquet specification:

> "Implementations MUST tolerate trailing data after the Parquet footer."

(See [Apache Parquet specification](https://github.com/apache/parquet-format), section "File Format" — the format defines a clear end marker but does not forbid additional bytes after it. Major implementations including parquet-mr, parquet-cpp, and pyarrow stop reading at the final `PAR1` marker.)

#### Parquet section structure

```
message ailake_schema {
  required int64 chunk_id;
  required binary chunk_text (STRING);
  optional binary section_path (STRING);
  required fixed_len_byte_array(3072) embedding
    [field_metadata: ailake.dim=1536, ailake.metric=cosine, ailake.precision=f16];
}
```

File-level `key_value_metadata`:
- `ailake.format_version` = `"1"`
- `ailake.hnsw_offset` = `"12582912"` (byte offset of AI-Lake footer)
- `ailake.hnsw_len` = `"4194304"`
- `ailake.precision` = `"f16"`
- `ailake.metric` = `"cosine"`

**Why this is safe**:
- `FIXED_LEN_BYTE_ARRAY` is a standard Parquet physical type.
- Parquet field metadata (`key_value_metadata` on `SchemaElement`) is explicitly defined in the Parquet spec as application-specific and silently ignored by readers that don't know the keys.
- File-level `key_value_metadata` is also application-specific.
- The bytes after the final `PAR1` are not part of the Parquet structure — readers terminate at the magic marker.

### 4. AI-Lake footer (after `PAR1`)

Contains the HNSW graph, centroid, radius, and supported distance metrics. **Standard Parquet readers never see this section** because they stop reading at the `PAR1` marker per spec.

See [`FILE_FORMAT.md`](./FILE_FORMAT.md) for the binary layout.

---

## What we do NOT write into Iceberg/Parquet structures

| Structure | AI-Lake additions | Rationale |
|---|---|---|
| `metadata.json` root fields | None | Root fields are spec-validated |
| Avro manifest record fields | None (only `custom-properties`) | Extra fields in Avro records break strict readers |
| Parquet schema | Only field metadata | Extra schema fields would change column count |
| Parquet row groups | None | Row group metadata is spec-validated |
| Parquet footer structure | Only `key_value_metadata` | Other footer fields are spec-validated |

---

## Iceberg Spec v2 compliance checklist

The `ailake-catalog` crate is responsible for maintaining this list. Every catalog write operation must pass these checks before committing:

- [ ] `format-version` is `2`
- [ ] `table-uuid` is a valid UUID v4
- [ ] `current-schema-id` references a valid schema in `schemas`
- [ ] `current-snapshot-id` references a valid snapshot in `snapshots`
- [ ] Each snapshot has `manifest-list` pointing to a valid `.avro` file
- [ ] Each `DataFile` entry has `file-path`, `file-format`, `partition`, `record-count`, `file-size-in-bytes`
- [ ] `file-format` is `PARQUET` (no ORC, no Avro data)
- [ ] Sequence numbers are monotonically increasing
- [ ] All `ailake.*` keys in `properties` have string values
- [ ] `key_metadata` bytes in each `DataFile` entry, when non-null, deserialize as valid `AilakeEntryExt` JSON
- [ ] Centroid value in `key_metadata` is valid base64-encoded f32 array of the expected length
- [ ] `schemas[0].fields` includes all Parquet columns with correct Iceberg types and field-ids, OR `schema.name-mapping.default` is set in `properties` so PyIceberg can resolve columns without Parquet field-ids

### Validation via PyIceberg (Phase 2 integration test)

```python
# tests/compat/test_pyiceberg_read.py
import pyiceberg.catalog
import pytest
import pyarrow as pa

def test_pyiceberg_reads_ailake_table(tmp_path):
    """Written by AI-Lake SDK, read by PyIceberg — must not raise."""
    table_uri = str(tmp_path / "test_table")

    # Write via AI-Lake Rust SDK (called via subprocess in Phase 1)
    write_ailake_table(table_uri, rows=1000, dim=128)

    # Read via PyIceberg — no AI-Lake plugin
    catalog = pyiceberg.catalog.load_catalog("local", **{"type": "rest", ...})
    table = catalog.load_table("test_table")
    df = table.scan().to_arrow()

    assert len(df) == 1000
    assert "chunk_text" in df.schema.names
    assert "embedding" in df.schema.names
    # embedding column reads as bytes — this is correct and expected
    assert df.schema.field("embedding").type == pa.large_binary()
```

### Validation that PyArrow ignores the trailing AI-Lake footer

```python
# tests/compat/test_parquet_trailing_bytes.py
import pyarrow.parquet as pq

def test_pyarrow_reads_ailake_parquet(ailake_parquet_path):
    """PyArrow reads a Parquet file with AI-Lake footer appended."""
    table = pq.read_table(ailake_parquet_path)
    assert table.num_rows > 0
    # PyArrow does not error on trailing bytes after the final PAR1
```

---

## Version evolution strategy

When the AI-Lake format needs a breaking change in how vector metadata is stored:

1. Increment `ailake.format-version` in `properties` (e.g. `"2"`).
2. Also increment the version in the AI-Lake footer header.
3. Old SDK versions read the version and raise `AilakeError::UnsupportedFormatVersion`.
4. Iceberg readers ignore the version key entirely — the table itself remains readable regardless.
5. New format versions MUST NOT change the Parquet schema layout, Avro manifest structure, or the position of the AI-Lake trailer (last 24 bytes of file).

The Iceberg `format-version` (currently `2`) is NOT our version number and MUST NOT be changed.

---

## Known compatibility notes by framework

### PyIceberg
- Reads `FIXED_LEN_BYTE_ARRAY` as `pa.fixed_size_binary(N)` or `pa.large_binary()`.
- Unknown `properties` keys: silently passed through to `table.properties` dict.
- Unknown field metadata on schema elements: silently ignored.
- Trailing bytes after `PAR1`: PyArrow (used by PyIceberg) terminates at the final magic marker.
- `key_metadata` on `DataFile`: passed through as bytes; AI-Lake SDK decodes the JSON.
- Tables require either Parquet field-ids or `schema.name-mapping.default` in `properties` for `StaticTable.scan()`. AI-Lake fixture sets both.
- **Status**: fully compatible — `StaticTable.from_metadata` + `.scan().to_arrow()` passes with 1 000-row fixture.

### Apache Spark (iceberg-spark 1.5+)
- `FIXED_LEN_BYTE_ARRAY` mapped to `BinaryType` in Spark SQL.
- `properties` exposed via `SHOW TBLPROPERTIES`.
- `custom-properties` accessible via `table.snapshot().allManifests().dataFiles().properties()`.
- **Status**: compatible. Verify with Spark integration test in Phase 3.

### Trino (iceberg connector)
- `FIXED_LEN_BYTE_ARRAY` mapped to `VARBINARY`.
- Table properties visible via `SHOW CREATE TABLE`.
- **Status**: compatible. Verify with Trino integration test in Phase 3.

### DuckDB (iceberg extension 0.10+)
- Full Iceberg Spec v2 support added in 0.10.
- `FIXED_LEN_BYTE_ARRAY` → `BLOB`.
- DuckDB Parquet reader uses `parquet-cpp`, which stops at `PAR1`.
- **Status**: compatible.

### Snowflake (Iceberg external tables)
- Reads Parquet directly via Iceberg catalog on S3.
- `FIXED_LEN_BYTE_ARRAY` → `BINARY`.
- **Status**: compatible by design (Parquet physical types are standard, trailing bytes are ignored).

### AWS Athena
- Uses Glue Data Catalog + Iceberg Parquet reader.
- `FIXED_LEN_BYTE_ARRAY` → `BINARY`.
- **Status**: compatible, no known issues.

---

## Verifying compatibility in CI

Phase 1 CI must include these tests:

1. **PyArrow read test**: write a small AI-Lake file, open it with `pyarrow.parquet.read_table`. Verify no errors, correct row count, vector column as binary.
2. **PyIceberg scan test**: write an AI-Lake table (multiple files + metadata), load with PyIceberg, scan to PyArrow. Verify schema and row counts match.
3. **Parquet stripped test**: write an AI-Lake file, truncate everything after the final `PAR1`, verify the truncated file is also a valid Parquet file with identical Parquet-level content.

Phase 3 CI adds:

4. **Spark read test**: same as PyIceberg but via Spark SQL.
5. **Trino read test**: same via Trino.
6. **DuckDB read test**: same via DuckDB.

Failure of any of these tests is a release blocker.
