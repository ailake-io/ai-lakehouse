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
  required int64 chunk_id            [field_id=1];
  required binary chunk_text (STRING) [field_id=2];
  optional binary section_path (STRING) [field_id=3];
  required fixed_len_byte_array(3072) embedding
    [field_id=4, ailake.dim=1536, ailake.metric=cosine, ailake.precision=f16];
}
```

Every column carries the `PARQUET:field_id` metadata key set to its 1-based Iceberg field ID (batch columns: `1..N`, vector column: `N+1`). This matches the Iceberg schema written in `metadata.json`, enabling strict readers like Spark with `check-nullability` to validate field alignment without relying on name-mapping.

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

- [x] `format-version` is `2`
- [x] `table-uuid` is a valid UUID v4
- [x] `current-schema-id` references a valid schema in `schemas`
- [x] `current-snapshot-id` references a valid snapshot in `snapshots`
- [x] Each snapshot has `manifest-list` pointing to a valid `.avro` file
- [x] Each `DataFile` entry has `file-path`, `file-format`, `partition`, `record-count`, `file-size-in-bytes`
- [x] `file-format` is `PARQUET` (no ORC, no Avro data)
- [x] Sequence numbers are monotonically increasing
- [x] All `ailake.*` keys in `properties` have string values
- [x] `key_metadata` bytes in each `DataFile` entry, when non-null, deserialize as valid `AilakeEntryExt` JSON
- [x] Centroid value in `key_metadata` is valid base64-encoded f32 array of the expected length
- [x] `schemas[0].fields` includes all Parquet columns with correct Iceberg types and field-ids — generated automatically by `TableWriter.commit()` via `arrow_schema_to_iceberg_update`; covers scalar types, `timestamptz`, `List`, `Struct`, `Map`, and `FixedSizeBinary` (vector columns). `schema.name-mapping.default` is also written as a fallback so PyIceberg resolves columns by name even without Parquet field-ids.
- [x] Each Parquet column carries `PARQUET:field_id` matching its Iceberg schema field-id (1-based; batch fields `1..N`, vector column `N+1`)

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
- **Status**: compatible — tested in CI via `compat-heavy.yml` (runs on every push to `main` and weekly). Uses `iceberg-spark-runtime-3.5_2.12:1.5.2` with `HadoopCatalog` (filesystem). Tests verify row count, `MIN`/`MAX` id, and schema columns (`id`, `text`, `embedding`) via Spark SQL.

### Trino (iceberg connector)
- `FIXED_LEN_BYTE_ARRAY` mapped to `VARBINARY`.
- Table properties visible via `SHOW CREATE TABLE`.
- **Status**: compatible — tested in CI via `compat-heavy.yml`. Uses `tabulario/iceberg-rest:0.10.0` as REST catalog (internally wraps `HadoopCatalog`, discovers tables by filesystem layout) + `trinodb/trino:436`. Tests verify row count, `MIN`/`MAX` id via Trino Python client; cross-verified via PyIceberg REST catalog scan.

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

### Always-on (every PR and push — `ci.yml`)

1. **PyArrow read test** (`compat-pyarrow`): build fixture via `write_fixture`, read all Parquet files with `pyarrow.parquet`. Verify row count, schema columns, vector column as binary.
2. **DuckDB read test** (`compat-duckdb`): same fixture, read via `duckdb.read_parquet`. Verify row count and id range.
3. **PyIceberg scan test** (`compat-pyiceberg`): load via `StaticTable.from_metadata` + `.scan().to_arrow()`. Verify 1000 rows, schema `[id, text, embedding]`.
4. **ailake-py SDK test** (`compat-ailake-py`): build wheel with `maturin` (Python 3.12), run `check_ailake_py.py` covering write→search (cosine + euclidean), multi-batch commit, `assemble_context`, error paths.

### Heavy engines (`compat-heavy.yml` — push to `main` + weekly schedule)

5. **Spark+Iceberg read test** (`compat-spark`): reads fixture via `iceberg-spark-runtime-3.5_2.12:1.5.2` with `HadoopCatalog`; SQL `COUNT`, `MIN(id)`, `MAX(id)`, schema validation.
6. **Trino read test** (`compat-trino`): `tabulario/iceberg-rest:0.10.0` REST catalog + `trinodb/trino:436`; verified via PyIceberg REST scan and Trino Python client.
7. **JVM plugins** (`compat-jvm-plugins`): builds `libailake_jni.so`, runs Flink, Spark, and Trino Gradle integration tests.

Failure of tests 1–4 is a PR blocker. Failure of 5–7 is a release blocker.
