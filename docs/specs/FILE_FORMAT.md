# AI-Lake File Format Specification — v1

**Status**: Stable (format-version = 1)
**Magic bytes**: `AILK` (0x41 0x49 0x4C 0x4B)
**Byte order**: little-endian throughout

---

## 1. Overview

An AI-Lake file is a standard Apache Parquet file extended with one or more
**AILK sections** embedded between the last row group and the Parquet footer.
The extension is invisible to standard Parquet readers: they follow row-group
offsets in the footer, which point before the AILK section, and never
encounter the extension bytes.

Every AI-Lake file is independently self-contained. It carries:

- tabular data (standard Parquet row groups)
- one or more vector columns encoded as `FIXED_LEN_BYTE_ARRAY`
- one AILK section per vector column (centroid + HNSW index)

---

## 2. File Layout

```
Byte 0
┌─────────────────────────────────────────────────────────────────┐
│  Parquet magic (4 bytes)  "PAR1"                                │
├─────────────────────────────────────────────────────────────────┤
│  Parquet row groups                                             │
│  (standard columnar data; vector column stored as               │
│   FIXED_LEN_BYTE_ARRAY with dim × precision bytes per row)      │
│                                                                 │
│  ← Parquet footer row-group offsets point here                  │
├─────────────────────────────────────────────────────────────────┤
│  AILK section — primary vector column                           │
│    64-byte AILK header                                          │
│    centroid blob  (dim × 4 bytes F32 + 4-byte radius F32)       │
│    HNSW index blob (bincode-serialized hnsw_rs graph)           │
│    24-byte AILK trailer                                         │
├─────────────────────────────────────────────────────────────────┤
│  AILK section — secondary vector column (optional, repeating)   │
│    (same structure as above)                                    │
├─────────────────────────────────────────────────────────────────┤
│  Parquet footer (schema, row-group metadata, KV metadata)       │
│    KV entry:  ailake.footer_offset = <decimal byte offset of    │
│               primary AILK section>                             │
│    KV entry:  ailake.<col>.footer_offset = <decimal byte offset>│
│               (one per secondary column)                        │
│  4-byte footer length (little-endian u32)                       │
│  Parquet magic (4 bytes)  "PAR1"                                │
└─────────────────────────────────────────────────────────────────┘
EOF
```

The last 4 bytes of the file are always `PAR1` (Parquet spec).
AILK sections are placed **before** the Parquet footer so the file remains a
valid, self-consistent Parquet file.

---

## 3. AILK Header (64 bytes)

Starts at byte 0 of every AILK section. All integer fields little-endian.

| Offset | Size | Type  | Field             | Description |
|--------|------|-------|-------------------|-------------|
| 0      | 4    | bytes | `magic`           | `AILK` (0x41 0x49 0x4C 0x4B) |
| 4      | 2    | u16   | `format_version`  | Must be `1` for this spec |
| 6      | 2    | u16   | `flags`           | Reserved; must be `0` |
| 8      | 4    | u32   | `dim`             | Vector dimensionality |
| 12     | 1    | u8    | `precision`       | See §3.1 |
| 13     | 1    | u8    | `distance_metric` | See §3.2 |
| 14     | 2    | —     | reserved          | Must be `0` |
| 16     | 8    | u64   | `record_count`    | Number of vectors in this section |
| 24     | 8    | u64   | `centroid_offset` | Byte offset of centroid blob relative to AILK section start |
| 32     | 8    | u64   | `centroid_len`    | Byte length of centroid blob |
| 40     | 8    | u64   | `hnsw_offset`     | Byte offset of HNSW blob relative to AILK section start |
| 48     | 8    | u64   | `hnsw_len`        | Byte length of HNSW blob |
| 56     | 8    | —     | reserved          | Must be `0` |

### 3.1 `precision` values

| Value | Encoding | Bytes/element |
|-------|----------|---------------|
| `0`   | F32      | 4 |
| `1`   | F16      | 2 (default) |
| `2`   | I8       | 1 |
| `3`   | Binary   | `ceil(dim/8)` per vector |

The `precision` field describes the encoding stored in the **Parquet column**.
The centroid blob in the AILK section is always F32 (4 bytes per element)
regardless of this field.

### 3.2 `distance_metric` values

| Value | Metric      | Distance definition |
|-------|-------------|---------------------|
| `0`   | Cosine      | `1 - dot(a,b) / (\|a\| × \|b\|)` — range [0, 2] |
| `1`   | Euclidean   | `sqrt(Σ (aᵢ - bᵢ)²)` |
| `2`   | DotProduct  | `-dot(a, b)` — negated so lower = more similar |

All distance functions follow the convention **lower value = more similar**.

---

## 4. AILK Trailer (24 bytes)

Located at the end of every AILK section, immediately before the next AILK
section or the Parquet footer.

| Offset | Size | Type  | Field            | Description |
|--------|------|-------|------------------|-------------|
| 0      | 8    | u64   | `footer_offset`  | Absolute byte offset of **this** AILK header within the file |
| 8      | 8    | u64   | `footer_len`     | Total byte length of this AILK section (header + centroid + HNSW + trailer) |
| 16     | 2    | u16   | `format_version` | Must be `1` |
| 18     | 2    | u16   | `flags`          | Reserved; must be `0` |
| 20     | 4    | bytes | `magic`          | `AILK` |

---

## 5. Centroid Blob

Immediately follows the AILK header (at `centroid_offset` from section start;
always equals `HEADER_SIZE = 64` for format_version 1).

```
[ f32 × dim (little-endian) ] [ f32 radius (little-endian) ]
```

- `dim × 4` bytes: centroid vector. Arithmetic mean of all raw F32 vectors in
  the file, computed before quantization.
- `4` bytes: radius — maximum distance from any vector in the file to the
  centroid, computed with the same distance metric as the column.

Total: `dim × 4 + 4` bytes.

**Use**: geometric file pruning. A file is skipped when
`distance(query, centroid) - radius > pruning_threshold` without
downloading or opening the file beyond the manifest.

---

## 6. HNSW Index Blob

Starts at `hnsw_offset` relative to AILK section start
(= `HEADER_SIZE + centroid_len = 64 + dim × 4 + 4`).

The blob is a **bincode v1** serialization of an `hnsw_rs::Hnsw` graph.
Readers deserialize with:

```rust
let hnsw: hnsw_rs::Hnsw<f32, hnsw_rs::dist::DistCosine> =
    bincode::deserialize(&blob)?;
```

(Distance type varies with `distance_metric`; see §3.2.)

Key invariants of the serialized graph:

- Node IDs are `u64` values equal to the **0-based Parquet row index**
  within the same file. Result `row_id` can be used directly to fetch the
  corresponding Parquet row.
- The graph contains exactly `record_count` nodes (from the AILK header).
- Readers MUST verify `hnsw_graph.node_count() == header.record_count`.

---

## 7. Vector Column Encoding (Parquet)

The vector column is stored as `FIXED_LEN_BYTE_ARRAY` in Parquet.

```
byte_width = dim × precision.bytes_per_element()
```

Encoding per row:

| Precision | Encoding | Per-row bytes |
|-----------|----------|---------------|
| F16       | IEEE 754 half-precision, elements LE | `dim × 2` |
| F32       | IEEE 754 single-precision, elements LE | `dim × 4` |
| I8        | Symmetric scalar quantization, signed int8 | `dim × 1` |

The Parquet column carries **field-level KV metadata**:

| Key                     | Example    | Description |
|-------------------------|------------|-------------|
| `ailake.dim`            | `1536`     | Dimensionality |
| `ailake.precision`      | `f16`      | Encoding (`f32`, `f16`, `i8`) |
| `ailake.metric`         | `cosine`   | Distance metric |
| `ailake.vector_column`  | `embedding`| Column name hint |
| `ailake.record_count`   | `50000`    | Row count |
| `ailake.format_version` | `1`        | Format version |
| `ailake.footer_offset`  | `12582912` | Absolute byte offset of primary AILK section |
| `ailake.<col>.footer_offset` | `...` | Offset for secondary column `<col>` |

All KV values are UTF-8 decimal strings (no quoting, no JSON encoding).

---

## 8. Catalog Metadata (Iceberg)

AI-Lake tables are managed by an Iceberg Spec v2 catalog
(`metadata/current.json` + per-snapshot manifests).

### 8.1 Table-level properties (`metadata.json`)

Stored in the Iceberg `properties` map:

| Key                       | Example      | Description |
|---------------------------|--------------|-------------|
| `ailake.format-version`   | `1`          | AI-Lake format version |
| `ailake.vector-column`    | `embedding`  | Primary vector column name |
| `ailake.vector-dim`       | `1536`       | Vector dimensionality |
| `ailake.vector-metric`    | `cosine`     | Distance metric (`cosine`, `euclidean`, `dotproduct`) |
| `ailake.vector-precision` | `f16`        | Precision (`f32`, `f16`, `i8`) |

### 8.2 File-level manifest entry fields

Each `DataFileEntry` in the manifest carries per-file geometric statistics
used for pruning.  Current implementation uses JSON manifests
(`metadata/snap-<id>.json`); a future version will use Avro per Iceberg spec.

| Field                  | Type     | Description |
|------------------------|----------|-------------|
| `path`                 | string   | Relative path from warehouse root |
| `record_count`         | u64      | Number of rows |
| `file_size_bytes`      | u64      | Total file size in bytes |
| `centroid_b64`         | string?  | Base64-encoded F32 centroid (primary column) |
| `radius`               | f32?     | Maximum distance from centroid to any vector |
| `hnsw_offset`          | u64?     | Absolute byte offset of primary AILK section |
| `hnsw_len`             | u64?     | Byte length of primary AILK section |
| `vector_column`        | string?  | Primary vector column name |
| `vector_dim`           | u32?     | Vector dimensionality |
| `extra_vector_indexes` | array    | Secondary columns (same fields per entry) |

### 8.3 Manifest example

```json
{
  "snapshot_id": 1,
  "files": [
    {
      "path": "data/part-00000.parquet",
      "record_count": 50000,
      "file_size_bytes": 67108864,
      "centroid_b64": "AAAA...",
      "radius": 0.342,
      "hnsw_offset": 12582912,
      "hnsw_len": 4194304,
      "vector_column": "embedding",
      "vector_dim": 1536,
      "extra_vector_indexes": []
    }
  ]
}
```

---

## 9. Read Algorithm

### 9.1 Catalog scan + geometric pruning

```
1. Read metadata/current.json  →  current_snapshot_id
2. Read metadata/snap-<id>.json  →  list of DataFileEntry
3. For each DataFileEntry:
   a. Decode centroid_b64  →  F32 centroid vector
   b. d = distance(query, centroid, metric)
   c. if d - radius > pruning_threshold  →  skip file (no I/O)
4. Surviving files proceed to §9.2
```

### 9.2 Per-file HNSW search

```
For each surviving file (parallelizable):
  1. Load file bytes (full file, or ranged GET for S3)
  2. Parse Parquet footer  →  read ailake.footer_offset KV
  3. Parse 64-byte AILK header at that absolute offset
  4. Slice HNSW bytes: [ailk_start + hnsw_offset, +hnsw_len)
  5. bincode::deserialize  →  HnswIndex
  6. index.search(query, candidate_k, ef_search)
     where candidate_k = top_k × rerank_factor (or top_k if no reranking)
  7. Optional reranking:
     a. Decode Parquet vector column  →  Vec<Vec<f32>>
     b. For each candidate (row_id, approx_dist):
        exact_dist = distance(query, raw_vectors[row_id], metric)
     c. Re-sort by exact_dist
```

### 9.3 Global merge

```
Collect all per-file results.
Sort ascending by distance.
Truncate to top_k.
```

---

## 10. Integrity Invariants

Conforming implementations MUST verify:

1. `ailk_header.magic == b"AILK"` and `ailk_trailer.magic == b"AILK"`
2. `ailk_header.format_version == 1`
3. `parquet_record_count == ailk_header.record_count == hnsw_graph.node_count()`
4. `ailk_header.centroid_len == dim × 4 + 4`
5. HNSW `row_id` values are in `[0, record_count)`

Violation of invariant (3) indicates a partially-written or corrupted file.
Readers MUST return an error rather than silently returning wrong results.

---

## 11. Multi-Column Files

A file may embed more than one vector column (e.g., `embedding` and
`context_embedding`). Each column gets its own AILK section.

Layout:
```
[PAR1][row groups][AILK-primary][AILK-secondary…][Parquet footer][PAR1]
```

Parquet KV entries:
- Primary column (first): `ailake.footer_offset = <offset>`
- Each secondary column: `ailake.<column_name>.footer_offset = <offset>`

Readers looking for column `ctx` MUST check `ailake.ctx.footer_offset` first;
fall back to `ailake.footer_offset` only for single-column files.

---

## 12. Versioning and Compatibility

| `format_version` | Status  | Notes |
|------------------|---------|-------|
| `1`              | Current | This document |

**Forward compatibility**: readers MUST reject unknown `format_version` values
with an error. They MUST NOT attempt to parse an unknown section layout.

**Parquet compatibility**: every AI-Lake file is a valid Parquet file.
Standard readers (Spark, Trino, DuckDB, PyIceberg) read tabular columns
normally. The vector column appears as opaque `FIXED_LEN_BYTE_ARRAY`;
the AILK sections are invisible.

---

## 13. Constants Summary

| Constant                | Value |
|-------------------------|-------|
| `AILAKE_MAGIC`          | `AILK` = `0x41 0x49 0x4C 0x4B` |
| `AILAKE_FORMAT_VERSION` | `1` |
| `HEADER_SIZE`           | `64` bytes |
| `TRAILER_SIZE`          | `24` bytes |
| Centroid blob size      | `dim × 4 + 4` bytes |
| HNSW serializer         | `bincode` v1 + `hnsw_rs` v0.3 |

---

## 14. Reference Implementation

Canonical implementation: `ailake-file` Rust crate.

| Module                  | Role |
|-------------------------|------|
| `ailake_file::footer`   | `AilakeHeader`, `AilakeTrailer` encoding/decoding |
| `ailake_file::writer`   | `AilakeFileWriter` — produces conforming files |
| `ailake_file::reader`   | `AilakeFileReader` — reads and verifies files |
| `ailake_vec::distance`  | Distance functions (`cosine_distance`, `euclidean_distance`, `dot_product`, `exact_distance`) |
| `ailake_index`          | `HnswBuilder`, `HnswIndex`, `HnswSerializer` |
| `ailake_catalog`        | Iceberg catalog metadata |
| `ailake_query::scanner` | `search()`, `SearchConfig`, pruning + reranking |
