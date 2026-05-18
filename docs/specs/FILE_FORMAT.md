# FILE_FORMAT.md — Unified File Binary Specification

Version: 1  
File extension: `.parquet` (no new extension — the file is a valid Parquet file with an AI-Lake footer appended)  
AI-Lake magic: `0x41 0x49 0x4C 0x4B` ("AILK")

---

## Overview

An AI-Lake file is a single self-contained physical file composed of two sections:

1. **Parquet section** — standard Apache Parquet file, including header (`PAR1`), row groups with columnar data (including the vector column as `FIXED_LEN_BYTE_ARRAY`), and footer (schema + metadata + final `PAR1`).
2. **AI-Lake footer extension** — appended after the Parquet section. Contains the HNSW graph, centroid, radius, and supported distance metrics.

Standard Parquet readers (PyIceberg, Spark, Trino, DuckDB) stop reading at the final `PAR1` marker per the Parquet specification. They never see the AI-Lake footer. This is the compatibility guarantee.

---

## File layout

```
┌─────────────────────────────────────────────────────────────────┐
│ PARQUET HEADER (4 bytes: "PAR1")                                │
├─────────────────────────────────────────────────────────────────┤
│ ROW GROUPS                                                      │
│   - column chunks: id, text, metadata, embedding, ...           │
│   - embedding column: FIXED_LEN_BYTE_ARRAY(dim * bytes_per_el)  │
├─────────────────────────────────────────────────────────────────┤
│ PARQUET FOOTER                                                  │
│   - schema (with ailake.* field metadata)                       │
│   - row group statistics                                        │
│   - key_value_metadata:                                         │
│       ailake.format_version = "1"                               │
│       ailake.hnsw_offset = "12582912"                           │
│       ailake.hnsw_len = "4194304"                               │
│       ailake.precision = "f16"                                  │
│       ailake.metric = "cosine"                                  │
│   - footer length (4 bytes)                                     │
│   - PARQUET MAGIC END (4 bytes: "PAR1")  ← standard readers stop here
├─────────────────────────────────────────────────────────────────┤
│ ▼▼▼ AI-LAKE FOOTER EXTENSION (invisible to standard readers)  ▼▼▼
│                                                                 │
│ AI-LAKE HEADER (64 bytes)                                       │
│   - magic ("AILK") | version | flags                            │
│   - dim, metric, precision, record_count                        │
│   - offsets to subsections                                      │
├─────────────────────────────────────────────────────────────────┤
│ CENTROID SECTION                                                │
│   - centroid: [f32; dim] (raw bytes, native endian)             │
│   - radius: f32                                                  │
├─────────────────────────────────────────────────────────────────┤
│ HNSW GRAPH SECTION                                              │
│   - bincode-serialized HNSW graph                               │
│   - includes: hierarchical layers, connections, vectors         │
│   - vectors stored at HNSW level use the same precision         │
│     as the Parquet column (F16 default)                         │
├─────────────────────────────────────────────────────────────────┤
│ AI-LAKE FOOTER TRAILER (24 bytes)                               │
│   - footer_offset: u64 (where AI-Lake header starts)            │
│   - footer_len: u64 (total length of AI-Lake extension)         │
│   - format_version: u16                                         │
│   - flags: u16                                                  │
│   - magic ("AILK"): [u8; 4]                                     │
└─────────────────────────────────────────────────────────────────┘
```

The trailer is always exactly the last 24 bytes of the file. A reader checks for `"AILK"` at the last 4 bytes to detect whether the file has an AI-Lake extension.

---

## AI-Lake header (64 bytes, little-endian)

| Offset | Size | Type | Field | Description |
|---|---|---|---|---|
| 0 | 4 | `[u8; 4]` | `magic` | `0x41 0x49 0x4C 0x4B` ("AILK") |
| 4 | 2 | `u16` | `format_version` | Must be `1` |
| 6 | 2 | `u16` | `flags` | Bit 0: PQ enabled. Bit 1: multi-column. Rest reserved |
| 8 | 4 | `u32` | `dim` | Vector dimensionality (e.g. 1536) |
| 12 | 1 | `u8` | `precision` | `0`=F32, `1`=F16, `2`=I8, `3`=Binary |
| 13 | 1 | `u8` | `distance_metric` | `0`=Cosine, `1`=Euclidean, `2`=DotProduct |
| 14 | 2 | `u16` | `_reserved` | Must be `0` |
| 16 | 8 | `u64` | `record_count` | Total vectors indexed |
| 24 | 8 | `u64` | `centroid_offset` | Byte offset of centroid section from AI-Lake header start |
| 32 | 8 | `u64` | `centroid_len` | Byte length of centroid section |
| 40 | 8 | `u64` | `hnsw_offset` | Byte offset of HNSW graph section from AI-Lake header start |
| 48 | 8 | `u64` | `hnsw_len` | Byte length of HNSW graph section |
| 56 | 8 | `[u8; 8]` | `_reserved` | Must be zero |

Total: 64 bytes.

---

## Centroid section

```
┌─────────────────────────────────┐
│ centroid: [f32; dim]            │   ← dim × 4 bytes, native little-endian
├─────────────────────────────────┤
│ radius: f32                     │   ← 4 bytes
└─────────────────────────────────┘
```

Total length: `dim * 4 + 4` bytes.

The centroid is always stored as F32 (not quantized), regardless of the vector column's storage precision. This is intentional — centroid precision matters for pruning accuracy, and the centroid is a single vector per file.

The radius is the maximum distance from any indexed vector to the centroid, using the file's `distance_metric`.

---

## HNSW graph section

The HNSW graph is serialized via `bincode` from the `hnsw_rs::Hnsw` type. The exact binary layout is owned by `hnsw_rs` and is not specified here — it is treated as an opaque byte blob from the file format's perspective.

The graph contains:
- All vectors that were inserted (at the precision configured for the file)
- Hierarchical layers with neighbor lists
- Entry point reference
- Distance metric configuration

**Key invariant**: every node in the HNSW graph has a `RowId` key that corresponds to its row position in the Parquet section. After loading the graph, `hnsw_graph.node_count() == parquet_record_count` must hold.

---

## AI-Lake footer trailer (24 bytes, last 24 bytes of file)

| Offset (from end) | Size | Type | Field | Description |
|---|---|---|---|---|
| -24 | 8 | `u64` | `footer_offset` | Absolute byte offset of AI-Lake header in file |
| -16 | 8 | `u64` | `footer_len` | Total length of AI-Lake extension (header + sections + trailer) |
| -8 | 2 | `u16` | `format_version` | Same as header version |
| -6 | 2 | `u16` | `flags` | Same as header flags |
| -4 | 4 | `[u8; 4]` | `magic` | `0x41 0x49 0x4C 0x4B` ("AILK") |

Total: 24 bytes.

The trailer is the bootstrap for AI-Lake reads. A reader does:

```
1. seek(SeekFrom::End(-24))
2. read 24 bytes → trailer
3. if trailer.magic == "AILK":
     seek(SeekFrom::Start(trailer.footer_offset))
     read trailer.footer_len bytes → full AI-Lake extension
   else:
     file is a standard Parquet file with no AI-Lake extension
```

---

## Parquet field metadata for the vector column

The vector column's schema element carries field-level `key_value_metadata`:

```
field name: embedding
physical type: FIXED_LEN_BYTE_ARRAY(3072)  # for dim=1536 in F16
key_value_metadata:
  - ailake.dim = "1536"
  - ailake.metric = "cosine"
  - ailake.precision = "f16"
```

Standard Parquet readers expose these as opaque string-keyed metadata. Readers that don't know them ignore them.

---

## Parquet file-level metadata

The Parquet footer's file-level `key_value_metadata`:

| Key | Value example | Purpose |
|---|---|---|
| `ailake.format_version` | `"1"` | AI-Lake format version |
| `ailake.hnsw_offset` | `"12582912"` | Absolute byte offset of AI-Lake header (= trailer.footer_offset) |
| `ailake.hnsw_len` | `"4194304"` | Length of AI-Lake extension |
| `ailake.precision` | `"f16"` | Vector precision in the file |
| `ailake.metric` | `"cosine"` | Distance metric |
| `ailake.record_count` | `"50000"` | Vectors indexed (for sanity checks) |

A reader can extract `ailake.hnsw_offset` from the Parquet footer to skip the trailer lookup. Both paths (trailer-based and Parquet-metadata-based) point to the same location.

---

## Partial-read strategy from S3

The unified file layout enables efficient partial reads:

```
1. HEAD object → file_size
2. GET range [file_size - 8192, file_size)  → fetches Parquet footer + AI-Lake trailer
3. Parse Parquet footer:
     - read ailake.hnsw_offset, ailake.hnsw_len from key_value_metadata
4. Read centroid (cheap — small):
     GET range [ailake.hnsw_offset, ailake.hnsw_offset + 64 + centroid_section_len)
5. Compute distance(query, centroid). If pruned, stop here.
6. If not pruned:
     GET range [ailake.hnsw_offset, ailake.hnsw_offset + ailake.hnsw_len)
     → full AI-Lake extension
7. Load HNSW via memmap2 from temp file, search top-k.
8. For top-k RowIds, fetch the relevant Parquet row group:
     standard Parquet projection + predicate pushdown
```

Total cost for a pruned file: ~16 KB downloaded (one footer fetch + one centroid fetch). For 10,000 files pruned, that's 160 MB total — well within network budget for a single query.

For a non-pruned file with HNSW size ~10 MB: one extra range GET, then mmap-backed graph traversal.

---

## Naming convention

```
data/part-{NNNNN}.parquet
```

Standard Iceberg Parquet naming. The `.parquet` extension is preserved because the file IS a valid Parquet file. Tools that filter by extension will see and accept these files.

---

## Integrity checks

On open, a reader MUST verify:
1. File ends with `"PAR1"` (Parquet magic). If not, it's not a valid Parquet file.
2. If the last 4 bytes are `"AILK"`, the file has an AI-Lake extension. Parse the trailer.
3. AI-Lake header magic at `trailer.footer_offset` must be `"AILK"`.
4. `header.format_version == 1`
5. `header.record_count == parquet_record_count` (positional invariant check)
6. `header.dim` matches the Parquet field metadata `ailake.dim`

After loading the HNSW:
7. `hnsw_graph.node_count() == header.record_count`

---

## Rust types

```rust
// ailake-file/src/footer.rs

pub const AILAKE_MAGIC: [u8; 4] = *b"AILK";
pub const AILAKE_FORMAT_VERSION: u16 = 1;
pub const TRAILER_SIZE: usize = 24;
pub const HEADER_SIZE: usize = 64;

#[derive(Debug, Clone)]
pub struct AilakeHeader {
    pub format_version: u16,
    pub flags: u16,
    pub dim: u32,
    pub precision: Precision,
    pub distance_metric: DistanceMetric,
    pub record_count: u64,
    pub centroid_offset: u64,
    pub centroid_len: u64,
    pub hnsw_offset: u64,
    pub hnsw_len: u64,
}

#[derive(Debug, Clone)]
pub struct AilakeTrailer {
    pub footer_offset: u64,
    pub footer_len: u64,
    pub format_version: u16,
    pub flags: u16,
}

#[repr(u8)]
#[derive(Debug, Clone, Copy)]
pub enum Precision { F32 = 0, F16 = 1, I8 = 2, Binary = 3 }

#[repr(u8)]
#[derive(Debug, Clone, Copy)]
pub enum DistanceMetric { Cosine = 0, Euclidean = 1, DotProduct = 2 }
```

---

## Multi-column vectors (`embedding` + `context_embedding`)

When a table has multiple vector columns (e.g. `LlmContextSchema` with both `embedding` and `context_embedding`), the file format extends as follows:

- The Parquet section has two `FIXED_LEN_BYTE_ARRAY` columns, each with its own field metadata.
- The AI-Lake header sets `flags` bit 1 (multi-column).
- The centroid section contains two centroids (one per column) and two radii.
- The HNSW graph section contains two serialized HNSW graphs (one per column), back-to-back.
- Header offsets/lengths refer to the combined sections; sub-offsets within are computed from the column-name → index mapping in an extended header (Phase 3).

For Phase 1 and Phase 2, only single-column files are supported.
