# DATA_FLOW.md — End-to-End Data Flows

## Write path

```
Caller
  │
  │  write_batch(record_batch: RecordBatch, embeddings: &[f32])
  ▼
ailake-query / TableWriter
  │
  ▼
ailake-file / AilakeFileWriter
  │
  ├─► ailake-vec
  │     │  1. quantize f32 → f16  (VectorPrecision::F16)
  │     │  2. compute centroid (mean) and radius (max distance to centroid)
  │     └─► returns: (centroid: Vec<f32>, radius: f32, quantized: Vec<u8>)
  │
  ├─► ailake-parquet / ParquetVectorWriter
  │     │  3. encode RecordBatch with embedding column as
  │     │     FIXED_LEN_BYTE_ARRAY(dim * 2), F16 bytes
  │     │     field metadata: {"ailake.dim": "1536", "ailake.metric": "cosine",
  │     │                       "ailake.precision": "f16"}
  │     │  4. write Parquet to part-NNNNN.parquet (Parquet section only)
  │     │  5. record offset where Parquet ended (just after final PAR1)
  │     └─► returns: parquet_end_offset
  │
  ├─► ailake-index / HnswBuilder
  │     │  6. build HNSW: insert each (RowId(N), vector_f32) pair
  │     │     (vectors expanded F16 → F32 for hnsw_rs)
  │     │  7. serialize HNSW via bincode
  │     └─► returns: hnsw_bytes: Vec<u8>
  │
  └─► ailake-file (back in AilakeFileWriter)
        │  8. append AI-Lake header (64 bytes) at parquet_end_offset
        │  9. append centroid section: centroid bytes + radius
        │ 10. append HNSW graph section: hnsw_bytes
        │ 11. append AI-Lake trailer (24 bytes) at end of file
        │ 12. rewrite Parquet footer with updated key_value_metadata:
        │       ailake.hnsw_offset = parquet_end_offset
        │       ailake.hnsw_len = (file_size - parquet_end_offset)
        │     (this is a footer rewrite, not a file rewrite)
        └─► returns: AilakeFileMeta { path, record_count, centroid, radius,
                                       hnsw_offset, hnsw_len }

Back in TableWriter:
  │
  └─► ailake-catalog / SnapshotManager.commit()
        │ 13. create new Iceberg DataFile entry with custom_properties:
        │       ailake.centroid = base64(centroid)
        │       ailake.radius = "0.342"
        │       ailake.hnsw_offset = ...
        │       ailake.hnsw_len = ...
        │ 14. append DataFile to snap-NNN.avro
        │ 15. atomically update metadata/v{N+1}.metadata.json
        └─► returns: SnapshotId

Invariant after commit:
  row N in part-NNNNN.parquet
    == HNSW node with key RowId(N) in the AI-Lake footer of the same file
```

**Atomicity**: steps 13–15 are a single logical transaction managed by Iceberg. If the process dies before step 15, the `.parquet` file exists but is not referenced by the manifest — it's an orphan, cleaned by a future vacuum. The Iceberg snapshot commit (step 15) is the only commit point.

**Important**: step 12 (Parquet footer rewrite) is necessary because the Parquet footer offsets must be written before we know the HNSW size. Sequence:
- Write Parquet with placeholder footer metadata
- Append AI-Lake extension
- Seek back, rewrite Parquet footer with correct `ailake.hnsw_offset/len`
- The rewrite is bounded — only the last few KB of the file change

---

## Read path — vector search

```
Caller
  │
  │  search(table_uri, query: &[f32], top_k: usize, filter: Option<&str>)
  ▼
ailake-query / VectorScanner
  │
  ├─► ailake-catalog / SnapshotManager
  │     │  1. read metadata.json → find current snapshot
  │     │  2. read snap-NNN.avro → load all DataFile entries
  │     │  3. for each entry, decode custom_properties:
  │     │       centroid = base64_decode(ailake.centroid) → [f32; dim]
  │     │       radius = parse_f32(ailake.radius)
  │     │       hnsw_offset, hnsw_len
  │     └─► returns: Vec<FileCandidate>
  │
  ├─► ailake-query / VectorPruner
  │     │  4. for each candidate:
  │     │       dist = cosine_distance(query, candidate.centroid)
  │     │       if dist - candidate.radius > search_threshold → PRUNE
  │     └─► returns: Vec<FileCandidate>  (only survivors)
  │
  ├─► [for each surviving candidate, in parallel via Tokio]
  │   │
  │   ├─► ailake-store / Store
  │   │     │  5a. GET range [hnsw_offset, hnsw_offset + hnsw_len)
  │   │     │      → fetches only the AI-Lake footer extension
  │   │     │  5b. write bytes to temp file
  │   │     └─► returns: tmp_file_path
  │   │
  │   ├─► ailake-index / MmapLoader
  │   │     │  6a. open tmp_file with memmap2::Mmap
  │   │     │  6b. parse AI-Lake header (first 64 bytes)
  │   │     │  6c. deserialize HNSW from graph section via bincode
  │   │     │       (mmap-backed — only touched pages are loaded)
  │   │     └─► returns: HnswIndex
  │   │
  │   └─► ailake-index / HnswIndex.search()
  │         │  6d. run HNSW search: query → top_k × oversampling candidates
  │         └─► returns: Vec<(RowId, f32)>
  │
  ├─► merge results across all surviving files, global top-k sort
  │
  ├─► ailake-store / Store
  │     │  7. for each winning RowId, identify which Parquet file owns it
  │     │     (each RowId is scoped to its file)
  │     └─► returns: Vec<(file_path, RowId, distance)>
  │
  ├─► ailake-parquet / ParquetVectorReader
  │     │  8. for each (file_path, RowId): read the specific row group
  │     │     containing that row, with predicate pushdown for `filter`
  │     │     (Parquet row group statistics enable this skip)
  │     └─► returns: RecordBatch with full row data
  │
  └─► return RecordBatch (columns: all table columns + _distance: f32)
```

**Performance note**: step 4 (centroid pruning) is the critical path for petabyte scale. The centroid array for 10,000 files × 1536 dims × 4 bytes = ~60 MB, fits comfortably in memory. Zero file I/O for pruned files.

**Network cost analysis** (per file):
- Pruned file: 0 bytes from S3 (centroid read from Avro manifest, no Parquet/footer access)
- Surviving file (no match): ~10-15 MB (HNSW footer fetch)
- Surviving file (with match): ~10-15 MB + ~1 MB row group fetch = ~16 MB

---

## Read path — standard Iceberg (no AI-Lake plugin)

```
Spark / Trino / DuckDB / PyIceberg
  │
  │  1. read metadata/v{N}.metadata.json
  │     → sees ailake.* properties, does not understand them, ignores
  │
  │  2. read snap-NNN.avro  (standard Iceberg manifest)
  │     → lists DataFile entries: data/part-NNNNN.parquet
  │     → custom_properties contains ailake.* keys
  │     → framework exposes them as opaque string map or ignores
  │
  │  3. read data/part-NNNNN.parquet
  │     → Parquet reader sees the final PAR1 magic and stops
  │     → never touches the AI-Lake footer (after the final PAR1)
  │     → schema has column "embedding" as FIXED_LEN_BYTE_ARRAY
  │     → framework reads it as raw bytes or skips if not projected
  │
  └─► returns all non-vector columns normally; "embedding" column = bytes
```

No errors. No surprises. The AI-Lake footer is invisible — Parquet specification mandates that readers stop at the final `PAR1` marker.

---

## Compaction flow

Triggered when the snapshot accumulates many small files (default threshold: 16 files smaller than 64 MB).

```
Compactor (background Tokio task)
  │
  │  1. identify small files to compact (read DataFile sizes from manifest)
  │
  ├─► ailake-file / AilakeFileReader
  │     │  2. for each input file: read full Parquet content + read full HNSW
  │     └─► returns: Vec<RecordBatch> + Vec<HnswIndex>
  │
  ├─► ailake-file / AilakeFileWriter
  │     │  3. concatenate RecordBatches
  │     │  4. recompute centroid + radius for the merged batch
  │     │  5. rebuild HNSW from scratch with all vectors
  │     │       (HNSW merge is non-trivial; we rebuild for simplicity in Phase 2.
  │     │        Phase 4 may implement true incremental HNSW merge.)
  │     │  6. write new unified file part-MERGED-NNN.parquet
  │     └─► returns: AilakeFileMeta
  │
  └─► ailake-catalog
        │  7. create new Iceberg snapshot:
        │       - REPLACE the N input files with the 1 output file
        │  8. old files become unreferenced — vacuum will remove them after retention period
        └─► returns: SnapshotId

Concurrent reads during compaction:
  - Readers use the snapshot at the time of their query start (Iceberg snapshot isolation)
  - New merged file is not visible until the catalog commit
  - Old files remain readable until vacuum (typically 7 days retention)
```

**Why rebuild HNSW instead of merging?** hnsw_rs does not provide a primitive for merging two indexes that preserves graph quality. Rebuilding from scratch is O(N log N) but only runs at compaction time (not on the hot write path). The merge produces a single high-quality HNSW with better search recall than the union of two smaller indexes.

---

## Centroid computation (on write)

Called in step 1 of the write path. Must be fast — runs synchronously before writing.

```rust
// ailake-vec/src/distance.rs
pub fn compute_centroid_and_radius(
    vectors: &[Vec<f32>],
    metric: VectorMetric,
) -> (Vec<f32>, f32) {
    let dim = vectors[0].len();

    // mean of all vectors
    let mut centroid = vec![0.0_f32; dim];
    for v in vectors {
        for (i, &x) in v.iter().enumerate() {
            centroid[i] += x;
        }
    }
    let n = vectors.len() as f32;
    for x in &mut centroid {
        *x /= n;
    }

    // normalize for cosine metric
    if matches!(metric, VectorMetric::Cosine) {
        let norm: f32 = centroid.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 0.0 {
            for x in &mut centroid {
                *x /= norm;
            }
        }
    }

    // radius = max distance from any vector to centroid
    let radius = vectors
        .iter()
        .map(|v| metric.distance(v, &centroid))
        .fold(0.0_f32, f32::max);

    (centroid, radius)
}
```

The centroid is base64-encoded into the Avro manifest. For dim=1536, this is ~8 KB per file entry — acceptable for tens of thousands of files.

---

## Context assembly flow (RAG use case)

```
Caller
  │
  │  assembler.assemble_chunks(chunks: Vec<Chunk>) -> AssembledContext
  ▼
ailake-query / ContextAssembler
  │
  │  1. deduplication
  │       sort by distance (ascending — most relevant first)
  │       for each pair (a, b): if cosine_distance(a.embedding, b.embedding) < dedup_threshold
  │         keep the chunk that appeared first (already sorted by relevance)
  │
  │  2. group by document_id, sort each group by chunk_index ascending
  │       cap each group at max_chunks_per_document (default: 10)
  │
  │  3. token budget (char budget = max_tokens × 4)
  │       greedily include chunks until char_budget would be exceeded
  │
  │  4. render XML
  │
  │       <context>
  │         <document id="{doc_id}" title="{title}" source="{uri}">
  │           <chunk index="{n}" section="{section}">
  │             <text>{chunk_text}</text>
  │           </chunk>
  │           ...
  │         </document>
  │         ...
  │       </context>
  │
  └─► return AssembledContext { text: String, token_estimate: usize, chunk_count: usize }
```
