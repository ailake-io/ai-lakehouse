# COMPACTION.md — Compaction Job Design

## Why compaction is required

AI-Lake files are write-once immutable. Every `write_batch` produces a new `.parquet` file with its own HNSW graph in the footer. Over time a table accumulates many small files. Problems:

- **Fan-out cost**: a vector search opens one HNSW per surviving file. 500 files × 10 MB HNSW = 5 GB of footer reads per query.
- **S3 request cost**: each file requires at least two range GETs (Parquet footer + HNSW footer).
- **Parquet read overhead**: many small row groups = poor predicate pushdown.

Compaction merges N small files into one large file with a single unified index (HNSW or IVF-PQ — determined by the table's `VectorStoragePolicy`), eliminating fan-out.

---

## Compaction is the index rebuild trigger

During streaming ingest or batch micro-batches, new files arrive with per-batch indexes (HNSW or IVF-PQ). Compaction is the **only** moment a full-quality index is built for a merged set of records. This is by design:

- Write path stays O(N) — just append Parquet + compute centroid.
- HNSW build is O(N log N) and CPU-heavy — happens once, asynchronously, at compaction time.

Records in un-compacted files are still searchable (their indexes exist, just small), but recall may be lower for very small batches. The "blind window" is bounded by the compaction interval.

> **Tip**: `write_batch_auto_deferred` (Python / Rust SDK) builds the per-shard index in a background Tokio task immediately after Parquet commit. New shards are served via flat scan until `IndexStatus::Ready`, then switch to HNSW/IVF-PQ automatically. This narrows the blind window to the index build time (~30-165 s per shard) without waiting for the compaction job.

---

## Compaction triggers

Three conditions trigger compaction, evaluated by the `Compactor` Tokio task on a schedule (default: every 5 minutes):

### Trigger 1 — File count threshold
```
if snapshot.file_count(partition) > MAX_FILES_PER_PARTITION:
    compact(partition)
```
Default: `MAX_FILES_PER_PARTITION = 16`. After 16 files accumulate, merge them all into one.

### Trigger 2 — File size threshold
```
if any(file.size_bytes < MIN_FILE_SIZE for file in snapshot.files()):
    compact_small_files(partition)
```
Default: `MIN_FILE_SIZE = 64 MB`. Files smaller than 64 MB are merged.

### Trigger 3 — Time-based
```
if time_since_last_compaction(partition) > MAX_COMPACTION_INTERVAL:
    compact(partition)
```
Default: `MAX_COMPACTION_INTERVAL = 1 hour`. Ensures the HNSW is refreshed regularly even if file count stays low.

### Manual trigger
```bash
# CLI
ailake compact s3://my-bucket/warehouse/db/my_table/

# Python SDK
import ailake
ailake.compact("s3://my-bucket/warehouse/db/my_table/", partition_filter="year=2024")

# Rust SDK
compactor.compact_now(&table_uri, &partition_filter).await?;
```

---

## Compaction algorithm

```
Input: List<DataFileEntry> (files to compact, from one partition)

1. PLAN
   Sort input files by record_count descending.
   Target output file size: MAX_OUTPUT_FILE_BYTES (default: 512 MB of Parquet data).
   Bin-pack files into groups each fitting the target size.

2. FOR EACH GROUP (in parallel, bounded by MAX_CONCURRENT_COMPACTIONS=4):

   a. COLLECT
      For each file in group:
        - GET Parquet section via partial read → RecordBatch
        - GET HNSW footer (optional — not strictly needed for rebuild)
      Concatenate all RecordBatches.

   b. FILTER DELETES
      Load Position Delete Files for this snapshot.
      Remove rows referenced by position delete files.
      Reindex surviving rows: new RowId(0..N) where N = surviving row count.

   c. COMPUTE CENTROID
      centroid, radius = compute_centroid_and_radius(&all_vectors, metric)

   d. QUANTIZE
      Apply F16 quantization (or configured precision) to all vectors.

   e. BUILD HNSW
      Run HnswBuilder.build(all_vectors_f32) — CPU-bound, in spawn_blocking.
      This is O(N log N) where N = total rows in the group.
      Default HNSW parameters: M=16, ef_construction=200.

   f. WRITE
      AilakeFileWriter.write(record_batch, hnsw, centroid, radius)
      → produces one new .parquet file with HNSW in footer.

   g. COMMIT
      SnapshotManager.replace_files(old_files, new_file, custom_properties)
      → atomic Iceberg snapshot: removes old DataFile entries, adds new one.
      → old files enter retention period (default: 7 days before vacuum).

3. UPDATE METRICS
   Log: files_in, files_out, rows_in, rows_out (deleted rows), hnsw_build_time_ms
```

### Why rebuild HNSW from scratch instead of merging

`hnsw_rs` does not expose a merge primitive. Merging two HNSW graphs while maintaining quality invariants is a non-trivial research problem. Rebuilding from scratch:

- Is correct by definition.
- Produces optimal recall at the configured `M` and `ef_construction`.
- Is bounded in cost: compaction runs on a batch, not on the entire table. The batch size is bounded by `MAX_OUTPUT_FILE_BYTES`.
- Is parallelizable: groups compact independently.

For a 512 MB Parquet file containing ~200k rows of dim=1536 F16 vectors, HNSW rebuild takes approximately 45–90 seconds on a 4-core worker. This is acceptable for a background job.

---

## Compaction and snapshot isolation

Iceberg's snapshot model provides full isolation:

```
Timeline:

T0: Snapshot S1 → files: [A, B, C, D]   (readers use S1)
T1: Compactor starts, reads A+B+C+D
T2: New write arrives → Snapshot S2 → files: [A, B, C, D, E]
T3: Compactor finishes → Snapshot S3 → files: [ABCD_merged, E]
    (A, B, C, D marked for deletion after retention period)

Readers at T2 see S2 (A+B+C+D+E), not affected by compaction.
Readers after T3 see S3 (merged+E), better performance.
```

A reader that started at S1 and is still running at T3 continues to use S1's files (A, B, C, D) until its query completes. They remain in S3-compatible storage until vacuum runs after the retention period.

### Conflict detection

If two compaction jobs race on the same partition (should not happen in normal operation but must be safe):

```
Job 1 plans: compact [A, B, C]
Job 2 plans: compact [A, B, D]

Job 1 commits first → creates snapshot with merged_ABC
Job 2 tries to commit → detects that A and B are no longer in the latest snapshot
                     → RETRY: re-plan using the current snapshot
```

This is the standard Iceberg optimistic concurrency pattern. `ailake-catalog` wraps the Iceberg commit and retries on conflict up to `MAX_COMMIT_RETRIES=5`.

---

## Compaction modes

### Mode 1 — `full` (default)
Rebuilds HNSW from scratch for every output file. Highest quality, highest CPU cost.

### Mode 2 — `index_only`
Rewrites HNSW for existing Parquet files without changing the Parquet data. Used when:
- The Parquet data is already well-compacted (large files, no small file problem).
- The HNSW was never built (streaming ingest without compaction).
- HNSW parameters changed (e.g. M or ef_construction were updated).

```bash
ailake compact --mode index_only s3://my-bucket/warehouse/db/my_table/
```

Produces new `.parquet` files identical in data content to the originals, with HNSW appended. No rows are moved or merged. Fast: skips the HNSW rebuild and only reads/writes the HNSW footer.

### Mode 3 — `data_only`
Runs standard Iceberg `rewrite_data_files` compaction (merges small Parquet files) but does NOT rebuild HNSW. The resulting merged Parquet file has no AI-Lake footer. Run `index_only` compaction afterwards to add HNSW.

Useful when: existing Spark/Trino `OPTIMIZE` job already handles Parquet compaction, and AI-Lake only needs to add the HNSW on top.

---

## Compaction in streaming pipelines

### Beam / Dataflow pattern

```python
# Dataflow pipeline with periodic compaction trigger
import apache_beam as beam
from apache_beam.transforms.trigger import AfterProcessingTime, Repeatedly

# Streaming write (data lands in small files)
records \
    | beam.WindowInto(
        beam.window.GlobalWindows(),
        trigger=Repeatedly(AfterProcessingTime(5 * 60)),  # every 5 min
        accumulation_mode=beam.trigger.AccumulationMode.DISCARDING
    ) \
    | Managed.write(Managed.ICEBERG, config={...})

# Compaction runs as a separate scheduled Dataflow job:
# ailake compact s3://... triggered every 30 minutes via Cloud Scheduler
```

### Spark Structured Streaming pattern

```python
# After each micro-batch, check if compaction is needed
def process_batch(df, batch_id):
    df.write.format("iceberg").mode("append").save("catalog.db.table")
    if batch_id % 20 == 0:  # every 20 micro-batches
        compact_async("catalog.db.table")  # non-blocking

query = df_stream \
    .writeStream \
    .foreachBatch(process_batch) \
    .start()
```

---

## Vacuum — cleaning up old files after compaction

Files superseded by compaction are not deleted immediately — they remain in object storage for the retention period, allowing time-travel queries and in-flight readers to complete.

Vacuum removes files that are:
1. Not referenced by any snapshot within the retention window.
2. Older than `retention_period` (default: 7 days).

```bash
# CLI
ailake vacuum s3://my-bucket/warehouse/db/my_table/ --older-than 7d

# Python
ailake.vacuum("s3://my-bucket/warehouse/db/my_table/", retention_days=7)
```

Vacuum is separate from compaction. It should be run after compaction on the same schedule (weekly is typical).

---

## Configuration reference

Set in `metadata.json` `properties` (applied to all future compaction runs):

| Property | Default | Description |
|---|---|---|
| `ailake.compaction.max-files-per-partition` | `16` | Trigger threshold (file count) |
| `ailake.compaction.min-file-size-bytes` | `67108864` (64 MB) | Trigger threshold (small file) |
| `ailake.compaction.max-output-file-bytes` | `536870912` (512 MB) | Target output file size |
| `ailake.compaction.max-concurrent-jobs` | `4` | Parallel compaction groups |
| `ailake.compaction.hnsw-m` | `16` | HNSW M parameter for rebuilt indexes |
| `ailake.compaction.hnsw-ef-construction` | `200` | HNSW ef_construction for rebuilt indexes |
| `ailake.compaction.mode` | `full` | `full`, `index_only`, or `data_only` |
| `ailake.vacuum.retention-days` | `7` | Days to retain superseded files |

---

## Rust implementation structure

```
ailake-query/src/
└── compaction.rs
```

```rust
pub struct CompactionConfig {
    /// Minimum number of files to trigger compaction (default: 4)
    pub min_files_to_compact: usize,
    /// Files below this size are candidates for compaction (default: 128 MB)
    pub target_file_size_bytes: u64,
}

pub struct CompactionPlanner {
    pub config: CompactionConfig,
}

impl CompactionPlanner {
    /// Returns files smaller than target_file_size_bytes if their count
    /// meets min_files_to_compact. Returns empty vec otherwise.
    pub fn plan(&self, files: &[DataFileEntry]) -> Vec<DataFileEntry>;
}

pub struct CompactionExecutor {
    store: Arc<dyn Store>,
    policy: VectorStoragePolicy,
}

impl CompactionExecutor {
    /// Read N files, concat RecordBatches, rebuild HNSW, write one unified file.
    pub async fn compact(
        &self,
        files: &[DataFileEntry],
        output_path: &str,
    ) -> AilakeResult<DataFileEntry>;

    /// Full cycle: plan → compact → catalog.commit_snapshot → delete old files.
    /// Returns None if planner finds nothing to compact.
    pub async fn run(
        &self,
        planner: &CompactionPlanner,
        table: &TableIdent,
        catalog: Arc<dyn CatalogProvider>,
        output_prefix: &str,
    ) -> AilakeResult<Option<DataFileEntry>>;
}
```
