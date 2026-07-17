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

### Failed index recovery

When a deferred background index build fails (e.g. k-means divergence, OOM), `patch_index_failed()` transitions the file's catalog entry to `IndexStatus::Failed` with an `index_error` reason string. Files in `Failed` state:

1. **Continue serving reads** via flat scan (exact O(N) brute-force) — no data loss, no downtime.
2. **Compete for a slot like any other file** — `CompactionPlanner::plan()` gives priority scheduling only to **foreign** files (missing `centroid_b64` — never written by the AI-Lake SDK, see below). A `Failed` file already has a centroid (computed before the HNSW build that failed), so it is not foreign, and is only picked up once it's part of the normal size-eligible candidate pool — it is not fast-tracked ahead of other files.
3. **Get their index rebuilt** whenever next included in a compaction pass — the executor reads their Parquet row data and rebuilds HNSW/IVF-PQ from scratch (or extends a dominant file's graph, see below), transitioning them back to `Ready`.

This makes compaction the **self-healing** path for transient build failures (transient GPU OOM, k-means seed instability). No operator intervention required — the file is repaired automatically the next time it's swept into a compaction pass.

---

## Compaction triggers

There is no built-in scheduler in `ailake-query` — no background Tokio task polls file counts, sizes, or elapsed time. Compaction only runs when explicitly invoked: CLI (`ailake compact`), the Python binding (`ailake.compact()`), the Rust API (`CompactionExecutor::run()`), or a caller-provided external scheduler (cron, an Airflow `AilakeCompactOperator` DAG, a periodic Spark/Beam job — see below). Once invoked, `CompactionPlanner::plan()` decides which files are eligible for that pass using `CompactionConfig`:

- `min_files_to_compact` (default: `4`) — skip the pass if fewer than this many size-eligible files exist (foreign files bypass this floor — see "Failed index recovery" above).
- `target_file_size_bytes` (default: `128 MB` in the Rust struct default; `ailake compact`/`ailake.compact()` default to `512 MB` instead) — files below this size are candidates.
- `max_files_per_pass` (default: `20`) — candidates are sorted smallest-first and capped here, bounding peak RAM and HNSW-rebuild cost per pass.

### Manual trigger
```bash
# CLI
ailake compact s3://my-bucket/warehouse/db/my_table/

# Python SDK
import ailake
ailake.compact("s3://my-bucket/warehouse/db/my_table/", min_files=4, target_size_bytes=536_870_912)

# Rust SDK
let planner = CompactionPlanner::new(CompactionConfig::default());
executor.run(&planner, &table, catalog, "data").await?;
```

---

## Compaction algorithm

One `compact()`/`compact_incremental()` call always merges its whole input into **exactly one** output file — there is no bin-packing into multiple parallel output groups. `CompactionExecutor::run()` fetches the table's full file list, hands it to `CompactionPlanner::plan()` (see "Compaction triggers" above), and merges the resulting single batch of eligible files (at most `max_files_per_pass`) into one new `.parquet` file per invocation.

```
Input: List<DataFileEntry> (the files selected by CompactionPlanner::plan(), one pass)

1. COLLECT
   Read all selected files in parallel (futures::future::try_join_all) via
   AilakeFileReader::read_parquet — decodes rows + vector column directly
   from Parquet, independent of whether the file's AILK footer/index exists.
   Concatenate all RecordBatches.

2. FILTER DELETES
   Deletion-vector-masked rows are dropped before the merge (row positions are
   about to change, so old bitmaps can't be carried forward).

3. COMPUTE CENTROID
   centroid, radius = compute_centroid_and_radius(&all_vectors, metric)

4. QUANTIZE
   Apply F16 quantization (or configured precision) to all vectors.

5. BUILD INDEX — two paths:
   - compact() (explicit full rebuild): always builds a brand-new HNSW/IVF-PQ
     index from every input vector. Algorithm choice is CompactionExecutor's
     index_strategy: Auto (detects hardware — IVF-PQ on GPU/many-core, HNSW
     otherwise; the default), ForceHnsw, or ForceIvfPq. Not exposed via CLI
     or Python today — always Auto through those bindings.
   - compact_incremental() (what CompactionExecutor::run() calls by default):
     when one input file holds more than 40% of the merged row count and has
     no deletion-vector-masked rows, its existing HNSW graph is reused and
     extended in place (HnswIndex::insert_node for every non-dominant row)
     instead of rebuilt from scratch. Falls back to a full compact() rebuild
     when there's no dominant file, the dominant file has DV-masked rows, or
     index_strategy == ForceIvfPq (incremental extension only ever produces
     HNSW, never IVF-PQ).
   Default HNSW parameters: M=16, ef_construction=150 (HnswConfig::default()).

6. WRITE
   AilakeFileWriter writes the merged RecordBatch + index + centroid/radius
   → one new .parquet file with the index in its footer.

7. COMMIT
   catalog.commit_snapshot(..., SnapshotOperation::Replace)
   → atomic Iceberg snapshot: removes old DataFile entries, adds the new one.
   → for catalog backends where retires_files_physically() is true (the
     default HadoopCatalog), the old files are deleted from object storage
     immediately after the commit succeeds — no retention window (see
     "Compaction and snapshot isolation" below). Backends that manage
     physical reclamation themselves (e.g. DuckLake) skip this step.
```

### Why rebuild HNSW from scratch instead of merging

The HNSW implementation (`ailake-index`, pure Rust — no longer built on the external `hnsw_rs` crate) does not expose a primitive for merging two independently-built graphs while maintaining quality invariants — that stays a non-trivial research problem, and `compact()`'s full rebuild remains correct by construction for the general case. The one merge shortcut that does exist is narrower: `compact_incremental()`'s dominant-file extension (see above), which *extends* one file's existing graph node-by-node rather than merging two separate graphs — it only applies when a single input file already accounts for most of the rows and has no pending deletes, and it always falls back to a full rebuild otherwise.

A full rebuild:

- Produces optimal recall at the configured `M` and `ef_construction`.
- Is bounded in cost: one compaction pass merges at most `max_files_per_pass` files, not the entire table.

For a 512 MB Parquet file containing ~200k rows of dim=1536 F16 vectors, HNSW rebuild takes approximately 45–90 seconds on a 4-core worker. This is acceptable for a background job.

---

## Compaction and snapshot isolation

Iceberg's snapshot model provides full isolation:

```
Timeline:

T0: Snapshot S1 → files: [A, B, C, D]   (readers use S1)
T1: Compaction starts, reads A+B+C+D
T2: New write arrives → Snapshot S2 → files: [A, B, C, D, E]
T3: Compaction commits → Snapshot S3 → files: [ABCD_merged, E]
    (A, B, C, D deleted from object storage right after this commit — see caveat below)

Readers at T2 see S2 (A+B+C+D+E), not affected by compaction.
Readers after T3 see S3 (merged+E), better performance.
```

**Caveat — no retention window for the default catalog backend**: for `HadoopCatalog` (the default) and any other backend where `retires_files_physically()` is true, superseded files (A, B, C, D above) are deleted from object storage immediately once the `Replace` snapshot commits — there is no grace period. A reader still mid-query against S1 at that point can hit a missing-file error rather than transparently finishing against S1's files. Backends that manage physical reclamation themselves (e.g. DuckLake) are unaffected the same way. There is no vacuum step and no configurable retention window today.

### Conflict detection

Optimistic-concurrency retry-on-conflict is implemented per catalog backend, not as a shared `ailake-catalog`-level wrapper, and coverage differs:

- **Glue, JDBC, REST catalogs** each have their own local commit-retry loop (read → re-apply → conditional write), capped at a backend-local `MAX_RETRIES = 5`.
- **`HadoopCatalog`** (the default) instead serializes commits through an in-process `tokio::sync::Mutex` — this prevents two compactions racing *within the same process*, but provides no cross-process/cross-machine optimistic-concurrency retry. Two compactors on different hosts committing against the same Hadoop-style warehouse concurrently are not automatically reconciled by a retry loop the way Glue/JDBC/REST are.

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

## No separate vacuum step

There is no `vacuum` command (CLI, Python, or Rust) and no `ailake.vacuum.*`/`ailake.compaction.*` property is read by any code path — `metadata.json` `properties` do not configure compaction behavior. As noted above, physical deletion of superseded files happens as the last step of compaction itself (`delete_old_files`, immediately after the `Replace` snapshot commits, for backends where `retires_files_physically()` is true), not as a separate scheduled job with a retention window.

---

## Rust implementation structure

```
ailake-query/src/
└── compaction.rs
```

```rust
pub enum CompactionIndexStrategy {
    Auto,       // detect hardware, pick HNSW vs IVF-PQ (default)
    ForceHnsw,
    ForceIvfPq,
}

pub struct CompactionConfig {
    /// Minimum number of files to trigger compaction (default: 4)
    pub min_files_to_compact: usize,
    /// Files below this size are candidates for compaction (default: 128 MB)
    pub target_file_size_bytes: u64,
    /// Index algorithm for the merged output file (default: Auto)
    pub index_strategy: CompactionIndexStrategy,
    /// Maximum files merged in a single pass (default: 20)
    pub max_files_per_pass: usize,
}

pub struct CompactionPlanner {
    config: CompactionConfig, // private; set via CompactionPlanner::new(config)
}

impl CompactionPlanner {
    /// Foreign files (missing centroid_b64) first, then size-eligible files
    /// smallest-first, capped at max_files_per_pass. Empty vec if fewer than
    /// min_files_to_compact size-eligible files exist and there are no
    /// foreign files to repair.
    pub fn plan(&self, files: &[DataFileEntry]) -> Vec<DataFileEntry>;
}

pub struct CompactionExecutor {
    store: Arc<dyn Store>,
    policy: VectorStoragePolicy,
    index_strategy: CompactionIndexStrategy,
    fts_config: Option<ailake_fts::FtsConfig>,
}

impl CompactionExecutor {
    /// Read N files, concat RecordBatches, rebuild HNSW/IVF-PQ from scratch,
    /// write one unified file.
    pub async fn compact(
        &self,
        files: &[DataFileEntry],
        output_path: &str,
    ) -> AilakeResult<DataFileEntry>;

    /// Same output contract as compact(), but reuses/extends a dominant
    /// input file's existing HNSW graph when eligible instead of rebuilding
    /// from scratch (falls back to compact() otherwise).
    pub async fn compact_incremental(
        &self,
        files: &[DataFileEntry],
        output_path: &str,
    ) -> AilakeResult<DataFileEntry>;

    /// Full cycle: plan → compact_incremental → catalog.commit_snapshot →
    /// delete old files (when the catalog backend retires files physically).
    /// Returns None if the planner finds nothing to compact.
    pub async fn run(
        &self,
        planner: &CompactionPlanner,
        table: &TableIdent,
        catalog: Arc<dyn CatalogProvider>,
        output_prefix: &str,
    ) -> AilakeResult<Option<DataFileEntry>>;
}
```
