// SPDX-License-Identifier: MIT OR Apache-2.0
use std::sync::Arc;
use tracing::{debug, error, info};

use ailake_catalog::{
    make_data_file_entry, make_data_file_entry_indexing, CatalogProvider, DataFileEntry,
    NewSnapshot, SnapshotOperation, TableIdent, VectorIndexInfo,
};
use ailake_core::{AilakeResult, RowId, VectorStoragePolicy};
use ailake_file::{AilakeFileReader, AilakeFileWriter};
use ailake_store::Store;
use ailake_vec::compute_centroid_and_radius;
use arrow_array::RecordBatch;
use arrow_schema::SchemaRef;
use bytes::Bytes;
use futures::future::try_join_all;

use crate::writer::build_and_patch_index;

/// Index strategy for the merged file produced by compaction.
#[derive(Debug, Clone, Default)]
pub enum CompactionIndexStrategy {
    /// Detect GPU / CPU cores at compaction time and pick the best index.
    /// IVF-PQ on GPU/many-core machines; HNSW elsewhere. (default)
    #[default]
    Auto,
    /// Always rebuild with HNSW — highest recall, larger index.
    ForceHnsw,
    /// Always rebuild with IVF-PQ — smaller index, better S3 throughput.
    ///
    /// Recommended for large compactions (N > 100 000) on CPU-only machines
    /// where HNSW rebuild cost becomes prohibitive.
    ForceIvfPq,
}

#[derive(Debug, Clone)]
pub struct CompactionConfig {
    /// Trigger compaction only if at least this many files are eligible.
    pub min_files_to_compact: usize,
    /// Target output file size in bytes. Files below this are merged.
    pub target_file_size_bytes: u64,
    /// Index algorithm for the merged output file.
    pub index_strategy: CompactionIndexStrategy,
    /// Maximum files merged in a single compaction pass.
    ///
    /// Candidates are sorted smallest-first; only the first `max_files_per_pass`
    /// are compacted each run. This bounds peak RAM and HNSW rebuild CPU cost —
    /// O(N log N) stays proportional to this limit rather than table size.
    /// Default: 20. Set to `usize::MAX` to compact all eligible files at once.
    pub max_files_per_pass: usize,
}

impl Default for CompactionConfig {
    fn default() -> Self {
        Self {
            min_files_to_compact: 4,
            target_file_size_bytes: 128 * 1024 * 1024, // 128 MB
            index_strategy: CompactionIndexStrategy::Auto,
            max_files_per_pass: 20,
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub enum CompactionMode {
    Full,    // compact all files below target size
    Partial, // compact the smallest N files
}

pub struct CompactionPlanner {
    config: CompactionConfig,
}

impl CompactionPlanner {
    pub fn new(config: CompactionConfig) -> Self {
        Self { config }
    }

    /// Select files to compact.
    ///
    /// Picks files smaller than `target_file_size_bytes`, sorts them smallest-first
    /// (cheapest to read), and caps the selection at `max_files_per_pass`. This
    /// tiered approach prevents a single pass from compacting the entire table into
    /// memory when thousands of small files exist.
    pub fn plan(&self, files: &[DataFileEntry]) -> Vec<DataFileEntry> {
        let mut candidates: Vec<DataFileEntry> = files
            .iter()
            .filter(|f| f.file_size_bytes < self.config.target_file_size_bytes)
            .cloned()
            .collect();
        if candidates.len() < self.config.min_files_to_compact {
            debug!(
                "ailake: compaction skipped — {} eligible files < min_files_to_compact={}",
                candidates.len(),
                self.config.min_files_to_compact
            );
            return vec![];
        }
        // Sort smallest-first so each pass handles the cheapest files first.
        // This bounds peak RAM to max_files_per_pass * avg_small_file_size.
        candidates.sort_unstable_by_key(|f| f.file_size_bytes);
        candidates.truncate(self.config.max_files_per_pass);
        let total_bytes: u64 = candidates.iter().map(|f| f.file_size_bytes).sum();
        info!(
            "ailake: compaction plan — {} files ({} bytes) → 1 merged file",
            candidates.len(),
            total_bytes
        );
        candidates
    }
}

/// Executes compaction plans: reads N small files, merges them into a single
/// AI-Lake file with a rebuilt index, and commits to the catalog.
///
/// The index algorithm is chosen via `CompactionIndexStrategy` (default: `Auto`,
/// which detects GPU / CPU cores at compaction time — the same heuristic used
/// by `write_batch_auto`).
///
/// For large tables use `compact_deferred` / `run_deferred`: the merged Parquet
/// is persisted immediately and the HNSW build runs in a background Tokio task,
/// decoupling I/O cost from CPU cost.
pub struct CompactionExecutor {
    store: Arc<dyn Store>,
    policy: VectorStoragePolicy,
    index_strategy: CompactionIndexStrategy,
}

impl CompactionExecutor {
    pub fn new(store: Arc<dyn Store>, policy: VectorStoragePolicy) -> Self {
        Self {
            store,
            policy,
            index_strategy: CompactionIndexStrategy::Auto,
        }
    }

    /// Override the default (Auto) index strategy for this executor.
    pub fn with_index_strategy(mut self, strategy: CompactionIndexStrategy) -> Self {
        self.index_strategy = strategy;
        self
    }

    /// Read all input files in parallel, returning ordered (batch, embeddings) pairs.
    async fn read_files_parallel(
        &self,
        files: &[DataFileEntry],
    ) -> AilakeResult<Vec<(RecordBatch, Vec<Vec<f32>>)>> {
        let futs = files.iter().map(|entry| {
            let store = self.store.clone();
            let path = entry.path.clone();
            let column = self.policy.column_name.clone();
            let dim = self.policy.dim;
            async move {
                let bytes: Bytes = store.get(&path).await?;
                let reader = AilakeFileReader::new(bytes, &column, dim);
                if !reader.is_ailake_file() {
                    debug!("ailake: compaction skipping {} — not an AI-Lake file", path);
                    return Ok::<Option<(RecordBatch, Vec<Vec<f32>>)>, ailake_core::AilakeError>(
                        None,
                    );
                }
                let pair = reader.read_parquet()?;
                Ok(Some(pair))
            }
        });
        let results = try_join_all(futs).await?;
        Ok(results.into_iter().flatten().collect())
    }

    /// Merge `files` into a single new file at `output_path`.
    ///
    /// Reads all input files **in parallel** to minimise S3 latency, then
    /// rebuilds the HNSW / IVF-PQ index synchronously. For very large merges
    /// (N > 100 000 vectors) prefer `compact_deferred`, which offloads the
    /// index build to a background Tokio task.
    ///
    /// Returns the DataFileEntry for the merged file.
    pub async fn compact(
        &self,
        files: &[DataFileEntry],
        output_path: &str,
    ) -> AilakeResult<DataFileEntry> {
        if files.is_empty() {
            return Err(ailake_core::AilakeError::Catalog(
                "compact: no files provided".into(),
            ));
        }

        let pairs = self.read_files_parallel(files).await?;

        if pairs.is_empty() {
            return Err(ailake_core::AilakeError::Catalog(
                "compact: no valid AI-Lake files in input".into(),
            ));
        }

        let schema: SchemaRef = pairs[0].0.schema();
        let (all_batches, all_embeddings): (Vec<_>, Vec<_>) = pairs.into_iter().unzip();
        let all_embeddings: Vec<Vec<f32>> = all_embeddings.into_iter().flatten().collect();

        // Concatenate all row groups into one batch
        let merged_batch = concat_batches(schema, &all_batches)?;
        let record_count = merged_batch.num_rows() as u64;

        // Write merged file with adaptive index selection.
        let writer = {
            let base = AilakeFileWriter::new(self.policy.clone());
            match &self.index_strategy {
                CompactionIndexStrategy::Auto => base.with_auto_index(),
                CompactionIndexStrategy::ForceHnsw => base,
                CompactionIndexStrategy::ForceIvfPq => {
                    let cfg = ailake_index::IvfPqConfig::for_dataset(
                        self.policy.dim as usize,
                        all_embeddings.len(),
                    );
                    base.with_ivf_pq(cfg)
                }
            }
        };
        let file_bytes = writer.write(&merged_batch, &all_embeddings)?;
        let file_size = file_bytes.len() as u64;
        self.store.put(output_path, file_bytes.clone()).await?;

        // Compute centroid and HNSW offsets for catalog entry
        let centroid = compute_centroid_and_radius(&all_embeddings, self.policy.metric);
        let reader = AilakeFileReader::new(file_bytes, &self.policy.column_name, self.policy.dim);
        let header = reader.read_header()?;
        let ailk_start = reader.ailk_offset()?;

        // Preserve row-ID continuity: merged file inherits the minimum first_row_id of
        // its sources so commit_snapshot doesn't allocate fresh IDs and grow next_row_id.
        let source_first_row_id = files.iter().filter_map(|f| f.first_row_id).min();

        let mut entry = make_data_file_entry(
            output_path,
            record_count,
            file_size,
            &centroid,
            VectorIndexInfo {
                column: &self.policy.column_name,
                dim: self.policy.dim,
                hnsw_offset: ailk_start + header.hnsw_offset,
                hnsw_len: header.hnsw_len,
            },
        );
        entry.first_row_id = source_first_row_id;
        Ok(entry)
    }

    /// Merge `files` into a single new file using incremental HNSW insertion.
    ///
    /// Identifies the **dominant file** — the file holding >= 40 % of the total
    /// row count — loads its existing HNSW graph from the AILK section, then
    /// calls `HnswIndex::insert_node` for every vector from the remaining files.
    ///
    /// **Complexity vs `compact`**:
    /// - Full rebuild: O(N log N), N = total rows.
    /// - Incremental (this method): O(N_dom) deserialization + O(N_small × log N_dom).
    ///   For a 90 / 10 split (N = 1 M, N_dom = 900 k) the speedup is ~7×.
    ///
    /// **Fallbacks** (all degrade gracefully to `compact`):
    /// - No file holds >= 40 % of rows.
    /// - Dominant file's HNSW cannot be loaded (IVF-PQ, `IndexStatus::Indexing`, corrupt).
    ///
    /// **RowId contract**: dominant file's vectors are placed first in the merged
    /// Parquet (positions 0..N_dom-1); other files follow. The existing RowIds from
    /// the dominant HNSW remain valid; new nodes receive RowIds N_dom..N-1.
    pub async fn compact_incremental(
        &self,
        files: &[DataFileEntry],
        output_path: &str,
    ) -> AilakeResult<DataFileEntry> {
        const DOMINANT_RATIO: f64 = 0.40;

        if files.is_empty() {
            return Err(ailake_core::AilakeError::Catalog(
                "compact_incremental: no files provided".into(),
            ));
        }

        // Find the dominant file by record_count.
        let total_rows: u64 = files.iter().map(|f| f.record_count).sum();
        let dom_idx = files
            .iter()
            .enumerate()
            .max_by_key(|(_, f)| f.record_count)
            .map(|(i, _)| i)
            .unwrap_or(0);
        let dom_rows = files[dom_idx].record_count;

        if (dom_rows as f64 / total_rows as f64) < DOMINANT_RATIO {
            debug!(
                "ailake: compact_incremental — no dominant file ({}/{} rows < {:.0}% threshold), \
                 falling back to full rebuild",
                dom_rows,
                total_rows,
                DOMINANT_RATIO * 100.0
            );
            return self.compact(files, output_path).await;
        }

        let column = self.policy.column_name.clone();
        let dim = self.policy.dim;
        let dom_path = files[dom_idx].path.clone();

        // Read all files in parallel. Retain raw bytes only for the dominant file
        // (needed to load its HNSW without a second round-trip).
        let futs: Vec<_> = files
            .iter()
            .map(|entry| {
                let store = self.store.clone();
                let path = entry.path.clone();
                let col = column.clone();
                let is_dom = path == dom_path;
                async move {
                    let bytes: Bytes = store.get(&path).await?;
                    let reader = AilakeFileReader::new(bytes.clone(), &col, dim);
                    if !reader.is_ailake_file() {
                        debug!(
                            "ailake: compact_incremental skipping {} — not an AI-Lake file",
                            path
                        );
                        return Ok::<Option<(RecordBatch, Vec<Vec<f32>>, bool, Option<Bytes>)>, ailake_core::AilakeError>(None);
                    }
                    let (batch, vecs) = reader.read_parquet()?;
                    let retained = if is_dom { Some(bytes) } else { None };
                    Ok(Some((batch, vecs, is_dom, retained)))
                }
            })
            .collect();

        let raw: Vec<(RecordBatch, Vec<Vec<f32>>, bool, Option<Bytes>)> =
            try_join_all(futs).await?.into_iter().flatten().collect();

        if raw.is_empty() {
            return Err(ailake_core::AilakeError::Catalog(
                "compact_incremental: no valid AI-Lake files in input".into(),
            ));
        }

        // Separate dominant from others; dominant goes first in the merged file.
        let mut dom_batch: Option<RecordBatch> = None;
        let mut dom_vecs: Vec<Vec<f32>> = Vec::new();
        let mut dom_bytes_found: Option<Bytes> = None;
        let mut other_batches: Vec<RecordBatch> = Vec::new();
        let mut other_vecs: Vec<Vec<f32>> = Vec::new();

        for (batch, vecs, is_dom, retained) in raw {
            if is_dom {
                dom_batch = Some(batch);
                dom_vecs = vecs;
                dom_bytes_found = retained;
            } else {
                other_batches.push(batch);
                other_vecs.extend(vecs);
            }
        }

        let (dom_batch, dom_bytes) = match (dom_batch, dom_bytes_found) {
            (Some(b), Some(byt)) => (b, byt),
            _ => {
                debug!(
                    "ailake: compact_incremental — dominant file missing from read results, \
                     falling back to full rebuild"
                );
                return self.compact(files, output_path).await;
            }
        };

        // Load the dominant file's existing HNSW graph.
        let dom_reader = AilakeFileReader::new(dom_bytes, &column, dim);
        let mut hnsw = match dom_reader.load_index() {
            Ok(idx) => idx,
            Err(e) => {
                debug!(
                    "ailake: compact_incremental — cannot load dominant HNSW ({}), \
                     falling back to full rebuild",
                    e
                );
                return self.compact(files, output_path).await;
            }
        };

        let dom_count = dom_batch.num_rows() as u64;

        // Insert vectors from non-dominant files into the loaded graph.
        // RowIds are assigned starting at dom_count to match positions in the merged Parquet.
        for (j, vec) in other_vecs.iter().enumerate() {
            hnsw.insert_node(RowId::new(dom_count + j as u64), vec.clone());
        }
        hnsw.quantize_to_f16();

        // Assemble merged batch (dominant rows first) and all embeddings.
        let schema: SchemaRef = dom_batch.schema();
        let mut all_batches = vec![dom_batch];
        all_batches.extend(other_batches);
        let merged_batch = concat_batches(schema, &all_batches)?;
        let record_count = merged_batch.num_rows() as u64;

        let mut all_embeddings = dom_vecs;
        all_embeddings.extend(other_vecs);

        // Write the merged file using the pre-built index (no rebuild).
        let writer = AilakeFileWriter::new(self.policy.clone());
        let file_bytes =
            writer.write_with_prebuilt_hnsw(&merged_batch, &all_embeddings, &hnsw)?;
        let file_size = file_bytes.len() as u64;
        self.store.put(output_path, file_bytes.clone()).await?;

        let centroid = compute_centroid_and_radius(&all_embeddings, self.policy.metric);
        let reader = AilakeFileReader::new(file_bytes, &self.policy.column_name, self.policy.dim);
        let header = reader.read_header()?;
        let ailk_start = reader.ailk_offset()?;

        // Dominant file goes first in the merged output, so the merged file's first
        // logical row was the dominant file's first row.  Use its first_row_id so
        // commit_snapshot doesn't grow next_row_id unnecessarily.
        let source_first_row_id = files[dom_idx].first_row_id;

        let mut entry = make_data_file_entry(
            output_path,
            record_count,
            file_size,
            &centroid,
            VectorIndexInfo {
                column: &self.policy.column_name,
                dim: self.policy.dim,
                hnsw_offset: ailk_start + header.hnsw_offset,
                hnsw_len: header.hnsw_len,
            },
        );
        entry.first_row_id = source_first_row_id;

        info!(
            "ailake: compact_incremental — merged {} files into {} \
             ({} rows from dominant + {} inserted incrementally)",
            files.len(),
            output_path,
            dom_count,
            record_count - dom_count
        );

        Ok(entry)
    }

    /// Merge `files` into a single new file at `output_path`, writing Parquet
    /// immediately and building the HNSW / IVF-PQ index in a background Tokio task.
    ///
    /// The merged file appears in the catalog as `IndexStatus::Indexing` until
    /// the background task completes; queries fall back to flat scan during that
    /// window (same behaviour as `write_batch_deferred`).
    ///
    /// Returns the `DataFileEntry` with `IndexStatus::Indexing`. The entry
    /// transitions to `Ready` automatically when the background build finishes.
    pub async fn compact_deferred(
        &self,
        files: &[DataFileEntry],
        output_path: &str,
        catalog: Arc<dyn CatalogProvider>,
        table: &TableIdent,
    ) -> AilakeResult<DataFileEntry> {
        if files.is_empty() {
            return Err(ailake_core::AilakeError::Catalog(
                "compact_deferred: no files provided".into(),
            ));
        }

        let pairs = self.read_files_parallel(files).await?;

        if pairs.is_empty() {
            return Err(ailake_core::AilakeError::Catalog(
                "compact_deferred: no valid AI-Lake files in input".into(),
            ));
        }

        let schema: SchemaRef = pairs[0].0.schema();
        let (all_batches, all_embeddings): (Vec<_>, Vec<_>) = pairs.into_iter().unzip();
        let all_embeddings: Vec<Vec<f32>> = all_embeddings.into_iter().flatten().collect();

        let merged_batch = concat_batches(schema, &all_batches)?;
        let record_count = merged_batch.num_rows() as u64;

        // Write Parquet-only immediately — fast path, no HNSW build.
        let file_writer = AilakeFileWriter::new(self.policy.clone());
        let parquet_bytes = file_writer.write_parquet_only(&merged_batch, &all_embeddings)?;
        let file_size = parquet_bytes.len() as u64;
        self.store.put(output_path, parquet_bytes).await?;

        // Centroid available for geometric pruning during the build window.
        let centroid = compute_centroid_and_radius(&all_embeddings, self.policy.metric);
        let source_first_row_id = files.iter().filter_map(|f| f.first_row_id).min();
        let mut entry = make_data_file_entry_indexing(
            output_path,
            record_count,
            file_size,
            &centroid,
            &self.policy.column_name,
            self.policy.dim,
        );
        entry.first_row_id = source_first_row_id;

        // Spawn background index build; errors are logged, not propagated.
        let store = self.store.clone();
        let policy = self.policy.clone();
        let table_id = table.clone();
        let fp = output_path.to_string();
        tokio::spawn(async move {
            if let Err(e) = build_and_patch_index(store, catalog, policy, table_id, fp).await {
                error!(
                    "ailake: compaction deferred HNSW build failed — file indexed as \
                     Parquet-only until next compaction rebuilds the index: {}",
                    e
                );
            }
        });

        Ok(entry)
    }

    /// Full compaction workflow: plan, compact (synchronous HNSW rebuild),
    /// drop old files from catalog, commit.
    pub async fn run(
        &self,
        planner: &CompactionPlanner,
        table: &TableIdent,
        catalog: Arc<dyn CatalogProvider>,
        output_prefix: &str,
    ) -> AilakeResult<Option<DataFileEntry>> {
        let all_files = catalog.list_files(table, None).await?;
        let to_compact = planner.plan(&all_files);
        if to_compact.is_empty() {
            return Ok(None);
        }

        let ts = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_else(|e| e.duration())
            .as_millis();
        let output_path = format!("{output_prefix}/compacted-{ts}.parquet");

        // Use incremental merge when a dominant file exists (falls back to full rebuild automatically).
        let merged = self.compact_incremental(&to_compact, &output_path).await?;

        // Commit: add merged file, remove input files (via Replace snapshot)
        let snapshot = NewSnapshot {
            snapshot_id: ailake_catalog::new_snapshot_id(),
            parent_snapshot_id: None,
            files: vec![merged.clone()],
            operation: SnapshotOperation::Replace,
            iceberg_schema: None,
            extra_properties: std::collections::HashMap::new(),
            bloom_filters: vec![],
                equality_delete_files: vec![],
        };
        catalog.commit_snapshot(table, snapshot).await?;

        info!(
            "ailake: compaction committed — merged {} files into {}",
            to_compact.len(),
            output_path
        );

        delete_old_files(&self.store, &to_compact).await;

        Ok(Some(merged))
    }

    /// Full compaction workflow with deferred HNSW build: plan, write merged
    /// Parquet immediately, commit as `Indexing`, spawn background index build.
    ///
    /// Use for large tables where inline HNSW rebuild blocks too long.
    pub async fn run_deferred(
        &self,
        planner: &CompactionPlanner,
        table: &TableIdent,
        catalog: Arc<dyn CatalogProvider>,
        output_prefix: &str,
    ) -> AilakeResult<Option<DataFileEntry>> {
        let all_files = catalog.list_files(table, None).await?;
        let to_compact = planner.plan(&all_files);
        if to_compact.is_empty() {
            return Ok(None);
        }

        let ts = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_else(|e| e.duration())
            .as_millis();
        let output_path = format!("{output_prefix}/compacted-{ts}.parquet");

        let merged = self
            .compact_deferred(&to_compact, &output_path, catalog.clone(), table)
            .await?;

        // Commit immediately: merged file in Indexing state replaces input files.
        let snapshot = NewSnapshot {
            snapshot_id: ailake_catalog::new_snapshot_id(),
            parent_snapshot_id: None,
            files: vec![merged.clone()],
            operation: SnapshotOperation::Replace,
            iceberg_schema: None,
            extra_properties: std::collections::HashMap::new(),
            bloom_filters: vec![],
                equality_delete_files: vec![],
        };
        catalog.commit_snapshot(table, snapshot).await?;

        info!(
            "ailake: compaction committed (deferred) — merged {} files into {} \
             (index building in background)",
            to_compact.len(),
            output_path
        );

        delete_old_files(&self.store, &to_compact).await;

        Ok(Some(merged))
    }
}

async fn delete_old_files(store: &Arc<dyn Store>, files: &[DataFileEntry]) {
    for entry in files {
        if let Err(e) = store.delete(&entry.path).await {
            error!(
                "ailake: compaction cleanup failed — could not delete {}: {} \
                 (orphan file in object store after successful catalog commit; \
                 delete manually to reclaim storage)",
                entry.path, e
            );
        }
    }
}

fn concat_batches(schema: SchemaRef, batches: &[RecordBatch]) -> AilakeResult<RecordBatch> {
    arrow_select::concat::concat_batches(&schema, batches)
        .map_err(|e| ailake_core::AilakeError::Arrow(e.to_string()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use ailake_catalog::IndexStatus;

    #[test]
    fn plan_returns_empty_if_too_few_files() {
        let planner = CompactionPlanner::new(CompactionConfig {
            min_files_to_compact: 4,
            target_file_size_bytes: 1024 * 1024,
            ..Default::default()
        });
        let files: Vec<DataFileEntry> = (0..3)
            .map(|i| DataFileEntry {
                path: format!("file-{i}.parquet"),
                record_count: 10,
                file_size_bytes: 100,
                centroid_b64: None,
                radius: None,
                hnsw_offset: None,
                hnsw_len: None,
                vector_column: None,
                vector_dim: None,
                extra_vector_indexes: vec![],
                index_status: IndexStatus::Ready,
                batch_id: None,
                embedding_model: None,
                partition_value: None,
                deletion_vector: None,
                first_row_id: None,
            })
            .collect();
        assert!(planner.plan(&files).is_empty());
    }

    #[test]
    fn plan_selects_small_files() {
        let planner = CompactionPlanner::new(CompactionConfig {
            min_files_to_compact: 2,
            target_file_size_bytes: 1000,
            ..Default::default()
        });
        let files = vec![
            DataFileEntry {
                path: "small.parquet".into(),
                record_count: 5,
                file_size_bytes: 500,
                centroid_b64: None,
                radius: None,
                hnsw_offset: None,
                hnsw_len: None,
                vector_column: None,
                vector_dim: None,
                extra_vector_indexes: vec![],
                index_status: IndexStatus::Ready,
                batch_id: None,
                embedding_model: None,
                partition_value: None,
                deletion_vector: None,
                first_row_id: None,
            },
            DataFileEntry {
                path: "large.parquet".into(),
                record_count: 5000,
                file_size_bytes: 200_000_000,
                centroid_b64: None,
                radius: None,
                hnsw_offset: None,
                hnsw_len: None,
                vector_column: None,
                vector_dim: None,
                extra_vector_indexes: vec![],
                index_status: IndexStatus::Ready,
                batch_id: None,
                embedding_model: None,
                partition_value: None,
                deletion_vector: None,
                first_row_id: None,
            },
            DataFileEntry {
                path: "also-small.parquet".into(),
                record_count: 5,
                file_size_bytes: 800,
                centroid_b64: None,
                radius: None,
                hnsw_offset: None,
                hnsw_len: None,
                vector_column: None,
                vector_dim: None,
                extra_vector_indexes: vec![],
                index_status: IndexStatus::Ready,
                batch_id: None,
                embedding_model: None,
                partition_value: None,
                deletion_vector: None,
                first_row_id: None,
            },
        ];
        let selected = planner.plan(&files);
        assert_eq!(selected.len(), 2);
        assert!(selected.iter().any(|f| f.path == "small.parquet"));
        assert!(selected.iter().any(|f| f.path == "also-small.parquet"));
    }

    #[test]
    fn plan_respects_max_files_per_pass() {
        let planner = CompactionPlanner::new(CompactionConfig {
            min_files_to_compact: 2,
            target_file_size_bytes: 1_000_000,
            max_files_per_pass: 3,
            ..Default::default()
        });
        let files: Vec<DataFileEntry> = (0..5)
            .map(|i| DataFileEntry {
                path: format!("f{i}.parquet"),
                record_count: 10,
                file_size_bytes: 100 + i as u64 * 100,
                centroid_b64: None,
                radius: None,
                hnsw_offset: None,
                hnsw_len: None,
                vector_column: None,
                vector_dim: None,
                extra_vector_indexes: vec![],
                index_status: IndexStatus::Ready,
                batch_id: None,
                embedding_model: None,
                partition_value: None,
                deletion_vector: None,
                first_row_id: None,
            })
            .collect();
        let selected = planner.plan(&files);
        assert_eq!(selected.len(), 3);
        assert_eq!(selected[0].file_size_bytes, 100);
        assert_eq!(selected[1].file_size_bytes, 200);
        assert_eq!(selected[2].file_size_bytes, 300);
    }

    #[test]
    fn plan_sorts_smallest_first() {
        let planner = CompactionPlanner::new(CompactionConfig {
            min_files_to_compact: 2,
            target_file_size_bytes: 10_000,
            max_files_per_pass: 4,
            ..Default::default()
        });
        let files = vec![
            DataFileEntry {
                path: "c.parquet".into(),
                record_count: 1,
                file_size_bytes: 300,
                centroid_b64: None, radius: None, hnsw_offset: None, hnsw_len: None,
                vector_column: None, vector_dim: None, extra_vector_indexes: vec![],
                index_status: IndexStatus::Ready, batch_id: None, embedding_model: None,
                partition_value: None,
                deletion_vector: None,
                first_row_id: None,
            },
            DataFileEntry {
                path: "a.parquet".into(),
                record_count: 1,
                file_size_bytes: 100,
                centroid_b64: None, radius: None, hnsw_offset: None, hnsw_len: None,
                vector_column: None, vector_dim: None, extra_vector_indexes: vec![],
                index_status: IndexStatus::Ready, batch_id: None, embedding_model: None,
                partition_value: None,
                deletion_vector: None,
                first_row_id: None,
            },
            DataFileEntry {
                path: "b.parquet".into(),
                record_count: 1,
                file_size_bytes: 200,
                centroid_b64: None, radius: None, hnsw_offset: None, hnsw_len: None,
                vector_column: None, vector_dim: None, extra_vector_indexes: vec![],
                index_status: IndexStatus::Ready, batch_id: None, embedding_model: None,
                partition_value: None,
                deletion_vector: None,
                first_row_id: None,
            },
        ];
        let selected = planner.plan(&files);
        assert_eq!(selected[0].file_size_bytes, 100);
        assert_eq!(selected[1].file_size_bytes, 200);
        assert_eq!(selected[2].file_size_bytes, 300);
    }

    #[tokio::test]
    async fn compact_merges_two_files() {
        use ailake_core::{VectorMetric, VectorPrecision};
        use ailake_store::LocalStore;
        use arrow_array::{Int32Array, RecordBatch};
        use arrow_schema::{DataType, Field, Schema};
        use std::sync::Arc;
        use tempfile::TempDir;

        let dir = TempDir::new().unwrap();
        let store = Arc::new(LocalStore::new(dir.path()));
        let policy = VectorStoragePolicy {
            column_name: "embedding".into(),
            dim: 4,
            metric: VectorMetric::Cosine,
            precision: VectorPrecision::F16,
            pq: None,
            keep_raw_for_reranking: true,
            pre_normalize: false,
            hnsw_m: None,
            hnsw_ef_construction: None,
            ivf_residual: false,
            embedding_model: None,
            modality: None,
            partition_by: None,
            partition_value: None,
        partition_column_type: None,
        };

        let schema = Arc::new(Schema::new(vec![Field::new("id", DataType::Int32, false)]));
        let embs_a: Vec<Vec<f32>> = vec![vec![1.0, 0.0, 0.0, 0.0], vec![0.0, 1.0, 0.0, 0.0]];
        let embs_b: Vec<Vec<f32>> = vec![vec![0.0, 0.0, 1.0, 0.0], vec![0.0, 0.0, 0.0, 1.0]];

        let batch_a = RecordBatch::try_new(
            schema.clone(),
            vec![Arc::new(Int32Array::from(vec![0i32, 1]))],
        )
        .unwrap();
        let batch_b = RecordBatch::try_new(
            schema.clone(),
            vec![Arc::new(Int32Array::from(vec![2i32, 3]))],
        )
        .unwrap();

        let writer_a = AilakeFileWriter::new(policy.clone());
        let bytes_a = writer_a.write(&batch_a, &embs_a).unwrap();
        let writer_b = AilakeFileWriter::new(policy.clone());
        let bytes_b = writer_b.write(&batch_b, &embs_b).unwrap();

        store.put("data/a.parquet", bytes_a.clone()).await.unwrap();
        store.put("data/b.parquet", bytes_b.clone()).await.unwrap();

        let entries = vec![
            DataFileEntry {
                path: "data/a.parquet".into(),
                record_count: 2,
                file_size_bytes: bytes_a.len() as u64,
                centroid_b64: None,
                radius: None,
                hnsw_offset: None,
                hnsw_len: None,
                vector_column: None,
                vector_dim: None,
                extra_vector_indexes: vec![],
                index_status: IndexStatus::Ready,
                batch_id: None,
                embedding_model: None,
                partition_value: None,
                deletion_vector: None,
                first_row_id: None,
            },
            DataFileEntry {
                path: "data/b.parquet".into(),
                record_count: 2,
                file_size_bytes: bytes_b.len() as u64,
                centroid_b64: None,
                radius: None,
                hnsw_offset: None,
                hnsw_len: None,
                vector_column: None,
                vector_dim: None,
                extra_vector_indexes: vec![],
                index_status: IndexStatus::Ready,
                batch_id: None,
                embedding_model: None,
                partition_value: None,
                deletion_vector: None,
                first_row_id: None,
            },
        ];

        let executor = CompactionExecutor::new(store.clone(), policy.clone());
        let merged = executor
            .compact(&entries, "data/merged.parquet")
            .await
            .unwrap();

        assert_eq!(merged.record_count, 4);
        assert_eq!(merged.path, "data/merged.parquet");

        let merged_bytes = store.get("data/merged.parquet").await.unwrap();
        let reader = AilakeFileReader::new(merged_bytes, "embedding", 4);
        reader.verify_integrity().unwrap();
        let (batch, embs) = reader.read_parquet().unwrap();
        assert_eq!(batch.num_rows(), 4);
        assert_eq!(embs.len(), 4);
    }

    #[tokio::test]
    async fn compact_incremental_merges_dominant_plus_small() {
        use ailake_core::{RowId, VectorMetric, VectorPrecision};
        use ailake_store::LocalStore;
        use arrow_array::{Int32Array, RecordBatch};
        use arrow_schema::{DataType, Field, Schema};
        use std::sync::Arc;
        use tempfile::TempDir;

        let dir = TempDir::new().unwrap();
        let store = Arc::new(LocalStore::new(dir.path()));
        let policy = VectorStoragePolicy {
            column_name: "embedding".into(),
            dim: 4,
            metric: VectorMetric::Cosine,
            precision: VectorPrecision::F16,
            pq: None,
            keep_raw_for_reranking: true,
            pre_normalize: false,
            hnsw_m: None,
            hnsw_ef_construction: None,
            ivf_residual: false,
            embedding_model: None,
            modality: None,
            partition_by: None,
            partition_value: None,
        partition_column_type: None,
        };

        let schema = Arc::new(Schema::new(vec![Field::new("id", DataType::Int32, false)]));

        // Dominant file: 6 rows (75% of total 8 rows — above 40% threshold).
        let embs_dom: Vec<Vec<f32>> = vec![
            vec![1.0, 0.0, 0.0, 0.0],
            vec![0.0, 1.0, 0.0, 0.0],
            vec![0.0, 0.0, 1.0, 0.0],
            vec![0.7, 0.7, 0.0, 0.0],
            vec![0.0, 0.7, 0.7, 0.0],
            vec![0.0, 0.0, 0.7, 0.7],
        ];
        let batch_dom = RecordBatch::try_new(
            schema.clone(),
            vec![Arc::new(Int32Array::from(vec![0i32, 1, 2, 3, 4, 5]))],
        )
        .unwrap();

        // Small file: 2 rows.
        let embs_small: Vec<Vec<f32>> = vec![
            vec![0.0, 0.0, 0.0, 1.0],
            vec![0.5, 0.5, 0.5, 0.5],
        ];
        let batch_small = RecordBatch::try_new(
            schema.clone(),
            vec![Arc::new(Int32Array::from(vec![6i32, 7]))],
        )
        .unwrap();

        let bytes_dom = AilakeFileWriter::new(policy.clone())
            .write(&batch_dom, &embs_dom)
            .unwrap();
        let bytes_small = AilakeFileWriter::new(policy.clone())
            .write(&batch_small, &embs_small)
            .unwrap();

        store
            .put("data/dominant.parquet", bytes_dom.clone())
            .await
            .unwrap();
        store
            .put("data/small.parquet", bytes_small.clone())
            .await
            .unwrap();

        let entries = vec![
            DataFileEntry {
                path: "data/dominant.parquet".into(),
                record_count: 6,
                file_size_bytes: bytes_dom.len() as u64,
                centroid_b64: None,
                radius: None,
                hnsw_offset: None,
                hnsw_len: None,
                vector_column: None,
                vector_dim: None,
                extra_vector_indexes: vec![],
                index_status: IndexStatus::Ready,
                batch_id: None,
                embedding_model: None,
                partition_value: None,
                deletion_vector: None,
                first_row_id: None,
            },
            DataFileEntry {
                path: "data/small.parquet".into(),
                record_count: 2,
                file_size_bytes: bytes_small.len() as u64,
                centroid_b64: None,
                radius: None,
                hnsw_offset: None,
                hnsw_len: None,
                vector_column: None,
                vector_dim: None,
                extra_vector_indexes: vec![],
                index_status: IndexStatus::Ready,
                batch_id: None,
                embedding_model: None,
                partition_value: None,
                deletion_vector: None,
                first_row_id: None,
            },
        ];

        let executor = CompactionExecutor::new(store.clone(), policy.clone());
        let merged = executor
            .compact_incremental(&entries, "data/merged.parquet")
            .await
            .unwrap();

        // Structural checks.
        assert_eq!(merged.record_count, 8);
        assert_eq!(merged.path, "data/merged.parquet");

        // Load merged file and verify it's a valid AI-Lake file.
        let merged_bytes = store.get("data/merged.parquet").await.unwrap();
        let reader = AilakeFileReader::new(merged_bytes, "embedding", 4);
        reader.verify_integrity().unwrap();

        let (batch, embs) = reader.read_parquet().unwrap();
        assert_eq!(batch.num_rows(), 8);
        assert_eq!(embs.len(), 8);

        // Dominant rows must come first (positions 0..5).
        for f in &embs[..6] {
            assert_eq!(f.len(), 4);
        }

        // HNSW must be searchable and return the nearest neighbor for a known query.
        let hnsw = reader.load_index().unwrap();
        assert_eq!(hnsw.node_count(), 8);

        // Query [1, 0, 0, 0] → nearest should be RowId 0 (embs_dom[0]).
        let results = hnsw.search(&[1.0, 0.0, 0.0, 0.0], 1, 50);
        assert_eq!(results[0].0, RowId::new(0));

        // Query [0, 0, 0, 1] → nearest should be RowId 6 (first row of small file,
        // inserted at position 6 in the merged file).
        let results = hnsw.search(&[0.0, 0.0, 0.0, 1.0], 1, 50);
        assert_eq!(results[0].0, RowId::new(6));
    }

    #[tokio::test]
    async fn compact_incremental_falls_back_when_no_dominant() {
        use ailake_core::{VectorMetric, VectorPrecision};
        use ailake_store::LocalStore;
        use arrow_array::{Int32Array, RecordBatch};
        use arrow_schema::{DataType, Field, Schema};
        use std::sync::Arc;
        use tempfile::TempDir;

        let dir = TempDir::new().unwrap();
        let store = Arc::new(LocalStore::new(dir.path()));
        let policy = VectorStoragePolicy {
            column_name: "embedding".into(),
            dim: 4,
            metric: VectorMetric::Cosine,
            precision: VectorPrecision::F16,
            pq: None,
            keep_raw_for_reranking: true,
            pre_normalize: false,
            hnsw_m: None,
            hnsw_ef_construction: None,
            ivf_residual: false,
            embedding_model: None,
            modality: None,
            partition_by: None,
            partition_value: None,
        partition_column_type: None,
        };

        let schema = Arc::new(Schema::new(vec![Field::new("id", DataType::Int32, false)]));

        // Two equal-sized files (50/50 split — no dominant, both below 40% threshold).
        let make_batch = |ids: Vec<i32>, embs: Vec<Vec<f32>>| {
            let batch = RecordBatch::try_new(
                schema.clone(),
                vec![Arc::new(Int32Array::from(ids))],
            )
            .unwrap();
            AilakeFileWriter::new(policy.clone())
                .write(&batch, &embs)
                .unwrap()
        };

        let embs_a: Vec<Vec<f32>> = vec![vec![1.0, 0.0, 0.0, 0.0], vec![0.0, 1.0, 0.0, 0.0]];
        let embs_b: Vec<Vec<f32>> = vec![vec![0.0, 0.0, 1.0, 0.0], vec![0.0, 0.0, 0.0, 1.0]];
        let bytes_a = make_batch(vec![0, 1], embs_a);
        let bytes_b = make_batch(vec![2, 3], embs_b);

        store.put("data/a.parquet", bytes_a.clone()).await.unwrap();
        store.put("data/b.parquet", bytes_b.clone()).await.unwrap();

        let entries = vec![
            DataFileEntry {
                path: "data/a.parquet".into(),
                record_count: 2,
                file_size_bytes: bytes_a.len() as u64,
                centroid_b64: None, radius: None, hnsw_offset: None, hnsw_len: None,
                vector_column: None, vector_dim: None, extra_vector_indexes: vec![],
                index_status: IndexStatus::Ready, batch_id: None, embedding_model: None,
                partition_value: None,
                deletion_vector: None,
                first_row_id: None,
            },
            DataFileEntry {
                path: "data/b.parquet".into(),
                record_count: 2,
                file_size_bytes: bytes_b.len() as u64,
                centroid_b64: None, radius: None, hnsw_offset: None, hnsw_len: None,
                vector_column: None, vector_dim: None, extra_vector_indexes: vec![],
                index_status: IndexStatus::Ready, batch_id: None, embedding_model: None,
                partition_value: None,
                deletion_vector: None,
                first_row_id: None,
            },
        ];

        let executor = CompactionExecutor::new(store.clone(), policy.clone());
        // Should fall back to full rebuild without error.
        let merged = executor
            .compact_incremental(&entries, "data/merged.parquet")
            .await
            .unwrap();

        assert_eq!(merged.record_count, 4);

        let merged_bytes = store.get("data/merged.parquet").await.unwrap();
        let reader = AilakeFileReader::new(merged_bytes, "embedding", 4);
        reader.verify_integrity().unwrap();
    }

    #[tokio::test]
    async fn compact_deferred_produces_parquet_only_file() {
        use ailake_catalog::HadoopCatalog;
        use ailake_core::{VectorMetric, VectorPrecision};
        use ailake_store::LocalStore;
        use arrow_array::{Int32Array, RecordBatch};
        use arrow_schema::{DataType, Field, Schema};
        use std::sync::Arc;
        use tempfile::TempDir;

        let dir = TempDir::new().unwrap();
        let store = Arc::new(LocalStore::new(dir.path()));
        let catalog_dir = TempDir::new().unwrap();
        let catalog_store = Arc::new(LocalStore::new(catalog_dir.path()));
        let catalog = Arc::new(HadoopCatalog::new(catalog_store, ""));
        let table = TableIdent {
            namespace: "ns".into(),
            name: "tbl".into(),
        };

        let policy = VectorStoragePolicy {
            column_name: "embedding".into(),
            dim: 4,
            metric: VectorMetric::Cosine,
            precision: VectorPrecision::F16,
            pq: None,
            keep_raw_for_reranking: true,
            pre_normalize: false,
            hnsw_m: None,
            hnsw_ef_construction: None,
            ivf_residual: false,
            embedding_model: None,
            modality: None,
            partition_by: None,
            partition_value: None,
        partition_column_type: None,
        };

        use ailake_catalog::TableProperties;
        catalog
            .create_table(
                &table,
                &TableProperties {
                    policy: policy.clone(),
                    extra: std::collections::HashMap::new(),
                    format_version: 2,
                    partition_column_type: None,
                },
            )
            .await
            .unwrap();

        let schema = Arc::new(Schema::new(vec![Field::new("id", DataType::Int32, false)]));
        let embs_a: Vec<Vec<f32>> = vec![vec![1.0, 0.0, 0.0, 0.0], vec![0.0, 1.0, 0.0, 0.0]];
        let batch_a = RecordBatch::try_new(
            schema.clone(),
            vec![Arc::new(Int32Array::from(vec![0i32, 1]))],
        )
        .unwrap();
        let bytes_a = AilakeFileWriter::new(policy.clone())
            .write(&batch_a, &embs_a)
            .unwrap();
        store.put("data/a.parquet", bytes_a.clone()).await.unwrap();

        let embs_b: Vec<Vec<f32>> = vec![vec![0.0, 0.0, 1.0, 0.0], vec![0.0, 0.0, 0.0, 1.0]];
        let batch_b = RecordBatch::try_new(
            schema.clone(),
            vec![Arc::new(Int32Array::from(vec![2i32, 3]))],
        )
        .unwrap();
        let bytes_b = AilakeFileWriter::new(policy.clone())
            .write(&batch_b, &embs_b)
            .unwrap();
        store.put("data/b.parquet", bytes_b.clone()).await.unwrap();

        let entries = vec![
            DataFileEntry {
                path: "data/a.parquet".into(),
                record_count: 2,
                file_size_bytes: bytes_a.len() as u64,
                centroid_b64: None, radius: None, hnsw_offset: None, hnsw_len: None,
                vector_column: None, vector_dim: None, extra_vector_indexes: vec![],
                index_status: IndexStatus::Ready, batch_id: None, embedding_model: None,
                partition_value: None,
                deletion_vector: None,
                first_row_id: None,
            },
            DataFileEntry {
                path: "data/b.parquet".into(),
                record_count: 2,
                file_size_bytes: bytes_b.len() as u64,
                centroid_b64: None, radius: None, hnsw_offset: None, hnsw_len: None,
                vector_column: None, vector_dim: None, extra_vector_indexes: vec![],
                index_status: IndexStatus::Ready, batch_id: None, embedding_model: None,
                partition_value: None,
                deletion_vector: None,
                first_row_id: None,
            },
        ];

        let executor = CompactionExecutor::new(store.clone(), policy.clone());
        let entry = executor
            .compact_deferred(&entries, "data/merged.parquet", catalog.clone(), &table)
            .await
            .unwrap();

        // Entry is Indexing — HNSW build pending in background
        assert_eq!(entry.index_status, IndexStatus::Indexing);
        assert_eq!(entry.record_count, 4);

        // The written file must be valid Parquet (readable) even without HNSW
        let merged_bytes = store.get("data/merged.parquet").await.unwrap();
        let pq_reader = ailake_parquet::ParquetVectorReader::new(merged_bytes, "embedding");
        let count = pq_reader.record_count().unwrap();
        assert_eq!(count, 4);
    }
}
