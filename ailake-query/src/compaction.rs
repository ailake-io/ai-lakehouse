use std::sync::Arc;

use ailake_catalog::{
    make_data_file_entry, CatalogProvider, DataFileEntry, NewSnapshot, SnapshotOperation,
    TableIdent, VectorIndexInfo,
};
use ailake_core::{AilakeResult, VectorStoragePolicy};
use ailake_file::{AilakeFileReader, AilakeFileWriter};
use ailake_store::Store;
use ailake_vec::compute_centroid_and_radius;
use arrow_array::RecordBatch;
use arrow_schema::SchemaRef;
use bytes::Bytes;

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
}

impl Default for CompactionConfig {
    fn default() -> Self {
        Self {
            min_files_to_compact: 4,
            target_file_size_bytes: 128 * 1024 * 1024, // 128 MB
            index_strategy: CompactionIndexStrategy::Auto,
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

    /// Select files to compact: all files smaller than `target_file_size_bytes`,
    /// provided at least `min_files_to_compact` qualify.
    pub fn plan(&self, files: &[DataFileEntry]) -> Vec<DataFileEntry> {
        let candidates: Vec<DataFileEntry> = files
            .iter()
            .filter(|f| f.file_size_bytes < self.config.target_file_size_bytes)
            .cloned()
            .collect();
        if candidates.len() < self.config.min_files_to_compact {
            return vec![];
        }
        candidates
    }
}

/// Executes compaction plans: reads N small files, merges them into a single
/// AI-Lake file with a rebuilt index, and commits to the catalog.
///
/// The index algorithm is chosen via `CompactionIndexStrategy` (default: `Auto`,
/// which detects GPU / CPU cores at compaction time — the same heuristic used
/// by `write_batch_auto`).
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

    /// Merge `files` into a single new file at `output_path`.
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

        let mut all_batches: Vec<RecordBatch> = Vec::new();
        let mut all_embeddings: Vec<Vec<f32>> = Vec::new();
        let mut schema: Option<SchemaRef> = None;

        for entry in files {
            let bytes: Bytes = self.store.get(&entry.path).await?;
            let reader = AilakeFileReader::new(bytes, &self.policy.column_name, self.policy.dim);
            if !reader.is_ailake_file() {
                continue;
            }
            let (batch, embs) = reader.read_parquet()?;
            if schema.is_none() {
                schema = Some(batch.schema());
            }
            all_batches.push(batch);
            all_embeddings.extend(embs);
        }

        if all_batches.is_empty() {
            return Err(ailake_core::AilakeError::Catalog(
                "compact: no valid AI-Lake files in input".into(),
            ));
        }

        // Concatenate all row groups into one batch
        let merged_batch = concat_batches(schema.unwrap(), &all_batches)?;
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

        let entry = make_data_file_entry(
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
        Ok(entry)
    }

    /// Full compaction workflow: plan, compact, drop old files from catalog, commit.
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
            .unwrap()
            .as_millis();
        let output_path = format!("{output_prefix}/compacted-{ts}.parquet");

        let merged = self.compact(&to_compact, &output_path).await?;

        // Commit: add merged file, remove input files (via Overwrite snapshot)
        let snapshot = NewSnapshot {
            snapshot_id: ailake_catalog::new_snapshot_id(),
            parent_snapshot_id: None,
            files: vec![merged.clone()],
            operation: SnapshotOperation::Replace,
            iceberg_schema: None,
        };
        catalog.commit_snapshot(table, snapshot).await?;

        // Delete old files from store
        for entry in &to_compact {
            let _ = self.store.delete(&entry.path).await;
        }

        Ok(Some(merged))
    }
}

fn concat_batches(schema: SchemaRef, batches: &[RecordBatch]) -> AilakeResult<RecordBatch> {
    arrow_select::concat::concat_batches(&schema, batches)
        .map_err(|e| ailake_core::AilakeError::Arrow(e.to_string()))
}

#[cfg(test)]
mod tests {
    use super::*;

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
                file_size_bytes: 100, // below target
                centroid_b64: None,
                radius: None,
                hnsw_offset: None,
                hnsw_len: None,
                vector_column: None,
                vector_dim: None,
                extra_vector_indexes: vec![],
                index_status: ailake_catalog::IndexStatus::Ready,
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
                index_status: ailake_catalog::IndexStatus::Ready,
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
                index_status: ailake_catalog::IndexStatus::Ready,
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
                index_status: ailake_catalog::IndexStatus::Ready,
            },
        ];
        let selected = planner.plan(&files);
        assert_eq!(selected.len(), 2);
        assert!(selected.iter().any(|f| f.path == "small.parquet"));
        assert!(selected.iter().any(|f| f.path == "also-small.parquet"));
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
            keep_raw_for_reranking: false,
        };

        // Write two small files
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
                index_status: ailake_catalog::IndexStatus::Ready,
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
                index_status: ailake_catalog::IndexStatus::Ready,
            },
        ];

        let executor = CompactionExecutor::new(store.clone(), policy.clone());
        let merged = executor
            .compact(&entries, "data/merged.parquet")
            .await
            .unwrap();

        assert_eq!(merged.record_count, 4);
        assert_eq!(merged.path, "data/merged.parquet");

        // Verify merged file is a valid AI-Lake file with all 4 rows
        let merged_bytes = store.get("data/merged.parquet").await.unwrap();
        let reader = AilakeFileReader::new(merged_bytes, "embedding", 4);
        reader.verify_integrity().unwrap();
        let (batch, embs) = reader.read_parquet().unwrap();
        assert_eq!(batch.num_rows(), 4);
        assert_eq!(embs.len(), 4);
    }
}
