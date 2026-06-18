// SPDX-License-Identifier: MIT OR Apache-2.0
//! Embedding model migration for AI-Lake tables.
//!
//! Reads all chunks from a table, re-embeds them with a new model, and writes
//! new files with the updated embedding column. Two strategies are supported:
//!
//! - `AtomicReplace`: replaces each file one at a time. Lower peak storage, but
//!   during the migration window different shards may have different columns.
//! - `DualWriteThenCutover`: writes new files containing both old and new columns,
//!   then atomically replaces all old files. Higher peak storage, zero downtime.

use std::sync::Arc;

use ailake_catalog::{
    make_data_file_entry, new_snapshot_id, CatalogProvider, DataFileEntry, NewSnapshot,
    SnapshotOperation, TableIdent, VectorIndexInfo,
};
use ailake_core::{AilakeError, AilakeResult, EmbeddingModelInfo, VectorStoragePolicy};
use ailake_file::{AilakeFileReader, AilakeFileWriter};
use ailake_store::Store;
use ailake_vec::compute_centroid_and_radius;
use arrow_array::{Array, RecordBatch, StringArray};
use tracing::info;

pub type EmbedFn = Arc<dyn Fn(&[String]) -> AilakeResult<Vec<Vec<f32>>> + Send + Sync>;
pub type ProgressFn = Arc<dyn Fn(MigrationProgress) + Send + Sync>;

/// How files are replaced during migration.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MigrationStrategy {
    /// Write new files file-by-file, replacing each old file as it completes.
    /// Lower peak storage. During migration, some shards have old column, others new.
    AtomicReplace,
    /// Write all new files first (old files untouched), then commit a single Replace
    /// snapshot swapping all old files for new ones atomically.
    /// Higher peak storage (2× during migration), but reads always see a consistent view.
    DualWriteThenCutover,
}

/// Progress reported via `on_progress` callback.
#[derive(Debug, Clone)]
pub struct MigrationProgress {
    pub files_done: usize,
    pub files_total: usize,
    pub rows_migrated: u64,
}

/// Migrates embedding columns in an AI-Lake table to a new model.
///
/// Usage:
/// ```ignore
/// let job = MigrationJob {
///     table: TableIdent::new("default", "docs"),
///     old_column: "embedding".to_string(),
///     new_column: "embedding_v2".to_string(),
///     text_column: "chunk_text".to_string(),
///     embed_fn: Arc::new(|texts| Ok(my_model.encode(texts))),
///     strategy: MigrationStrategy::DualWriteThenCutover,
///     batch_size: 10_000,
///     new_model: Some(EmbeddingModelInfo::new("my-model-v2")),
///     on_progress: None,
/// };
/// job.run(catalog, store).await?;
/// ```
pub struct MigrationJob {
    pub table: TableIdent,
    /// Name of the embedding column to replace (e.g., "embedding").
    pub old_column: String,
    /// Name to give the new embedding column (e.g., "embedding_v2").
    /// Can be the same as `old_column` to do an in-place model upgrade.
    pub new_column: String,
    /// Column in the Parquet files that holds the text to re-embed.
    /// Defaults to `chunk_text` (the `LlmContextSchema` canonical name).
    pub text_column: String,
    /// Callable that converts a slice of texts to embeddings.
    /// Must return exactly `texts.len()` vectors, all of the same dimension.
    pub embed_fn: EmbedFn,
    pub strategy: MigrationStrategy,
    /// How many rows to embed per `embed_fn` call. Tune based on model batch size.
    pub batch_size: usize,
    /// Metadata for the new embedding model — stored in Iceberg properties after migration.
    pub new_model: Option<EmbeddingModelInfo>,
    /// Optional callback called after each file completes.
    pub on_progress: Option<ProgressFn>,
}

impl MigrationJob {
    pub async fn run(
        self,
        catalog: Arc<dyn CatalogProvider>,
        store: Arc<dyn Store>,
    ) -> AilakeResult<()> {
        match self.strategy {
            MigrationStrategy::AtomicReplace => self.run_atomic_replace(catalog, store).await,
            MigrationStrategy::DualWriteThenCutover => self.run_dual_write(catalog, store).await,
        }
    }

    /// AtomicReplace: process and commit each file one at a time.
    async fn run_atomic_replace(
        &self,
        catalog: Arc<dyn CatalogProvider>,
        store: Arc<dyn Store>,
    ) -> AilakeResult<()> {
        let table_meta = catalog.load_table(&self.table).await?;
        let old_files = catalog
            .list_files(&self.table, table_meta.current_snapshot_id)
            .await?;
        let total = old_files.len();
        let mut rows_migrated: u64 = 0;

        // Derive new policy from table properties + new model info
        let new_policy = self.new_policy_from_metadata(&table_meta.properties)?;

        let mut parent_snap = table_meta.current_snapshot_id;

        for (idx, old_entry) in old_files.iter().enumerate() {
            let (batch, texts) = self
                .read_file_texts(&old_entry.path, &store, &new_policy)
                .await?;
            let new_embeddings = self.embed_in_batches(&texts)?;

            let new_entry = self
                .write_new_file(&batch, &new_embeddings, &new_policy, &store, idx)
                .await?;

            rows_migrated += new_entry.record_count;

            let snap_id = new_snapshot_id();
            catalog
                .commit_snapshot(
                    &self.table,
                    NewSnapshot {
                        snapshot_id: snap_id,
                        parent_snapshot_id: parent_snap,
                        files: vec![new_entry],
                        operation: SnapshotOperation::Replace,
                        iceberg_schema: None,
                        extra_properties: std::collections::HashMap::new(),
                        bloom_filters: vec![],
                equality_delete_files: vec![],
                    },
                )
                .await?;
            parent_snap = Some(snap_id);

            if let Some(cb) = &self.on_progress {
                cb(MigrationProgress {
                    files_done: idx + 1,
                    files_total: total,
                    rows_migrated,
                });
            }

            info!(
                "ailake migration: AtomicReplace {}/{} files done, {} rows migrated",
                idx + 1,
                total,
                rows_migrated
            );
        }

        Ok(())
    }

    /// DualWriteThenCutover: write all new files first, then commit one Replace snapshot.
    async fn run_dual_write(
        &self,
        catalog: Arc<dyn CatalogProvider>,
        store: Arc<dyn Store>,
    ) -> AilakeResult<()> {
        let table_meta = catalog.load_table(&self.table).await?;
        let old_files = catalog
            .list_files(&self.table, table_meta.current_snapshot_id)
            .await?;
        let total = old_files.len();
        let mut rows_migrated: u64 = 0;

        let new_policy = self.new_policy_from_metadata(&table_meta.properties)?;
        let mut new_entries: Vec<DataFileEntry> = Vec::with_capacity(total);

        for (idx, old_entry) in old_files.iter().enumerate() {
            let (batch, texts) = self
                .read_file_texts(&old_entry.path, &store, &new_policy)
                .await?;
            let new_embeddings = self.embed_in_batches(&texts)?;

            let entry = self
                .write_new_file(&batch, &new_embeddings, &new_policy, &store, idx)
                .await?;

            rows_migrated += entry.record_count;
            new_entries.push(entry);

            if let Some(cb) = &self.on_progress {
                cb(MigrationProgress {
                    files_done: idx + 1,
                    files_total: total,
                    rows_migrated,
                });
            }

            info!(
                "ailake migration: DualWrite phase {}/{} files ready",
                idx + 1,
                total
            );
        }

        // Single atomic cutover: replace all old files with all new files.
        let snap_id = new_snapshot_id();
        catalog
            .commit_snapshot(
                &self.table,
                NewSnapshot {
                    snapshot_id: snap_id,
                    parent_snapshot_id: table_meta.current_snapshot_id,
                    files: new_entries,
                    operation: SnapshotOperation::Replace,
                    iceberg_schema: None,
                    extra_properties: std::collections::HashMap::new(),
                    bloom_filters: vec![],
                equality_delete_files: vec![],
                },
            )
            .await?;

        info!(
            "ailake migration: DualWriteThenCutover complete — {} files, {} rows",
            total, rows_migrated
        );
        Ok(())
    }

    /// Read Parquet bytes from store, decode the text column.
    async fn read_file_texts(
        &self,
        path: &str,
        store: &Arc<dyn Store>,
        policy: &VectorStoragePolicy,
    ) -> AilakeResult<(RecordBatch, Vec<String>)> {
        let bytes = store.get(path).await?;
        let reader = AilakeFileReader::new(bytes, &self.old_column, policy.dim);
        let (batch, _) = reader.read_parquet()?;

        let texts = extract_string_column(&batch, &self.text_column)?;
        Ok((batch, texts))
    }

    /// Call embed_fn in chunks of batch_size.
    fn embed_in_batches(&self, texts: &[String]) -> AilakeResult<Vec<Vec<f32>>> {
        let mut all: Vec<Vec<f32>> = Vec::with_capacity(texts.len());
        for chunk in texts.chunks(self.batch_size) {
            let mut chunk_vecs = (self.embed_fn)(chunk)?;
            all.append(&mut chunk_vecs);
        }
        Ok(all)
    }

    /// Write a new AI-Lake file with the re-embedded vectors, return its catalog entry.
    async fn write_new_file(
        &self,
        batch: &RecordBatch,
        embeddings: &[Vec<f32>],
        policy: &VectorStoragePolicy,
        store: &Arc<dyn Store>,
        idx: usize,
    ) -> AilakeResult<DataFileEntry> {
        let file_path = format!("data/migrated-{:05}.parquet", idx);

        let writer = AilakeFileWriter::new(policy.clone());
        let file_bytes = writer.write(batch, embeddings)?;
        let file_size = file_bytes.len() as u64;

        store.put(&file_path, file_bytes.clone()).await?;

        let centroid = compute_centroid_and_radius(embeddings, policy.metric);
        let reader = AilakeFileReader::new(file_bytes, &policy.column_name, policy.dim);
        let header = reader.read_header()?;
        let ailk_start = reader.ailk_offset()?;
        let hnsw_abs = ailk_start + header.hnsw_offset;

        Ok(make_data_file_entry(
            &file_path,
            embeddings.len() as u64,
            file_size,
            &centroid,
            VectorIndexInfo {
                column: &policy.column_name,
                dim: policy.dim,
                hnsw_offset: hnsw_abs,
                hnsw_len: header.hnsw_len,
            },
        ))
    }

    /// Build the new `VectorStoragePolicy` from existing table properties,
    /// overriding the column name and embedding model.
    fn new_policy_from_metadata(
        &self,
        props: &std::collections::HashMap<String, String>,
    ) -> AilakeResult<VectorStoragePolicy> {
        use ailake_core::{VectorMetric, VectorPrecision};

        let dim: u32 = props
            .get("ailake.vector-dim")
            .and_then(|s| s.parse().ok())
            .ok_or_else(|| {
                AilakeError::InvalidArgument("table missing ailake.vector-dim property".into())
            })?;

        let metric = match props
            .get("ailake.vector-metric")
            .map(|s| s.as_str())
            .unwrap_or("cosine")
        {
            "euclidean" => VectorMetric::Euclidean,
            "dotproduct" | "dot_product" => VectorMetric::DotProduct,
            "normalizedcosine" | "normalized_cosine" => VectorMetric::NormalizedCosine,
            _ => VectorMetric::Cosine,
        };

        let precision = match props
            .get("ailake.vector-precision")
            .map(|s| s.as_str())
            .unwrap_or("f16")
        {
            "f32" => VectorPrecision::F32,
            "i8" => VectorPrecision::I8,
            _ => VectorPrecision::F16,
        };

        Ok(VectorStoragePolicy {
            column_name: self.new_column.clone(),
            dim,
            metric,
            precision,
            pq: None,
            keep_raw_for_reranking: true,
            pre_normalize: props
                .get("ailake.pre-normalize")
                .map(|s| s == "true")
                .unwrap_or(false),
            hnsw_m: props.get("ailake.hnsw-m").and_then(|s| s.parse().ok()),
            hnsw_ef_construction: props
                .get("ailake.hnsw-ef-construction")
                .and_then(|s| s.parse().ok()),
            ivf_residual: false,
            embedding_model: self.new_model.clone(),
            modality: None,
            partition_by: None,
            partition_value: None,
            partition_column_type: None,
                partition_fields: vec![],
})
    }
}

fn extract_string_column(batch: &RecordBatch, column_name: &str) -> AilakeResult<Vec<String>> {
    let col = batch.column_by_name(column_name).ok_or_else(|| {
        AilakeError::InvalidArgument(format!(
            "text column '{}' not found in Parquet file; available: {}",
            column_name,
            batch
                .schema()
                .fields()
                .iter()
                .map(|f| f.name().as_str())
                .collect::<Vec<_>>()
                .join(", ")
        ))
    })?;

    let arr = col.as_any().downcast_ref::<StringArray>().ok_or_else(|| {
        AilakeError::InvalidArgument(format!(
            "column '{}' is not a Utf8/String column",
            column_name
        ))
    })?;

    Ok((0..arr.len())
        .map(|i| {
            if arr.is_null(i) {
                String::new()
            } else {
                arr.value(i).to_string()
            }
        })
        .collect())
}
