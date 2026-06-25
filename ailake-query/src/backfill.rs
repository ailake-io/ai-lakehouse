// SPDX-License-Identifier: MIT OR Apache-2.0
//! Backfill job: adds a new vector column to existing files in an AI-Lake table.
//!
//! Reads each file, generates embeddings for the new column via `embed_fn`, and
//! rewrites the file with both the original vector column and the new one (using
//! `write_multi`). Commits an Overwrite snapshot per file (AtomicReplace semantics).
//!
//! Idempotent: files that already contain the new column (detected via
//! `ailk_offset_for_column`) are skipped.

use std::sync::Arc;

use ailake_catalog::{
    encode_centroid_b64, make_multi_column_data_file_entry, new_snapshot_id,
    provider::{CatalogProvider, DataFileEntry, ExtraVectorIndex, NewSnapshot, SnapshotOperation},
    TableIdent, VectorIndexInfo,
};
use ailake_core::{AilakeError, AilakeResult, VectorColSpec, VectorStoragePolicy};
use ailake_file::{AilakeFileReader, AilakeFileWriter, VectorColumnBatch};
use ailake_store::Store;
use ailake_vec::compute_centroid_and_radius;
use arrow_array::{Array, RecordBatch, StringArray};
use bytes::Bytes;
use tracing::info;

pub use crate::migration::EmbedFn;

pub type BackfillProgressFn = Arc<dyn Fn(BackfillProgress) + Send + Sync>;

/// Progress reported after each file is backfilled.
#[derive(Debug, Clone)]
pub struct BackfillProgress {
    pub files_done: usize,
    pub files_total: usize,
    pub files_skipped: usize,
    pub rows_backfilled: u64,
}

/// Adds a new vector column to all existing files in a table.
///
/// Does not touch files that already have the column (idempotent).
/// Concurrent new writes (after `add_vector_column` was called) already include
/// the column and are also skipped.
pub struct BackfillJob {
    pub table: TableIdent,
    /// Column in the Parquet files that holds the text to embed.
    pub text_column: String,
    /// Specification for the new vector column.
    pub new_col: VectorColSpec,
    /// Callable: given a slice of text strings, returns one F32 vector per text.
    pub embed_fn: EmbedFn,
    /// How many texts to embed per `embed_fn` call.
    pub batch_size: usize,
    /// Optional progress callback.
    pub on_progress: Option<BackfillProgressFn>,
}

impl BackfillJob {
    pub async fn run(
        self,
        catalog: Arc<dyn CatalogProvider>,
        store: Arc<dyn Store>,
    ) -> AilakeResult<()> {
        let table_meta = catalog.load_table(&self.table).await?;
        let files = catalog
            .list_files(&self.table, table_meta.current_snapshot_id)
            .await?;
        let total = files.len();
        let mut rows_backfilled: u64 = 0;
        let mut files_skipped: usize = 0;

        // Build policy for the new column from the VectorColSpec.
        let new_policy = VectorStoragePolicy {
            column_name: self.new_col.column_name.clone(),
            dim: self.new_col.dim,
            metric: self.new_col.metric,
            precision: self.new_col.precision,
            pre_normalize: self.new_col.pre_normalize,
            hnsw_m: self.new_col.hnsw_m,
            hnsw_ef_construction: self.new_col.hnsw_ef_construction,
            pq: None,
            keep_raw_for_reranking: true,
            ivf_residual: false,
            embedding_model: None,
            modality: None,
            partition_by: None,
            partition_value: None,
            partition_column_type: None,
            partition_fields: vec![],
        };

        // Derive primary policy from table properties.
        let primary_policy = primary_policy_from_props(&table_meta.properties)?;

        let mut parent_snap = table_meta.current_snapshot_id;

        for (idx, entry) in files.iter().enumerate() {
            let file_bytes = store.get(&entry.path).await?;

            // Idempotency: skip if new column already has an AILK section.
            let reader =
                AilakeFileReader::new(file_bytes.clone(), &primary_policy.column_name, primary_policy.dim);
            if reader
                .ailk_offset_for_column(&self.new_col.column_name)
                .is_ok()
            {
                files_skipped += 1;
                info!(
                    "ailake backfill: skipping {} — column '{}' already present ({}/{})",
                    entry.path,
                    self.new_col.column_name,
                    idx + 1,
                    total
                );
                continue;
            }

            // Read Parquet data, text column, and existing primary embeddings.
            let (batch, texts, primary_embeddings) = read_batch_texts_and_embeddings(
                file_bytes,
                &primary_policy.column_name,
                primary_policy.dim,
                &self.text_column,
            )?;

            // Generate embeddings for new column in batches.
            let new_embeddings = embed_in_batches(&self.embed_fn, &texts, self.batch_size)?;

            // Write new file with both columns.
            let new_entry = write_backfilled_file(
                &batch,
                &primary_embeddings,
                &new_embeddings,
                &primary_policy,
                &new_policy,
                &store,
                idx,
            )
            .await?;

            rows_backfilled += new_entry.record_count;

            // Commit Overwrite snapshot: replaces old file with new multi-column file.
            let snap_id = new_snapshot_id();
            catalog
                .commit_snapshot(
                    &self.table,
                    NewSnapshot {
                        snapshot_id: snap_id,
                        parent_snapshot_id: parent_snap,
                        files: vec![new_entry],
                        operation: SnapshotOperation::Overwrite,
                        iceberg_schema: None,
                        extra_properties: std::collections::HashMap::new(),
                        bloom_filters: vec![],
                        equality_delete_files: vec![],
                    },
                )
                .await?;
            parent_snap = Some(snap_id);

            if let Some(cb) = &self.on_progress {
                cb(BackfillProgress {
                    files_done: idx + 1 - files_skipped,
                    files_total: total,
                    files_skipped,
                    rows_backfilled,
                });
            }

            info!(
                "ailake backfill: {}/{} files done ({} skipped), {} rows",
                idx + 1,
                total,
                files_skipped,
                rows_backfilled
            );
        }

        info!(
            "ailake backfill complete — column='{}', files={}, skipped={}, rows={}",
            self.new_col.column_name,
            total - files_skipped,
            files_skipped,
            rows_backfilled
        );
        Ok(())
    }
}

fn read_batch_texts_and_embeddings(
    bytes: Bytes,
    vector_column: &str,
    dim: u32,
    text_column: &str,
) -> AilakeResult<(RecordBatch, Vec<String>, Vec<Vec<f32>>)> {
    let reader = AilakeFileReader::new(bytes, vector_column, dim);
    let (batch, primary_embeddings) = reader.read_parquet()?;

    let col = batch.column_by_name(text_column).ok_or_else(|| {
        AilakeError::InvalidArgument(format!(
            "text column '{}' not found; available: {}",
            text_column,
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
        AilakeError::InvalidArgument(format!("column '{text_column}' is not a String column"))
    })?;

    let texts: Vec<String> = (0..arr.len())
        .map(|i| {
            if arr.is_null(i) {
                String::new()
            } else {
                arr.value(i).to_string()
            }
        })
        .collect();

    Ok((batch, texts, primary_embeddings))
}

fn embed_in_batches(
    embed_fn: &EmbedFn,
    texts: &[String],
    batch_size: usize,
) -> AilakeResult<Vec<Vec<f32>>> {
    let mut all: Vec<Vec<f32>> = Vec::with_capacity(texts.len());
    for chunk in texts.chunks(batch_size) {
        let mut vecs = embed_fn(chunk)?;
        all.append(&mut vecs);
    }
    Ok(all)
}

async fn write_backfilled_file(
    batch: &RecordBatch,
    primary_embeddings: &[Vec<f32>],
    new_embeddings: &[Vec<f32>],
    primary_policy: &VectorStoragePolicy,
    new_policy: &VectorStoragePolicy,
    store: &Arc<dyn Store>,
    idx: usize,
) -> AilakeResult<DataFileEntry> {

    let file_path = format!("data/backfill-{:05}.parquet", idx);

    let writer = AilakeFileWriter::new(primary_policy.clone());
    let file_bytes = writer.write_multi(
        batch,
        &[
            VectorColumnBatch {
                policy: primary_policy,
                embeddings: &primary_embeddings,
            },
            VectorColumnBatch {
                policy: new_policy,
                embeddings: new_embeddings,
            },
        ],
    )?;
    let file_size = file_bytes.len() as u64;
    store.put(&file_path, file_bytes.clone()).await?;

    // Read back AILK offsets for both columns using read_header_for_column.
    let reader = AilakeFileReader::new(file_bytes, &primary_policy.column_name, primary_policy.dim);
    let primary_ailk_offset = reader.ailk_offset()?;
    let primary_header = reader.read_header()?;
    let primary_hnsw_abs = primary_ailk_offset + primary_header.hnsw_offset;

    let new_ailk_offset = reader.ailk_offset_for_column(&new_policy.column_name)?;
    let new_header = reader.read_header_for_column(&new_policy.column_name)?;
    let new_hnsw_abs = new_ailk_offset + new_header.hnsw_offset;

    let primary_centroid = compute_centroid_and_radius(&primary_embeddings, primary_policy.metric);
    let new_centroid = compute_centroid_and_radius(new_embeddings, new_policy.metric);

    let extra = vec![ExtraVectorIndex {
        column: new_policy.column_name.clone(),
        dim: new_policy.dim,
        hnsw_offset: new_hnsw_abs,
        hnsw_len: new_header.hnsw_len,
        centroid_b64: Some(encode_centroid_b64(&new_centroid)),
        radius: Some(new_centroid.radius),
    }];

    Ok(make_multi_column_data_file_entry(
        &file_path,
        new_embeddings.len() as u64,
        file_size,
        &primary_centroid,
        VectorIndexInfo {
            column: &primary_policy.column_name,
            dim: primary_policy.dim,
            hnsw_offset: primary_hnsw_abs,
            hnsw_len: primary_header.hnsw_len,
        },
        &extra,
    ))
}

fn primary_policy_from_props(
    props: &std::collections::HashMap<String, String>,
) -> AilakeResult<VectorStoragePolicy> {
    use ailake_core::{VectorMetric, VectorPrecision};

    let column_name = props
        .get("ailake.vector-column")
        .cloned()
        .unwrap_or_else(|| "embedding".to_string());

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
        column_name,
        dim,
        metric,
        precision,
        pre_normalize: props
            .get("ailake.pre-normalize")
            .map(|s| s == "true")
            .unwrap_or(false),
        hnsw_m: props.get("ailake.hnsw-m").and_then(|s| s.parse().ok()),
        hnsw_ef_construction: props
            .get("ailake.hnsw-ef-construction")
            .and_then(|s| s.parse().ok()),
        pq: None,
        keep_raw_for_reranking: true,
        ivf_residual: false,
        embedding_model: None,
        modality: None,
        partition_by: None,
        partition_value: None,
        partition_column_type: None,
        partition_fields: vec![],
    })
}
