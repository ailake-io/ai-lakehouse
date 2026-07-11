// SPDX-License-Identifier: MIT OR Apache-2.0
//! Backfill job: adds a new vector column to existing files in an AI-Lake table.
//!
//! Reads each file, generates embeddings for the new column via `embed_fn`, and
//! rewrites the file with both the original vector column and the new one (using
//! `write_multi`). Commits an Overwrite snapshot per file (AtomicReplace semantics).
//!
//! Idempotent: files that already contain the new column (detected via
//! `has_column_footer`) are skipped.

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
        // Running view of every file currently in the table. `Overwrite` does not
        // inherit the previous manifest (same contract as `Replace` — see
        // `HadoopCatalog::commit_snapshot`), so each commit below must carry the
        // complete current state, not just the one file that changed this iteration,
        // or every file backfilled (or not yet reached) in a prior iteration would
        // vanish from the table on this commit.
        let mut current_files = files.clone();

        for (idx, entry) in files.iter().enumerate() {
            let file_bytes = store.get(&entry.path).await?;

            // Idempotency: skip if new column already has its own AILK section.
            // `has_column_footer` checks the per-column KV key directly — unlike
            // `ailk_offset_for_column`, it doesn't fall back to the primary column's
            // footer, which would make every AI-Lake file look like it already has
            // the new column and skip backfilling entirely.
            let reader = AilakeFileReader::new(
                file_bytes.clone(),
                &primary_policy.column_name,
                primary_policy.dim,
            );
            if reader.has_column_footer(&self.new_col.column_name) {
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

            // Drop DV-masked rows before re-embedding — the backfilled file is brand-new,
            // so a deleted row must not get a fresh new-column embedding and resurrect.
            let (batch, texts, primary_embeddings) = if let Some(dv) = &entry.deletion_vector {
                let bitmap = crate::dv::load_deletion_vector(&store, dv).await?;
                let combined: Vec<(String, Vec<f32>)> =
                    texts.into_iter().zip(primary_embeddings).collect();
                let (batch, combined) = crate::dv::filter_deleted_rows(batch, combined, &bitmap)?;
                let (texts, primary_embeddings): (Vec<String>, Vec<Vec<f32>>) =
                    combined.into_iter().unzip();
                (batch, texts, primary_embeddings)
            } else {
                (batch, texts, primary_embeddings)
            };

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

            // Swap this file's entry in place; every other file (already backfilled in
            // a prior iteration, skipped as idempotent, or not yet reached) is carried
            // forward unchanged.
            current_files[idx] = new_entry;

            // Commit Overwrite snapshot: replaces old file with new multi-column file,
            // carrying forward the full current file list (see comment above the loop).
            let snap_id = new_snapshot_id();
            catalog
                .commit_snapshot(
                    &self.table,
                    NewSnapshot {
                        snapshot_id: snap_id,
                        parent_snapshot_id: parent_snap,
                        files: current_files.clone(),
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
    // Timestamped so a second backfill run (another column, or a retry) never
    // reuses a path from an earlier run — the old plain-index name made run 2
    // overwrite the committed backfill-00000 it was itself reading, an in-place
    // rewrite of a live file (breaks readers holding the old entry; hard error
    // under catalogs with supports_in_place_rewrite() == false).
    let file_path = format!(
        "data/backfill-{}-{:05}.parquet",
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_else(|e| e.duration())
            .as_millis(),
        idx
    );

    let writer = AilakeFileWriter::new(primary_policy.clone());
    let file_bytes = writer.write_multi(
        batch,
        &[
            VectorColumnBatch {
                policy: primary_policy,
                embeddings: primary_embeddings,
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

    let primary_centroid = compute_centroid_and_radius(primary_embeddings, primary_policy.metric);
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

#[cfg(test)]
mod tests {
    use super::*;
    use ailake_catalog::{HadoopCatalog, TableProperties};
    use ailake_core::{VectorMetric, VectorPrecision};
    use ailake_store::LocalStore;
    use arrow_array::Int32Array;
    use arrow_schema::{DataType, Field, Schema};
    use tempfile::TempDir;

    fn make_policy(dim: u32) -> VectorStoragePolicy {
        VectorStoragePolicy {
            column_name: "embedding".into(),
            dim,
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
            partition_fields: vec![],
        }
    }

    /// Regression test: `BackfillJob::run` used to commit `SnapshotOperation::Overwrite`
    /// with `files: vec![new_entry]` per loop iteration. `Overwrite` doesn't inherit the
    /// previous manifest (same contract as `Replace` — see
    /// `hadoop.rs::replace_does_not_inherit_previous_manifest`), so every iteration after
    /// the first silently discarded every file backfilled by prior iterations. Uses 3
    /// files specifically because the bug is invisible with only 1 (replacing "the only
    /// file" with a partial list is coincidentally correct).
    #[tokio::test]
    async fn run_preserves_all_files_not_just_the_last() {
        let dir = TempDir::new().unwrap();
        let store: Arc<dyn Store> = Arc::new(LocalStore::new(dir.path()));
        let catalog_dir = TempDir::new().unwrap();
        let catalog_store = Arc::new(LocalStore::new(catalog_dir.path()));
        let catalog: Arc<dyn CatalogProvider> = Arc::new(HadoopCatalog::new(catalog_store, ""));
        let table = TableIdent::new("ns", "tbl");

        let dim = 4u32;
        let policy = make_policy(dim);
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

        let schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int32, false),
            Field::new("chunk_text", DataType::Utf8, false),
        ]));

        // Three files — each its own commit, so list_files() returns 3 entries.
        let mut parent_snap = None;
        for (i, (ids, texts)) in [
            (vec![0i32, 1], vec!["a0", "a1"]),
            (vec![2, 3], vec!["b0", "b1"]),
            (vec![4, 5], vec!["c0", "c1"]),
        ]
        .into_iter()
        .enumerate()
        {
            let embs: Vec<Vec<f32>> = ids.iter().map(|&v| vec![v as f32; dim as usize]).collect();
            let batch = RecordBatch::try_new(
                schema.clone(),
                vec![
                    Arc::new(Int32Array::from(ids.clone())),
                    Arc::new(StringArray::from(texts)),
                ],
            )
            .unwrap();
            let bytes = AilakeFileWriter::new(policy.clone())
                .write(&batch, &embs)
                .unwrap();
            let path = format!("data/old_{i}.parquet");
            store.put(&path, bytes.clone()).await.unwrap();

            let centroid = compute_centroid_and_radius(&embs, VectorMetric::Cosine);
            let reader = AilakeFileReader::new(bytes.clone(), "embedding", dim);
            let header = reader.read_header().unwrap();
            let ailk_start = reader.ailk_offset().unwrap();
            let entry = ailake_catalog::make_data_file_entry(
                &path,
                ids.len() as u64,
                bytes.len() as u64,
                &centroid,
                VectorIndexInfo {
                    column: "embedding",
                    dim,
                    hnsw_offset: ailk_start + header.hnsw_offset,
                    hnsw_len: header.hnsw_len,
                },
            );
            let snap_id = new_snapshot_id();
            catalog
                .commit_snapshot(
                    &table,
                    NewSnapshot {
                        snapshot_id: snap_id,
                        parent_snapshot_id: parent_snap,
                        files: vec![entry],
                        operation: SnapshotOperation::Append,
                        iceberg_schema: None,
                        extra_properties: std::collections::HashMap::new(),
                        bloom_filters: vec![],
                        equality_delete_files: vec![],
                    },
                )
                .await
                .unwrap();
            parent_snap = Some(snap_id);
        }

        let files_before = catalog.list_files(&table, None).await.unwrap();
        assert_eq!(
            files_before.len(),
            3,
            "sanity: 3 files committed via Append"
        );

        let job = BackfillJob {
            table: table.clone(),
            text_column: "chunk_text".into(),
            new_col: ailake_core::VectorColSpec {
                column_name: "embedding_v2".into(),
                dim,
                metric: VectorMetric::Cosine,
                precision: VectorPrecision::F16,
                pre_normalize: false,
                hnsw_m: None,
                hnsw_ef_construction: None,
            },
            embed_fn: Arc::new(|texts: &[String]| {
                Ok(texts.iter().map(|_| vec![9.0f32; 4]).collect())
            }),
            batch_size: 10,
            on_progress: None,
        };
        job.run(catalog.clone(), store.clone()).await.unwrap();

        let files_after = catalog.list_files(&table, None).await.unwrap();
        assert_eq!(
            files_after.len(),
            3,
            "BUG: expected all 3 backfilled files to remain visible, got {:?}",
            files_after.iter().map(|f| &f.path).collect::<Vec<_>>()
        );
        let total_rows: u64 = files_after.iter().map(|f| f.record_count).sum();
        assert_eq!(total_rows, 6, "all 6 original rows must survive backfill");

        // Every file must actually have been rewritten (not skipped as a false-positive
        // "already has the column" idempotency match) and be independently readable
        // with both columns present.
        for entry in &files_after {
            assert!(
                entry.path.starts_with("data/backfill-"),
                "BUG: {} was never backfilled (idempotency check false-skipped it)",
                entry.path
            );
            let bytes = store.get(&entry.path).await.unwrap();
            let reader = AilakeFileReader::new(bytes, "embedding", dim);
            assert!(
                reader.has_column_footer("embedding_v2"),
                "file {} missing backfilled column",
                entry.path
            );
            let (batch, _) = reader.read_parquet().unwrap();
            assert_eq!(batch.num_rows(), 2);
        }
    }
}
