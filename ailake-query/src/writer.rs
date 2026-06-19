// SPDX-License-Identifier: MIT OR Apache-2.0
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::Arc;

use ailake_catalog::{
    encode_centroid_b64, make_data_file_entry, make_data_file_entry_indexing,
    make_multi_column_data_file_entry, new_snapshot_id, CatalogProvider, DataFileEntry,
    ExtraVectorIndex, IcebergSchemaUpdate, IndexStatus, NewSnapshot, SnapshotId, SnapshotOperation,
    TableIdent, TableProperties, VectorIndexInfo,
};
use ailake_core::{AilakeError, AilakeResult, EmbeddingModelInfo, VectorStoragePolicy};
use ailake_file::{AilakeFileReader, AilakeFileWriter, IndexType, VectorColumnBatch};
use ailake_index::{IvfPqCodebook, IvfPqConfig};
use ailake_store::Store;
use ailake_vec::compute_centroid_and_radius;
use arrow_array::Array;
use arrow_array::RecordBatch;
use arrow_schema::SchemaRef;
use bytes::Bytes;
use serde_json;
use tracing::{error, info, warn};

/// Apply partition transforms and return the final stored value.
/// For multi-column specs, raw must be \x1f-separated; each part is transformed
/// independently and the result is rejoined with \x1f.
/// For single-column (partition_by path), raw is returned as-is (identity only).
fn apply_partition_transforms(policy: &VectorStoragePolicy, raw: Option<&str>) -> Option<String> {
    let raw = raw?;
    if policy.partition_fields.is_empty() {
        return Some(raw.to_string());
    }
    let parts: Vec<&str> = raw.split('\x1f').collect();
    let transformed: Vec<String> = policy
        .partition_fields
        .iter()
        .enumerate()
        .map(|(i, pf)| {
            let v = parts.get(i).copied().unwrap_or("");
            pf.apply(v)
        })
        .collect();
    Some(transformed.join("\x1f"))
}

/// One vector column for a multi-column write batch.
pub struct MultiVectorBatch<'a> {
    pub policy: VectorStoragePolicy,
    pub embeddings: &'a [Vec<f32>],
}

pub struct TableWriter {
    catalog: Arc<dyn CatalogProvider>,
    store: Arc<dyn Store>,
    policy: VectorStoragePolicy,
    table: TableIdent,
    part_counter: Arc<AtomicU32>,
    pending_files: Vec<DataFileEntry>,
    parent_snapshot_id: Option<SnapshotId>,
    /// Arrow schema captured from the first write_batch call; used to populate
    /// Iceberg schema fields and schema.name-mapping.default on commit.
    captured_schema: Option<SchemaRef>,
    /// Extra vector column policies from write_batch_multi (columns beyond primary).
    extra_vec_policies: Vec<VectorStoragePolicy>,
    /// IVF-PQ codebook trained on the first shard and reused for all subsequent shards.
    /// Ensures cross-shard ADC distances are comparable — no reranking needed.
    cached_ivf_codebook: Option<Arc<IvfPqCodebook>>,
    /// Shared codebook cell for deferred IVF-PQ builds. Cloneable Arc so each
    /// background task can access it; OnceCell guarantees training runs exactly once.
    deferred_ivf_codebook: Arc<tokio::sync::OnceCell<IvfPqCodebook>>,
    /// When set, BM25 IDF stats are accumulated from this Parquet column on each
    /// write_batch call and persisted to `metadata/ailake_bm25_stats.bin`.
    /// Enables hybrid vector+BM25 search via `SearchConfig::hybrid`.
    bm25_text_column: Option<String>,
    /// Per-file Bloom filters built during write_batch when bm25_text_column is set.
    /// Flushed to NewSnapshot::bloom_filters on commit (Phase F Puffin stats).
    pending_blooms: Vec<(String, Vec<u8>)>,
}

impl TableWriter {
    pub fn new(
        catalog: Arc<dyn CatalogProvider>,
        store: Arc<dyn Store>,
        policy: VectorStoragePolicy,
        table: TableIdent,
    ) -> Self {
        Self {
            catalog,
            store,
            policy,
            table,
            part_counter: Arc::new(AtomicU32::new(0)),
            pending_files: Vec::new(),
            parent_snapshot_id: None,
            captured_schema: None,
            extra_vec_policies: Vec::new(),
            cached_ivf_codebook: None,
            deferred_ivf_codebook: Arc::new(tokio::sync::OnceCell::new()),
            bm25_text_column: None,
            pending_blooms: Vec::new(),
        }
    }

    /// Enable BM25 hybrid search by accumulating IDF stats from `column` on each write.
    ///
    /// After calling this, every `write_batch*` call will tokenize the specified column,
    /// update the corpus IDF stats, and persist them to `metadata/ailake_bm25_stats.bin`.
    /// This file is then loaded automatically by `SearchConfig::hybrid` at query time.
    ///
    /// Typical usage: `TableWriter::new(...).with_bm25("chunk_text")`.
    pub fn with_bm25(mut self, text_column: impl Into<String>) -> Self {
        self.bm25_text_column = Some(text_column.into());
        self
    }

    pub fn with_parent_snapshot(mut self, id: SnapshotId) -> Self {
        self.parent_snapshot_id = Some(id);
        self
    }

    /// Write batch as Parquet-only immediately, build HNSW in background.
    ///
    /// Returns after the Parquet file is persisted (~LanceDB write speed).
    /// A tokio task runs concurrently to build the HNSW index, rewrite the
    /// file with the AILK section, and update the catalog entry.
    ///
    /// During the build window, `SearchSession` serves this shard via flat scan
    /// (brute-force, exact) instead of HNSW. The transition is automatic once
    /// the background task commits the updated manifest entry.
    pub async fn write_batch_deferred(
        &mut self,
        batch: &RecordBatch,
        embeddings: &[Vec<f32>],
    ) -> AilakeResult<()> {
        self.validate_embedding_dim(embeddings)?;
        if self.captured_schema.is_none() {
            self.captured_schema = Some(batch.schema());
        }
        let part_num = self.part_counter.fetch_add(1, Ordering::SeqCst);
        let file_path = format!("data/part-{:05}.parquet", part_num);

        // Fast path: persist Parquet without HNSW.
        let file_writer = AilakeFileWriter::new(self.policy.clone());
        let parquet_bytes = file_writer.write_parquet_only(batch, embeddings)?;
        let file_size = parquet_bytes.len() as u64;
        self.store.put(&file_path, parquet_bytes).await?;

        // Centroid needed immediately for geometric pruning during the build window.
        let centroid = compute_centroid_and_radius(embeddings, self.policy.metric);
        let mut entry = make_data_file_entry_indexing(
            &file_path,
            embeddings.len() as u64,
            file_size,
            &centroid,
            &self.policy.column_name,
            self.policy.dim,
        );
        entry.embedding_model = self
            .policy
            .embedding_model
            .as_ref()
            .map(|m| m.to_property_value());
        entry.partition_value =
            apply_partition_transforms(&self.policy, self.policy.partition_value.as_deref());
        self.pending_files.push(entry);

        // Spawn background HNSW build (fire-and-forget; errors are logged).
        let store = self.store.clone();
        let catalog = self.catalog.clone();
        let policy = self.policy.clone();
        let table = self.table.clone();
        let fp = file_path.clone();
        tokio::spawn(async move {
            if let Err(e) = build_and_patch_index(store, catalog, policy, table, fp).await {
                error!(
                    "ailake: deferred HNSW build failed — file is indexed as Parquet-only until \
                     next compaction rebuilds the index: {}",
                    e
                );
            }
        });

        // Update BM25 IDF stats + build Bloom filter (Phase F) for the new file.
        if self.bm25_text_column.is_some() {
            self.update_bm25_stats_from_batch(batch).await?;
            self.build_bloom_for_file(batch, &file_path);
        }

        Ok(())
    }

    /// Write batch as Parquet-only immediately; train IVF-PQ index in background.
    ///
    /// The first shard trains the shared codebook (k-means). All subsequent shards
    /// reuse it via `OnceCell` — build is O(n) assign+encode, not O(n×k) k-means.
    /// Returns after Parquet is persisted. Index transitions Indexing → Ready async.
    pub async fn write_batch_ivf_pq_deferred(
        &mut self,
        batch: &RecordBatch,
        embeddings: &[Vec<f32>],
        ivf_config: IvfPqConfig,
    ) -> AilakeResult<()> {
        if self.captured_schema.is_none() {
            self.captured_schema = Some(batch.schema());
        }
        let part_num = self.part_counter.fetch_add(1, Ordering::SeqCst);
        let file_path = format!("data/part-{:05}.parquet", part_num);

        let file_writer = AilakeFileWriter::new(self.policy.clone());
        let parquet_bytes = file_writer.write_parquet_only(batch, embeddings)?;
        let file_size = parquet_bytes.len() as u64;
        self.store.put(&file_path, parquet_bytes).await?;

        let centroid = compute_centroid_and_radius(embeddings, self.policy.metric);
        let mut entry = make_data_file_entry_indexing(
            &file_path,
            embeddings.len() as u64,
            file_size,
            &centroid,
            &self.policy.column_name,
            self.policy.dim,
        );
        entry.embedding_model = self
            .policy
            .embedding_model
            .as_ref()
            .map(|m| m.to_property_value());
        entry.partition_value =
            apply_partition_transforms(&self.policy, self.policy.partition_value.as_deref());
        self.pending_files.push(entry);

        let store = self.store.clone();
        let catalog = self.catalog.clone();
        let policy = self.policy.clone();
        let table = self.table.clone();
        let fp = file_path.clone();
        let codebook_cell = self.deferred_ivf_codebook.clone();
        tokio::spawn(async move {
            if let Err(e) = build_ivf_pq_and_patch_index(
                store,
                catalog,
                policy,
                table,
                fp,
                ivf_config,
                codebook_cell,
            )
            .await
            {
                error!(
                    "ailake: deferred IVF-PQ build failed — file is indexed as Parquet-only until \
                     next compaction rebuilds the index: {}",
                    e
                );
            }
        });

        Ok(())
    }

    /// Idempotent variant of `write_batch`.
    ///
    /// Before any I/O, checks if `batch_id` already appears in the current
    /// snapshot. If it does, this is a no-op — safe for Airflow/Kestra retries.
    /// If not found, writes the batch and tags the `DataFileEntry` with `batch_id`
    /// so future retries can detect it.
    ///
    /// `commit()` is likewise a no-op when `pending_files` is empty.
    pub async fn write_batch_idempotent(
        &mut self,
        batch: &RecordBatch,
        embeddings: &[Vec<f32>],
        batch_id: &str,
    ) -> AilakeResult<()> {
        let existing = self.catalog.list_files(&self.table, None).await?;
        if existing
            .iter()
            .any(|f| f.batch_id.as_deref() == Some(batch_id))
        {
            return Ok(());
        }
        self.write_batch_with_id(batch, embeddings, Some(batch_id.to_string()))
            .await
    }

    /// Write a batch to a new AI-Lake file and stage it for commit.
    /// Validates that provided embeddings match the table's configured dimension.
    /// Returns `ModelMismatch` error when dim differs — prevents silently mixing
    /// incompatible vectors (same error type used across write paths for consistency).
    fn validate_embedding_dim(&self, embeddings: &[Vec<f32>]) -> AilakeResult<()> {
        if let Some(first) = embeddings.first() {
            let actual = first.len() as u32;
            if actual != self.policy.dim {
                let table_model = self
                    .policy
                    .embedding_model
                    .as_ref()
                    .map(|m| m.to_property_value())
                    .unwrap_or_else(|| format!("dim={}", self.policy.dim));
                return Err(AilakeError::ModelMismatch {
                    table_model,
                    table_dim: self.policy.dim,
                    batch_model: format!("dim={}", actual),
                    batch_dim: actual,
                });
            }
        }
        Ok(())
    }

    pub async fn write_batch(
        &mut self,
        batch: &RecordBatch,
        embeddings: &[Vec<f32>],
    ) -> AilakeResult<()> {
        self.write_batch_with_id(batch, embeddings, None).await
    }

    async fn write_batch_with_id(
        &mut self,
        batch: &RecordBatch,
        embeddings: &[Vec<f32>],
        batch_id: Option<String>,
    ) -> AilakeResult<()> {
        self.validate_embedding_dim(embeddings)?;
        if self.captured_schema.is_none() {
            self.captured_schema = Some(batch.schema());
        }
        let part_num = self.part_counter.fetch_add(1, Ordering::SeqCst);
        let file_path = format!("data/part-{:05}.parquet", part_num);

        // Write AI-Lake file
        let file_writer = AilakeFileWriter::new(self.policy.clone());
        let file_bytes: Bytes = file_writer.write(batch, embeddings)?;
        let file_size = file_bytes.len() as u64;

        // Store the file
        self.store.put(&file_path, file_bytes.clone()).await?;

        // Compute centroid for catalog entry
        let centroid = compute_centroid_and_radius(embeddings, self.policy.metric);

        // Read back the HNSW offsets from the written file
        let reader = ailake_file::AilakeFileReader::new(
            file_bytes,
            &self.policy.column_name,
            self.policy.dim,
        );
        let header = reader.read_header()?;
        let ailk_start = reader.ailk_offset()?;
        let hnsw_abs_offset = ailk_start + header.hnsw_offset;
        let hnsw_len = header.hnsw_len;

        let mut entry = make_data_file_entry(
            &file_path,
            embeddings.len() as u64,
            file_size,
            &centroid,
            VectorIndexInfo {
                column: &self.policy.column_name,
                dim: self.policy.dim,
                hnsw_offset: hnsw_abs_offset,
                hnsw_len,
            },
        );
        entry.batch_id = batch_id;
        entry.embedding_model = self
            .policy
            .embedding_model
            .as_ref()
            .map(|m| m.to_property_value());
        entry.partition_value =
            apply_partition_transforms(&self.policy, self.policy.partition_value.as_deref());
        self.pending_files.push(entry);

        // Update BM25 IDF stats + build Bloom filter (Phase F).
        if self.bm25_text_column.is_some() {
            self.update_bm25_stats_from_batch(batch).await?;
            self.build_bloom_for_file(batch, &file_path);
        }
        Ok(())
    }

    /// Write batch, auto-selecting the index based on detected hardware.
    ///
    /// Picks IVF-PQ when a CUDA GPU or ≥8 CPU cores are present AND the batch
    /// has ≥5 000 vectors. Falls back to HNSW for weaker / local hardware.
    /// Uses `IvfPqConfig::for_dataset` to scale nlist with dataset size.
    pub async fn write_batch_auto(
        &mut self,
        batch: &RecordBatch,
        embeddings: &[Vec<f32>],
    ) -> AilakeResult<()> {
        let profile = ailake_index::HardwareProfile::detect();
        if profile.recommend_ivf_pq(embeddings.len()) {
            let mut ivf_config =
                ailake_index::IvfPqConfig::for_dataset(self.policy.dim as usize, embeddings.len());
            if self.policy.ivf_residual {
                ivf_config = ivf_config.with_residual();
            }
            self.write_batch_ivf_pq(batch, embeddings, ivf_config).await
        } else {
            self.write_batch(batch, embeddings).await
        }
    }

    /// Write batch, auto-selecting the index based on detected hardware — deferred variant.
    ///
    /// Same hardware detection as `write_batch_auto`: picks IVF-PQ when a CUDA GPU or
    /// ≥8 CPU cores are present AND the batch has ≥5 000 vectors; falls back to HNSW.
    ///
    /// Unlike `write_batch_auto`, the index is built in a background tokio task:
    /// - Parquet is persisted immediately (~200k vec/s, same as write_parquet_only).
    /// - HNSW or IVF-PQ index built asynchronously; shard served via flat scan meanwhile.
    ///
    /// Use this when ingest throughput matters more than immediate searchability.
    pub async fn write_batch_auto_deferred(
        &mut self,
        batch: &RecordBatch,
        embeddings: &[Vec<f32>],
    ) -> AilakeResult<()> {
        let profile = ailake_index::HardwareProfile::detect();
        if profile.recommend_ivf_pq(embeddings.len()) {
            let mut ivf_config =
                ailake_index::IvfPqConfig::for_dataset(self.policy.dim as usize, embeddings.len());
            if self.policy.ivf_residual {
                ivf_config = ivf_config.with_residual();
            }
            self.write_batch_ivf_pq_deferred(batch, embeddings, ivf_config)
                .await
        } else {
            self.write_batch_deferred(batch, embeddings).await
        }
    }

    /// Write batch with IVF-PQ index built synchronously (no background task).
    ///
    /// Smaller index than HNSW; better for S3 sequential-scan workloads.
    pub async fn write_batch_ivf_pq(
        &mut self,
        batch: &RecordBatch,
        embeddings: &[Vec<f32>],
        ivf_config: IvfPqConfig,
    ) -> AilakeResult<()> {
        if self.captured_schema.is_none() {
            self.captured_schema = Some(batch.schema());
        }
        let part_num = self.part_counter.fetch_add(1, Ordering::SeqCst);
        let file_path = format!("data/part-{:05}.parquet", part_num);

        // Train codebook once on the first shard; all subsequent shards reuse it.
        // This makes cross-shard ADC distances comparable, eliminating the need
        // for exact reranking during multi-shard search.
        if self.cached_ivf_codebook.is_none() {
            let codebook = tokio::task::spawn_blocking({
                let embeddings = embeddings.to_vec();
                let metric = self.policy.metric;
                let config = ivf_config.clone();
                move || ailake_index::IvfPqIndex::train_codebook(&embeddings, metric, &config)
            })
            .await
            .map_err(|e| ailake_core::AilakeError::Store(format!("spawn_blocking panic: {e}")))??;
            self.cached_ivf_codebook = Some(Arc::new(codebook));
        }
        // SAFETY: set to Some in the block above (either pre-existing or just trained).
        let codebook = self
            .cached_ivf_codebook
            .as_ref()
            .expect("IVF-PQ codebook must be Some after training block")
            .clone();

        let file_writer = AilakeFileWriter::new(self.policy.clone())
            .with_index_type(IndexType::IvfPq(ivf_config))
            .with_shared_ivf_codebook(codebook);
        let file_bytes: Bytes = file_writer.write(batch, embeddings)?;
        let file_size = file_bytes.len() as u64;

        self.store.put(&file_path, file_bytes.clone()).await?;

        let centroid = compute_centroid_and_radius(embeddings, self.policy.metric);

        let reader = ailake_file::AilakeFileReader::new(
            file_bytes,
            &self.policy.column_name,
            self.policy.dim,
        );
        let header = reader.read_header()?;
        let ailk_start = reader.ailk_offset()?;
        let index_abs_offset = ailk_start + header.hnsw_offset;
        let index_len = header.hnsw_len;

        let mut entry = make_data_file_entry(
            &file_path,
            embeddings.len() as u64,
            file_size,
            &centroid,
            VectorIndexInfo {
                column: &self.policy.column_name,
                dim: self.policy.dim,
                hnsw_offset: index_abs_offset,
                hnsw_len: index_len,
            },
        );
        entry.embedding_model = self
            .policy
            .embedding_model
            .as_ref()
            .map(|m| m.to_property_value());
        entry.partition_value =
            apply_partition_transforms(&self.policy, self.policy.partition_value.as_deref());
        self.pending_files.push(entry);
        Ok(())
    }

    /// Write a batch with multiple vector columns into a single AI-Lake file.
    ///
    /// The first entry in `columns` is treated as the primary column (used for
    /// geometric pruning). Additional columns each get their own HNSW section.
    pub async fn write_batch_multi(
        &mut self,
        batch: &RecordBatch,
        columns: &[MultiVectorBatch<'_>],
    ) -> AilakeResult<()> {
        use ailake_core::AilakeError;
        if self.captured_schema.is_none() {
            self.captured_schema = Some(batch.schema());
        }
        if self.extra_vec_policies.is_empty() && columns.len() > 1 {
            self.extra_vec_policies = columns[1..].iter().map(|c| c.policy.clone()).collect();
        }

        if columns.is_empty() {
            return Err(AilakeError::InvalidArgument(
                "write_batch_multi requires at least one column".into(),
            ));
        }

        let part_num = self.part_counter.fetch_add(1, Ordering::SeqCst);
        let file_path = format!("data/part-{:05}.parquet", part_num);

        let col_batches: Vec<VectorColumnBatch<'_>> = columns
            .iter()
            .map(|c| VectorColumnBatch {
                policy: &c.policy,
                embeddings: c.embeddings,
            })
            .collect();

        let primary_policy = &columns[0].policy;
        let file_writer = AilakeFileWriter::new(primary_policy.clone());
        let file_bytes: Bytes = file_writer.write_multi(batch, &col_batches)?;
        let file_size = file_bytes.len() as u64;

        self.store.put(&file_path, file_bytes.clone()).await?;

        // Primary centroid for pruning
        let primary_centroid =
            compute_centroid_and_radius(columns[0].embeddings, primary_policy.metric);

        // Read primary AILK header for offsets
        let reader = ailake_file::AilakeFileReader::new(
            file_bytes.clone(),
            &primary_policy.column_name,
            primary_policy.dim,
        );
        let primary_ailk_start = reader.ailk_offset()?;
        let primary_header = {
            use ailake_file::HEADER_SIZE;
            let start = primary_ailk_start as usize;
            let hdr_bytes: &[u8; HEADER_SIZE] = file_bytes[start..start + HEADER_SIZE]
                .try_into()
                .map_err(|_| AilakeError::NotAnAilakeFile)?;
            ailake_file::AilakeHeader::from_bytes(hdr_bytes)?
        };
        let primary_hnsw_abs = primary_ailk_start + primary_header.hnsw_offset;

        // Extra column index metadata
        let mut extra: Vec<ExtraVectorIndex> = Vec::new();
        for col in columns.iter().skip(1) {
            let col_ailk_start = reader.ailk_offset_for_column(&col.policy.column_name)?;
            let col_header = {
                use ailake_file::HEADER_SIZE;
                let start = col_ailk_start as usize;
                let hdr_bytes: &[u8; HEADER_SIZE] = file_bytes[start..start + HEADER_SIZE]
                    .try_into()
                    .map_err(|_| AilakeError::NotAnAilakeFile)?;
                ailake_file::AilakeHeader::from_bytes(hdr_bytes)?
            };
            let col_centroid = compute_centroid_and_radius(col.embeddings, col.policy.metric);
            extra.push(ExtraVectorIndex {
                column: col.policy.column_name.clone(),
                dim: col.policy.dim,
                hnsw_offset: col_ailk_start + col_header.hnsw_offset,
                hnsw_len: col_header.hnsw_len,
                centroid_b64: Some(encode_centroid_b64(&col_centroid)),
                radius: Some(col_centroid.radius),
            });
        }

        let mut entry = make_multi_column_data_file_entry(
            &file_path,
            columns[0].embeddings.len() as u64,
            file_size,
            &primary_centroid,
            VectorIndexInfo {
                column: &primary_policy.column_name,
                dim: primary_policy.dim,
                hnsw_offset: primary_hnsw_abs,
                hnsw_len: primary_header.hnsw_len,
            },
            &extra,
        );
        entry.embedding_model = self
            .policy
            .embedding_model
            .as_ref()
            .map(|m| m.to_property_value());
        entry.partition_value =
            apply_partition_transforms(&self.policy, self.policy.partition_value.as_deref());
        self.pending_files.push(entry);
        Ok(())
    }

    /// Write a multi-column batch as Parquet-only immediately; build all N column
    /// HNSW indexes in a single background task.
    ///
    /// Same semantics as `write_batch_deferred` but for N vector columns:
    /// - Parquet (primary column bytes) is persisted immediately (~200k vec/s).
    /// - A background tokio task rebuilds the full AILK file via `write_multi` and
    ///   patches the catalog entry with primary + extra column offsets once ready.
    /// - During the build window, `SearchSession` serves this shard via GPU/CPU flat
    ///   scan. Transition to HNSW-indexed search is automatic on `IndexStatus::Ready`.
    ///
    /// All N column embeddings are cloned into the background task; choose batch size
    /// so that N×rows×dim×4 bytes fits comfortably in RAM while the task runs.
    pub async fn write_batch_multi_deferred(
        &mut self,
        batch: &RecordBatch,
        columns: &[MultiVectorBatch<'_>],
    ) -> AilakeResult<()> {
        use ailake_core::AilakeError;
        if columns.is_empty() {
            return Err(AilakeError::InvalidArgument(
                "write_batch_multi_deferred requires at least one column".into(),
            ));
        }
        if self.captured_schema.is_none() {
            self.captured_schema = Some(batch.schema());
        }
        if self.extra_vec_policies.is_empty() && columns.len() > 1 {
            self.extra_vec_policies = columns[1..].iter().map(|c| c.policy.clone()).collect();
        }

        let part_num = self.part_counter.fetch_add(1, Ordering::SeqCst);
        let file_path = format!("data/part-{:05}.parquet", part_num);

        // Immediate path: write Parquet with primary column only (no AILK sections yet).
        let primary_policy = &columns[0].policy;
        let file_writer = AilakeFileWriter::new(primary_policy.clone());
        let parquet_bytes = file_writer.write_parquet_only(batch, columns[0].embeddings)?;
        let file_size = parquet_bytes.len() as u64;
        self.store.put(&file_path, parquet_bytes).await?;

        // Primary centroid enables geometric pruning during the build window.
        let primary_centroid =
            compute_centroid_and_radius(columns[0].embeddings, primary_policy.metric);
        let mut entry = make_data_file_entry_indexing(
            &file_path,
            columns[0].embeddings.len() as u64,
            file_size,
            &primary_centroid,
            &primary_policy.column_name,
            primary_policy.dim,
        );
        // Populate extra_vector_indexes with centroids/radii for pruning.
        // hnsw_offset/len are 0 until the background task patches them to non-zero.
        entry.extra_vector_indexes = columns[1..]
            .iter()
            .map(|c| {
                let col_centroid = compute_centroid_and_radius(c.embeddings, c.policy.metric);
                ExtraVectorIndex {
                    column: c.policy.column_name.clone(),
                    dim: c.policy.dim,
                    hnsw_offset: 0,
                    hnsw_len: 0,
                    centroid_b64: Some(encode_centroid_b64(&col_centroid)),
                    radius: Some(col_centroid.radius),
                }
            })
            .collect();
        entry.embedding_model = self
            .policy
            .embedding_model
            .as_ref()
            .map(|m| m.to_property_value());
        entry.partition_value =
            apply_partition_transforms(&self.policy, self.policy.partition_value.as_deref());
        self.pending_files.push(entry);

        // Clone all column data for the background task.
        let all_policies: Vec<VectorStoragePolicy> =
            columns.iter().map(|c| c.policy.clone()).collect();
        let all_embeddings: Vec<Vec<Vec<f32>>> =
            columns.iter().map(|c| c.embeddings.to_vec()).collect();
        let store = self.store.clone();
        let catalog = self.catalog.clone();
        let table = self.table.clone();
        let fp = file_path.clone();
        tokio::spawn(async move {
            if let Err(e) =
                build_and_patch_multi_index(store, catalog, all_policies, table, fp, all_embeddings)
                    .await
            {
                error!(
                    "ailake: deferred multi-column HNSW build failed — shard stays in flat-scan \
                     mode until next compaction rebuilds the index: {}",
                    e
                );
            }
        });

        Ok(())
    }

    /// Commit all staged files as a new Iceberg snapshot.
    ///
    /// No-op when `pending_files` is empty (e.g., all `write_batch_idempotent`
    /// calls were skipped because their `batch_id` was already committed).
    /// Returns the current snapshot id in that case (or 0 if no snapshot exists yet).
    /// Build a Bloom filter from the BM25 text column and store it for the given file.
    /// Called alongside `update_bm25_stats_from_batch` for every write_batch. The filter
    /// is flushed to the Puffin stats file at commit time (Phase F).
    fn build_bloom_for_file(&mut self, batch: &RecordBatch, file_path: &str) {
        use arrow_array::cast::AsArray;
        let col_name = match &self.bm25_text_column {
            Some(c) => c.clone(),
            None => return,
        };
        let col = match batch.column_by_name(&col_name) {
            Some(c) => c,
            None => return,
        };
        let str_arr = match col.as_string_opt::<i32>() {
            Some(a) => a,
            None => return,
        };
        // Size the filter for ~10× unique terms per row at 1% FPR.
        let cap = (batch.num_rows() * 10).max(128);
        let mut bloom = crate::bloom::BloomFilter::with_capacity(cap, 0.01);
        for i in 0..str_arr.len() {
            if str_arr.is_valid(i) {
                for term in crate::bm25::tokenize(str_arr.value(i)) {
                    bloom.insert(&term);
                }
            }
        }
        self.pending_blooms
            .push((file_path.to_string(), bloom.to_bytes()));
    }

    /// Update BM25 IDF stats from a batch's text column and persist to storage.
    ///
    /// Read-modify-write: loads existing stats (if any), merges new DF counts,
    /// writes back. Concurrent writers may lose some DF deltas; acceptable for
    /// approximate BM25 (same as Iceberg without OCC). Compaction rebuilds accurately.
    async fn update_bm25_stats_from_batch(&self, batch: &RecordBatch) -> AilakeResult<()> {
        use arrow_array::cast::AsArray;

        let col_name = match &self.bm25_text_column {
            Some(c) => c.as_str(),
            None => return Ok(()),
        };
        let col = match batch.column_by_name(col_name) {
            Some(c) => c,
            None => {
                tracing::warn!(
                    "ailake: BM25 text column '{}' not found in batch — skipping IDF update",
                    col_name
                );
                return Ok(());
            }
        };
        let str_arr = match col.as_string_opt::<i32>() {
            Some(a) => a,
            None => {
                tracing::warn!(
                    "ailake: BM25 text column '{}' is not a Utf8 column — skipping",
                    col_name
                );
                return Ok(());
            }
        };

        let texts: Vec<&str> = (0..str_arr.len())
            .filter(|&i| str_arr.is_valid(i))
            .map(|i| str_arr.value(i))
            .collect();

        // Load existing stats
        let stats_path = crate::bm25::BM25_STATS_FILE;
        let mut stats: crate::bm25::IdfStats = match self.store.get(stats_path).await {
            Ok(bytes) => crate::bm25::IdfStats::from_bytes(&bytes).unwrap_or_default(),
            Err(_) => crate::bm25::IdfStats::default(),
        };

        stats.merge_batch(&texts);

        let bytes = stats.to_bytes()?;
        self.store
            .put(stats_path, bytes::Bytes::from(bytes))
            .await?;
        Ok(())
    }

    pub async fn commit(mut self) -> AilakeResult<SnapshotId> {
        if self.pending_files.is_empty() {
            let current = self
                .catalog
                .load_table(&self.table)
                .await
                .ok()
                .and_then(|m| m.current_snapshot_id)
                .unwrap_or(0);
            return Ok(current);
        }
        let iceberg_schema = self
            .captured_schema
            .as_deref()
            .map(|s| arrow_schema_to_iceberg_update(s, &self.policy, &self.extra_vec_policies));
        // Store secondary column dims/metrics as table-level properties so
        // search_multimodal can discover them without reading Parquet files.
        let mut extra_properties = std::collections::HashMap::new();
        for ep in &self.extra_vec_policies {
            extra_properties.insert(format!("ailake.dim-{}", ep.column_name), ep.dim.to_string());
            extra_properties.insert(
                format!("ailake.metric-{}", ep.column_name),
                ailake_parquet::schema::metric_str(ep.metric).to_string(),
            );
            if let Some(modality) = ep.modality {
                extra_properties.insert(
                    format!("ailake.modality-{}", ep.column_name),
                    modality.as_str().to_string(),
                );
            }
        }
        let snapshot = NewSnapshot {
            snapshot_id: new_snapshot_id(),
            parent_snapshot_id: self.parent_snapshot_id,
            files: std::mem::take(&mut self.pending_files),
            operation: SnapshotOperation::Append,
            iceberg_schema,
            extra_properties,
            bloom_filters: std::mem::take(&mut self.pending_blooms),
            equality_delete_files: vec![],
        };
        self.catalog.commit_snapshot(&self.table, snapshot).await
    }

    /// Create a table if it doesn't exist, then return a writer for it.
    pub async fn create_or_open(
        catalog: Arc<dyn CatalogProvider>,
        store: Arc<dyn Store>,
        policy: VectorStoragePolicy,
        table: TableIdent,
        format_version: u8,
    ) -> AilakeResult<Self> {
        // Track existing file count so new writers start their part counter past
        // any already-committed files, preventing name collisions on sequential writes.
        let existing_file_count: u32;

        match catalog.load_table(&table).await {
            Ok(existing_meta) => {
                // Hard error: dim stored in table metadata must match the policy dim.
                // validate_embedding_dim() only checks vectors vs policy.dim; without this
                // check a caller can open with dim=16 on a dim=8 table and silently corrupt it.
                if let Some(stored_dim_str) = existing_meta.properties.get("ailake.vector-dim") {
                    if let Ok(stored_dim) = stored_dim_str.parse::<u32>() {
                        if stored_dim != policy.dim {
                            let table_model = policy
                                .embedding_model
                                .as_ref()
                                .map(|m| m.to_property_value())
                                .unwrap_or_else(|| format!("dim={}", stored_dim));
                            return Err(AilakeError::ModelMismatch {
                                table_model,
                                table_dim: stored_dim,
                                batch_model: format!("dim={}", policy.dim),
                                batch_dim: policy.dim,
                            });
                        }
                    }
                }
                // Warn when writing with a different model name into an existing table.
                // Name divergence is softer — same dim, different model (e.g. fine-tune vs
                // base) — warn only.
                if let Some(incoming) = &policy.embedding_model {
                    if let Some(stored_val) = existing_meta
                        .properties
                        .get(EmbeddingModelInfo::property_key())
                    {
                        let stored = EmbeddingModelInfo::from_property_value(stored_val);
                        if stored.name != incoming.name {
                            warn!(
                                "ailake: embedding model name changed: table has '{}', writing with '{}' \
                                 (dim={}). Vectors may be incompatible for similarity search.",
                                stored.name, incoming.name, policy.dim
                            );
                        }
                    }
                }
                existing_file_count = catalog
                    .list_files(&table, None)
                    .await
                    .unwrap_or_default()
                    .len() as u32;
            }
            Err(_) => {
                catalog
                    .create_table(
                        &table,
                        &TableProperties {
                            partition_column_type: policy.partition_column_type.clone(),
                            policy: policy.clone(),
                            extra: std::collections::HashMap::new(),
                            format_version,
                        },
                    )
                    .await?;
                existing_file_count = 0;
            }
        }
        let mut writer = Self::new(catalog, store, policy, table);
        writer.part_counter = Arc::new(AtomicU32::new(existing_file_count));
        Ok(writer)
    }
}

/// Convert an Arrow schema to an Iceberg schema update for catalog commits.
///
/// Top-level field IDs are assigned sequentially (1-based) and match the
/// `PARQUET:field_id` stamps written by `ParquetVectorWriter`. Nested element
/// IDs (inside List/Struct/Map) are assigned after all top-level IDs are
/// pre-reserved, so they never collide with Parquet column field IDs.
fn arrow_schema_to_iceberg_update(
    schema: &arrow_schema::Schema,
    policy: &VectorStoragePolicy,
    extra_vec_policies: &[VectorStoragePolicy],
) -> IcebergSchemaUpdate {
    let bytes_per_dim = policy.precision.bytes_per_element() as u32;
    let vec_fixed_len = policy.dim * bytes_per_dim;

    // Collect all vector column names that will appear in the final schema.
    let has_primary_in_batch = schema
        .fields()
        .iter()
        .any(|f| f.name() == &policy.column_name);
    let vec_cols: Vec<(String, u32)> = {
        let mut v = Vec::new();
        if !has_primary_in_batch {
            v.push((policy.column_name.clone(), vec_fixed_len));
        }
        for ep in extra_vec_policies {
            let ep_fixed_len = ep.dim * ep.precision.bytes_per_element() as u32;
            if !schema.fields().iter().any(|f| f.name() == &ep.column_name) {
                v.push((ep.column_name.clone(), ep_fixed_len));
            }
        }
        v
    };

    // Total top-level columns = batch fields + appended vec columns.
    let top_level_count = schema.fields().len() + vec_cols.len();
    // Nested element IDs start after all top-level IDs are pre-reserved.
    let mut nested_id = top_level_count as i32;

    let mut fields: Vec<serde_json::Value> = Vec::new();
    let mut name_mapping: Vec<serde_json::Value> = Vec::new();

    for (idx, field) in schema.fields().iter().enumerate() {
        let field_id = (idx + 1) as i32;
        let iceberg_type = arrow_type_to_iceberg(field.data_type(), &mut nested_id);
        fields.push(serde_json::json!({
            "id": field_id,
            "name": field.name(),
            "required": false,
            "type": iceberg_type,
        }));
        name_mapping.push(serde_json::json!({
            "field-id": field_id,
            "names": [field.name()],
        }));
    }

    // Append vector columns that live outside the RecordBatch schema.
    for (i, (col_name, fixed_len)) in vec_cols.iter().enumerate() {
        let field_id = (schema.fields().len() + 1 + i) as i32;
        fields.push(serde_json::json!({
            "id": field_id,
            "name": col_name,
            "required": false,
            "type": format!("fixed[{fixed_len}]"),
        }));
        name_mapping.push(serde_json::json!({
            "field-id": field_id,
            "names": [col_name],
        }));
    }

    let last_column_id = nested_id;
    let name_mapping_json = serde_json::to_string(&name_mapping).unwrap_or_else(|_| "[]".into());

    IcebergSchemaUpdate {
        fields,
        last_column_id,
        name_mapping_json,
    }
}

/// Map an Arrow DataType to an Iceberg schema type value (string or JSON object).
///
/// `nested_id` is a shared counter for generating unique element/field IDs inside
/// List, Struct, and Map types. It must start beyond all pre-reserved top-level IDs.
fn arrow_type_to_iceberg(dt: &arrow_schema::DataType, nested_id: &mut i32) -> serde_json::Value {
    use arrow_schema::DataType;
    match dt {
        DataType::Boolean => serde_json::json!("boolean"),
        DataType::Int8 | DataType::Int16 | DataType::Int32 | DataType::UInt8 | DataType::UInt16 => {
            serde_json::json!("int")
        }
        DataType::Int64 | DataType::UInt32 | DataType::UInt64 => serde_json::json!("long"),
        DataType::Float16 | DataType::Float32 => serde_json::json!("float"),
        DataType::Float64 => serde_json::json!("double"),
        DataType::Utf8 | DataType::LargeUtf8 | DataType::Utf8View => serde_json::json!("string"),
        DataType::Binary | DataType::LargeBinary | DataType::BinaryView => {
            serde_json::json!("binary")
        }
        DataType::Date32 | DataType::Date64 => serde_json::json!("date"),
        // Timestamp with timezone → timestamptz; without → timestamp.
        DataType::Timestamp(_, Some(_)) => serde_json::json!("timestamptz"),
        DataType::Timestamp(_, None) => serde_json::json!("timestamp"),
        DataType::Time32(_) | DataType::Time64(_) => serde_json::json!("time"),
        DataType::FixedSizeBinary(n) => serde_json::json!(format!("fixed[{n}]")),
        DataType::Decimal128(p, s) | DataType::Decimal256(p, s) => {
            serde_json::json!(format!("decimal({p}, {s})"))
        }
        DataType::List(inner)
        | DataType::LargeList(inner)
        | DataType::ListView(inner)
        | DataType::FixedSizeList(inner, _) => {
            *nested_id += 1;
            let element_id = *nested_id;
            let element_type = arrow_type_to_iceberg(inner.data_type(), nested_id);
            serde_json::json!({
                "type": "list",
                "element-id": element_id,
                "element": element_type,
                "element-required": !inner.is_nullable(),
            })
        }
        DataType::Struct(arrow_fields) => {
            let struct_fields: Vec<serde_json::Value> = arrow_fields
                .iter()
                .map(|f| {
                    *nested_id += 1;
                    let fid = *nested_id;
                    let ftype = arrow_type_to_iceberg(f.data_type(), nested_id);
                    serde_json::json!({
                        "id": fid,
                        "name": f.name(),
                        "required": !f.is_nullable(),
                        "type": ftype,
                    })
                })
                .collect();
            serde_json::json!({ "type": "struct", "fields": struct_fields })
        }
        DataType::Map(entries, _) => {
            // Arrow Map is List<Struct<key: K, value: V>>.
            *nested_id += 1;
            let key_id = *nested_id;
            *nested_id += 1;
            let val_id = *nested_id;
            if let DataType::Struct(kv_fields) = entries.data_type() {
                let key_f = kv_fields
                    .iter()
                    .find(|f| f.name() == "key" || f.name() == "keys");
                let val_f = kv_fields
                    .iter()
                    .find(|f| f.name() == "value" || f.name() == "values");
                let key_type = key_f
                    .map(|f| arrow_type_to_iceberg(f.data_type(), nested_id))
                    .unwrap_or(serde_json::json!("binary"));
                let val_type = val_f
                    .map(|f| arrow_type_to_iceberg(f.data_type(), nested_id))
                    .unwrap_or(serde_json::json!("binary"));
                let val_required = val_f.map(|f| !f.is_nullable()).unwrap_or(false);
                serde_json::json!({
                    "type": "map",
                    "key-id": key_id,
                    "key": key_type,
                    "value-id": val_id,
                    "value": val_type,
                    "value-required": val_required,
                })
            } else {
                serde_json::json!("binary")
            }
        }
        _ => serde_json::json!("binary"),
    }
}

/// Background task: reads a Parquet-only shard, builds full AILK file, patches catalog.
pub(crate) async fn build_and_patch_index(
    store: Arc<dyn Store>,
    catalog: Arc<dyn CatalogProvider>,
    policy: VectorStoragePolicy,
    table: TableIdent,
    file_path: String,
) -> AilakeResult<()> {
    // Read the Parquet-only bytes already stored.
    let parquet_bytes = store.get(&file_path).await?;
    let reader = AilakeFileReader::new(parquet_bytes, &policy.column_name, policy.dim);
    let (batch, embeddings) = reader.read_parquet()?;

    // Build the full AILK file (Parquet + HNSW) — CPU-intensive; run on blocking pool
    // so the tokio async threads aren't starved when many shards build concurrently.
    let full_bytes = tokio::task::spawn_blocking({
        let policy = policy.clone();
        move || {
            let file_writer = AilakeFileWriter::new(policy);
            file_writer.write(&batch, &embeddings)
        }
    })
    .await
    .map_err(|e| ailake_core::AilakeError::Store(format!("spawn_blocking panic: {e}")))??;

    // Extract HNSW offsets from the newly written file.
    let full_reader = AilakeFileReader::new(full_bytes.clone(), &policy.column_name, policy.dim);
    let header = full_reader.read_header()?;
    let ailk_start = full_reader.ailk_offset()?;
    let hnsw_abs_offset = ailk_start + header.hnsw_offset;
    let hnsw_len = header.hnsw_len;

    // Overwrite the Parquet-only file with the full AILK version.
    store.put(&file_path, full_bytes).await?;

    // Wait for the initial writer commit to appear (max 60 s).
    // HNSW builds can finish before the main write loop calls commit_snapshot.
    let mut committed = false;
    for _ in 0..120u32 {
        match catalog.load_table(&table).await {
            Ok(meta) if meta.current_snapshot_id.is_some() => {
                committed = true;
                break;
            }
            _ => tokio::time::sleep(std::time::Duration::from_millis(500)).await,
        }
    }
    if !committed {
        return Err(ailake_core::AilakeError::Store(format!(
            "deferred HNSW build: no snapshot committed for {file_path} after 60 s — \
             did you call TableWriter::commit()?"
        )));
    }

    // Update the catalog with CAS-like retry to handle concurrent background tasks.
    // Multiple tasks can race on commit_snapshot(Replace): the last writer wins and
    // may overwrite a sibling task's Ready status. Retry until we confirm our file
    // is marked Ready in the current snapshot.
    for attempt in 0..50u32 {
        let table_meta = catalog.load_table(&table).await?;
        let parent_snapshot_id = table_meta.current_snapshot_id;
        let mut files = catalog.list_files(&table, None).await?;

        // Already marked Ready by a previous successful attempt.
        if files
            .iter()
            .any(|f| f.path == file_path && f.index_status == IndexStatus::Ready)
        {
            break;
        }

        for f in &mut files {
            if f.path == file_path {
                f.hnsw_offset = Some(hnsw_abs_offset);
                f.hnsw_len = Some(hnsw_len);
                f.index_status = IndexStatus::Ready;
                break;
            }
        }
        catalog
            .commit_snapshot(
                &table,
                NewSnapshot {
                    snapshot_id: new_snapshot_id(),
                    parent_snapshot_id,
                    files,
                    operation: SnapshotOperation::Replace,
                    iceberg_schema: None,
                    extra_properties: std::collections::HashMap::new(),
                    bloom_filters: vec![],
                    equality_delete_files: vec![],
                },
            )
            .await?;

        // Brief yield so sibling tasks can commit, then verify our change survived.
        tokio::time::sleep(std::time::Duration::from_millis(10 + attempt as u64 * 5)).await;

        let verify = catalog.list_files(&table, None).await?;
        if verify
            .iter()
            .any(|f| f.path == file_path && f.index_status == IndexStatus::Ready)
        {
            break;
        }
        // Another task overwrote us — retry.
    }

    info!(
        "ailake: deferred HNSW index built for {} (offset={}, len={})",
        file_path, hnsw_abs_offset, hnsw_len
    );
    Ok(())
}

/// Background task: train IVF-PQ (using shared codebook) and patch catalog entry.
///
/// The OnceCell guarantees that k-means training runs exactly once across all
/// concurrent background tasks — subsequent tasks skip directly to assign+encode.
async fn build_ivf_pq_and_patch_index(
    store: Arc<dyn Store>,
    catalog: Arc<dyn CatalogProvider>,
    policy: VectorStoragePolicy,
    table: TableIdent,
    file_path: String,
    ivf_config: IvfPqConfig,
    codebook_cell: Arc<tokio::sync::OnceCell<IvfPqCodebook>>,
) -> AilakeResult<()> {
    let parquet_bytes = store.get(&file_path).await?;
    let reader = AilakeFileReader::new(parquet_bytes, &policy.column_name, policy.dim);
    let (batch, embeddings) = reader.read_parquet()?;

    // Get or train the shared codebook. First task trains; all others await the result.
    let codebook = codebook_cell
        .get_or_try_init(|| async {
            let vecs = embeddings.clone();
            let metric = policy.metric;
            let cfg = ivf_config.clone();
            tokio::task::spawn_blocking(move || {
                ailake_index::IvfPqIndex::train_codebook(&vecs, metric, &cfg)
            })
            .await
            .map_err(|e| ailake_core::AilakeError::Store(format!("spawn_blocking panic: {e}")))?
        })
        .await?;

    let full_bytes = tokio::task::spawn_blocking({
        let policy = policy.clone();
        let codebook = codebook.clone();
        move || {
            let file_writer = AilakeFileWriter::new(policy)
                .with_index_type(IndexType::IvfPq(ivf_config))
                .with_shared_ivf_codebook(Arc::new(codebook));
            file_writer.write(&batch, &embeddings)
        }
    })
    .await
    .map_err(|e| ailake_core::AilakeError::Store(format!("spawn_blocking panic: {e}")))??;

    let full_reader = AilakeFileReader::new(full_bytes.clone(), &policy.column_name, policy.dim);
    let header = full_reader.read_header()?;
    let ailk_start = full_reader.ailk_offset()?;
    let hnsw_abs_offset = ailk_start + header.hnsw_offset;
    let hnsw_len = header.hnsw_len;

    store.put(&file_path, full_bytes).await?;

    // Wait for initial commit to appear then patch IndexStatus::Ready (max 60 s).
    let mut committed = false;
    for _ in 0..120u32 {
        match catalog.load_table(&table).await {
            Ok(meta) if meta.current_snapshot_id.is_some() => {
                committed = true;
                break;
            }
            _ => tokio::time::sleep(std::time::Duration::from_millis(500)).await,
        }
    }
    if !committed {
        return Err(ailake_core::AilakeError::Store(format!(
            "deferred IVF-PQ build: no snapshot committed for {file_path} after 60 s — \
             did you call TableWriter::commit()?"
        )));
    }

    for attempt in 0..50u32 {
        let table_meta = catalog.load_table(&table).await?;
        let parent_snapshot_id = table_meta.current_snapshot_id;
        let mut files = catalog.list_files(&table, None).await?;

        if files
            .iter()
            .any(|f| f.path == file_path && f.index_status == IndexStatus::Ready)
        {
            break;
        }

        for f in &mut files {
            if f.path == file_path {
                f.hnsw_offset = Some(hnsw_abs_offset);
                f.hnsw_len = Some(hnsw_len);
                f.index_status = IndexStatus::Ready;
                break;
            }
        }
        catalog
            .commit_snapshot(
                &table,
                NewSnapshot {
                    snapshot_id: new_snapshot_id(),
                    parent_snapshot_id,
                    files,
                    operation: SnapshotOperation::Replace,
                    iceberg_schema: None,
                    extra_properties: std::collections::HashMap::new(),
                    bloom_filters: vec![],
                    equality_delete_files: vec![],
                },
            )
            .await?;

        tokio::time::sleep(std::time::Duration::from_millis(10 + attempt as u64 * 5)).await;

        let verify = catalog.list_files(&table, None).await?;
        if verify
            .iter()
            .any(|f| f.path == file_path && f.index_status == IndexStatus::Ready)
        {
            break;
        }
    }

    info!(
        "ailake: deferred IVF-PQ index built for {} (offset={}, len={})",
        file_path, hnsw_abs_offset, hnsw_len
    );
    Ok(())
}

/// Background task: rebuild full multi-column AILK file and patch all column offsets.
///
/// Reads the Parquet-only shard, calls `write_multi` with all N column embeddings
/// (cloned from the caller), extracts per-column HNSW offsets, overwrites the file,
/// then applies the same CAS retry loop used by single-column deferred tasks.
async fn build_and_patch_multi_index(
    store: Arc<dyn Store>,
    catalog: Arc<dyn CatalogProvider>,
    policies: Vec<VectorStoragePolicy>,
    table: TableIdent,
    file_path: String,
    all_embeddings: Vec<Vec<Vec<f32>>>,
) -> AilakeResult<()> {
    // Read the Parquet-only shard (primary column only).
    let parquet_bytes = store.get(&file_path).await?;
    let primary_reader =
        AilakeFileReader::new(parquet_bytes, &policies[0].column_name, policies[0].dim);
    let (batch, _) = primary_reader.read_parquet()?;

    // Build full AILK file with all N column HNSW sections on the blocking pool.
    let full_bytes = tokio::task::spawn_blocking({
        let policies = policies.clone();
        let all_embeddings = all_embeddings.clone();
        move || {
            let col_batches: Vec<VectorColumnBatch<'_>> = policies
                .iter()
                .zip(all_embeddings.iter())
                .map(|(p, embs)| VectorColumnBatch {
                    policy: p,
                    embeddings: embs.as_slice(),
                })
                .collect();
            let file_writer = AilakeFileWriter::new(policies[0].clone());
            file_writer.write_multi(&batch, &col_batches)
        }
    })
    .await
    .map_err(|e| ailake_core::AilakeError::Store(format!("spawn_blocking panic: {e}")))??;

    // Extract primary HNSW offsets.
    let primary_reader = AilakeFileReader::new(
        full_bytes.clone(),
        &policies[0].column_name,
        policies[0].dim,
    );
    let primary_header = primary_reader.read_header()?;
    let primary_ailk_start = primary_reader.ailk_offset()?;
    let primary_hnsw_abs = primary_ailk_start + primary_header.hnsw_offset;
    let primary_hnsw_len = primary_header.hnsw_len;

    // Extract extra column HNSW offsets (one reader per column).
    // Must use ailk_offset_for_column / read_header_for_column so each column's
    // own `ailake.{col}.footer_offset` is used — ailk_offset() always returns the
    // primary column offset, which is wrong for extra columns.
    let mut extra_offsets: Vec<(u64, u64)> = Vec::with_capacity(policies.len().saturating_sub(1));
    for col_policy in policies.iter().skip(1) {
        let col_reader =
            AilakeFileReader::new(full_bytes.clone(), &col_policy.column_name, col_policy.dim);
        let col_ailk_start = col_reader.ailk_offset_for_column(&col_policy.column_name)?;
        let col_header = col_reader.read_header_for_column(&col_policy.column_name)?;
        extra_offsets.push((col_ailk_start + col_header.hnsw_offset, col_header.hnsw_len));
    }

    // Overwrite the Parquet-only shard with the full AILK file.
    store.put(&file_path, full_bytes).await?;

    // Wait for the initial writer commit to appear (max 60 s).
    let mut committed = false;
    for _ in 0..120u32 {
        match catalog.load_table(&table).await {
            Ok(meta) if meta.current_snapshot_id.is_some() => {
                committed = true;
                break;
            }
            _ => tokio::time::sleep(std::time::Duration::from_millis(500)).await,
        }
    }
    if !committed {
        return Err(ailake_core::AilakeError::Store(format!(
            "deferred index build: no snapshot committed for {file_path} after 60 s — \
             did you call TableWriter::commit()?"
        )));
    }

    // CAS retry loop: patch primary offsets + extra_vector_indexes + IndexStatus::Ready.
    for attempt in 0..50u32 {
        let table_meta = catalog.load_table(&table).await?;
        let parent_snapshot_id = table_meta.current_snapshot_id;
        let mut files = catalog.list_files(&table, None).await?;

        if files
            .iter()
            .any(|f| f.path == file_path && f.index_status == IndexStatus::Ready)
        {
            break;
        }

        for f in &mut files {
            if f.path == file_path {
                f.hnsw_offset = Some(primary_hnsw_abs);
                f.hnsw_len = Some(primary_hnsw_len);
                f.index_status = IndexStatus::Ready;
                for (i, &(off, len)) in extra_offsets.iter().enumerate() {
                    if let Some(xi) = f.extra_vector_indexes.get_mut(i) {
                        xi.hnsw_offset = off;
                        xi.hnsw_len = len;
                    }
                }
                break;
            }
        }
        catalog
            .commit_snapshot(
                &table,
                NewSnapshot {
                    snapshot_id: new_snapshot_id(),
                    parent_snapshot_id,
                    files,
                    operation: SnapshotOperation::Replace,
                    iceberg_schema: None,
                    extra_properties: std::collections::HashMap::new(),
                    bloom_filters: vec![],
                    equality_delete_files: vec![],
                },
            )
            .await?;

        tokio::time::sleep(std::time::Duration::from_millis(10 + attempt as u64 * 5)).await;

        let verify = catalog.list_files(&table, None).await?;
        if verify
            .iter()
            .any(|f| f.path == file_path && f.index_status == IndexStatus::Ready)
        {
            break;
        }
    }

    info!(
        "ailake: deferred multi-column HNSW built for {} ({} cols, primary offset={})",
        file_path,
        policies.len(),
        primary_hnsw_abs
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use ailake_core::{VectorMetric, VectorPrecision};
    use arrow_schema::{DataType, Field, Schema, TimeUnit};

    fn policy(col: &str, dim: u32) -> VectorStoragePolicy {
        VectorStoragePolicy {
            column_name: col.to_string(),
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

    fn update_for(schema: &Schema, pol: &VectorStoragePolicy) -> IcebergSchemaUpdate {
        arrow_schema_to_iceberg_update(schema, pol, &[])
    }

    #[test]
    fn simple_schema_produces_correct_fields() {
        let schema = Schema::new(vec![
            Field::new("id", DataType::Int32, false),
            Field::new("text", DataType::Utf8, false),
        ]);
        let pol = policy("embedding", 8);
        let upd = update_for(&schema, &pol);

        assert_eq!(upd.fields.len(), 3);
        assert_eq!(upd.fields[0]["id"], 1);
        assert_eq!(upd.fields[0]["type"], "int");
        assert_eq!(upd.fields[1]["id"], 2);
        assert_eq!(upd.fields[1]["type"], "string");
        assert_eq!(upd.fields[2]["id"], 3);
        assert_eq!(upd.fields[2]["type"], "fixed[16]"); // dim=8, F16=2 bytes

        let nm: Vec<serde_json::Value> = serde_json::from_str(&upd.name_mapping_json).unwrap();
        assert_eq!(nm.len(), 3);
        assert_eq!(nm[2]["field-id"], 3);
        assert_eq!(nm[2]["names"][0], "embedding");
        assert_eq!(upd.last_column_id, 3);
    }

    #[test]
    fn timestamp_without_tz_maps_to_timestamp_not_timestamptz() {
        let schema = Schema::new(vec![
            Field::new(
                "created_at",
                DataType::Timestamp(TimeUnit::Microsecond, None),
                true,
            ),
            Field::new(
                "updated_at",
                DataType::Timestamp(TimeUnit::Microsecond, Some("UTC".into())),
                true,
            ),
        ]);
        let pol = policy("vec", 4);
        let upd = update_for(&schema, &pol);

        assert_eq!(upd.fields[0]["type"], "timestamp");
        assert_eq!(upd.fields[1]["type"], "timestamptz");
    }

    #[test]
    fn list_type_produces_iceberg_list_object() {
        let schema = Schema::new(vec![Field::new(
            "tags",
            DataType::List(std::sync::Arc::new(Field::new(
                "item",
                DataType::Utf8,
                true,
            ))),
            true,
        )]);
        let pol = policy("vec", 4);
        let upd = update_for(&schema, &pol);

        let t = &upd.fields[0]["type"];
        assert_eq!(t["type"], "list");
        assert_eq!(t["element"], "string");
        // element-id must be > top-level field count (2: tags + vec)
        assert!(t["element-id"].as_i64().unwrap() > 2);
    }

    #[test]
    fn struct_type_produces_nested_fields() {
        let schema = Schema::new(vec![Field::new(
            "meta",
            DataType::Struct(
                vec![
                    Field::new("key", DataType::Utf8, false),
                    Field::new("val", DataType::Int64, false),
                ]
                .into(),
            ),
            true,
        )]);
        let pol = policy("vec", 4);
        let upd = update_for(&schema, &pol);

        let t = &upd.fields[0]["type"];
        assert_eq!(t["type"], "struct");
        let nested = t["fields"].as_array().unwrap();
        assert_eq!(nested.len(), 2);
        assert_eq!(nested[0]["name"], "key");
        assert_eq!(nested[0]["type"], "string");
        assert_eq!(nested[1]["name"], "val");
        assert_eq!(nested[1]["type"], "long");
        // Nested IDs must be > top-level count (2: meta + vec)
        assert!(nested[0]["id"].as_i64().unwrap() > 2);
    }

    #[test]
    fn no_duplicate_vec_column_when_already_in_batch() {
        // If for some reason the vec column is in the batch schema, don't add it twice.
        let schema = Schema::new(vec![
            Field::new("id", DataType::Int32, false),
            Field::new("embedding", DataType::FixedSizeBinary(16), false),
        ]);
        let pol = policy("embedding", 8);
        let upd = update_for(&schema, &pol);

        assert_eq!(upd.fields.len(), 2, "should not add embedding twice");
        let names: Vec<&str> = upd
            .fields
            .iter()
            .map(|f| f["name"].as_str().unwrap())
            .collect();
        assert_eq!(names.iter().filter(|&&n| n == "embedding").count(), 1);
    }

    #[test]
    fn multi_vec_policies_all_appended() {
        let schema = Schema::new(vec![Field::new("id", DataType::Int32, false)]);
        let primary = policy("embedding", 4);
        let extra = vec![policy("context_embedding", 4)];
        let upd = arrow_schema_to_iceberg_update(&schema, &primary, &extra);

        assert_eq!(upd.fields.len(), 3); // id + embedding + context_embedding
        let names: Vec<&str> = upd
            .fields
            .iter()
            .map(|f| f["name"].as_str().unwrap())
            .collect();
        assert!(names.contains(&"embedding"));
        assert!(names.contains(&"context_embedding"));
    }

    #[test]
    fn top_level_field_ids_match_parquet_stamp_sequence() {
        // Top-level IDs must be 1, 2, ..., N regardless of nested element IDs.
        let schema = Schema::new(vec![
            Field::new("id", DataType::Int64, false),
            Field::new(
                "tags",
                DataType::List(std::sync::Arc::new(Field::new(
                    "item",
                    DataType::Utf8,
                    true,
                ))),
                true,
            ),
        ]);
        let pol = policy("vec", 4);
        let upd = update_for(&schema, &pol);

        // Top-level: id=1, tags=2, vec=3
        assert_eq!(upd.fields[0]["id"], 1);
        assert_eq!(upd.fields[1]["id"], 2);
        assert_eq!(upd.fields[2]["id"], 3);

        // Nested element ID must be > 3
        assert!(upd.fields[1]["type"]["element-id"].as_i64().unwrap() > 3);
    }

    /// Smoke-test write_batch_auto_deferred: verifies that it completes without error
    /// and stages a pending file entry (index built asynchronously in background).
    #[tokio::test]
    async fn write_batch_auto_deferred_stages_file() {
        use ailake_catalog::{HadoopCatalog, TableIdent};
        use ailake_store::LocalStore;
        use arrow_schema::{DataType, Field, Schema};

        let dir = tempfile::tempdir().unwrap();
        let store: std::sync::Arc<dyn ailake_store::Store> =
            std::sync::Arc::new(LocalStore::new(dir.path().to_str().unwrap()));
        let catalog = std::sync::Arc::new(HadoopCatalog::new(std::sync::Arc::clone(&store), ""));
        let pol = policy("embedding", 4);
        let ident = TableIdent::new("default", "t");

        let mut writer = TableWriter::create_or_open(catalog, store, pol, ident, 2)
            .await
            .unwrap();

        let schema =
            std::sync::Arc::new(Schema::new(vec![Field::new("text", DataType::Utf8, false)]));
        let batch = arrow_array::RecordBatch::try_new(
            schema,
            vec![std::sync::Arc::new(arrow_array::StringArray::from(vec![
                "hello",
            ]))],
        )
        .unwrap();
        let embeddings = vec![vec![1.0f32, 0.0, 0.0, 0.0]];

        writer
            .write_batch_auto_deferred(&batch, &embeddings)
            .await
            .unwrap();

        // One pending file should be staged even before commit.
        assert_eq!(writer.pending_files.len(), 1);
    }

    /// Smoke-test write_batch_multi_deferred: verifies Parquet staged immediately,
    /// placeholder extra_vector_indexes populated, and background task spawned.
    #[tokio::test]
    async fn write_batch_multi_deferred_stages_file_with_extra_indexes() {
        use ailake_catalog::{HadoopCatalog, IndexStatus, TableIdent};
        use ailake_store::LocalStore;
        use arrow_schema::{DataType, Field, Schema};

        let dir = tempfile::tempdir().unwrap();
        let store: std::sync::Arc<dyn ailake_store::Store> =
            std::sync::Arc::new(LocalStore::new(dir.path().to_str().unwrap()));
        let catalog = std::sync::Arc::new(HadoopCatalog::new(std::sync::Arc::clone(&store), ""));
        let primary_pol = policy("embedding", 4);
        let ident = TableIdent::new("default", "t");

        let mut writer = TableWriter::create_or_open(catalog, store, primary_pol, ident, 2)
            .await
            .unwrap();

        let schema =
            std::sync::Arc::new(Schema::new(vec![Field::new("text", DataType::Utf8, false)]));
        let batch = arrow_array::RecordBatch::try_new(
            schema,
            vec![std::sync::Arc::new(arrow_array::StringArray::from(vec![
                "hello", "world",
            ]))],
        )
        .unwrap();

        let text_embs = vec![vec![1.0f32, 0.0, 0.0, 0.0], vec![0.0, 1.0, 0.0, 0.0]];
        let img_embs = vec![vec![1.0f32, 0.0], vec![0.0, 1.0]];

        let columns = vec![
            MultiVectorBatch {
                policy: policy("embedding", 4),
                embeddings: &text_embs,
            },
            MultiVectorBatch {
                policy: policy("img_embedding", 2),
                embeddings: &img_embs,
            },
        ];

        writer
            .write_batch_multi_deferred(&batch, &columns)
            .await
            .unwrap();

        assert_eq!(writer.pending_files.len(), 1);
        let entry = &writer.pending_files[0];
        // IndexStatus::Indexing — index build is async
        assert_eq!(entry.index_status, IndexStatus::Indexing);
        // Primary centroid populated for pruning during build window
        assert!(entry.centroid_b64.is_some());
        // Placeholder extra column entry (centroid present, offsets zero)
        assert_eq!(entry.extra_vector_indexes.len(), 1);
        let xi = &entry.extra_vector_indexes[0];
        assert_eq!(xi.column, "img_embedding");
        assert_eq!(xi.dim, 2);
        assert_eq!(xi.hnsw_offset, 0); // not yet built
        assert_eq!(xi.hnsw_len, 0); // not yet built
        assert!(xi.centroid_b64.is_some());
    }
}
