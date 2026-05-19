use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::Arc;

use ailake_catalog::{
    encode_centroid_b64, make_data_file_entry, make_multi_column_data_file_entry, new_snapshot_id,
    CatalogProvider, DataFileEntry, ExtraVectorIndex, NewSnapshot, SnapshotId, SnapshotOperation,
    TableIdent, TableProperties, VectorIndexInfo,
};
use ailake_core::{AilakeResult, VectorStoragePolicy};
use ailake_file::{AilakeFileWriter, VectorColumnBatch};
use ailake_store::Store;
use ailake_vec::compute_centroid_and_radius;
use arrow_array::RecordBatch;
use bytes::Bytes;

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
        }
    }

    pub fn with_parent_snapshot(mut self, id: SnapshotId) -> Self {
        self.parent_snapshot_id = Some(id);
        self
    }

    /// Write a batch to a new AI-Lake file and stage it for commit.
    pub async fn write_batch(
        &mut self,
        batch: &RecordBatch,
        embeddings: &[Vec<f32>],
    ) -> AilakeResult<()> {
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

        let entry = make_data_file_entry(
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

        let entry = make_multi_column_data_file_entry(
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
        self.pending_files.push(entry);
        Ok(())
    }

    /// Commit all staged files as a new Iceberg snapshot.
    pub async fn commit(mut self) -> AilakeResult<SnapshotId> {
        let snapshot = NewSnapshot {
            snapshot_id: new_snapshot_id(),
            parent_snapshot_id: self.parent_snapshot_id,
            files: std::mem::take(&mut self.pending_files),
            operation: SnapshotOperation::Append,
        };
        self.catalog.commit_snapshot(&self.table, snapshot).await
    }

    /// Create a table if it doesn't exist, then return a TableWriter for it.
    pub async fn create_or_open(
        catalog: Arc<dyn CatalogProvider>,
        store: Arc<dyn Store>,
        policy: VectorStoragePolicy,
        table: TableIdent,
    ) -> AilakeResult<Self> {
        // Try to load; if not found, create
        if catalog.load_table(&table).await.is_err() {
            catalog
                .create_table(
                    &table,
                    &TableProperties {
                        policy: policy.clone(),
                        extra: std::collections::HashMap::new(),
                    },
                )
                .await?;
        }
        Ok(Self::new(catalog, store, policy, table))
    }
}
