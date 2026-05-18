use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::Arc;

use ailake_catalog::{
    make_data_file_entry, new_snapshot_id, CatalogProvider, DataFileEntry, NewSnapshot, SnapshotId,
    SnapshotOperation, TableIdent, TableProperties,
};
use ailake_core::{AilakeResult, VectorStoragePolicy};
use ailake_file::AilakeFileWriter;
use ailake_store::Store;
use ailake_vec::compute_centroid_and_radius;
use arrow_array::RecordBatch;
use bytes::Bytes;

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
        let trailer = reader.read_trailer()?;
        let hnsw_abs_offset = trailer.footer_offset + header.hnsw_offset;
        let hnsw_len = header.hnsw_len;

        let entry = make_data_file_entry(
            &file_path,
            embeddings.len() as u64,
            file_size,
            &centroid,
            hnsw_abs_offset,
            hnsw_len,
            &self.policy.column_name,
            self.policy.dim,
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
