// SPDX-License-Identifier: MIT OR Apache-2.0
use std::sync::Arc;
use std::time::{Duration, Instant};

use ailake_catalog::{CatalogProvider, SnapshotId, TableIdent};
use ailake_core::{AilakeResult, VectorStoragePolicy};
use ailake_store::Store;
use arrow_array::RecordBatch;
use arrow_select::concat::concat_batches;

use crate::writer::TableWriter;

/// Tuning knobs for `MemTableWriter`.
#[derive(Debug, Clone)]
pub struct MemTableConfig {
    /// Flush when accumulated embedding bytes exceed this threshold.
    /// Default: 64 MiB.
    pub flush_size_bytes: usize,
    /// Flush when the row count exceeds this value regardless of byte size.
    /// Default: 100,000 rows.
    pub flush_max_rows: usize,
    /// Maximum age of unflushed data before `flush_if_due` triggers a flush.
    /// Default: 30 seconds.
    pub flush_interval: Duration,
}

impl Default for MemTableConfig {
    fn default() -> Self {
        Self {
            flush_size_bytes: 64 * 1024 * 1024,
            flush_max_rows: 100_000,
            flush_interval: Duration::from_secs(30),
        }
    }
}

/// In-memory write buffer that batches small inserts before persisting.
///
/// Problem: streaming pipelines (Flink, Spark Streaming) emit small
/// RecordBatches every few seconds. Calling `write_batch_deferred` on each
/// micro-batch creates many tiny Parquet files and triggers repeated HNSW
/// builds — both are expensive.
///
/// Solution: buffer rows in RAM, flush to a single Parquet shard only when
/// the buffer reaches the configured size/row/time threshold. The deferred
/// HNSW build runs once per flush, not once per micro-batch.
///
/// # Usage
///
/// ```ignore
/// let mut mt = MemTableWriter::new(catalog, store, policy, table, MemTableConfig::default());
/// loop {
///     mt.insert(&batch, &embeddings).await.unwrap();
///     mt.flush_if_due().await.unwrap();
/// }
/// mt.flush().await.unwrap();
/// ```
pub struct MemTableWriter {
    catalog: Arc<dyn CatalogProvider>,
    store: Arc<dyn Store>,
    policy: VectorStoragePolicy,
    table: TableIdent,
    config: MemTableConfig,

    // Accumulated micro-batches waiting for flush
    pending_batches: Vec<RecordBatch>,
    pending_embeddings: Vec<Vec<f32>>,
    buffered_bytes: usize,
    last_flush: Instant,
}

impl MemTableWriter {
    pub fn new(
        catalog: Arc<dyn CatalogProvider>,
        store: Arc<dyn Store>,
        policy: VectorStoragePolicy,
        table: TableIdent,
        config: MemTableConfig,
    ) -> Self {
        Self {
            catalog,
            store,
            policy,
            table,
            config,
            pending_batches: Vec::new(),
            pending_embeddings: Vec::new(),
            buffered_bytes: 0,
            last_flush: Instant::now(),
        }
    }

    /// Buffer a micro-batch. Flushes automatically if size or row threshold exceeded.
    /// Returns `Some(snapshot_id)` when an automatic flush occurred, `None` otherwise.
    pub async fn insert(
        &mut self,
        batch: &RecordBatch,
        embeddings: &[Vec<f32>],
    ) -> AilakeResult<Option<SnapshotId>> {
        let row_bytes =
            embeddings.len() * self.policy.dim as usize * self.policy.precision.bytes_per_element();

        self.pending_batches.push(batch.clone());
        self.pending_embeddings.extend_from_slice(embeddings);
        self.buffered_bytes += row_bytes;

        if self.buffered_bytes >= self.config.flush_size_bytes
            || self.pending_embeddings.len() >= self.config.flush_max_rows
        {
            Ok(Some(self.flush().await?))
        } else {
            Ok(None)
        }
    }

    /// Flush if `flush_interval` has elapsed since the last flush.
    /// Returns `Some(snapshot_id)` when a flush occurred, `None` otherwise.
    pub async fn flush_if_due(&mut self) -> AilakeResult<Option<SnapshotId>> {
        if self.pending_embeddings.is_empty() {
            self.last_flush = Instant::now();
            return Ok(None);
        }
        if self.last_flush.elapsed() >= self.config.flush_interval {
            Ok(Some(self.flush().await?))
        } else {
            Ok(None)
        }
    }

    /// Flush all buffered data immediately, even if thresholds are not met.
    /// Returns the committed `SnapshotId`. Calling on an empty buffer is a no-op
    /// and returns `SnapshotId::default()`.
    pub async fn flush(&mut self) -> AilakeResult<SnapshotId> {
        if self.pending_embeddings.is_empty() {
            self.last_flush = Instant::now();
            return Ok(0);
        }

        // Concatenate accumulated micro-batches into one shard batch.
        let merged = if self.pending_batches.len() == 1 {
            self.pending_batches.remove(0)
        } else {
            concat_batches(&self.pending_batches[0].schema(), &self.pending_batches)
                .map_err(|e| ailake_core::AilakeError::Arrow(e.to_string()))?
        };
        let embeddings = std::mem::take(&mut self.pending_embeddings);
        self.pending_batches.clear();
        self.buffered_bytes = 0;
        self.last_flush = Instant::now();

        // Delegate to TableWriter: Parquet-only write + deferred HNSW build.
        let mut writer = TableWriter::new(
            self.catalog.clone(),
            self.store.clone(),
            self.policy.clone(),
            self.table.clone(),
        );
        writer.write_batch_deferred(&merged, &embeddings).await?;
        let snap = writer.commit().await?;
        Ok(snap)
    }

    /// Number of rows currently in the buffer.
    pub fn buffered_rows(&self) -> usize {
        self.pending_embeddings.len()
    }

    /// Estimated byte size of the embedding data currently in the buffer.
    pub fn buffered_bytes(&self) -> usize {
        self.buffered_bytes
    }

    /// True when a size or row-count threshold has been reached.
    pub fn is_full(&self) -> bool {
        self.buffered_bytes >= self.config.flush_size_bytes
            || self.pending_embeddings.len() >= self.config.flush_max_rows
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

    use ailake_catalog::{HadoopCatalog, TableIdent};
    use ailake_core::{VectorMetric, VectorPrecision};
    use ailake_store::{LocalStore, Store};
    use arrow_array::{Int32Array, StringArray};
    use arrow_schema::{DataType, Field, Schema};

    fn make_policy() -> VectorStoragePolicy {
        VectorStoragePolicy {
            column_name: "embedding".to_string(),
            dim: 4,
            metric: VectorMetric::Euclidean,
            precision: VectorPrecision::F16,
            pq: None,
            keep_raw_for_reranking: false,
        }
    }

    fn make_batch(ids: &[i32]) -> RecordBatch {
        let schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int32, false),
            Field::new("text", DataType::Utf8, false),
        ]));
        let texts: Vec<&str> = ids.iter().map(|_| "chunk").collect();
        RecordBatch::try_new(
            schema,
            vec![
                Arc::new(Int32Array::from(ids.to_vec())),
                Arc::new(StringArray::from(texts)),
            ],
        )
        .unwrap()
    }

    fn make_embeddings(n: usize, dim: usize) -> Vec<Vec<f32>> {
        (0..n)
            .map(|i| (0..dim).map(|d| i as f32 + d as f32 * 0.1).collect())
            .collect()
    }

    async fn setup_table(catalog: &HadoopCatalog, table: &TableIdent) {
        catalog
            .create_table(
                table,
                &ailake_catalog::TableProperties {
                    policy: make_policy(),
                    extra: Default::default(),
                },
            )
            .await
            .unwrap();
    }

    #[tokio::test]
    async fn mem_table_insert_and_flush() {
        let dir = tempfile::tempdir().unwrap();
        let store: Arc<dyn Store> = Arc::new(LocalStore::new(dir.path()));
        let catalog = Arc::new(HadoopCatalog::new(store.clone(), "warehouse"));
        let table = TableIdent::new("default", "test_mem");
        setup_table(&catalog, &table).await;

        let config = MemTableConfig {
            flush_size_bytes: 1024 * 1024,
            flush_max_rows: 1000,
            flush_interval: Duration::from_secs(60),
        };
        let mut mt = MemTableWriter::new(
            catalog.clone(),
            store.clone(),
            make_policy(),
            table.clone(),
            config,
        );

        for i in 0..3 {
            let ids: Vec<i32> = (i * 5..(i + 1) * 5).collect();
            let batch = make_batch(&ids);
            let embs = make_embeddings(5, 4);
            let snap = mt.insert(&batch, &embs).await.unwrap();
            assert!(snap.is_none(), "should not auto-flush yet");
        }
        assert_eq!(mt.buffered_rows(), 15);

        let snap = mt.flush().await.unwrap();
        assert!(snap > 0, "snapshot id should be non-zero");
        assert_eq!(mt.buffered_rows(), 0, "buffer should be empty after flush");
        assert_eq!(mt.buffered_bytes(), 0);
    }

    #[tokio::test]
    async fn mem_table_auto_flush_on_row_limit() {
        let dir = tempfile::tempdir().unwrap();
        let store: Arc<dyn Store> = Arc::new(LocalStore::new(dir.path()));
        let catalog = Arc::new(HadoopCatalog::new(store.clone(), "warehouse"));
        let table = TableIdent::new("default", "test_auto");
        setup_table(&catalog, &table).await;

        let config = MemTableConfig {
            flush_size_bytes: 1024 * 1024 * 1024,
            flush_max_rows: 8,
            flush_interval: Duration::from_secs(60),
        };
        let mut mt = MemTableWriter::new(
            catalog.clone(),
            store.clone(),
            make_policy(),
            table.clone(),
            config,
        );

        let batch = make_batch(&[1, 2, 3, 4, 5]);
        let embs = make_embeddings(5, 4);
        assert!(mt.insert(&batch, &embs).await.unwrap().is_none());

        let batch2 = make_batch(&[6, 7, 8, 9, 10]);
        let snap = mt.insert(&batch2, &embs).await.unwrap();
        assert!(snap.is_some(), "should auto-flush when row limit exceeded");
        assert_eq!(mt.buffered_rows(), 0);
    }

    #[tokio::test]
    async fn mem_table_flush_if_due() {
        let dir = tempfile::tempdir().unwrap();
        let store: Arc<dyn Store> = Arc::new(LocalStore::new(dir.path()));
        let catalog = Arc::new(HadoopCatalog::new(store.clone(), "warehouse"));
        let table = TableIdent::new("default", "test_due");
        setup_table(&catalog, &table).await;

        let config = MemTableConfig {
            flush_size_bytes: 1024 * 1024 * 1024,
            flush_max_rows: 1000,
            flush_interval: Duration::from_millis(1),
        };
        let mut mt = MemTableWriter::new(
            catalog.clone(),
            store.clone(),
            make_policy(),
            table.clone(),
            config,
        );

        let batch = make_batch(&[1, 2, 3]);
        let embs = make_embeddings(3, 4);
        mt.insert(&batch, &embs).await.unwrap();

        tokio::time::sleep(Duration::from_millis(5)).await;

        let snap = mt.flush_if_due().await.unwrap();
        assert!(snap.is_some(), "should flush because interval elapsed");
        assert_eq!(mt.buffered_rows(), 0);
    }
}
