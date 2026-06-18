// SPDX-License-Identifier: MIT OR Apache-2.0
use std::collections::VecDeque;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tracing::info;

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
            info!(
                "ailake: MemTable auto-flush triggered — {} rows / {} bytes buffered",
                self.pending_embeddings.len(),
                self.buffered_bytes
            );
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
            info!(
                "ailake: MemTable time-based flush — {} rows / {} bytes buffered (interval={}s)",
                self.pending_embeddings.len(),
                self.buffered_bytes,
                self.config.flush_interval.as_secs()
            );
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

// ─── WorkingMemoryBuffer ──────────────────────────────────────────────────────

/// Single entry in a `WorkingMemoryBuffer`.
#[derive(Debug, Clone)]
pub struct WorkingMemoryEntry {
    /// Text content (chunk_text or tool call summary).
    pub text: String,
    /// Embedding vector for similarity search.
    pub embedding: Vec<f32>,
    /// Agent-assigned importance score (0.0–1.0). Influences hybrid scoring.
    pub importance: f32,
}

/// Bounded in-memory buffer for agent short-term memory.
///
/// Stores the N most recent entries (text + embedding). When full, the oldest
/// entry is evicted on each `push`. Supports brute-force flat scan (`search`)
/// and draining all entries to an AI-Lake table (`drain_to_table`).
///
/// # Cascade pattern
///
/// Short-term agents use only `WorkingMemoryBuffer`. Long-term agents cascade:
/// 1. `search` queries the buffer first (fast, recent).
/// 2. When full, `drain_to_table` persists to AI-Lake; continue searching via
///    `ailake::search` for historical context.
///
/// # Example
///
/// ```ignore
/// let mut wm = WorkingMemoryBuffer::new(100);
/// wm.push("Meeting notes: …", embedding, 0.8);
/// let top3 = wm.search(&query_vec, 3);
/// if wm.is_full() {
///     wm.drain_to_table(&mut writer).await?;
///     writer.commit().await?;
/// }
/// ```
pub struct WorkingMemoryBuffer {
    max_rows: usize,
    entries: VecDeque<WorkingMemoryEntry>,
}

impl WorkingMemoryBuffer {
    /// Create buffer with at most `max_rows` entries. Evicts oldest on overflow.
    pub fn new(max_rows: usize) -> Self {
        Self {
            max_rows: max_rows.max(1),
            entries: VecDeque::with_capacity(max_rows.min(4096)),
        }
    }

    /// Add entry. If at capacity, evicts the oldest entry (FIFO).
    pub fn push(&mut self, text: impl Into<String>, embedding: Vec<f32>, importance: f32) {
        if self.entries.len() >= self.max_rows {
            self.entries.pop_front();
        }
        self.entries.push_back(WorkingMemoryEntry {
            text: text.into(),
            embedding,
            importance: importance.clamp(0.0, 1.0),
        });
    }

    /// Brute-force cosine similarity scan. Returns `(distance, entry)` pairs
    /// sorted ascending (smallest distance = most similar). Distances in `[0, 2]`.
    pub fn search(&self, query: &[f32], top_k: usize) -> Vec<(f32, &WorkingMemoryEntry)> {
        if self.entries.is_empty() || top_k == 0 {
            return vec![];
        }
        let q_norm = l2_norm(query);
        let mut scored: Vec<(f32, usize)> = self
            .entries
            .iter()
            .enumerate()
            .map(|(i, e)| {
                let v_norm = l2_norm(&e.embedding);
                let dot: f32 = query.iter().zip(&e.embedding).map(|(a, b)| a * b).sum();
                let cos_sim = if q_norm * v_norm < f32::EPSILON {
                    0.0
                } else {
                    dot / (q_norm * v_norm)
                };
                // cosine distance: lower = more similar
                (1.0 - cos_sim, i)
            })
            .collect();

        scored.sort_unstable_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
        scored.truncate(top_k);
        scored
            .into_iter()
            .map(|(dist, idx)| (dist, &self.entries[idx]))
            .collect()
    }

    /// Write all buffered entries to an AI-Lake table and clear the buffer.
    ///
    /// Creates a RecordBatch with a `chunk_text` column. Call `writer.commit()`
    /// after to persist the snapshot.
    pub async fn drain_to_table(&mut self, writer: &mut TableWriter) -> AilakeResult<()> {
        use arrow_array::StringArray;
        use arrow_schema::{DataType, Field, Schema};

        if self.entries.is_empty() {
            return Ok(());
        }

        let texts: Vec<&str> = self.entries.iter().map(|e| e.text.as_str()).collect();
        let embeddings: Vec<Vec<f32>> = self.entries.iter().map(|e| e.embedding.clone()).collect();
        let importance_vals: Vec<f32> = self.entries.iter().map(|e| e.importance).collect();

        let schema = Arc::new(Schema::new(vec![
            Field::new("chunk_text", DataType::Utf8, false),
            Field::new("importance_score", DataType::Float32, false),
        ]));
        let batch = arrow_array::RecordBatch::try_new(
            schema,
            vec![
                Arc::new(StringArray::from(texts)) as _,
                Arc::new(arrow_array::Float32Array::from(importance_vals)) as _,
            ],
        )
        .map_err(|e| ailake_core::AilakeError::Arrow(e.to_string()))?;

        writer.write_batch(&batch, &embeddings).await?;
        self.entries.clear();
        Ok(())
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// True when the buffer has reached its configured capacity.
    pub fn is_full(&self) -> bool {
        self.entries.len() >= self.max_rows
    }
}

fn l2_norm(v: &[f32]) -> f32 {
    v.iter().map(|x| x * x).sum::<f32>().sqrt()
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
                    format_version: 2,
                    partition_column_type: None,
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

    #[test]
    fn working_memory_evicts_oldest() {
        let mut wm = WorkingMemoryBuffer::new(3);
        wm.push("a", vec![1.0, 0.0], 1.0);
        wm.push("b", vec![0.0, 1.0], 1.0);
        wm.push("c", vec![1.0, 1.0], 1.0);
        assert_eq!(wm.len(), 3);
        assert!(wm.is_full());

        wm.push("d", vec![0.5, 0.5], 1.0);
        assert_eq!(wm.len(), 3);
        // "a" should be evicted
        assert!(!wm.entries.iter().any(|e| e.text == "a"));
        assert!(wm.entries.iter().any(|e| e.text == "d"));
    }

    #[test]
    fn working_memory_search_ranks_similar_first() {
        let mut wm = WorkingMemoryBuffer::new(10);
        wm.push("near", vec![1.0, 0.0, 0.0], 1.0);
        wm.push("far",  vec![0.0, 1.0, 0.0], 1.0);
        wm.push("very far", vec![0.0, 0.0, 1.0], 1.0);

        let query = vec![1.0, 0.0, 0.0];
        let results = wm.search(&query, 3);
        assert_eq!(results.len(), 3);
        // "near" should have smallest cosine distance
        assert_eq!(results[0].1.text, "near");
        assert!(results[0].0 < results[1].0);
    }

    #[test]
    fn working_memory_search_empty() {
        let wm = WorkingMemoryBuffer::new(10);
        assert!(wm.search(&[1.0, 0.0], 5).is_empty());
    }

    #[tokio::test]
    async fn working_memory_drain_to_table() {
        let dir = tempfile::tempdir().unwrap();
        let store: Arc<dyn Store> = Arc::new(LocalStore::new(dir.path()));
        let catalog = Arc::new(HadoopCatalog::new(store.clone(), "warehouse"));
        let table = TableIdent::new("default", "test_wm_drain");

        let policy = VectorStoragePolicy {
            column_name: "embedding".to_string(),
            dim: 3,
            metric: ailake_core::VectorMetric::Cosine,
            precision: ailake_core::VectorPrecision::F16,
            pq: None,
            keep_raw_for_reranking: false,
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
        catalog
            .create_table(
                &table,
                &ailake_catalog::TableProperties {
                    policy: policy.clone(),
                    extra: Default::default(),
                    format_version: 2,
                    partition_column_type: None,
                },
            )
            .await
            .unwrap();

        let mut wm = WorkingMemoryBuffer::new(5);
        wm.push("memory one", vec![1.0, 0.0, 0.0], 0.9);
        wm.push("memory two", vec![0.0, 1.0, 0.0], 0.5);

        let mut writer = TableWriter::new(
            Arc::clone(&catalog) as Arc<dyn ailake_catalog::CatalogProvider>,
            Arc::clone(&store) as Arc<dyn ailake_store::Store>,
            policy,
            table,
        );
        wm.drain_to_table(&mut writer).await.unwrap();
        assert!(wm.is_empty());
        let snap = writer.commit().await.unwrap();
        assert!(snap > 0);
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
