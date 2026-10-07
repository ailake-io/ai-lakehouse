// SPDX-License-Identifier: MIT OR Apache-2.0
//! Deterministic recall/latency smoke test for the CPU search path.

mod fixtures;

use std::collections::HashSet;
use std::ops::Range;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use ailake_catalog::{
    new_snapshot_id, CatalogProvider, HadoopCatalog, IndexStatus, NewSnapshot, SnapshotOperation,
    TableIdent,
};
use ailake_core::{VectorMetric, VectorPrecision, VectorStoragePolicy};
use ailake_query::{search, SearchConfig, TableWriter};
use ailake_store::{LocalStore, Store};
use async_trait::async_trait;
use bytes::Bytes;
use tempfile::TempDir;

fn percentile_ms(values: &mut [f64], percentile: f64) -> f64 {
    values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let index = ((values.len() as f64 * percentile).ceil() as usize).saturating_sub(1);
    values[index.min(values.len().saturating_sub(1))]
}

fn exact_top_k(query: &[f32], vectors: &[Vec<f32>], top_k: usize) -> HashSet<u64> {
    let mut scored: Vec<(usize, f32)> = vectors
        .iter()
        .enumerate()
        .map(|(index, vector)| {
            let dot: f32 = query
                .iter()
                .zip(vector)
                .map(|(left, right)| left * right)
                .sum();
            (index, 1.0 - dot)
        })
        .collect();
    scored.sort_unstable_by(|left, right| {
        left.1
            .partial_cmp(&right.1)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    scored
        .into_iter()
        .take(top_k)
        .map(|(index, _)| index as u64)
        .collect()
}

#[derive(Default)]
struct ReadStats {
    requests: AtomicU64,
    bytes: AtomicU64,
}

struct LatencyStore {
    inner: Arc<dyn Store>,
    per_read_delay: Duration,
    stats: Arc<ReadStats>,
}

#[async_trait]
impl Store for LatencyStore {
    async fn get(&self, path: &str) -> ailake_core::AilakeResult<Bytes> {
        tokio::time::sleep(self.per_read_delay).await;
        let bytes = self.inner.get(path).await?;
        self.stats.requests.fetch_add(1, Ordering::Relaxed);
        self.stats
            .bytes
            .fetch_add(bytes.len() as u64, Ordering::Relaxed);
        Ok(bytes)
    }

    async fn get_range(&self, path: &str, range: Range<u64>) -> ailake_core::AilakeResult<Bytes> {
        tokio::time::sleep(self.per_read_delay).await;
        let bytes = self.inner.get_range(path, range).await?;
        self.stats.requests.fetch_add(1, Ordering::Relaxed);
        self.stats
            .bytes
            .fetch_add(bytes.len() as u64, Ordering::Relaxed);
        Ok(bytes)
    }

    async fn put(&self, path: &str, data: Bytes) -> ailake_core::AilakeResult<()> {
        self.inner.put(path, data).await
    }

    async fn list(&self, prefix: &str) -> ailake_core::AilakeResult<Vec<String>> {
        self.inner.list(prefix).await
    }

    async fn file_size(&self, path: &str) -> ailake_core::AilakeResult<u64> {
        self.inner.file_size(path).await
    }

    async fn exists(&self, path: &str) -> ailake_core::AilakeResult<bool> {
        self.inner.exists(path).await
    }

    async fn delete(&self, path: &str) -> ailake_core::AilakeResult<()> {
        self.inner.delete(path).await
    }
}

#[tokio::test]
async fn cpu_search_recall_and_p95_are_regression_safe() {
    let dir = TempDir::new().unwrap();
    let store: Arc<dyn ailake_store::Store> = Arc::new(LocalStore::new(dir.path()));
    let catalog: Arc<dyn CatalogProvider> =
        Arc::new(HadoopCatalog::new(Arc::clone(&store), "warehouse"));
    let table = TableIdent::new("default", "performance_regression");
    let dim = 128u32;
    let policy = VectorStoragePolicy {
        column_name: "embedding".into(),
        dim,
        metric: VectorMetric::Cosine,
        precision: VectorPrecision::F16,
        pq: None,
        keep_raw_for_reranking: true,
        pre_normalize: false,
        hnsw_m: Some(16),
        hnsw_ef_construction: Some(150),
        ivf_residual: false,
        embedding_model: None,
        modality: None,
        partition_by: None,
        partition_value: None,
        partition_column_type: None,
        partition_fields: vec![],
    };

    let (batch, vectors) = fixtures::generate_batch(10_000, dim as usize);
    let mut writer = TableWriter::create_or_open(
        Arc::clone(&catalog),
        Arc::clone(&store),
        policy,
        table.clone(),
        2,
    )
    .await
    .unwrap();
    writer.write_batch(&batch, &vectors).await.unwrap();
    writer.commit().await.unwrap();

    // A second store view adds a fixed delay per object read for an isolated
    // remote-latency simulation. The regression gate below remains local.
    let read_stats = Arc::new(ReadStats::default());
    let read_store: Arc<dyn Store> = Arc::new(LatencyStore {
        inner: Arc::clone(&store),
        per_read_delay: Duration::from_millis(5),
        stats: Arc::clone(&read_stats),
    });

    let query_indices: Vec<usize> = (0..32).map(|i| (i * 313) % vectors.len()).collect();
    let top_k = 10usize;
    let search_config = SearchConfig {
        top_k,
        ef_search: 50,
        pruning_threshold: f32::INFINITY,
        rerank_factor: None,
        score_fn: None,
        partition_filter: None,
        hybrid: None,
        column_filter: None,
        strict_deletes: true,
    };

    let first_search_started = Instant::now();
    search(
        &table,
        &vectors[query_indices[0]],
        search_config.clone(),
        "embedding",
        dim,
        Arc::clone(&catalog),
        Arc::clone(&store),
    )
    .await
    .unwrap();
    let first_search_ms = first_search_started.elapsed().as_secs_f64() * 1_000.0;

    // Warm the catalog and index path before collecting steady-state samples.
    for &query_index in query_indices.iter().take(4) {
        search(
            &table,
            &vectors[query_index],
            search_config.clone(),
            "embedding",
            dim,
            Arc::clone(&catalog),
            Arc::clone(&store),
        )
        .await
        .unwrap();
    }

    let mut latencies = Vec::with_capacity(query_indices.len());
    let mut hits = 0usize;
    for &query_index in &query_indices {
        let query = &vectors[query_index];
        let started = Instant::now();
        let results = search(
            &table,
            query,
            search_config.clone(),
            "embedding",
            dim,
            Arc::clone(&catalog),
            Arc::clone(&store),
        )
        .await
        .unwrap();
        latencies.push(started.elapsed().as_secs_f64() * 1_000.0);

        let expected = exact_top_k(query, &vectors, top_k);
        let actual: HashSet<u64> = results.iter().map(|result| result.row_id.0).collect();
        hits += actual.intersection(&expected).count();
    }

    let recall = hits as f64 / (query_indices.len() * top_k) as f64;
    let p95_ms = percentile_ms(&mut latencies, 0.95);
    let min_recall = std::env::var("AILAKE_MIN_RECALL")
        .ok()
        .and_then(|value| value.parse::<f64>().ok())
        .unwrap_or(0.95);
    let max_p95_ms = std::env::var("AILAKE_MAX_P95_MS")
        .ok()
        .and_then(|value| value.parse::<f64>().ok())
        .unwrap_or(500.0);

    println!(
        "PERF_RECALL_JSON={{\"recall_at_10\":{recall:.6},\"p95_ms\":{p95_ms:.3},\"first_search_ms\":{first_search_ms:.3},\"queries\":{},\"rows\":{}}}",
        query_indices.len(),
        vectors.len()
    );

    read_stats.requests.store(0, Ordering::Relaxed);
    read_stats.bytes.store(0, Ordering::Relaxed);
    let mut delayed_latencies = Vec::new();
    for &query_index in query_indices.iter().skip(8).take(8) {
        let started = Instant::now();
        search(
            &table,
            &vectors[query_index],
            search_config.clone(),
            "embedding",
            dim,
            Arc::clone(&catalog),
            Arc::clone(&read_store),
        )
        .await
        .unwrap();
        delayed_latencies.push(started.elapsed().as_secs_f64() * 1_000.0);
    }
    let delayed_p95_ms = percentile_ms(&mut delayed_latencies, 0.95);
    println!(
        "PERF_OBJECT_LATENCY_JSON={{\"simulated_read_delay_ms\":5,\"p95_ms\":{delayed_p95_ms:.3},\"read_requests\":{},\"read_bytes\":{},\"queries\":8}}",
        read_stats.requests.load(Ordering::Relaxed),
        read_stats.bytes.load(Ordering::Relaxed)
    );

    let table_metadata = catalog.load_table(&table).await.unwrap();
    let mut files = catalog.list_files(&table, None).await.unwrap();
    assert_eq!(files.len(), 1, "fallback profile expects one data shard");
    files[0].index_status = IndexStatus::Indexing;
    catalog
        .commit_snapshot(
            &table,
            NewSnapshot {
                snapshot_id: new_snapshot_id(),
                parent_snapshot_id: table_metadata.current_snapshot_id,
                files,
                operation: SnapshotOperation::Overwrite,
                iceberg_schema: None,
                extra_properties: Default::default(),
                bloom_filters: vec![],
                equality_delete_files: vec![],
            },
        )
        .await
        .unwrap();

    read_stats.requests.store(0, Ordering::Relaxed);
    read_stats.bytes.store(0, Ordering::Relaxed);
    let fallback_queries = query_indices.iter().take(8);
    let mut fallback_latencies = Vec::new();
    let mut fallback_hits = 0usize;
    for &query_index in fallback_queries {
        let query = &vectors[query_index];
        let started = Instant::now();
        let results = search(
            &table,
            query,
            search_config.clone(),
            "embedding",
            dim,
            Arc::clone(&catalog),
            Arc::clone(&store),
        )
        .await
        .unwrap();
        fallback_latencies.push(started.elapsed().as_secs_f64() * 1_000.0);
        let expected = exact_top_k(query, &vectors, top_k);
        let actual: HashSet<u64> = results.iter().map(|result| result.row_id.0).collect();
        fallback_hits += actual.intersection(&expected).count();
    }
    let fallback_query_count = fallback_latencies.len();
    let fallback_recall = fallback_hits as f64 / (fallback_query_count * top_k) as f64;
    let fallback_p95_ms = percentile_ms(&mut fallback_latencies, 0.95);
    println!(
        "PERF_FALLBACK_JSON={{\"fallback_recall_at_10\":{fallback_recall:.6},\"fallback_p95_ms\":{fallback_p95_ms:.3},\"queries\":{fallback_query_count},\"rows_per_query\":{}}}",
        vectors.len()
    );
    assert!(
        recall >= min_recall,
        "Recall@10 {recall:.4} below configured minimum {min_recall:.4}"
    );
    assert!(
        p95_ms <= max_p95_ms,
        "search p95 {p95_ms:.2} ms above configured maximum {max_p95_ms:.2} ms"
    );
    assert!(
        fallback_recall >= min_recall,
        "flat-scan Recall@10 {fallback_recall:.4} below configured minimum {min_recall:.4}"
    );
}
