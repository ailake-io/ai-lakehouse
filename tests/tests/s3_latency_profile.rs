// SPDX-License-Identifier: MIT OR Apache-2.0
//! S3-compatible object-store latency and range-read profile.

mod fixtures;

use std::ops::Range;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Instant;

use ailake_catalog::{CatalogProvider, HadoopCatalog, TableIdent};
use ailake_core::{VectorMetric, VectorPrecision, VectorStoragePolicy};
use ailake_query::{search, SearchConfig, TableWriter};
use ailake_store::s3::{s3_store, S3Config, S3Credentials};
use ailake_store::Store;
use async_trait::async_trait;
use bytes::Bytes;
use secrecy::SecretString;

#[derive(Default)]
struct ReadStats {
    requests: AtomicU64,
    bytes: AtomicU64,
}

struct CountingStore {
    inner: Arc<dyn Store>,
    stats: Arc<ReadStats>,
}

#[async_trait]
impl Store for CountingStore {
    async fn get(&self, path: &str) -> ailake_core::AilakeResult<Bytes> {
        let bytes = self.inner.get(path).await?;
        self.stats.requests.fetch_add(1, Ordering::Relaxed);
        self.stats
            .bytes
            .fetch_add(bytes.len() as u64, Ordering::Relaxed);
        Ok(bytes)
    }

    async fn get_range(&self, path: &str, range: Range<u64>) -> ailake_core::AilakeResult<Bytes> {
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
async fn s3_compatible_search_reports_real_object_store_reads() {
    let Ok(endpoint) = std::env::var("AILAKE_TEST_S3_ENDPOINT") else {
        return;
    };
    let bucket = std::env::var("AILAKE_TEST_S3_BUCKET").unwrap_or_else(|_| "ailake-perf".into());
    let prefix = format!("perf-{}/", uuid::Uuid::new_v4());
    let store: Arc<dyn Store> = Arc::new(
        s3_store(
            S3Config {
                bucket,
                region: "us-east-1".into(),
                endpoint: Some(endpoint),
                allow_http: true,
                credentials: S3Credentials::Static {
                    access_key_id: "test".into(),
                    secret_access_key: SecretString::from("test"),
                    session_token: None,
                },
            },
            prefix.clone(),
        )
        .unwrap(),
    );
    let catalog: Arc<dyn CatalogProvider> =
        Arc::new(HadoopCatalog::new(Arc::clone(&store), "warehouse"));
    let table = TableIdent::new("default", "s3_latency_profile");
    let dim = 32u32;
    let policy = VectorStoragePolicy {
        column_name: "embedding".into(),
        dim,
        metric: VectorMetric::Cosine,
        precision: VectorPrecision::F16,
        pq: None,
        keep_raw_for_reranking: true,
        pre_normalize: false,
        hnsw_m: Some(16),
        hnsw_ef_construction: Some(100),
        ivf_residual: false,
        embedding_model: None,
        modality: None,
        partition_by: None,
        partition_value: None,
        partition_column_type: None,
        partition_fields: vec![],
    };

    let (batch, vectors) = fixtures::generate_batch(512, dim as usize);
    let mut writer = TableWriter::create_or_open(
        Arc::clone(&catalog),
        Arc::clone(&store),
        policy,
        table.clone(),
        2,
    )
    .await
    .unwrap();
    for rows in 0..8 {
        let start = rows * 64;
        writer
            .write_batch(&batch.slice(start, 64), &vectors[start..start + 64])
            .await
            .unwrap();
    }
    writer.commit().await.unwrap();

    let stats = Arc::new(ReadStats::default());
    let counted_store: Arc<dyn Store> = Arc::new(CountingStore {
        inner: Arc::clone(&store),
        stats: Arc::clone(&stats),
    });
    let config = SearchConfig {
        top_k: 10,
        pruning_threshold: f32::INFINITY,
        ..SearchConfig::default()
    };
    let mut latency_ms = Vec::new();
    for query in vectors.iter().step_by(64).take(8) {
        let started = Instant::now();
        let results = search(
            &table,
            query,
            config.clone(),
            "embedding",
            dim,
            Arc::clone(&catalog),
            Arc::clone(&counted_store),
        )
        .await
        .unwrap();
        assert!(!results.is_empty());
        latency_ms.push(started.elapsed().as_secs_f64() * 1_000.0);
    }
    latency_ms.sort_by(f64::total_cmp);
    let p95_index =
        ((latency_ms.len() as f64 * 0.95).ceil() as usize - 1).min(latency_ms.len() - 1);
    let result = serde_json::json!({
        "profile": std::env::var("AILAKE_TEST_S3_PROFILE_NAME")
            .unwrap_or_else(|_| "s3-compatible".into()),
        "files": 8,
        "rows": 512,
        "queries": 8,
        "p95_ms": latency_ms[p95_index],
        "read_requests": stats.requests.load(Ordering::Relaxed),
        "read_bytes": stats.bytes.load(Ordering::Relaxed),
    });
    println!("PERF_S3_JSON={result}");
    if let Ok(output) = std::env::var("AILAKE_PERF_OUTPUT") {
        let output = std::path::PathBuf::from(output);
        if let Some(parent) = output.parent() {
            std::fs::create_dir_all(parent).unwrap();
        }
        std::fs::write(output, format!("{result}\n")).unwrap();
    }

    for path in store.list("").await.unwrap() {
        store.delete(&path).await.unwrap();
    }
}
