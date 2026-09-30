// SPDX-License-Identifier: MIT OR Apache-2.0
//! Deterministic recall/latency smoke test for the CPU search path.

mod fixtures;

use std::collections::HashSet;
use std::sync::Arc;
use std::time::Instant;

use ailake_catalog::{CatalogProvider, HadoopCatalog, TableIdent};
use ailake_core::{VectorMetric, VectorPrecision, VectorStoragePolicy};
use ailake_query::{search, SearchConfig, TableWriter};
use ailake_store::LocalStore;
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

#[tokio::test]
async fn cpu_search_recall_and_p95_are_regression_safe() {
    let dir = TempDir::new().unwrap();
    let store: Arc<dyn ailake_store::Store> = Arc::new(LocalStore::new(dir.path()));
    let catalog: Arc<dyn CatalogProvider> =
        Arc::new(HadoopCatalog::new(Arc::clone(&store), "warehouse"));
    let table = TableIdent::new("default", "performance_regression");
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
        hnsw_ef_construction: Some(150),
        ivf_residual: false,
        embedding_model: None,
        modality: None,
        partition_by: None,
        partition_value: None,
        partition_column_type: None,
        partition_fields: vec![],
    };

    let (batch, vectors) = fixtures::generate_batch(2_000, dim as usize);
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

    let queries = [0usize, 137, 511, 999, 1_337, 1_999];
    let top_k = 10usize;
    let mut latencies = Vec::with_capacity(queries.len());
    let mut hits = 0usize;
    for &query_index in &queries {
        let query = &vectors[query_index];
        let started = Instant::now();
        let results = search(
            &table,
            query,
            SearchConfig {
                top_k,
                ef_search: 50,
                pruning_threshold: f32::INFINITY,
                rerank_factor: None,
                score_fn: None,
                partition_filter: None,
                hybrid: None,
                column_filter: None,
                strict_deletes: false,
            },
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

    let recall = hits as f64 / (queries.len() * top_k) as f64;
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
        "PERF_RECALL_JSON={{\"recall_at_10\":{recall:.6},\"p95_ms\":{p95_ms:.3},\"queries\":{},\"rows\":{}}}",
        queries.len(),
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
}
