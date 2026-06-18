// SPDX-License-Identifier: MIT OR Apache-2.0
//! Tests that partition_by / partition_value / partition_filter isolate writes
//! and searches to per-agent file subsets, and that score_fn is called during search.

mod fixtures;

use std::sync::{
    atomic::{AtomicUsize, Ordering},
    Arc,
};

use ailake_catalog::{HadoopCatalog, TableIdent};
use ailake_core::{VectorMetric, VectorPrecision, VectorStoragePolicy};
use ailake_query::{search, ScoreFn, SearchConfig, TableWriter};
use ailake_store::LocalStore;
use tempfile::TempDir;

fn policy(partition_by: Option<String>, partition_value: Option<String>) -> VectorStoragePolicy {
    VectorStoragePolicy {
        column_name: "embedding".to_string(),
        dim: 16,
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
        partition_by,
        partition_value,
        partition_column_type: None,
    }
}

async fn write_agent_shard(
    catalog: Arc<HadoopCatalog>,
    store: Arc<LocalStore>,
    table: &TableIdent,
    agent: &str,
    embs: Vec<Vec<f32>>,
) {
    let dim = embs[0].len();
    use arrow_array::{Int32Array, RecordBatch};
    use arrow_schema::{DataType, Field, Schema};
    let schema = Arc::new(Schema::new(vec![Field::new("id", DataType::Int32, false)]));
    let ids: Vec<i32> = (0..embs.len() as i32).collect();
    let batch = RecordBatch::try_new(
        schema,
        vec![Arc::new(Int32Array::from(ids))],
    )
    .unwrap();

    let mut writer = TableWriter::create_or_open(
        Arc::clone(&catalog) as Arc<dyn ailake_catalog::CatalogProvider>,
        Arc::clone(&store) as Arc<dyn ailake_store::Store>,
        policy(Some("agent_id".into()), Some(agent.into())),
        table.clone(),
        2,
    )
    .await
    .unwrap();
    let _ = dim;
    writer.write_batch(&batch, &embs).await.unwrap();
    writer.commit().await.unwrap();
}

/// Write two batches tagged with different partition values (agent-A vs agent-B).
/// Each agent gets vectors from an orthogonal cluster so that:
///   - searching agent-A with agent-A's centroid → distance ~0
///   - searching agent-B with agent-A's centroid → distance ~1 (orthogonal)
#[tokio::test]
async fn partition_filter_isolates_per_agent_search() {
    let dir = TempDir::new().unwrap();
    let store = Arc::new(LocalStore::new(dir.path()));
    let catalog = Arc::new(HadoopCatalog::new(
        Arc::clone(&store) as Arc<dyn ailake_store::Store>,
        "warehouse",
    ));
    let table = TableIdent::new("default", "agent_table");
    let dim = 16usize;

    // Agent-A cluster: center near [1, 0, 0, ...]
    let mut center_a = vec![0.0f32; dim];
    center_a[0] = 1.0;
    let embs_a = fixtures::cluster_around(&center_a, dim, 50, 0.05);

    // Agent-B cluster: center near [0, 1, 0, ...] — orthogonal to agent-A
    let mut center_b = vec![0.0f32; dim];
    center_b[1] = 1.0;
    let embs_b = fixtures::cluster_around(&center_b, dim, 50, 0.05);

    write_agent_shard(Arc::clone(&catalog), Arc::clone(&store), &table, "agent-A", embs_a.clone()).await;
    write_agent_shard(Arc::clone(&catalog), Arc::clone(&store), &table, "agent-B", embs_b).await;

    // Query is agent-A's center vector.
    let query = center_a.clone();

    // Search restricted to agent-A → top-1 should be very close to center_a (dist ~0.05).
    let results_a = search(
        &table,
        &query,
        SearchConfig {
            top_k: 5,
            ef_search: 50,
            pruning_threshold: f32::INFINITY,
            rerank_factor: None,
            score_fn: None,
            partition_filter: Some("agent-A".to_string()),
            hybrid: None,
        },
        "embedding",
        dim as u32,
        Arc::clone(&catalog) as Arc<dyn ailake_catalog::CatalogProvider>,
        Arc::clone(&store) as Arc<dyn ailake_store::Store>,
    )
    .await
    .unwrap();

    assert!(!results_a.is_empty(), "agent-A search should return results");
    assert!(
        results_a[0].distance < 0.2,
        "top-1 for agent-A partition should be near center_a, got dist {}",
        results_a[0].distance
    );

    // Search restricted to agent-B with agent-A's query.
    // Agent-B vectors are near center_b (orthogonal to center_a) → distances must be large.
    let results_b = search(
        &table,
        &query,
        SearchConfig {
            top_k: 5,
            ef_search: 50,
            pruning_threshold: f32::INFINITY,
            rerank_factor: None,
            score_fn: None,
            partition_filter: Some("agent-B".to_string()),
            hybrid: None,
        },
        "embedding",
        dim as u32,
        Arc::clone(&catalog) as Arc<dyn ailake_catalog::CatalogProvider>,
        Arc::clone(&store) as Arc<dyn ailake_store::Store>,
    )
    .await
    .unwrap();

    // Agent-B results must be farther from center_a than agent-A results.
    if !results_b.is_empty() {
        assert!(
            results_b[0].distance > results_a[0].distance + 0.1,
            "agent-B (dist {}) should be farther than agent-A (dist {}) for agent-A query",
            results_b[0].distance,
            results_a[0].distance
        );
    }
}

/// partition_filter with no matching files returns empty result set (not an error).
#[tokio::test]
async fn partition_filter_nonexistent_returns_empty() {
    let dir = TempDir::new().unwrap();
    let store = Arc::new(LocalStore::new(dir.path()));
    let catalog = Arc::new(HadoopCatalog::new(
        Arc::clone(&store) as Arc<dyn ailake_store::Store>,
        "warehouse",
    ));
    let table = TableIdent::new("default", "agent_table2");
    let dim = 16usize;

    let mut center = vec![0.0f32; dim];
    center[0] = 1.0;
    let embs = fixtures::cluster_around(&center, dim, 10, 0.05);
    write_agent_shard(Arc::clone(&catalog), Arc::clone(&store), &table, "agent-A", embs).await;

    let results = search(
        &table,
        &center,
        SearchConfig {
            top_k: 5,
            ef_search: 50,
            pruning_threshold: f32::INFINITY,
            rerank_factor: None,
            score_fn: None,
            partition_filter: Some("nonexistent-agent".to_string()),
            hybrid: None,
        },
        "embedding",
        dim as u32,
        Arc::clone(&catalog) as Arc<dyn ailake_catalog::CatalogProvider>,
        Arc::clone(&store) as Arc<dyn ailake_store::Store>,
    )
    .await
    .unwrap();

    assert!(
        results.is_empty(),
        "nonexistent partition filter should return 0 results, got {}",
        results.len()
    );
}

/// Unfiltered search must return at least as many results as partition-filtered.
#[tokio::test]
async fn unfiltered_search_spans_all_partitions() {
    let dir = TempDir::new().unwrap();
    let store = Arc::new(LocalStore::new(dir.path()));
    let catalog = Arc::new(HadoopCatalog::new(
        Arc::clone(&store) as Arc<dyn ailake_store::Store>,
        "warehouse",
    ));
    let table = TableIdent::new("default", "multi_agent");
    let dim = 16usize;

    for (i, agent) in ["agent-X", "agent-Y"].iter().enumerate() {
        let mut center = vec![0.0f32; dim];
        center[i] = 1.0;
        let embs = fixtures::cluster_around(&center, dim, 30, 0.05);
        write_agent_shard(Arc::clone(&catalog), Arc::clone(&store), &table, agent, embs).await;
    }

    // Query near agent-X's cluster center
    let mut query = vec![0.0f32; dim];
    query[0] = 1.0;

    let base_cfg = SearchConfig {
        top_k: 20,
        ef_search: 50,
        pruning_threshold: f32::INFINITY,
        rerank_factor: None,
        score_fn: None,
        partition_filter: None,
        hybrid: None,
    };

    let results_all = search(
        &table,
        &query,
        base_cfg.clone(),
        "embedding",
        dim as u32,
        Arc::clone(&catalog) as Arc<dyn ailake_catalog::CatalogProvider>,
        Arc::clone(&store) as Arc<dyn ailake_store::Store>,
    )
    .await
    .unwrap();

    let results_x = search(
        &table,
        &query,
        SearchConfig { partition_filter: Some("agent-X".to_string()), ..base_cfg },
        "embedding",
        dim as u32,
        Arc::clone(&catalog) as Arc<dyn ailake_catalog::CatalogProvider>,
        Arc::clone(&store) as Arc<dyn ailake_store::Store>,
    )
    .await
    .unwrap();

    assert!(
        results_all.len() >= results_x.len(),
        "unfiltered ({}) < partition-filtered ({})",
        results_all.len(),
        results_x.len()
    );
}

/// score_fn is invoked for each candidate row during search.
#[tokio::test]
async fn score_fn_is_invoked_during_search() {
    let dir = TempDir::new().unwrap();
    let store = Arc::new(LocalStore::new(dir.path()));
    let catalog = Arc::new(HadoopCatalog::new(
        Arc::clone(&store) as Arc<dyn ailake_store::Store>,
        "warehouse",
    ));
    let table = TableIdent::new("default", "score_fn_test");
    let dim = 16usize;

    let (batch, embs) = fixtures::generate_batch(50, dim);
    let mut writer = TableWriter::create_or_open(
        Arc::clone(&catalog) as Arc<dyn ailake_catalog::CatalogProvider>,
        Arc::clone(&store) as Arc<dyn ailake_store::Store>,
        policy(None, None),
        table.clone(),
        2,
    )
    .await
    .unwrap();
    writer.write_batch(&batch, &embs).await.unwrap();
    writer.commit().await.unwrap();

    let call_count = Arc::new(AtomicUsize::new(0));
    let counter = Arc::clone(&call_count);

    let results = search(
        &table,
        &embs[0],
        SearchConfig {
            top_k: 5,
            ef_search: 50,
            pruning_threshold: f32::INFINITY,
            rerank_factor: None,
            score_fn: Some(ScoreFn::new(move |distance, _row| {
                counter.fetch_add(1, Ordering::Relaxed);
                distance
            })),
            partition_filter: None,
            hybrid: None,
        },
        "embedding",
        dim as u32,
        Arc::clone(&catalog) as Arc<dyn ailake_catalog::CatalogProvider>,
        Arc::clone(&store) as Arc<dyn ailake_store::Store>,
    )
    .await
    .unwrap();

    if !results.is_empty() {
        assert!(
            call_count.load(Ordering::Relaxed) > 0,
            "score_fn was never called despite {} results",
            results.len()
        );
    }
}

/// score_fn with constant 0.0 return: results have score=0.0 in distance field, no panic.
#[tokio::test]
async fn score_fn_constant_zero_returns_rows() {
    let dir = TempDir::new().unwrap();
    let store = Arc::new(LocalStore::new(dir.path()));
    let catalog = Arc::new(HadoopCatalog::new(
        Arc::clone(&store) as Arc<dyn ailake_store::Store>,
        "warehouse",
    ));
    let table = TableIdent::new("default", "score_fn_zero");
    let dim = 16usize;

    let (batch, embs) = fixtures::generate_batch(20, dim);
    let mut writer = TableWriter::create_or_open(
        Arc::clone(&catalog) as Arc<dyn ailake_catalog::CatalogProvider>,
        Arc::clone(&store) as Arc<dyn ailake_store::Store>,
        policy(None, None),
        table.clone(),
        2,
    )
    .await
    .unwrap();
    writer.write_batch(&batch, &embs).await.unwrap();
    writer.commit().await.unwrap();

    let results = search(
        &table,
        &embs[0],
        SearchConfig {
            top_k: 5,
            ef_search: 50,
            pruning_threshold: f32::INFINITY,
            rerank_factor: None,
            score_fn: Some(ScoreFn::new(|_d, _row| 0.0)),
            partition_filter: None,
            hybrid: None,
        },
        "embedding",
        dim as u32,
        Arc::clone(&catalog) as Arc<dyn ailake_catalog::CatalogProvider>,
        Arc::clone(&store) as Arc<dyn ailake_store::Store>,
    )
    .await
    .unwrap();

    assert!(!results.is_empty(), "expected results with constant score_fn");
    for r in &results {
        assert!(
            (r.distance - 0.0f32).abs() < 1e-6,
            "expected distance=0.0 from constant score_fn, got {}",
            r.distance
        );
    }
}
