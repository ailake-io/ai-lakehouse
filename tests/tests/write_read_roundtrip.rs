// SPDX-License-Identifier: MIT OR Apache-2.0
//! Integration test: write a table, read it back, verify results.

mod fixtures;

use ailake_catalog::{HadoopCatalog, TableIdent};
use ailake_core::{VectorMetric, VectorPrecision, VectorStoragePolicy};
use ailake_query::{search, SearchConfig, TableWriter};
use ailake_store::LocalStore;
use std::sync::Arc;
use tempfile::TempDir;

#[tokio::test]
async fn write_10k_rows_search_top10() {
    let dir = TempDir::new().unwrap();
    let store = Arc::new(LocalStore::new(dir.path()));
    let catalog = Arc::new(HadoopCatalog::new(
        Arc::clone(&store) as Arc<dyn ailake_store::Store>,
        "warehouse",
    ));
    let table = TableIdent::new("default", "test_table");
    let dim = 32u32;

    let policy = VectorStoragePolicy {
        column_name: "embedding".to_string(),
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
    };

    // Create table and write 10k rows split across 2 batches
    let mut writer = TableWriter::create_or_open(
        Arc::clone(&catalog) as Arc<dyn ailake_catalog::CatalogProvider>,
        Arc::clone(&store) as Arc<dyn ailake_store::Store>,
        policy.clone(),
        table.clone(),
    )
    .await
    .unwrap();

    let (batch1, embs1) = fixtures::generate_batch(5000, dim as usize);
    let (batch2, embs2) = fixtures::generate_batch(5000, dim as usize);
    writer.write_batch(&batch1, &embs1).await.unwrap();
    writer.write_batch(&batch2, &embs2).await.unwrap();
    writer.commit().await.unwrap();

    // Search for a known vector (first in batch1 should be top result)
    let query = embs1[0].clone();
    let results = search(
        &table,
        &query,
        SearchConfig {
            top_k: 10,
            ef_search: 50,
            pruning_threshold: f32::INFINITY,
            rerank_factor: None,
            score_fn: None,
            partition_filter: None,
            hybrid: None,
        },
        "embedding",
        dim,
        catalog as Arc<dyn ailake_catalog::CatalogProvider>,
        store as Arc<dyn ailake_store::Store>,
    )
    .await
    .unwrap();

    assert_eq!(results.len(), 10);
    // The query vector itself should be the closest match (distance ~0)
    assert!(
        results[0].distance < 0.01,
        "top result distance too high: {}",
        results[0].distance
    );
}
