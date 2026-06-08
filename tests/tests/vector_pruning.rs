// SPDX-License-Identifier: MIT OR Apache-2.0
//! Verifies that files with distant centroids are pruned before opening.

use std::sync::Arc;

use ailake_catalog::{HadoopCatalog, TableIdent};
use ailake_core::{VectorMetric, VectorPrecision, VectorStoragePolicy};
use ailake_query::{search, SearchConfig, TableWriter};
use ailake_store::LocalStore;
use arrow_array::{Int32Array, RecordBatch};
use arrow_schema::{DataType, Field, Schema};
use tempfile::TempDir;

#[tokio::test]
async fn pruning_eliminates_distant_file() {
    let dir = TempDir::new().unwrap();
    let root = dir.path().to_str().unwrap();

    let store = Arc::new(LocalStore::new(root));
    let catalog = Arc::new(HadoopCatalog::new(store.clone(), root));
    let table = TableIdent::new("default", "prune_test");

    let policy = VectorStoragePolicy {
        column_name: "embedding".into(),
        dim: 4,
        metric: VectorMetric::Cosine,
        precision: VectorPrecision::F16,
        pq: None,
        keep_raw_for_reranking: false,
        pre_normalize: false,
        hnsw_m: None,
        hnsw_ef_construction: None,
        rabitq: None,
        binary: None,
    };

    let schema = Arc::new(Schema::new(vec![Field::new("id", DataType::Int32, false)]));

    // File A: vectors near [1, 0, 0, 0]
    let embs_a = vec![
        vec![1.0f32, 0.0, 0.0, 0.0],
        vec![0.9, 0.1, 0.0, 0.0],
        vec![0.95, 0.05, 0.0, 0.0],
    ];
    let batch_a = RecordBatch::try_new(
        schema.clone(),
        vec![Arc::new(Int32Array::from(vec![0i32, 1, 2]))],
    )
    .unwrap();

    // File B: vectors near [0, 0, 0, 1] (far from query)
    let embs_b = vec![
        vec![0.0f32, 0.0, 0.0, 1.0],
        vec![0.0, 0.0, 0.1, 0.9],
        vec![0.0, 0.1, 0.0, 0.9],
    ];
    let batch_b = RecordBatch::try_new(
        schema.clone(),
        vec![Arc::new(Int32Array::from(vec![3i32, 4, 5]))],
    )
    .unwrap();

    let mut writer = TableWriter::create_or_open(
        catalog.clone(),
        store.clone(),
        policy.clone(),
        table.clone(),
    )
    .await
    .unwrap();
    writer.write_batch(&batch_a, &embs_a).await.unwrap();
    writer.write_batch(&batch_b, &embs_b).await.unwrap();
    writer.commit().await.unwrap();

    // Query near [1, 0, 0, 0] — file B should be pruned
    let query = vec![1.0f32, 0.0, 0.0, 0.0];
    let results = search(
        &table,
        &query,
        SearchConfig {
            top_k: 5,
            ef_search: 50,
            pruning_threshold: 0.5, // file B centroid is ~1.0 distance away → pruned
            rerank_factor: None,
        },
        "embedding",
        4,
        catalog as Arc<dyn ailake_catalog::CatalogProvider>,
        store as Arc<dyn ailake_store::Store>,
    )
    .await
    .unwrap();

    // All results must come from file A (data/part-00000.parquet)
    for r in &results {
        assert!(
            r.file_path.contains("part-00000"),
            "result {:?} should come from file A, not file B",
            r
        );
    }
    assert!(!results.is_empty(), "should find results in file A");
}
