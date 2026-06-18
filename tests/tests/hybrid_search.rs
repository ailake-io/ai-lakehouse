// SPDX-License-Identifier: MIT OR Apache-2.0
//! Integration tests: BM25 hybrid search and pure text search.

mod fixtures;

use std::sync::Arc;

use ailake_catalog::{CatalogProvider, HadoopCatalog, TableIdent};
use ailake_core::{VectorMetric, VectorPrecision, VectorStoragePolicy};
use ailake_query::{search_text, BM25Scorer, HybridConfig, IdfStats, SearchConfig, TableWriter};
use ailake_store::LocalStore;
use arrow_array::{RecordBatch, StringArray};
use arrow_schema::{DataType, Field, Schema};
use tempfile::TempDir;

fn make_policy(dim: u32) -> VectorStoragePolicy {
    VectorStoragePolicy {
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
    }
}

fn rand_unit_vec(dim: usize, seed: u64) -> Vec<f32> {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};
    let v: Vec<f32> = (0..dim)
        .map(|j| {
            let mut h = DefaultHasher::new();
            (seed * 1000 + j as u64).hash(&mut h);
            let bits = (h.finish() & 0x3FFF_FFFF) as u32;
            (bits as f32 / 0x3FFF_FFFF as f32) * 2.0 - 1.0
        })
        .collect();
    let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 0.0 {
        v.into_iter().map(|x| x / norm).collect()
    } else {
        v
    }
}

/// Write a table with `chunk_text` column and BM25 indexing enabled.
/// Returns (dir, catalog, store, table).
async fn setup_bm25_table(
    texts: &[&str],
    embeddings: &[Vec<f32>],
    dim: u32,
) -> (TempDir, Arc<HadoopCatalog>, Arc<LocalStore>, TableIdent) {
    let dir = TempDir::new().unwrap();
    let store = Arc::new(LocalStore::new(dir.path()));
    let catalog = Arc::new(HadoopCatalog::new(
        Arc::clone(&store) as Arc<dyn ailake_store::Store>,
        "warehouse",
    ));
    let table = TableIdent::new("default", "test_bm25");

    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int32, false),
        Field::new("chunk_text", DataType::Utf8, false),
    ]));
    let ids: Vec<i32> = (0..texts.len() as i32).collect();
    let batch = RecordBatch::try_new(
        schema,
        vec![
            Arc::new(arrow_array::Int32Array::from(ids)),
            Arc::new(StringArray::from(texts.to_vec())),
        ],
    )
    .unwrap();

    let mut writer = TableWriter::create_or_open(
        Arc::clone(&catalog) as Arc<dyn ailake_catalog::CatalogProvider>,
        Arc::clone(&store) as Arc<dyn ailake_store::Store>,
        make_policy(dim),
        table.clone(),
    )
    .await
    .unwrap();
    writer = writer.with_bm25("chunk_text".to_string());
    writer.write_batch(&batch, embeddings).await.unwrap();
    writer.commit().await.unwrap();

    (dir, catalog, store, table)
}

#[tokio::test]
async fn search_text_returns_most_relevant_doc() {
    let dim = 8u32;
    let texts = [
        "rust programming systems language memory safety",
        "python machine learning data science numpy",
        "rust async tokio concurrency futures",
        "javascript frontend web react components",
        "rust memory ownership borrowing lifetimes",
    ];
    let embeddings: Vec<Vec<f32>> = (0..texts.len())
        .map(|i| rand_unit_vec(dim as usize, i as u64))
        .collect();

    let (_dir, catalog, store, table) =
        setup_bm25_table(&texts, &embeddings, dim).await;

    let results = search_text(
        &table,
        "rust programming language",
        &["chunk_text"],
        3,
        Arc::clone(&catalog) as Arc<dyn ailake_catalog::CatalogProvider>,
        Arc::clone(&store) as Arc<dyn ailake_store::Store>,
        None,
    )
    .await
    .unwrap();

    assert!(!results.is_empty(), "should return at least one result");
    // All top results should have negative distance (BM25 > 0)
    let top = &results[0];
    assert!(
        top.distance < 0.0,
        "BM25 distance should be negative (negated score), got {}",
        top.distance
    );
    // Results should be sorted ascending (most relevant = most negative)
    for i in 1..results.len() {
        assert!(
            results[i].distance >= results[i - 1].distance,
            "results should be sorted ascending"
        );
    }
}

#[tokio::test]
async fn search_text_returns_top_k_limit() {
    let dim = 8u32;
    let texts = [
        "rust systems programming",
        "python data science",
        "go concurrency goroutines",
        "java jvm bytecode",
        "cpp templates metaprogramming",
    ];
    let embeddings: Vec<Vec<f32>> = (0..texts.len())
        .map(|i| rand_unit_vec(dim as usize, i as u64))
        .collect();

    let (_dir, catalog, store, table) =
        setup_bm25_table(&texts, &embeddings, dim).await;

    let results = search_text(
        &table,
        "rust",
        &["chunk_text"],
        2,
        Arc::clone(&catalog) as Arc<dyn ailake_catalog::CatalogProvider>,
        Arc::clone(&store) as Arc<dyn ailake_store::Store>,
        None,
    )
    .await
    .unwrap();

    assert!(
        results.len() <= 2,
        "should return at most top_k=2 results, got {}",
        results.len()
    );
}

#[tokio::test]
async fn hybrid_search_rrf_returns_top_k() {
    let dim = 8u32;
    let texts = [
        "rust programming systems language memory safety",
        "python machine learning data science",
        "rust async tokio concurrency",
        "javascript web frontend",
    ];
    let embeddings: Vec<Vec<f32>> = (0..texts.len())
        .map(|i| rand_unit_vec(dim as usize, i as u64 + 100))
        .collect();

    let (_dir, catalog, store, table) =
        setup_bm25_table(&texts, &embeddings, dim).await;

    // Use embedding of first doc as query — both vector and text point to "rust"
    let query = embeddings[0].clone();
    let config = SearchConfig {
        top_k: 3,
        ef_search: 50,
        pruning_threshold: f32::INFINITY,
        rerank_factor: None,
        score_fn: None,
        partition_filter: None,
        hybrid: Some(
            HybridConfig::new("rust programming language")
                .with_text_column("chunk_text")
                .with_bm25_weight(0.4),
        ),
    };

    let results = ailake_query::search(
        &table,
        &query,
        config,
        "embedding",
        dim,
        Arc::clone(&catalog) as Arc<dyn ailake_catalog::CatalogProvider>,
        Arc::clone(&store) as Arc<dyn ailake_store::Store>,
    )
    .await
    .unwrap();

    assert!(!results.is_empty(), "hybrid search should return results");
    assert!(
        results.len() <= 3,
        "should respect top_k=3, got {}",
        results.len()
    );
}

#[tokio::test]
async fn idf_stats_serialization_roundtrip() {
    let mut stats = IdfStats::default();
    stats.merge_batch(&[
        "rust programming language",
        "python machine learning",
        "rust memory safety",
    ]);
    let bytes = stats.to_bytes().unwrap();
    let restored = IdfStats::from_bytes(&bytes).unwrap();
    assert_eq!(restored.doc_count, 3);
    assert_eq!(restored.term_df.get("rust"), Some(&2));
    assert_eq!(restored.term_df.get("python"), Some(&1));
}

#[tokio::test]
async fn bm25_scorer_ranks_rust_docs_above_python() {
    let mut stats = IdfStats::default();
    let docs = [
        "rust programming systems language",
        "python machine learning neural",
        "rust memory safety ownership",
    ];
    stats.merge_batch(&docs);
    let scorer = BM25Scorer::new(&stats);
    let query = "rust systems";
    let s_rust1 = scorer.score(query, docs[0]);
    let s_python = scorer.score(query, docs[1]);
    let s_rust2 = scorer.score(query, docs[2]);
    assert!(s_rust1 > s_python, "rust doc > python doc: {s_rust1} > {s_python}");
    assert!(s_rust2 > s_python, "rust doc > python doc: {s_rust2} > {s_python}");
}

#[tokio::test]
async fn write_batch_auto_deferred_creates_file() {
    let dir = TempDir::new().unwrap();
    let store = Arc::new(LocalStore::new(dir.path()));
    let catalog = Arc::new(HadoopCatalog::new(
        Arc::clone(&store) as Arc<dyn ailake_store::Store>,
        "warehouse",
    ));
    let table = TableIdent::new("default", "test_auto_deferred");
    let dim = 8u32;

    let mut writer = TableWriter::create_or_open(
        Arc::clone(&catalog) as Arc<dyn ailake_catalog::CatalogProvider>,
        Arc::clone(&store) as Arc<dyn ailake_store::Store>,
        make_policy(dim),
        table.clone(),
    )
    .await
    .unwrap();

    let (batch, embeddings) = fixtures::generate_batch(50, dim as usize);
    writer
        .write_batch_auto_deferred(&batch, &embeddings)
        .await
        .unwrap();
    let snap = writer.commit().await.unwrap();
    assert!(snap > 0);

    let files = catalog.list_files(&table, None).await.unwrap();
    assert!(!files.is_empty(), "should have at least one file after deferred write");
}
