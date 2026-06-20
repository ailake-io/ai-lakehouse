// SPDX-License-Identifier: MIT OR Apache-2.0
//! Integration tests: Tantivy fast path vs BM25 fallback in search_text().
//!
//! Key distinction: files written with `with_fts_config()` use Tantivy O(log N);
//! legacy files (no FTS blob) fall back to BM25 O(N). Both paths must return
//! correct, sorted results.

mod fixtures;

use std::sync::Arc;

use ailake_catalog::{CatalogProvider, HadoopCatalog, TableIdent};
use ailake_core::{VectorMetric, VectorPrecision, VectorStoragePolicy};
use ailake_fts::FtsConfig;
use ailake_query::{search_text, TableWriter};
use ailake_store::LocalStore;
use arrow_array::{Int32Array, RecordBatch, StringArray};
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
        partition_column_type: None,
        partition_fields: vec![],
    }
}

fn fts_cfg(cols: &[&str]) -> FtsConfig {
    FtsConfig {
        text_columns: cols.iter().map(|s| s.to_string()).collect(),
        tokenizer: "default".into(),
        writer_heap_bytes: 16 * 1024 * 1024,
    }
}

// ── helpers ───────────────────────────────────────────────────────────────────

async fn make_table_with_fts(
    texts: &[&str],
    dim: u32,
    fts_cols: &[&str],
) -> (TempDir, Arc<HadoopCatalog>, Arc<LocalStore>, TableIdent) {
    let dir = TempDir::new().unwrap();
    let store = Arc::new(LocalStore::new(dir.path()));
    let catalog = Arc::new(HadoopCatalog::new(
        Arc::clone(&store) as Arc<dyn ailake_store::Store>,
        "warehouse",
    ));
    let table = TableIdent::new("default", "fts_table");

    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int32, false),
        Field::new("chunk_text", DataType::Utf8, false),
    ]));
    let ids: Vec<i32> = (0..texts.len() as i32).collect();
    let batch = RecordBatch::try_new(
        schema,
        vec![
            Arc::new(Int32Array::from(ids)),
            Arc::new(StringArray::from(texts.to_vec())),
        ],
    )
    .unwrap();

    let embeddings: Vec<Vec<f32>> = (0..texts.len())
        .map(|i| {
            let mut v = vec![0.0f32; dim as usize];
            v[0] = i as f32;
            v
        })
        .collect();

    let mut writer = TableWriter::create_or_open(
        Arc::clone(&catalog) as Arc<dyn ailake_catalog::CatalogProvider>,
        Arc::clone(&store) as Arc<dyn ailake_store::Store>,
        make_policy(dim),
        table.clone(),
        2,
    )
    .await
    .unwrap();

    writer = writer.with_fts_config(fts_cfg(fts_cols));
    writer.write_batch(&batch, &embeddings).await.unwrap();
    writer.commit().await.unwrap();

    (dir, catalog, store, table)
}

async fn make_legacy_table(
    texts: &[&str],
    dim: u32,
) -> (TempDir, Arc<HadoopCatalog>, Arc<LocalStore>, TableIdent) {
    let dir = TempDir::new().unwrap();
    let store = Arc::new(LocalStore::new(dir.path()));
    let catalog = Arc::new(HadoopCatalog::new(
        Arc::clone(&store) as Arc<dyn ailake_store::Store>,
        "warehouse",
    ));
    let table = TableIdent::new("default", "legacy_table");

    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int32, false),
        Field::new("chunk_text", DataType::Utf8, false),
    ]));
    let ids: Vec<i32> = (0..texts.len() as i32).collect();
    let batch = RecordBatch::try_new(
        schema,
        vec![
            Arc::new(Int32Array::from(ids)),
            Arc::new(StringArray::from(texts.to_vec())),
        ],
    )
    .unwrap();

    let embeddings: Vec<Vec<f32>> = (0..texts.len())
        .map(|i| {
            let mut v = vec![0.0f32; dim as usize];
            v[0] = i as f32;
            v
        })
        .collect();

    let mut writer = TableWriter::create_or_open(
        Arc::clone(&catalog) as Arc<dyn ailake_catalog::CatalogProvider>,
        Arc::clone(&store) as Arc<dyn ailake_store::Store>,
        make_policy(dim),
        table.clone(),
        2,
    )
    .await
    .unwrap();

    // NO with_fts_config → legacy file, BM25 fallback only
    writer.write_batch(&batch, &embeddings).await.unwrap();
    writer.commit().await.unwrap();

    (dir, catalog, store, table)
}

// ── tests ─────────────────────────────────────────────────────────────────────

/// File written with `with_fts_config()` must return hits via Tantivy fast path.
/// Scores come from Tantivy and are encoded as negative distance (lower = better).
#[tokio::test]
async fn search_text_uses_tantivy_when_fts_blob_present() {
    let texts = &[
        "rust programming language systems",
        "python machine learning data",
        "rust async tokio concurrency",
    ];
    let (_dir, catalog, store, table) = make_table_with_fts(texts, 8, &["chunk_text"]).await;

    let results = search_text(
        &table,
        "rust",
        &["chunk_text"],
        5,
        Arc::clone(&catalog) as Arc<dyn ailake_catalog::CatalogProvider>,
        Arc::clone(&store) as Arc<dyn ailake_store::Store>,
        None,
    )
    .await
    .unwrap();

    assert!(!results.is_empty(), "Tantivy search returned no results");
    // Both "rust" docs should rank above the python doc
    let rust_hits: Vec<_> = results
        .iter()
        .filter(|r| r.row_id.as_u64() == 0 || r.row_id.as_u64() == 2)
        .collect();
    assert!(
        !rust_hits.is_empty(),
        "expected at least one rust-related result"
    );
    // Tantivy distances: negated score, so negative = relevant
    assert!(
        results[0].distance < 0.0,
        "Tantivy hit should have negative distance, got {}",
        results[0].distance
    );
}

/// File written WITHOUT `with_fts_config()` must still return results via BM25.
#[tokio::test]
async fn search_text_falls_back_to_bm25_for_legacy_files() {
    let texts = &[
        "rust programming language",
        "python machine learning",
        "rust memory ownership",
    ];
    let (_dir, catalog, store, table) = make_legacy_table(texts, 8).await;

    let results = search_text(
        &table,
        "rust",
        &["chunk_text"],
        5,
        Arc::clone(&catalog) as Arc<dyn ailake_catalog::CatalogProvider>,
        Arc::clone(&store) as Arc<dyn ailake_store::Store>,
        None,
    )
    .await
    .unwrap();

    assert!(
        !results.is_empty(),
        "BM25 fallback returned no results for legacy file"
    );
    assert!(
        results[0].distance < 0.0,
        "BM25 distance should be negative (negated score)"
    );
}

/// Table with FTS: search must respect top_k limit.
#[tokio::test]
async fn search_text_tantivy_respects_top_k() {
    let texts = &[
        "rust programming language memory",
        "rust async tokio runtime",
        "rust cargo build system",
        "rust lifetime borrow checker",
        "rust traits generics polymorphism",
    ];
    let (_dir, catalog, store, table) = make_table_with_fts(texts, 8, &["chunk_text"]).await;

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
        "search_text should return at most top_k=2 results, got {}",
        results.len()
    );
}

/// Table with FTS: results must be sorted ascending by distance.
#[tokio::test]
async fn search_text_tantivy_results_are_sorted() {
    let texts = &[
        "rust programming language systems memory safety",
        "python machine learning data science",
        "rust async tokio concurrency futures",
    ];
    let (_dir, catalog, store, table) = make_table_with_fts(texts, 8, &["chunk_text"]).await;

    let results = search_text(
        &table,
        "rust",
        &["chunk_text"],
        10,
        Arc::clone(&catalog) as Arc<dyn ailake_catalog::CatalogProvider>,
        Arc::clone(&store) as Arc<dyn ailake_store::Store>,
        None,
    )
    .await
    .unwrap();

    for i in 1..results.len() {
        assert!(
            results[i].distance >= results[i - 1].distance,
            "results must be sorted ascending (most relevant first)"
        );
    }
}

/// Table written with `with_fts_config()`: no hits for a query that matches nothing.
#[tokio::test]
async fn search_text_tantivy_no_match_returns_empty() {
    let texts = &["rust programming", "python data science", "go concurrency"];
    let (_dir, catalog, store, table) = make_table_with_fts(texts, 8, &["chunk_text"]).await;

    let results = search_text(
        &table,
        "haskell monads category theory zzzyyyy",
        &["chunk_text"],
        5,
        Arc::clone(&catalog) as Arc<dyn ailake_catalog::CatalogProvider>,
        Arc::clone(&store) as Arc<dyn ailake_store::Store>,
        None,
    )
    .await
    .unwrap();

    assert!(
        results.is_empty(),
        "no-match query should return empty, got {} results",
        results.len()
    );
}

/// write_batch with `with_fts_config()` + commit works without error.
/// Verifies the TableWriter → file-level FTS integration end-to-end.
#[tokio::test]
async fn table_writer_with_fts_config_commits_successfully() {
    let dir = TempDir::new().unwrap();
    let store = Arc::new(LocalStore::new(dir.path()));
    let catalog = Arc::new(HadoopCatalog::new(
        Arc::clone(&store) as Arc<dyn ailake_store::Store>,
        "warehouse",
    ));
    let table = TableIdent::new("default", "fts_commit_test");

    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int32, false),
        Field::new("chunk_text", DataType::Utf8, false),
    ]));
    let batch = RecordBatch::try_new(
        schema,
        vec![
            Arc::new(Int32Array::from(vec![0i32, 1, 2])),
            Arc::new(StringArray::from(vec![
                "rust programming",
                "python learning",
                "go concurrency",
            ])),
        ],
    )
    .unwrap();
    let embeddings = vec![vec![1.0f32, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]; 3];

    let mut writer = TableWriter::create_or_open(
        Arc::clone(&catalog) as Arc<dyn ailake_catalog::CatalogProvider>,
        Arc::clone(&store) as Arc<dyn ailake_store::Store>,
        make_policy(8),
        table.clone(),
        2,
    )
    .await
    .unwrap();

    writer = writer.with_fts_config(fts_cfg(&["chunk_text"]));
    writer.write_batch(&batch, &embeddings).await.unwrap();
    let snap_id = writer.commit().await.unwrap();

    assert!(snap_id > 0, "snapshot_id must be > 0 after commit");

    let files = catalog.list_files(&table, None).await.unwrap();
    assert!(
        !files.is_empty(),
        "committed table must have at least one file"
    );
}
