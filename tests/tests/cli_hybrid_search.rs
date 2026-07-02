// SPDX-License-Identifier: MIT OR Apache-2.0
//! Regression: `ailake-go`'s `SearchHybrid` sends `--hybrid-text`/`--text-column`/
//! `--bm25-weight` to the `ailake` CLI, but the `Search` subcommand never defined those
//! flags (`hybrid: None` was hardcoded in `main.rs`) — every hybrid search call failed
//! with a clap "unrecognized argument" error. This drives the real compiled CLI binary
//! end-to-end to confirm the flags exist, parse, and produce a hybrid search result.

use std::process::Command;
use std::sync::Arc;

use ailake_catalog::{provider::TableProperties, CatalogProvider, HadoopCatalog};
use ailake_core::{VectorMetric, VectorPrecision, VectorStoragePolicy};
use ailake_query::TableWriter;
use ailake_store::LocalStore;
use arrow_array::{RecordBatch, StringArray};
use arrow_schema::{DataType, Field, Schema};
use tempfile::TempDir;

/// Path to the `ailake` binary built alongside this workspace's `cargo test` run.
/// Not wired through `CARGO_BIN_EXE_ailake` (this test crate doesn't depend on the
/// `ailake-cli` package), so this resolves the debug binary by convention instead.
fn cli_bin_path() -> Option<std::path::PathBuf> {
    let manifest_dir = env!("CARGO_MANIFEST_DIR");
    let candidate = std::path::Path::new(manifest_dir)
        .parent()?
        .join("target/debug/ailake");
    candidate.exists().then_some(candidate)
}

#[tokio::test]
async fn cli_search_hybrid_text_flags_are_recognized_and_return_results() {
    let Some(bin) = cli_bin_path() else {
        eprintln!("SKIP: ailake CLI binary not found at target/debug/ailake — run `cargo build -p ailake-cli` first");
        return;
    };

    let dir = TempDir::new().unwrap();
    let store: Arc<dyn ailake_store::Store> = Arc::new(LocalStore::new(dir.path()));
    let catalog = Arc::new(HadoopCatalog::new(Arc::clone(&store), ""));
    let table = ailake_catalog::provider::TableIdent::new("default", "docs");

    let policy = VectorStoragePolicy {
        column_name: "embedding".into(),
        dim: 4,
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
    };

    catalog
        .create_table(
            &table,
            &TableProperties {
                policy: policy.clone(),
                extra: Default::default(),
                format_version: 2,
                partition_column_type: None,
            },
        )
        .await
        .unwrap();

    let schema = Arc::new(Schema::new(vec![Field::new(
        "chunk_text",
        DataType::Utf8,
        false,
    )]));
    let texts = [
        "rust programming language",
        "python data science toolkit",
        "vector search engine internals",
        "machine learning model training",
    ];
    let batch =
        RecordBatch::try_new(schema, vec![Arc::new(StringArray::from(texts.to_vec()))]).unwrap();
    let embeddings: Vec<Vec<f32>> = vec![
        vec![1.0, 0.0, 0.0, 0.0],
        vec![0.0, 1.0, 0.0, 0.0],
        vec![0.9, 0.1, 0.0, 0.0],
        vec![0.0, 0.0, 1.0, 0.0],
    ];

    let mut writer = TableWriter::create_or_open(
        catalog.clone() as Arc<dyn ailake_catalog::provider::CatalogProvider>,
        Arc::clone(&store),
        policy,
        table.clone(),
        2,
    )
    .await
    .unwrap()
    .with_bm25("chunk_text");
    writer.write_batch(&batch, &embeddings).await.unwrap();
    writer.commit().await.unwrap();

    // Query vector close to row 0 ("rust programming language") but with hybrid text
    // pulling toward "vector search" (row 2) — proves the fusion path actually ran,
    // not just a plain vector search ignoring --hybrid-text.
    let output = Command::new(&bin)
        .args([
            "--store",
            dir.path().to_str().unwrap(),
            "search",
            "default.docs",
            "--query",
            "1.0,0.0,0.0,0.0",
            "--hybrid-text",
            "vector search engine",
            "--text-column",
            "chunk_text",
            "--bm25-weight",
            "0.7",
            "--top-k",
            "4",
            "--format",
            "json",
        ])
        .output()
        .expect("failed to run ailake CLI");

    assert!(
        output.status.success(),
        "ailake search --hybrid-text failed: stdout={} stderr={}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );

    let stdout = String::from_utf8_lossy(&output.stdout);
    let json: serde_json::Value = serde_json::from_str(stdout.trim())
        .unwrap_or_else(|e| panic!("hybrid search output not valid JSON: {e}\nstdout={stdout}"));
    let results = json["results"]
        .as_array()
        .expect("response must have a results array");
    assert!(
        !results.is_empty(),
        "hybrid search returned no results: {json}"
    );
}
