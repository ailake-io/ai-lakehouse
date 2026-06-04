// SPDX-License-Identifier: MIT OR Apache-2.0
//! Phase 1 local demo — write an AI-Lake table, search it, inspect the file layout.
//!
//! Usage:
//!   cargo run --example demo -p ailake-query

use std::sync::Arc;

use ailake_catalog::{CatalogProvider, HadoopCatalog, TableIdent};
use ailake_core::{VectorMetric, VectorPrecision, VectorStoragePolicy};
use ailake_file::AilakeFileReader;
use ailake_query::{search, SearchConfig, TableWriter};
use ailake_store::{LocalStore, Store};
use arrow_array::{Int32Array, RecordBatch, StringArray};
use arrow_schema::{DataType, Field, Schema};
use bytes::Bytes;
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

// ---------- data generation ----------

fn generate_batch(rows: usize, dim: usize) -> (RecordBatch, Vec<Vec<f32>>) {
    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int32, false),
        Field::new("text", DataType::Utf8, false),
    ]));
    let ids: Vec<i32> = (0..rows as i32).collect();
    let texts: Vec<String> = (0..rows).map(|i| format!("doc_{}", i)).collect();
    let texts_ref: Vec<&str> = texts.iter().map(|s| s.as_str()).collect();

    let batch = RecordBatch::try_new(
        schema,
        vec![
            Arc::new(Int32Array::from(ids)),
            Arc::new(StringArray::from(texts_ref)),
        ],
    )
    .unwrap();

    let embeddings: Vec<Vec<f32>> = (0..rows)
        .map(|i| {
            let mut v: Vec<f32> = (0..dim)
                .map(|j| {
                    let mut h = DefaultHasher::new();
                    (i * dim + j).hash(&mut h);
                    let bits = (h.finish() & 0x3FFF_FFFF) as u32;
                    (bits as f32 / 0x3FFF_FFFF as f32) * 2.0 - 1.0
                })
                .collect();
            let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
            if norm > 0.0 {
                for x in &mut v {
                    *x /= norm;
                }
            }
            v
        })
        .collect();

    (batch, embeddings)
}

// ---------- file inspector ----------

fn inspect_file(bytes: &Bytes) {
    let len = bytes.len();
    println!("\n  File layout ({} bytes):", len);

    // Find PAR1 positions
    let par1_pos: Vec<usize> = bytes
        .windows(4)
        .enumerate()
        .filter(|(_, w)| *w == b"PAR1")
        .map(|(i, _)| i)
        .collect();
    for (n, pos) in par1_pos.iter().enumerate() {
        println!("    PAR1 #{} at byte {}", n + 1, pos);
    }

    // AILK magic
    let ailk_positions: Vec<usize> = bytes
        .windows(4)
        .enumerate()
        .filter(|(_, w)| *w == b"AILK")
        .map(|(i, _)| i)
        .collect();
    for pos in &ailk_positions {
        println!("    AILK magic at byte {}", pos);
    }

    // Parse header via reader
    let reader = AilakeFileReader::new(bytes.clone(), "embedding", 0);
    if reader.is_ailake_file() {
        let ailk_start = reader.ailk_offset().unwrap();
        let header = reader.read_header().unwrap();
        println!(
            "    AILK section    : {}..{}",
            ailk_start,
            ailk_start + header.hnsw_offset + header.hnsw_len + 24
        );
        println!(
            "    Centroid section: {}..{}",
            ailk_start + header.centroid_offset,
            ailk_start + header.centroid_offset + header.centroid_len
        );
        println!(
            "    HNSW section    : {}..{} ({} bytes)",
            ailk_start + header.hnsw_offset,
            ailk_start + header.hnsw_offset + header.hnsw_len,
            header.hnsw_len
        );
        println!("    Record count    : {}", header.record_count);
        println!("    Dim             : {}", header.dim);
    }
}

// ---------- main ----------

#[tokio::main]
async fn main() {
    let dir = tempfile::TempDir::new().unwrap();
    println!("Workspace: {}", dir.path().display());

    let store: Arc<dyn Store> = Arc::new(LocalStore::new(dir.path()));
    let catalog = Arc::new(HadoopCatalog::new(Arc::clone(&store), "warehouse"));
    let table = TableIdent::new("default", "demo_table");
    let dim = 16u32;

    let policy = VectorStoragePolicy {
        column_name: "embedding".to_string(),
        dim,
        metric: VectorMetric::Cosine,
        precision: VectorPrecision::F16,
        pq: None,
        keep_raw_for_reranking: false,
        pre_normalize: false,
        hnsw_m: None,
        hnsw_ef_construction: None,
        rabitq: None,
    };

    // ---- write ----
    println!("\n=== Writing 2 batches (500 rows each) ===");

    let mut writer = TableWriter::create_or_open(
        Arc::clone(&catalog) as Arc<dyn ailake_catalog::CatalogProvider>,
        Arc::clone(&store),
        policy.clone(),
        table.clone(),
    )
    .await
    .unwrap();

    let (batch1, embs1) = generate_batch(500, dim as usize);
    let (batch2, embs2) = generate_batch(500, dim as usize);

    writer.write_batch(&batch1, &embs1).await.unwrap();
    println!("  part-00000.parquet written");
    writer.write_batch(&batch2, &embs2).await.unwrap();
    println!("  part-00001.parquet written");

    let snap_id = writer.commit().await.unwrap();
    println!("  Committed snapshot id={}", snap_id);

    // ---- inspect first file ----
    println!("\n=== File layout inspection (part-00000.parquet) ===");
    let file_bytes = store.get("data/part-00000.parquet").await.unwrap();
    inspect_file(&file_bytes);

    // ---- search ----
    println!("\n=== Search: query = embs1[0] (should be top result) ===");
    let query = embs1[0].clone();

    let results = search(
        &table,
        &query,
        SearchConfig {
            top_k: 5,
            ef_search: 50,
            pruning_threshold: f32::INFINITY,
            rerank_factor: None,
        },
        "embedding",
        dim,
        Arc::clone(&catalog) as Arc<dyn ailake_catalog::CatalogProvider>,
        Arc::clone(&store),
    )
    .await
    .unwrap();

    println!("  Top-{} results:", results.len());
    for (i, r) in results.iter().enumerate() {
        println!(
            "    #{}: row_id={} distance={:.6}  file={}",
            i + 1,
            r.row_id.as_u64(),
            r.distance,
            r.file_path
        );
    }

    assert!(
        results[0].distance < 0.01,
        "top result should be the query vector itself (distance ~0)"
    );
    println!(
        "\nPASS: top result distance = {:.2e} < 0.01",
        results[0].distance
    );

    // ---- integrity check ----
    println!("\n=== Integrity check on both files ===");
    for part in &["data/part-00000.parquet", "data/part-00001.parquet"] {
        let bytes = store.get(part).await.unwrap();
        let reader = AilakeFileReader::new(bytes, "embedding", dim);
        reader.verify_integrity().unwrap();
        let idx = reader.load_index().unwrap();
        println!("  {} — {} nodes, integrity OK", part, idx.node_count());
    }

    // ---- catalog listing ----
    println!("\n=== Catalog: list_files ===");
    let files = (Arc::clone(&catalog) as Arc<dyn CatalogProvider>)
        .list_files(&table, None)
        .await
        .unwrap();
    for f in &files {
        println!(
            "  {} — {} rows, hnsw_offset={}, hnsw_len={}",
            f.path,
            f.record_count,
            f.hnsw_offset.unwrap_or(0),
            f.hnsw_len.unwrap_or(0)
        );
    }

    println!("\nFase 1 demo concluída com sucesso.");
}
