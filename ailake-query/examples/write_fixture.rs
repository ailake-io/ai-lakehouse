// SPDX-License-Identifier: MIT OR Apache-2.0
//! Writes a reference AI-Lake table to disk for external compatibility tests.
//!
//! Output path: $COMPAT_FIXTURE_PATH or ./compat-fixture/
//! Table: default.compat_test — 1 000 rows, dim=8, cosine, F16
//!
//! Usage:
//!   cargo run --example write_fixture -p ailake-query

use std::sync::Arc;

use ailake_catalog::{CatalogProvider, HadoopCatalog, TableIdent};
use ailake_core::{VectorMetric, VectorPrecision, VectorStoragePolicy};
use ailake_query::TableWriter;
use ailake_store::{LocalStore, Store};
use arrow_array::{Int32Array, RecordBatch, StringArray};
use arrow_schema::{DataType, Field, Schema};

const DIM: usize = 8;
const ROWS_PER_BATCH: usize = 500;
const BATCHES: usize = 2;

fn make_batch(offset: usize) -> (RecordBatch, Vec<Vec<f32>>) {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};

    let rows = ROWS_PER_BATCH;
    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int32, false),
        Field::new("text", DataType::Utf8, false),
    ]));
    let ids: Vec<i32> = (offset as i32..(offset + rows) as i32).collect();
    let texts: Vec<String> = (offset..offset + rows)
        .map(|i| format!("doc_{i}"))
        .collect();
    let texts_ref: Vec<&str> = texts.iter().map(|s| s.as_str()).collect();

    let batch = RecordBatch::try_new(
        schema,
        vec![
            Arc::new(Int32Array::from(ids)),
            Arc::new(StringArray::from(texts_ref)),
        ],
    )
    .unwrap();

    let embeddings: Vec<Vec<f32>> = (offset..offset + rows)
        .map(|i| {
            let mut v: Vec<f32> = (0..DIM)
                .map(|j| {
                    let mut h = DefaultHasher::new();
                    (i * DIM + j).hash(&mut h);
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

#[tokio::main]
async fn main() {
    let out =
        std::env::var("COMPAT_FIXTURE_PATH").unwrap_or_else(|_| "./compat-fixture".to_string());

    std::fs::create_dir_all(&out).expect("create fixture dir");
    println!("writing fixture to: {out}");

    let abs_out = std::fs::canonicalize(&out).unwrap_or_else(|_| std::path::PathBuf::from(&out));
    let abs_out_str = abs_out.to_string_lossy().to_string();

    let store: Arc<dyn Store> = Arc::new(LocalStore::new(&abs_out_str));
    let catalog = Arc::new(HadoopCatalog::new(Arc::clone(&store), &abs_out_str));
    let table = TableIdent::new("default", "compat_test");

    let policy = VectorStoragePolicy {
        column_name: "embedding".to_string(),
        dim: DIM as u32,
        metric: VectorMetric::Cosine,
        precision: VectorPrecision::F16,
        pq: None,
        keep_raw_for_reranking: false,
        pre_normalize: false,
        hnsw_m: None,
        hnsw_ef_construction: None,
    };

    let mut writer = TableWriter::create_or_open(
        Arc::clone(&catalog) as Arc<dyn CatalogProvider>,
        Arc::clone(&store),
        policy,
        table.clone(),
    )
    .await
    .expect("create writer");

    let total_rows = ROWS_PER_BATCH * BATCHES;
    for b in 0..BATCHES {
        let (batch, embs) = make_batch(b * ROWS_PER_BATCH);
        writer
            .write_batch(&batch, &embs)
            .await
            .expect("write batch");
        println!(
            "  batch {}/{} written ({} rows)",
            b + 1,
            BATCHES,
            ROWS_PER_BATCH
        );
    }

    writer.commit().await.expect("commit");
    println!("committed — {total_rows} rows total");

    // Patch metadata: add schema fields + name-mapping so PyIceberg can scan without field-ids
    {
        let meta_dir = format!("{}/default/compat_test/metadata", abs_out_str);
        let hint_path = format!("{}/version-hint.text", meta_dir);
        let version: u32 = std::fs::read_to_string(&hint_path)
            .unwrap_or_else(|_| "1".to_string())
            .trim()
            .parse()
            .unwrap_or(1);
        let meta_path = format!("{}/v{}.metadata.json", meta_dir, version);
        let raw = std::fs::read_to_string(&meta_path).expect("read metadata");
        let mut meta: serde_json::Value = serde_json::from_str(&raw).expect("parse metadata");

        let embedding_type = format!("fixed[{}]", DIM * 2); // F16 = 2 bytes/dim
        meta["schemas"][0]["fields"] = serde_json::json!([
            {"id": 1, "name": "id",        "required": false, "type": "int"},
            {"id": 2, "name": "text",      "required": false, "type": "string"},
            {"id": 3, "name": "embedding", "required": false, "type": embedding_type},
        ]);
        meta["last-column-id"] = serde_json::json!(3);

        let name_mapping = serde_json::json!([
            {"field-id": 1, "names": ["id"]},
            {"field-id": 2, "names": ["text"]},
            {"field-id": 3, "names": ["embedding"]},
        ]);
        meta["properties"]["schema.name-mapping.default"] =
            serde_json::json!(name_mapping.to_string());

        let next = version + 1;
        let new_path = format!("{}/v{}.metadata.json", meta_dir, next);
        std::fs::write(&new_path, serde_json::to_string_pretty(&meta).unwrap())
            .expect("write patched metadata");
        std::fs::write(&hint_path, next.to_string()).expect("update version-hint");
        println!("patched metadata → v{next} (schema fields + name-mapping)");
    }

    // Print fixture manifest so CI can verify path
    let files = (catalog as Arc<dyn CatalogProvider>)
        .list_files(&table, None)
        .await
        .expect("list files");
    println!("files in catalog:");
    for f in &files {
        println!("  {}", f.path);
    }

    // Write a small manifest for Python scripts to consume without parsing Iceberg
    let manifest_txt = files
        .iter()
        .map(|f| f.path.clone())
        .collect::<Vec<_>>()
        .join("\n");
    std::fs::write(format!("{out}/fixture_files.txt"), manifest_txt)
        .expect("write fixture_files.txt");
    std::fs::write(format!("{out}/fixture_rows.txt"), total_rows.to_string())
        .expect("write fixture_rows.txt");

    println!("fixture ready.");
}
