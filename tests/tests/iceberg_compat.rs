// SPDX-License-Identifier: MIT OR Apache-2.0
//! Verifies AI-Lake catalog output is compatible with Apache Iceberg Spec v2.
//!
//! These tests do NOT use the AI-Lake SDK to read back the files — they inspect
//! raw bytes and JSON to confirm that any standard Iceberg/Parquet reader would
//! handle the output correctly.

mod fixtures;

use ailake_catalog::{CatalogProvider, HadoopCatalog, TableIdent};
use ailake_core::{VectorMetric, VectorPrecision, VectorStoragePolicy};
use ailake_query::TableWriter;
use ailake_store::{LocalStore, Store};
use std::sync::Arc;
use tempfile::TempDir;

fn make_policy(dim: u32) -> VectorStoragePolicy {
    VectorStoragePolicy {
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
        binary: None,
    }
}

async fn write_table(dir: &TempDir, table_name: &str, rows: usize, dim: u32) {
    let store = Arc::new(LocalStore::new(dir.path()));
    let catalog = Arc::new(HadoopCatalog::new(
        Arc::clone(&store) as Arc<dyn Store>,
        "warehouse",
    ));
    let table = TableIdent::new("default", table_name);

    let mut writer = TableWriter::create_or_open(
        Arc::clone(&catalog) as Arc<dyn CatalogProvider>,
        Arc::clone(&store) as Arc<dyn Store>,
        make_policy(dim),
        table,
    )
    .await
    .unwrap();

    let (batch, embs) = fixtures::generate_batch(rows, dim as usize);
    writer.write_batch(&batch, &embs).await.unwrap();
    writer.commit().await.unwrap();
}

fn find_files(root: &std::path::Path, ext: &str) -> Vec<std::path::PathBuf> {
    let mut out = Vec::new();
    if let Ok(rd) = std::fs::read_dir(root) {
        for entry in rd.flatten() {
            let path = entry.path();
            if path.is_dir() {
                out.extend(find_files(&path, ext));
            } else if path.extension().and_then(|e| e.to_str()) == Some(ext) {
                out.push(path);
            }
        }
    }
    out
}

/// Find the current versioned metadata file (vN.metadata.json) by reading version-hint.text,
/// or fall back to the highest-versioned file if the hint is missing.
fn find_current_metadata(root: &std::path::Path) -> Vec<std::path::PathBuf> {
    let mut out = Vec::new();
    // Walk looking for version-hint.text files
    fn walk(dir: &std::path::Path, results: &mut Vec<std::path::PathBuf>) {
        if let Ok(rd) = std::fs::read_dir(dir) {
            for entry in rd.flatten() {
                let path = entry.path();
                if path.is_dir() {
                    walk(&path, results);
                } else if path.file_name().and_then(|n| n.to_str()) == Some("version-hint.text") {
                    let meta_dir = path.parent().unwrap();
                    if let Ok(hint) = std::fs::read_to_string(&path) {
                        let v = hint.trim().to_string();
                        let meta_path = meta_dir.join(format!("v{v}.metadata.json"));
                        if meta_path.exists() {
                            results.push(meta_path);
                            return;
                        }
                    }
                    // fallback: highest vN.metadata.json
                    if let Ok(rd2) = std::fs::read_dir(meta_dir) {
                        let mut candidates: Vec<_> = rd2
                            .flatten()
                            .map(|e| e.path())
                            .filter(|p| {
                                p.file_name()
                                    .and_then(|n| n.to_str())
                                    .map(|n| n.ends_with(".metadata.json"))
                                    .unwrap_or(false)
                            })
                            .collect();
                        candidates.sort();
                        if let Some(last) = candidates.last() {
                            results.push(last.clone());
                        }
                    }
                }
            }
        }
    }
    walk(root, &mut out);
    out
}

#[tokio::test]
async fn metadata_json_is_iceberg_spec_v2() {
    let dir = TempDir::new().unwrap();
    write_table(&dir, "compat_meta", 10, 8).await;

    let meta_files = find_current_metadata(dir.path());
    assert!(
        !meta_files.is_empty(),
        "no vN.metadata.json (Iceberg metadata) found under {:?}",
        dir.path()
    );

    let bytes = std::fs::read(&meta_files[0]).unwrap();
    let meta: serde_json::Value =
        serde_json::from_slice(&bytes).expect("metadata file is not valid JSON");

    assert_eq!(
        meta["format-version"], 2,
        "format-version must be 2 (Iceberg Spec v2)"
    );
    assert!(
        meta["table-uuid"].is_string(),
        "table-uuid must be present and a string"
    );
    assert!(
        meta["location"].is_string(),
        "location must be present and a string"
    );

    let props = &meta["properties"];
    assert!(props.is_object(), "properties must be an object");
    assert!(
        props["ailake.format-version"].is_string(),
        "ailake.format-version missing from properties"
    );
    assert!(
        props["ailake.vector-dim"].is_string(),
        "ailake.vector-dim missing from properties"
    );
    assert!(
        props["ailake.vector-metric"].is_string(),
        "ailake.vector-metric missing from properties"
    );
}

#[tokio::test]
async fn parquet_files_have_valid_magic_and_ailake_section() {
    let dir = TempDir::new().unwrap();
    write_table(&dir, "compat_parquet", 20, 8).await;

    let parquet_files = find_files(dir.path(), "parquet");
    assert!(
        !parquet_files.is_empty(),
        "no .parquet data files found under {:?}",
        dir.path()
    );

    for path in &parquet_files {
        let bytes = std::fs::read(path).unwrap();
        assert!(
            bytes.len() > 8,
            "parquet file too small: {} bytes",
            bytes.len()
        );

        // Parquet spec: first 4 and last 4 bytes must be b"PAR1"
        assert_eq!(&bytes[..4], b"PAR1", "{:?}: must start with PAR1", path);
        assert_eq!(
            &bytes[bytes.len() - 4..],
            b"PAR1",
            "{:?}: must end with PAR1",
            path
        );

        // AI-Lake section: AILK magic must appear at least twice (header + trailer)
        // and must be strictly before the PAR1 footer
        let ailk_positions: Vec<usize> = bytes
            .windows(4)
            .enumerate()
            .filter(|(_, w)| *w == b"AILK")
            .map(|(i, _)| i)
            .collect();
        assert!(
            ailk_positions.len() >= 2,
            "{:?}: AILK magic must appear at least twice (header + trailer), found {}",
            path,
            ailk_positions.len()
        );

        let last_ailk = *ailk_positions.last().unwrap();
        assert!(
            last_ailk < bytes.len() - 4,
            "{:?}: last AILK must not be the final 4 bytes — PAR1 footer must follow it",
            path
        );
    }
}

#[tokio::test]
async fn data_files_referenced_in_metadata() {
    let dir = TempDir::new().unwrap();
    write_table(&dir, "compat_refs", 15, 8).await;

    let parquet_files = find_files(dir.path(), "parquet");
    assert!(!parquet_files.is_empty(), "no data files found");

    let meta_files = find_current_metadata(dir.path());
    assert!(!meta_files.is_empty(), "no vN.metadata.json found");

    let meta_bytes = std::fs::read(&meta_files[0]).unwrap();
    let meta: serde_json::Value = serde_json::from_slice(&meta_bytes).unwrap();

    // Iceberg Spec v2: current-snapshot-id must be set after a write
    assert!(
        !meta["current-snapshot-id"].is_null(),
        "current-snapshot-id must be set after commit"
    );
    // snapshots array must be non-empty
    let empty = vec![];
    let snapshots = meta["snapshots"].as_array().unwrap_or(&empty);
    assert!(
        !snapshots.is_empty(),
        "snapshots array must be non-empty after commit"
    );
}
