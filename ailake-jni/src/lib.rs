//! ailake-jni — uniffi JVM bindings
//!
//! Thin sync wrapper over ailake-query for Spark/Trino connector hot path.
//! Build with: cargo build --release -p ailake-jni
//! The cdylib is loaded by the Spark/Trino connector via System.loadLibrary.

use std::sync::Arc;

use ailake_catalog::{CatalogProvider, HadoopCatalog};
use ailake_core::VectorMetric;
use ailake_query::{
    search as rs_search, Chunk, ContextAssembler, ContextAssemblerConfig, SearchConfig,
};
use ailake_store::LocalStore;

uniffi::setup_scaffolding!();

// ── Types ─────────────────────────────────────────────────────────────────────

/// A single vector search result.
#[derive(uniffi::Record)]
pub struct RowResult {
    pub row_id: u64,
    pub distance: f32,
    pub file_path: String,
}

// ── Helpers ───────────────────────────────────────────────────────────────────

fn rt() -> tokio::runtime::Runtime {
    tokio::runtime::Runtime::new().expect("tokio runtime")
}

fn parse_metric(s: &str) -> VectorMetric {
    match s {
        "euclidean" => VectorMetric::Euclidean,
        "dot_product" | "dotproduct" => VectorMetric::DotProduct,
        _ => VectorMetric::Cosine,
    }
}

// ── Exports ───────────────────────────────────────────────────────────────────

/// Search a local AI-Lake table for the top-k nearest vectors.
///
/// `table_uri`   — local filesystem path to the table root
/// `query_bytes` — raw f32 values as little-endian bytes (4 bytes per dimension)
/// `top_k`       — number of nearest neighbors to return
///
/// Returns results sorted by ascending distance. Returns empty list on error.
#[uniffi::export]
pub fn vector_search(table_uri: String, query_bytes: Vec<u8>, top_k: u32) -> Vec<RowResult> {
    let query: Vec<f32> = query_bytes
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
        .collect();

    let store: Arc<dyn ailake_store::Store> = Arc::new(LocalStore::new(&table_uri));
    let catalog = Arc::new(HadoopCatalog::new(store.clone(), &table_uri));
    let table = ailake_catalog::TableIdent::new("default", "table");
    let rt = rt();

    let meta = match rt.block_on(catalog.load_table(&table)) {
        Ok(m) => m,
        Err(_) => return vec![],
    };
    let dim: u32 = meta
        .properties
        .get("ailake.vector-dim")
        .and_then(|s: &String| s.parse().ok())
        .unwrap_or(query.len() as u32);
    let vector_column = meta
        .properties
        .get("ailake.vector-column")
        .cloned()
        .unwrap_or_else(|| "embedding".into());
    let _metric = parse_metric(
        meta.properties
            .get("ailake.vector-metric")
            .map(String::as_str)
            .unwrap_or("cosine"),
    );

    let config = SearchConfig {
        top_k: top_k as usize,
        ef_search: 50,
        pruning_threshold: f32::INFINITY,
    };

    match rt.block_on(rs_search(
        &table,
        &query,
        config,
        &vector_column,
        dim,
        catalog,
        store,
    )) {
        Ok(results) => results
            .into_iter()
            .map(|r| RowResult {
                row_id: r.row_id.as_u64(),
                distance: r.distance,
                file_path: r.file_path,
            })
            .collect(),
        Err(_) => vec![],
    }
}

/// Assemble JSON-serialized chunks into structured XML context for LLM input.
///
/// Each element of `chunk_jsons` must be a JSON object containing at minimum:
///   `document_id` (str), `chunk_index` (int), `chunk_text` (str)
/// Optional: `document_title`, `section_path`, `source_uri`, `distance` (float)
///
/// Returns XML string ready for insertion into an LLM prompt.
#[uniffi::export]
pub fn assemble_context(chunk_jsons: Vec<String>, max_tokens: u64) -> String {
    let config = ContextAssemblerConfig {
        max_tokens: max_tokens as usize,
        ..Default::default()
    };
    let ca = ContextAssembler::new(config);

    let chunks: Vec<Chunk> = chunk_jsons
        .iter()
        .filter_map(|json| {
            let v: serde_json::Value = serde_json::from_str(json).ok()?;
            let get_str = |key: &str| -> String {
                v.get(key)
                    .and_then(|x| x.as_str())
                    .unwrap_or("")
                    .to_string()
            };
            let get_opt = |key: &str| -> Option<String> {
                v.get(key).and_then(|x| x.as_str()).map(|s| s.to_string())
            };
            Some(Chunk {
                document_id: get_str("document_id"),
                chunk_index: v.get("chunk_index").and_then(|x| x.as_u64()).unwrap_or(0) as u32,
                chunk_text: get_str("chunk_text"),
                document_title: get_opt("document_title"),
                section_path: get_opt("section_path"),
                source_uri: get_opt("source_uri"),
                distance: v.get("distance").and_then(|x| x.as_f64()).unwrap_or(0.0) as f32,
                embedding: None,
            })
        })
        .collect();

    ca.assemble_chunks(chunks).text
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn query_bytes_decode() {
        let v = vec![1.0f32, 2.0, 3.0];
        let bytes: Vec<u8> = v.iter().flat_map(|f| f.to_le_bytes()).collect();
        let decoded: Vec<f32> = bytes
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
            .collect();
        assert_eq!(decoded, v);
    }

    #[test]
    fn assemble_context_empty() {
        let result = assemble_context(vec![], 1024);
        assert!(result.contains("<context") || result.is_empty());
    }

    #[test]
    fn assemble_context_one_chunk() {
        let chunk = serde_json::json!({
            "document_id": "doc-1",
            "chunk_index": 0,
            "chunk_text": "Hello world",
            "document_title": "Test",
        })
        .to_string();
        let result = assemble_context(vec![chunk], 4096);
        assert!(result.contains("Hello world"));
    }
}
