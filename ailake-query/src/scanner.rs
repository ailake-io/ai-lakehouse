use std::sync::Arc;

use ailake_catalog::{CatalogProvider, TableIdent};
use ailake_core::{AilakeResult, RowId, VectorMetric};
use ailake_file::AilakeFileReader;
use ailake_store::Store;
use bytes::Bytes;

use crate::pruner::VectorPruner;

#[derive(Debug, Clone)]
pub struct SearchConfig {
    pub top_k: usize,
    pub ef_search: usize,
    /// Maximum distance from query to file centroid edge for a file to be searched.
    /// Files where `distance(query, centroid) - radius > pruning_threshold` are skipped.
    /// Set to `f32::INFINITY` to disable pruning (scan all files).
    pub pruning_threshold: f32,
}

impl Default for SearchConfig {
    fn default() -> Self {
        Self {
            top_k: 10,
            ef_search: 50,
            pruning_threshold: f32::INFINITY,
        }
    }
}

impl SearchConfig {
    pub fn with_pruning(mut self, threshold: f32) -> Self {
        self.pruning_threshold = threshold;
        self
    }
}

#[derive(Debug)]
pub struct SearchResult {
    pub row_id: RowId,
    pub distance: f32,
    pub file_path: String,
}

/// Search across all files in the latest snapshot, with geometric pruning.
///
/// Flow:
/// 1. Load file list from catalog (includes centroid metadata)
/// 2. Prune files whose centroid + radius cannot contain a result within `pruning_threshold`
/// 3. For surviving files: load bytes, deserialize HNSW, run top-k search
/// 4. Global merge of all per-file top-k lists, return global top-k
pub async fn search(
    table: &TableIdent,
    query: &[f32],
    config: SearchConfig,
    vector_column: &str,
    dim: u32,
    catalog: Arc<dyn CatalogProvider>,
    store: Arc<dyn Store>,
) -> AilakeResult<Vec<SearchResult>> {
    // Get file metadata (includes centroid info) without reading any data files
    let all_files = catalog.list_files(table, None).await?;

    // Determine vector metric from table metadata for correct distance computation
    let table_meta = catalog.load_table(table).await?;
    let metric = parse_metric(
        table_meta
            .properties
            .get("ailake.vector-metric")
            .map(String::as_str)
            .unwrap_or("cosine"),
    );

    // Geometric pruning: skip files whose centroid is too far from the query
    let surviving_files =
        VectorPruner::prune(all_files, query, metric, config.pruning_threshold);

    let mut all_results: Vec<SearchResult> = Vec::new();

    for file_entry in &surviving_files {
        let file_bytes: Bytes = store.get(&file_entry.path).await?;
        let reader = AilakeFileReader::new(file_bytes, vector_column, dim);

        if !reader.is_ailake_file() {
            continue;
        }

        let index = reader.load_index()?;
        let local_results = index.search(query, config.top_k, config.ef_search);

        for (row_id, distance) in local_results {
            all_results.push(SearchResult {
                row_id,
                distance,
                file_path: file_entry.path.clone(),
            });
        }
    }

    // Global merge: sort all candidates by distance, keep top-k
    all_results.sort_by(|a, b| {
        a.distance
            .partial_cmp(&b.distance)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    all_results.truncate(config.top_k);
    Ok(all_results)
}

fn parse_metric(s: &str) -> VectorMetric {
    match s {
        "euclidean" => VectorMetric::Euclidean,
        "dotproduct" | "dot_product" | "dot" => VectorMetric::DotProduct,
        _ => VectorMetric::Cosine,
    }
}
