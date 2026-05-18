// Phase 1: sequential scan of all files — no geometric pruning.
// Phase 2: add VectorPruner to skip files with distant centroids.

use std::sync::Arc;

use ailake_catalog::{CatalogProvider, TableIdent};
use ailake_core::{AilakeResult, RowId};
use ailake_file::AilakeFileReader;
use ailake_store::Store;
use bytes::Bytes;

#[derive(Debug, Clone)]
pub struct SearchConfig {
    pub top_k: usize,
    pub ef_search: usize,
}

impl Default for SearchConfig {
    fn default() -> Self {
        Self {
            top_k: 10,
            ef_search: 50,
        }
    }
}

#[derive(Debug)]
pub struct SearchResult {
    pub row_id: RowId,
    pub distance: f32,
    pub file_path: String,
}

/// Search across all files in the latest snapshot.
/// Returns top-k results globally sorted by distance ascending.
pub async fn search(
    table: &TableIdent,
    query: &[f32],
    config: SearchConfig,
    vector_column: &str,
    dim: u32,
    catalog: Arc<dyn CatalogProvider>,
    store: Arc<dyn Store>,
) -> AilakeResult<Vec<SearchResult>> {
    let files = catalog.list_files(table, None).await?;
    let mut all_results: Vec<SearchResult> = Vec::new();

    for file_entry in &files {
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
