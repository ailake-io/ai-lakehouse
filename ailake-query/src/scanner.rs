// SPDX-License-Identifier: MIT OR Apache-2.0
use std::sync::Arc;

use rayon::prelude::*;
use tracing::{debug, error};

use ailake_catalog::{CatalogProvider, DataFileEntry, IndexStatus, TableIdent};
use ailake_core::{AilakeError, AilakeResult, RowId, VectorMetric};
use ailake_file::AilakeFileReader;
use ailake_index::AnyIndex;
use ailake_store::Store;
use ailake_vec::exact_distance;
use arrow_array::RecordBatch;
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
    /// When `Some(factor)`, fetch `top_k * factor` candidates from the HNSW index and
    /// rerank them using exact F32 distances before truncating to `top_k`.
    /// Corrects the approximation error introduced by PQ-compressed HNSW distances.
    /// `None` (default) disables reranking.
    pub rerank_factor: Option<usize>,
}

impl Default for SearchConfig {
    fn default() -> Self {
        Self {
            top_k: 10,
            ef_search: 50,
            pruning_threshold: f32::INFINITY,
            rerank_factor: None,
        }
    }
}

impl SearchConfig {
    pub fn with_pruning(mut self, threshold: f32) -> Self {
        self.pruning_threshold = threshold;
        self
    }

    pub fn with_reranking(mut self, factor: usize) -> Self {
        self.rerank_factor = Some(factor);
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
    let total_files = all_files.len();
    let surviving_files = VectorPruner::prune(all_files, query, metric, config.pruning_threshold);
    debug!(
        "ailake: geometric pruning — {}/{} files survive (threshold={})",
        surviving_files.len(),
        total_files,
        config.pruning_threshold
    );

    let candidate_k = match config.rerank_factor {
        Some(factor) => config.top_k * factor,
        None => config.top_k,
    };

    let mut all_results: Vec<SearchResult> = Vec::new();

    for file_entry in &surviving_files {
        let file_bytes: Bytes = store.get(&file_entry.path).await?;
        let reader = AilakeFileReader::new(file_bytes, vector_column, dim);

        if file_entry.index_status == IndexStatus::Indexing || !reader.is_ailake_file() {
            // HNSW not yet built — flat scan over raw vectors.
            debug!(
                "ailake: flat scan fallback for {} (index_status={:?})",
                file_entry.path, file_entry.index_status
            );
            let (_, raw_vectors) = reader.read_parquet()?;
            for (row_id, distance) in flat_search(&raw_vectors, query, candidate_k, metric) {
                all_results.push(SearchResult {
                    row_id,
                    distance,
                    file_path: file_entry.path.clone(),
                });
            }
            continue;
        }

        let index = reader.load_any_index_for_column(vector_column)?;
        let local_results = index.search(query, candidate_k, config.ef_search);

        if config.rerank_factor.is_some() {
            // Read raw F32 vectors for exact distance reranking; file bytes already loaded.
            let (_, raw_vectors) = reader.read_parquet()?;
            for (row_id, _approx_dist) in local_results {
                let idx = row_id.as_u64() as usize;
                let exact_dist = match raw_vectors.get(idx) {
                    Some(v) => exact_distance(metric, query, v),
                    None => {
                        error!(
                            "ailake: invariant violated — row_id {} out of bounds \
                             (raw_vectors.len={}, file={}); \
                             Parquet row count and HNSW node count are out of sync; \
                             file may be corrupt — run compaction to rebuild",
                            idx,
                            raw_vectors.len(),
                            file_entry.path
                        );
                        f32::INFINITY
                    }
                };
                all_results.push(SearchResult {
                    row_id,
                    distance: exact_dist,
                    file_path: file_entry.path.clone(),
                });
            }
        } else {
            for (row_id, distance) in local_results {
                all_results.push(SearchResult {
                    row_id,
                    distance,
                    file_path: file_entry.path.clone(),
                });
            }
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

/// Brute-force top-k search over raw vectors. Used for Indexing shards.
fn flat_search(
    raw: &[Vec<f32>],
    query: &[f32],
    top_k: usize,
    metric: VectorMetric,
) -> Vec<(RowId, f32)> {
    let mut results: Vec<(RowId, f32)> = raw
        .iter()
        .enumerate()
        .map(|(i, v)| (RowId::new(i as u64), exact_distance(metric, query, v)))
        .collect();
    results.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
    results.truncate(top_k);
    results
}

fn parse_metric(s: &str) -> VectorMetric {
    match s {
        "euclidean" => VectorMetric::Euclidean,
        "dotproduct" | "dot_product" | "dot" => VectorMetric::DotProduct,
        _ => VectorMetric::Cosine,
    }
}

/// Pre-loaded search session: all HNSW indexes loaded into memory once.
///
/// Useful for benchmarks and servers that issue many queries against the same
/// snapshot. Avoids re-loading and re-deserializing indexes on every call.
pub struct SearchSession {
    shards: Vec<LoadedShard>,
    metric: VectorMetric,
}

struct LoadedShard {
    entry: DataFileEntry,
    /// None when the shard is still being indexed (IndexStatus::Indexing).
    index: Option<AnyIndex>,
    /// Raw F32 vectors: always present for Indexing shards (flat scan), optionally
    /// present for Ready shards when `load_raw = true` (reranking).
    raw_vectors: Option<Vec<Vec<f32>>>,
}

impl SearchSession {
    /// Load all indexes for the latest snapshot into memory.
    ///
    /// Pass `load_raw = true` when reranking will be used (`rerank_factor` is
    /// `Some`); it reads the full parquet columns so exact distances are
    /// available without extra I/O during `search_query`.
    pub async fn load(
        table: &TableIdent,
        vector_column: &str,
        dim: u32,
        catalog: Arc<dyn CatalogProvider>,
        store: Arc<dyn Store>,
        load_raw: bool,
    ) -> AilakeResult<Self> {
        let all_files = catalog.list_files(table, None).await?;
        let table_meta = catalog.load_table(table).await?;
        let metric = parse_metric(
            table_meta
                .properties
                .get("ailake.vector-metric")
                .map(String::as_str)
                .unwrap_or("cosine"),
        );

        let mut shards = Vec::with_capacity(all_files.len());
        for entry in all_files {
            let file_bytes: Bytes = store.get(&entry.path).await?;
            let reader = AilakeFileReader::new(file_bytes, vector_column, dim);

            if entry.index_status == IndexStatus::Indexing {
                // HNSW not yet built — load raw vectors for flat scan.
                let (_, raw_vecs) = reader.read_parquet()?;
                shards.push(LoadedShard {
                    entry,
                    index: None,
                    raw_vectors: Some(raw_vecs),
                });
            } else if reader.is_ailake_file() {
                let mut index = reader.load_any_index_for_column(vector_column)?;
                let raw_vectors = if load_raw {
                    index.quantize_to_f16();
                    let (_, vecs) = reader.read_parquet()?;
                    Some(vecs)
                } else {
                    None
                };
                shards.push(LoadedShard {
                    entry,
                    index: Some(index),
                    raw_vectors,
                });
            }
        }

        Ok(Self { shards, metric })
    }

    /// Number of loaded shards.
    pub fn shard_count(&self) -> usize {
        self.shards.len()
    }

    /// Search multiple queries in one call.
    ///
    /// For shards with raw vectors (Indexing or reranking): dispatches to GPU batch
    /// matmul when a CUDA device is available, falling back to CPU flat scan.
    /// For indexed shards (HNSW / IVF-PQ): rayon parallel-map over queries — graph
    /// traversal is inherently sequential and has no GPU batch path.
    ///
    /// Returns one `Vec<SearchResult>` per input query, in the same order.
    pub fn search_batch(
        &self,
        queries: &[Vec<f32>],
        config: &SearchConfig,
    ) -> Vec<Vec<SearchResult>> {
        if queries.is_empty() {
            return vec![];
        }

        let n_queries = queries.len();
        let candidate_k = match config.rerank_factor {
            Some(factor) => config.top_k * factor,
            None => config.top_k,
        };
        let use_nvidia = ailake_index::hardware::detect_cuda();
        let use_amd = ailake_index::hardware::detect_rocm();

        // Accumulate per-query results across all shards.
        let mut all_results: Vec<Vec<SearchResult>> = (0..n_queries).map(|_| Vec::new()).collect();

        for shard in &self.shards {
            if let Some(raw) = &shard.raw_vectors {
                // Flat-scan shard — try GPU batch path (NVIDIA first, then AMD ROCm).
                if !raw.is_empty() {
                    let dim = raw[0].len();
                    let flat: Vec<f32> = raw.iter().flat_map(|v| v.iter().copied()).collect();
                    let row_ids: Vec<u64> = (0..raw.len() as u64).collect();
                    let q_refs: Vec<&[f32]> = queries.iter().map(|q| q.as_slice()).collect();

                    let gpu_batch = if use_nvidia {
                        ailake_index::gpu::try_nvidia_search_batch(
                            &q_refs,
                            &row_ids,
                            &flat,
                            dim,
                            self.metric,
                            candidate_k,
                        )
                    } else if use_amd {
                        ailake_index::gpu::try_rocm_search_batch(
                            &q_refs,
                            &row_ids,
                            &flat,
                            dim,
                            self.metric,
                            candidate_k,
                        )
                    } else {
                        None
                    };

                    if let Some(batch) = gpu_batch {
                        for (qi, results) in batch.into_iter().enumerate() {
                            for (row_id, distance) in results {
                                all_results[qi].push(SearchResult {
                                    row_id,
                                    distance,
                                    file_path: shard.entry.path.clone(),
                                });
                            }
                        }
                        continue;
                    }
                }

                // CPU fallback for flat scan.
                for (qi, query) in queries.iter().enumerate() {
                    for (row_id, distance) in flat_search(raw, query, candidate_k, self.metric) {
                        all_results[qi].push(SearchResult {
                            row_id,
                            distance,
                            file_path: shard.entry.path.clone(),
                        });
                    }
                }
            } else if let Some(index) = &shard.index {
                // Indexed shard — rayon parallel-map over queries.
                let shard_results: Vec<Vec<SearchResult>> = queries
                    .par_iter()
                    .map(|query| {
                        index
                            .search(query, candidate_k, config.ef_search)
                            .into_iter()
                            .map(|(row_id, distance)| SearchResult {
                                row_id,
                                distance,
                                file_path: shard.entry.path.clone(),
                            })
                            .collect()
                    })
                    .collect();

                for (qi, results) in shard_results.into_iter().enumerate() {
                    all_results[qi].extend(results);
                }
            }
        }

        // Sort + truncate per query.
        for results in &mut all_results {
            results.sort_by(|a, b| {
                a.distance
                    .partial_cmp(&b.distance)
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
            results.truncate(config.top_k);
        }

        all_results
    }

    /// Search using pre-loaded indexes. No I/O — pure in-memory search.
    pub fn search_query(&self, query: &[f32], config: &SearchConfig) -> Vec<SearchResult> {
        let candidate_k = match config.rerank_factor {
            Some(factor) => config.top_k * factor,
            None => config.top_k,
        };

        let mut all_results: Vec<SearchResult> = self
            .shards
            .par_iter()
            .flat_map(|shard| {
                // Geometric pruning per shard.
                if let Some(centroid) = ailake_catalog::decode_centroid(&shard.entry, self.metric) {
                    let dist = match self.metric {
                        VectorMetric::Cosine | VectorMetric::NormalizedCosine => {
                            ailake_vec::cosine_distance(query, &centroid.values)
                        }
                        VectorMetric::Euclidean => {
                            ailake_vec::euclidean_distance(query, &centroid.values)
                        }
                        VectorMetric::DotProduct => {
                            -ailake_vec::dot_product(query, &centroid.values)
                        }
                    };
                    if dist - centroid.radius > config.pruning_threshold {
                        return vec![];
                    }
                }

                if let Some(index) = &shard.index {
                    // Ready shard: HNSW or IVF-PQ search (dispatched by AnyIndex).
                    let local_results = index.search(query, candidate_k, config.ef_search);
                    if config.rerank_factor.is_some() {
                        if let Some(raw) = &shard.raw_vectors {
                            local_results
                                .into_iter()
                                .map(|(row_id, _approx_dist)| {
                                    let idx = row_id.as_u64() as usize;
                                    let exact_dist = raw
                                        .get(idx)
                                        .map(|v| exact_distance(self.metric, query, v))
                                        .unwrap_or(f32::INFINITY);
                                    SearchResult {
                                        row_id,
                                        distance: exact_dist,
                                        file_path: shard.entry.path.clone(),
                                    }
                                })
                                .collect()
                        } else {
                            local_results
                                .into_iter()
                                .map(|(row_id, distance)| SearchResult {
                                    row_id,
                                    distance,
                                    file_path: shard.entry.path.clone(),
                                })
                                .collect()
                        }
                    } else {
                        local_results
                            .into_iter()
                            .map(|(row_id, distance)| SearchResult {
                                row_id,
                                distance,
                                file_path: shard.entry.path.clone(),
                            })
                            .collect()
                    }
                } else if let Some(raw) = &shard.raw_vectors {
                    // Indexing shard: exact flat scan.
                    flat_search(raw, query, candidate_k, self.metric)
                        .into_iter()
                        .map(|(row_id, distance)| SearchResult {
                            row_id,
                            distance,
                            file_path: shard.entry.path.clone(),
                        })
                        .collect()
                } else {
                    vec![]
                }
            })
            .collect();

        all_results.sort_by(|a, b| {
            a.distance
                .partial_cmp(&b.distance)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        all_results.truncate(config.top_k);
        all_results
    }
}

/// Fetch full row data for a slice of search results.
///
/// Groups results by Parquet file, reads each file once, extracts the matching rows
/// via `arrow_select::take`, then concatenates everything back in original top-k order
/// with a `_distance: Float32` column appended.
///
/// Use this immediately after `search()` to retrieve the actual text / metadata
/// columns (e.g. `chunk_text`, `document_title`) alongside the distance scores.
pub async fn fetch_rows(
    results: &[SearchResult],
    store: Arc<dyn Store>,
    vector_column: &str,
    dim: u32,
) -> AilakeResult<RecordBatch> {
    use std::collections::HashMap;

    use arrow_array::{ArrayRef, Float32Array, UInt32Array};
    use arrow_schema::{DataType, Field, Schema};
    use arrow_select::{concat::concat_batches, take::take};

    if results.is_empty() {
        return Ok(RecordBatch::new_empty(Arc::new(Schema::empty())));
    }

    // Group by file path; preserve original position for re-sorting.
    let mut by_file: HashMap<&str, Vec<(u64, f32, usize)>> = HashMap::new();
    for (i, r) in results.iter().enumerate() {
        by_file
            .entry(r.file_path.as_str())
            .or_default()
            .push((r.row_id.as_u64(), r.distance, i));
    }

    use arrow_array::FixedSizeListArray;

    // (original_index, distance, single-row RecordBatch, decoded F32 vector)
    let mut collected: Vec<(usize, f32, RecordBatch, Vec<f32>)> = Vec::with_capacity(results.len());

    for (file_path, rows) in &by_file {
        let bytes = store.get(file_path).await?;
        let reader = AilakeFileReader::new(bytes, vector_column, dim);
        let (batch, vectors) = reader.read_parquet()?;

        for &(row_id, distance, pos) in rows {
            let idx = row_id as usize;
            if idx >= batch.num_rows() {
                tracing::warn!(
                    "fetch_rows: row_id {} out of bounds (file_rows={}, file={}), skipping",
                    idx,
                    batch.num_rows(),
                    file_path
                );
                continue;
            }

            let indices = UInt32Array::from(vec![idx as u32]);
            let row_cols: Vec<ArrayRef> = batch
                .columns()
                .iter()
                .map(|col| {
                    take(col.as_ref(), &indices, None)
                        .map_err(|e| AilakeError::Arrow(e.to_string()))
                })
                .collect::<AilakeResult<Vec<_>>>()?;

            let row_batch = RecordBatch::try_new(batch.schema(), row_cols)
                .map_err(|e| AilakeError::Arrow(e.to_string()))?;

            // Capture decoded F32 vector for this row (empty vec if not available).
            let vec = vectors
                .get(idx)
                .cloned()
                .unwrap_or_else(|| vec![0.0f32; dim as usize]);

            collected.push((pos, distance, row_batch, vec));
        }
    }

    if collected.is_empty() {
        return Ok(RecordBatch::new_empty(Arc::new(Schema::empty())));
    }

    // Restore original top-k order from the search results slice.
    collected.sort_by_key(|(pos, _, _, _)| *pos);

    let distances: Vec<f32> = collected.iter().map(|(_, d, _, _)| *d).collect();
    let row_batches: Vec<&RecordBatch> = collected.iter().map(|(_, _, b, _)| b).collect();
    let base_schema = collected[0].2.schema();

    let combined = concat_batches(&base_schema, row_batches)
        .map_err(|e| AilakeError::Arrow(e.to_string()))?;

    // Build FixedSizeList<Float32> column with decoded vectors (F32, not raw F16 bytes).
    let flat_vecs: Vec<f32> = collected
        .iter()
        .flat_map(|(_, _, _, v)| v.iter().copied())
        .collect();
    let item_field = Arc::new(Field::new("item", DataType::Float32, false));
    let values_arr = Arc::new(Float32Array::from(flat_vecs)) as ArrayRef;
    let vec_col = FixedSizeListArray::new(item_field.clone(), dim as i32, values_arr, None);
    let vec_field = Arc::new(Field::new(
        vector_column,
        DataType::FixedSizeList(item_field, dim as i32),
        false,
    ));

    // Schema: tabular cols, then decoded vector col, then _distance.
    let mut fields: Vec<Arc<Field>> = base_schema.fields().to_vec();
    fields.push(vec_field);
    fields.push(Arc::new(Field::new("_distance", DataType::Float32, false)));
    let new_schema = Arc::new(Schema::new(fields));

    let mut columns: Vec<ArrayRef> = combined.columns().to_vec();
    columns.push(Arc::new(vec_col));
    columns.push(Arc::new(Float32Array::from(distances)));

    RecordBatch::try_new(new_schema, columns).map_err(|e| AilakeError::Arrow(e.to_string()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use ailake_catalog::{HadoopCatalog, TableIdent};
    use ailake_core::{VectorMetric, VectorPrecision, VectorStoragePolicy};
    use ailake_store::LocalStore;
    use arrow_array::{Int32Array, RecordBatch};
    use arrow_schema::{DataType, Field, Schema};
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
        }
    }

    async fn write_demo_table(dir: &TempDir, dim: usize, rows: usize) {
        let store: Arc<dyn Store> = Arc::new(LocalStore::new(dir.path()));
        let catalog = Arc::new(HadoopCatalog::new(store.clone(), "warehouse"));
        let table = TableIdent::new("default", "table");

        let schema = Arc::new(Schema::new(vec![Field::new("id", DataType::Int32, false)]));
        let ids: Vec<i32> = (0..rows as i32).collect();
        let batch = RecordBatch::try_new(schema, vec![Arc::new(Int32Array::from(ids))]).unwrap();

        // Each row i has embedding with 1.0 at dimension i and 0 elsewhere (unit basis vectors)
        let embeddings: Vec<Vec<f32>> = (0..rows)
            .map(|i| {
                let mut v = vec![0.0f32; dim];
                v[i % dim] = 1.0;
                v
            })
            .collect();

        let mut writer =
            crate::TableWriter::create_or_open(catalog, store, make_policy(dim as u32), table)
                .await
                .unwrap();
        writer.write_batch(&batch, &embeddings).await.unwrap();
        writer.commit().await.unwrap();
    }

    #[tokio::test]
    async fn rerank_returns_correct_top_k_count() {
        let dir = TempDir::new().unwrap();
        let dim = 8usize;
        write_demo_table(&dir, dim, 8).await;

        let store: Arc<dyn Store> = Arc::new(LocalStore::new(dir.path()));
        let catalog: Arc<dyn CatalogProvider> =
            Arc::new(HadoopCatalog::new(store.clone(), "warehouse"));
        let table = TableIdent::new("default", "table");

        let query = vec![1.0f32, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let config = SearchConfig {
            top_k: 3,
            ef_search: 50,
            pruning_threshold: f32::INFINITY,
            rerank_factor: Some(2),
        };

        let results = search(
            &table,
            &query,
            config,
            "embedding",
            dim as u32,
            catalog,
            store,
        )
        .await
        .unwrap();

        assert_eq!(results.len(), 3);
    }

    #[tokio::test]
    async fn rerank_nearest_is_exact_match() {
        let dir = TempDir::new().unwrap();
        let dim = 8usize;
        write_demo_table(&dir, dim, 8).await;

        let store: Arc<dyn Store> = Arc::new(LocalStore::new(dir.path()));
        let catalog: Arc<dyn CatalogProvider> =
            Arc::new(HadoopCatalog::new(store.clone(), "warehouse"));
        let table = TableIdent::new("default", "table");

        // Row 0 has [1,0,0,...] — cosine distance to same query is 0
        let query = vec![1.0f32, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let config = SearchConfig {
            top_k: 1,
            ef_search: 50,
            pruning_threshold: f32::INFINITY,
            rerank_factor: Some(4),
        };

        let results = search(
            &table,
            &query,
            config,
            "embedding",
            dim as u32,
            catalog,
            store,
        )
        .await
        .unwrap();

        assert_eq!(results.len(), 1);
        // Exact cosine distance between identical unit vectors is ~0 (F16 rounding allowed)
        assert!(
            results[0].distance < 1e-3,
            "distance was {}",
            results[0].distance
        );
        assert_eq!(results[0].row_id, RowId::new(0));
    }

    #[tokio::test]
    async fn no_rerank_matches_default_behavior() {
        let dir = TempDir::new().unwrap();
        let dim = 4usize;
        write_demo_table(&dir, dim, 4).await;

        let store_a: Arc<dyn Store> = Arc::new(LocalStore::new(dir.path()));
        let store_b: Arc<dyn Store> = Arc::new(LocalStore::new(dir.path()));
        let cat_a: Arc<dyn CatalogProvider> =
            Arc::new(HadoopCatalog::new(store_a.clone(), "warehouse"));
        let cat_b: Arc<dyn CatalogProvider> =
            Arc::new(HadoopCatalog::new(store_b.clone(), "warehouse"));
        let table = TableIdent::new("default", "table");

        let query = vec![1.0f32, 0.0, 0.0, 0.0];
        let cfg_plain = SearchConfig {
            top_k: 2,
            ef_search: 50,
            pruning_threshold: f32::INFINITY,
            rerank_factor: None,
        };
        let cfg_rerank = SearchConfig {
            top_k: 2,
            ef_search: 50,
            pruning_threshold: f32::INFINITY,
            rerank_factor: Some(2),
        };

        let plain = search(
            &table,
            &query,
            cfg_plain,
            "embedding",
            dim as u32,
            cat_a,
            store_a,
        )
        .await
        .unwrap();
        let reranked = search(
            &table,
            &query,
            cfg_rerank,
            "embedding",
            dim as u32,
            cat_b,
            store_b,
        )
        .await
        .unwrap();

        // Both should return same top-1 result (row 0, distance ~0)
        assert_eq!(plain[0].row_id, reranked[0].row_id);
    }
}
