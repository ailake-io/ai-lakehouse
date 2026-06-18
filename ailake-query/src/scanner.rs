// SPDX-License-Identifier: MIT OR Apache-2.0
use std::sync::Arc;

use rayon::prelude::*;
use tracing::{debug, error};

use ailake_catalog::{CatalogProvider, DataFileEntry, IndexStatus, TableIdent};
use ailake_core::{AilakeError, AilakeResult, EmbeddingModelInfo, RowId, VectorMetric};
use ailake_file::AilakeFileReader;
use ailake_index::AnyIndex;
use ailake_store::Store;
use ailake_vec::exact_distance;
use arrow_array::{Array, RecordBatch};
use bytes::Bytes;

use crate::pruner::VectorPruner;

/// Injectable per-result scoring function for hybrid ranking.
///
/// Called after HNSW retrieval with the HNSW distance and a single-row
/// `RecordBatch` containing all Parquet columns for that result. Returns a
/// replacement score (lower = better rank, same convention as distance).
///
/// Typical use: combine HNSW distance with recency and importance signals
/// from the `episodic_columns` for agent memory tables:
///
/// ```rust,no_run
/// use ailake_core::{hybrid_score, episodic_columns};
/// use ailake_query::scanner::ScoreFn;
/// use arrow_array::{RecordBatch, cast::AsArray};
/// use arrow_array::types::Float32Type;
///
/// let score_fn = ScoreFn::new(|distance, row| {
///     let recency = row
///         .column_by_name(episodic_columns::RECENCY_WEIGHT)
///         .and_then(|c| c.as_primitive_opt::<Float32Type>())
///         .and_then(|a| a.iter().next().flatten())
///         .unwrap_or(1.0);
///     let importance = row
///         .column_by_name(episodic_columns::IMPORTANCE_SCORE)
///         .and_then(|c| c.as_primitive_opt::<Float32Type>())
///         .and_then(|a| a.iter().next().flatten())
///         .unwrap_or(1.0);
///     hybrid_score(distance, recency, importance)
/// });
/// ```
pub struct ScoreFn(pub std::sync::Arc<dyn Fn(f32, &RecordBatch) -> f32 + Send + Sync>);

impl ScoreFn {
    pub fn new(f: impl Fn(f32, &RecordBatch) -> f32 + Send + Sync + 'static) -> Self {
        Self(std::sync::Arc::new(f))
    }

    #[inline]
    pub fn call(&self, distance: f32, row: &RecordBatch) -> f32 {
        (self.0)(distance, row)
    }
}

impl Clone for ScoreFn {
    fn clone(&self) -> Self {
        Self(std::sync::Arc::clone(&self.0))
    }
}

impl std::fmt::Debug for ScoreFn {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("ScoreFn(<fn>)")
    }
}

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
    /// Hybrid BM25+vector search configuration.
    ///
    /// When set, the pipeline loads global IDF stats from the table's BM25 stats file,
    /// fetches a larger candidate pool from HNSW (`candidate_pool` or `10 * top_k`),
    /// scores each candidate with BM25 against `query_text`, then fuses vector distance
    /// and BM25 score via RRF (default) or linear combination.
    ///
    /// The BM25 stats file (`metadata/ailake_bm25_stats.bin`) is populated automatically
    /// by `TableWriter` when `bm25_text_column` is configured. If absent, pure vector
    /// distances are used (BM25 scores default to 0).
    pub hybrid: Option<crate::bm25::HybridConfig>,
    /// Optional scoring function for hybrid ranking.
    ///
    /// When set, the search pipeline reads the Parquet row for each HNSW
    /// candidate and calls `score_fn(distance, &single_row_batch)`. The
    /// returned value replaces `distance` in `SearchResult` and determines
    /// final ranking (lower = better).
    ///
    /// If `rerank_factor` is also set, `score_fn` receives the exact
    /// (non-approximated) distance from the reranking step.
    ///
    /// Use `ScoreFn::new(|d, row| ...)` to construct. See `ScoreFn` docs
    /// for an example using `hybrid_score` with episodic memory columns.
    pub score_fn: Option<ScoreFn>,
    /// Partition filter: only search files whose `DataFileEntry::partition_value`
    /// matches this string. `None` searches all files (no partition pruning).
    /// Set to `agent_id` in Agent.recall() for per-agent isolated search.
    pub partition_filter: Option<String>,
}

impl Default for SearchConfig {
    fn default() -> Self {
        Self {
            top_k: 10,
            ef_search: 50,
            pruning_threshold: f32::INFINITY,
            rerank_factor: None,
            score_fn: None,
            partition_filter: None,
            hybrid: None,
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

    pub fn with_score_fn(mut self, f: impl Fn(f32, &RecordBatch) -> f32 + Send + Sync + 'static) -> Self {
        self.score_fn = Some(ScoreFn::new(f));
        self
    }

    pub fn with_hybrid(mut self, cfg: crate::bm25::HybridConfig) -> Self {
        self.hybrid = Some(cfg);
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

    // Validate query dim against the column's stored dim.
    // Primary column: use `ailake.vector-dim`. Secondary columns: use `ailake.dim-<col>`.
    // Skip validation when the column has no stored dim (e.g. old tables written before
    // multi-column support).
    let primary_col = table_meta
        .properties
        .get("ailake.vector-column")
        .map(String::as_str)
        .unwrap_or("");
    let stored_dim_key = if vector_column == primary_col {
        "ailake.vector-dim".to_string()
    } else {
        format!("ailake.dim-{vector_column}")
    };
    if let Some(table_dim_str) = table_meta.properties.get(&stored_dim_key) {
        if let Ok(table_dim) = table_dim_str.parse::<u32>() {
            let query_dim = query.len() as u32;
            if query_dim != table_dim {
                let table_model = table_meta
                    .properties
                    .get(EmbeddingModelInfo::property_key())
                    .cloned()
                    .unwrap_or_else(|| format!("dim={}", table_dim));
                return Err(AilakeError::ModelMismatch {
                    table_model,
                    table_dim,
                    batch_model: format!("query dim={}", query_dim),
                    batch_dim: query_dim,
                });
            }
        }
    }

    // Metric: prefer per-column `ailake.metric-<col>`, fall back to primary metric.
    let metric_key = if vector_column == primary_col {
        "ailake.vector-metric".to_string()
    } else {
        format!("ailake.metric-{vector_column}")
    };
    let metric = parse_metric(
        table_meta
            .properties
            .get(&metric_key)
            .or_else(|| table_meta.properties.get("ailake.vector-metric"))
            .map(String::as_str)
            .unwrap_or("cosine"),
    );

    // Partition pruning: skip files not belonging to the requested partition value.
    let all_files = if let Some(ref pv) = config.partition_filter {
        let before = all_files.len();
        let filtered: Vec<_> = all_files
            .into_iter()
            .filter(|f| f.partition_value.as_deref() == Some(pv.as_str()))
            .collect();
        debug!(
            "ailake: partition pruning '{}' — {}/{} files survive",
            pv,
            filtered.len(),
            before
        );
        filtered
    } else {
        all_files
    };

    // Geometric pruning: skip files whose centroid is too far from the query
    let total_files = all_files.len();
    let surviving_files = VectorPruner::prune(all_files, query, metric, config.pruning_threshold);
    debug!(
        "ailake: geometric pruning — {}/{} files survive (threshold={})",
        surviving_files.len(),
        total_files,
        config.pruning_threshold
    );

    // Compute candidate pool: hybrid needs a larger pool for BM25 re-ranking.
    let candidate_k = match (&config.hybrid, config.rerank_factor) {
        (Some(h), rf) => {
            let pool = h.candidate_pool.unwrap_or(config.top_k * 10);
            pool.max(rf.map_or(config.top_k, |f| f * config.top_k))
        }
        (None, Some(factor)) => config.top_k * factor,
        (None, None) => config.top_k,
    };

    let use_hybrid = config.hybrid.is_some();

    // Load BM25 stats from the table's stats file when hybrid search is active.
    let bm25_stats: Option<crate::bm25::IdfStats> = if let Some(ref h) = config.hybrid {
        if h.text_columns.is_empty() {
            None
        } else {
            let stats_path = table_meta
                .properties
                .get(crate::bm25::BM25_STATS_PATH_PROP)
                .map(String::as_str)
                .unwrap_or(crate::bm25::BM25_STATS_FILE);
            match store.get(stats_path).await {
                Ok(bytes) => crate::bm25::IdfStats::from_bytes(&bytes).ok(),
                Err(_) => {
                    debug!(
                        "ailake: BM25 stats not found at '{}' — falling back to empty corpus IDF",
                        stats_path
                    );
                    None
                }
            }
        }
    } else {
        None
    };

    // raw_candidates: (row_id, vec_dist, file_path, bm25_text) for hybrid re-ranking.
    // Only populated when use_hybrid = true; otherwise all_results is populated directly.
    let mut raw_candidates: Vec<(RowId, f32, String, String)> = Vec::new();
    let mut all_results: Vec<SearchResult> = Vec::new();

    for file_entry in &surviving_files {
        let file_bytes: Bytes = store.get(&file_entry.path).await?;
        let reader = AilakeFileReader::new(file_bytes, vector_column, dim);

        // Parquet read required for: flat scan fallback, exact reranking, score_fn, hybrid.
        let need_parquet = file_entry.index_status == IndexStatus::Indexing
            || !reader.is_ailake_file()
            || config.rerank_factor.is_some()
            || config.score_fn.is_some()
            || use_hybrid;

        if file_entry.index_status == IndexStatus::Indexing || !reader.is_ailake_file() {
            debug!(
                "ailake: flat scan fallback for {} (index_status={:?})",
                file_entry.path, file_entry.index_status
            );
            let (batch, raw_vectors) = reader.read_parquet()?;
            for (row_id, distance) in flat_search(&raw_vectors, query, candidate_k, metric) {
                if use_hybrid {
                    let text = extract_text_for_row(
                        &batch,
                        row_id.as_u64() as usize,
                        config.hybrid.as_ref().unwrap(),
                    );
                    raw_candidates.push((row_id, distance, file_entry.path.clone(), text));
                } else {
                    let final_score = apply_score_fn(&config.score_fn, distance, row_id, &batch);
                    all_results.push(SearchResult {
                        row_id,
                        distance: final_score,
                        file_path: file_entry.path.clone(),
                    });
                }
            }
            continue;
        }

        let index = reader.load_any_index_for_column(vector_column)?;
        let local_results = index.search(query, candidate_k, config.ef_search);

        let parquet_data = if need_parquet {
            Some(reader.read_parquet()?)
        } else {
            None
        };

        for (row_id, approx_dist) in local_results {
            let idx = row_id.as_u64() as usize;

            let distance = if config.rerank_factor.is_some() {
                match parquet_data.as_ref().and_then(|(_, vecs)| vecs.get(idx)) {
                    Some(v) => exact_distance(metric, query, v),
                    None => {
                        error!(
                            "ailake: invariant violated — row_id {} out of bounds \
                             (file={}); Parquet and HNSW node count out of sync; \
                             run compaction to rebuild",
                            idx, file_entry.path
                        );
                        f32::INFINITY
                    }
                }
            } else {
                approx_dist
            };

            if use_hybrid {
                let text = parquet_data.as_ref().map_or(String::new(), |(batch, _)| {
                    extract_text_for_row(batch, idx, config.hybrid.as_ref().unwrap())
                });
                raw_candidates.push((row_id, distance, file_entry.path.clone(), text));
            } else {
                let final_score = if let Some((ref batch, _)) = parquet_data {
                    apply_score_fn(&config.score_fn, distance, row_id, batch)
                } else {
                    distance
                };
                all_results.push(SearchResult {
                    row_id,
                    distance: final_score,
                    file_path: file_entry.path.clone(),
                });
            }
        }
    }

    // Hybrid BM25 fusion: applied after all HNSW candidates are collected.
    if let Some(ref h) = config.hybrid {
        let empty_stats = crate::bm25::IdfStats::default();
        let stats = bm25_stats.as_ref().unwrap_or(&empty_stats);
        let scorer = crate::bm25::BM25Scorer::new(stats);

        // Compute BM25 score for each candidate.
        let bm25_scores: Vec<f32> = raw_candidates
            .iter()
            .map(|(_, _, _, text)| scorer.score(&h.query_text, text))
            .collect();

        // Rank by vector distance (already sorted within each file, but merge globally).
        raw_candidates.sort_by(|a, b| a.1.total_cmp(&b.1));
        let vec_ranks: Vec<usize> = (0..raw_candidates.len()).collect();

        // Rank by BM25 score descending (higher BM25 = better).
        let mut bm25_indexed: Vec<(usize, f32)> = bm25_scores.iter().copied().enumerate().collect();
        bm25_indexed.sort_by(|a, b| b.1.total_cmp(&a.1));
        let mut bm25_rank_of = vec![0usize; raw_candidates.len()];
        for (rank, (orig_idx, _)) in bm25_indexed.iter().enumerate() {
            bm25_rank_of[*orig_idx] = rank;
        }

        use crate::bm25::{HybridFusion, linear_score, rrf_score};

        let fused: Vec<f32> = match h.fusion {
            HybridFusion::Rrf => vec_ranks
                .iter()
                .enumerate()
                .map(|(i, &vr)| rrf_score(vr, bm25_rank_of[i], h.bm25_weight))
                .collect(),
            HybridFusion::Linear => {
                let min_d = raw_candidates.iter().map(|r| r.1).fold(f32::INFINITY, f32::min);
                let max_d = raw_candidates.iter().map(|r| r.1).fold(f32::NEG_INFINITY, f32::max);
                let min_b = bm25_scores.iter().copied().fold(f32::INFINITY, f32::min);
                let max_b = bm25_scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                raw_candidates
                    .iter()
                    .enumerate()
                    .map(|(i, r)| {
                        linear_score(r.1, min_d, max_d, bm25_scores[i], min_b, max_b, h.bm25_weight)
                    })
                    .collect()
            }
        };

        for (i, (row_id, _, file_path, _)) in raw_candidates.into_iter().enumerate() {
            all_results.push(SearchResult { row_id, distance: fused[i], file_path });
        }

        // For RRF: lower (more negative) = better; for Linear: lower = better. Same convention.
        all_results.sort_by(|a, b| a.distance.total_cmp(&b.distance));
    } else {
        all_results.sort_by(|a, b| a.distance.total_cmp(&b.distance));
    }

    all_results.truncate(config.top_k);
    Ok(all_results)
}

/// Extract concatenated text from specified columns for a single row.
fn extract_text_for_row(
    batch: &RecordBatch,
    row_idx: usize,
    hybrid: &crate::bm25::HybridConfig,
) -> String {
    use arrow_array::cast::AsArray;
    hybrid
        .text_columns
        .iter()
        .filter_map(|col| {
            batch.column_by_name(col).and_then(|arr| {
                arr.as_string_opt::<i32>().and_then(|sa| {
                    if row_idx < sa.len() && sa.is_valid(row_idx) {
                        Some(sa.value(row_idx).to_string())
                    } else {
                        None
                    }
                })
            })
        })
        .collect::<Vec<_>>()
        .join(" ")
}

/// One query arm in a cross-modal search.
#[derive(Debug, Clone)]
pub struct ModalQuery<'a> {
    /// Vector column to search (must exist in the table).
    pub column: &'a str,
    /// Query vector for this modality.
    pub query: &'a [f32],
    /// Relative weight applied in the RRF formula: `weight / (k + rank)`.
    /// `1.0` means equal weight across all modalities.
    pub weight: f32,
    /// Dimensionality of this column's vectors. `0` = auto-detect from table metadata
    /// (`ailake.dim-<column>` for secondary columns, `ailake.vector-dim` for primary).
    pub dim: u32,
}

/// Fusion method for combining results from multiple vector columns.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FusionMethod {
    /// Reciprocal Rank Fusion: `score(d) = Σ weight_i / (k + rank_i(d))`.
    /// `k = 60` (standard). Returned `SearchResult.distance` = `-rrf_score`
    /// so that sort-ascending-by-distance gives the correct RRF ranking.
    Rrf,
}

/// Cross-modal search: run independent HNSW searches across N vector columns,
/// then fuse per-column ranked lists using Reciprocal Rank Fusion.
///
/// Each `ModalQuery` specifies a column name, its query vector, RRF weight, and dim.
/// When `ModalQuery.dim == 0`, the dim is auto-detected from `ailake.dim-<col>` /
/// `ailake.vector-dim` in table metadata.
/// Results are de-duplicated by `(file_path, row_id)` and ranked by aggregate
/// RRF score. `SearchResult.distance` stores `-rrf_score` (lower = better) so
/// existing sort-ascending callers get the correct ordering.
pub async fn search_multimodal(
    table: &TableIdent,
    queries: &[ModalQuery<'_>],
    config: SearchConfig,
    catalog: Arc<dyn CatalogProvider>,
    store: Arc<dyn Store>,
    fusion: FusionMethod,
) -> AilakeResult<Vec<SearchResult>> {
    use std::collections::HashMap;

    if queries.is_empty() {
        return Err(AilakeError::InvalidArgument(
            "search_multimodal requires at least one ModalQuery".into(),
        ));
    }

    // Load table metadata once for dim auto-detection and metric resolution.
    let table_meta = catalog.load_table(table).await?;
    let primary_col = table_meta
        .properties
        .get("ailake.vector-column")
        .cloned()
        .unwrap_or_default();
    let primary_dim: u32 = table_meta
        .properties
        .get("ailake.vector-dim")
        .and_then(|s| s.parse().ok())
        .unwrap_or(0);

    // Fetch more candidates per column so RRF has enough to fuse.
    let per_col_k = (config.top_k * queries.len().max(2)).min(1000);

    let mut per_col_results: Vec<(f32, Vec<SearchResult>)> = Vec::with_capacity(queries.len());
    for mq in queries {
        // Resolve dim: caller-supplied > per-column property > primary column dim.
        let resolved_dim = if mq.dim > 0 {
            mq.dim
        } else if mq.column == primary_col {
            primary_dim
        } else {
            table_meta
                .properties
                .get(&format!("ailake.dim-{}", mq.column))
                .and_then(|s| s.parse().ok())
                .unwrap_or(mq.query.len() as u32)
        };

        let col_config = SearchConfig {
            top_k: per_col_k,
            ef_search: config.ef_search,
            pruning_threshold: config.pruning_threshold,
            rerank_factor: config.rerank_factor,
            score_fn: None,
            partition_filter: config.partition_filter.clone(),
            hybrid: None,
        };
        let results = search(
            table,
            mq.query,
            col_config,
            mq.column,
            resolved_dim,
            catalog.clone(),
            store.clone(),
        )
        .await?;
        per_col_results.push((mq.weight, results));
    }

    // RRF fusion: accumulate score per (file_path, row_id).
    const K: f32 = 60.0;
    let mut scores: HashMap<(String, u64), f32> = HashMap::new();

    for (weight, results) in &per_col_results {
        for (rank, r) in results.iter().enumerate() {
            let key = (r.file_path.clone(), r.row_id.as_u64());
            let rrf = weight / (K + rank as f32 + 1.0);
            *scores.entry(key).or_insert(0.0) += rrf;
        }
    }

    // Build SearchResult list sorted by descending RRF score.
    // Store `-rrf_score` as `.distance` so callers sorting ascending get correct order.
    let all_files = catalog.list_files(table, None).await?;
    let _ = all_files; // centroid not needed for fusion — just need file_path+row_id

    // Collect unique candidates: prefer the row's appearance in the first column's results.
    let mut seen: HashMap<(String, u64), f32> = HashMap::new();
    for (_, results) in &per_col_results {
        for r in results {
            let key = (r.file_path.clone(), r.row_id.as_u64());
            let rrf_score = *scores.get(&key).unwrap_or(&0.0);
            seen.entry(key).or_insert(rrf_score);
        }
    }

    let mut fused: Vec<SearchResult> = seen
        .into_iter()
        .map(|((file_path, row_id_u64), rrf_score)| SearchResult {
            row_id: RowId::new(row_id_u64),
            distance: -rrf_score,
            file_path,
        })
        .collect();

    fused.sort_by(|a, b| {
        a.distance
            .partial_cmp(&b.distance)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    fused.truncate(config.top_k);

    let _ = fusion; // only RRF implemented; enum is extensible

    Ok(fused)
}

/// Apply `score_fn` to a single result row, or return `distance` unchanged.
///
/// Slices the batch to a 1-row RecordBatch at `row_id` and calls the fn.
/// If `score_fn` is `None` or the row index is out of bounds, returns `distance`.
#[inline]
fn apply_score_fn(
    score_fn: &Option<ScoreFn>,
    distance: f32,
    row_id: RowId,
    batch: &RecordBatch,
) -> f32 {
    match score_fn {
        None => distance,
        Some(f) => {
            let idx = row_id.as_u64() as usize;
            if idx < batch.num_rows() {
                f.call(distance, &batch.slice(idx, 1))
            } else {
                distance
            }
        }
    }
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

/// Pure BM25 full-text search across all Parquet files in the table.
///
/// Scans every surviving file (O(N) complexity), scores each row with BM25 against
/// `query_text`, and returns the global top-k by score. IDF stats are loaded from
/// `metadata/ailake_bm25_stats.bin` (written by `TableWriter` when `bm25_text_column`
/// is configured). If the stats file is absent, IDF defaults to an empty corpus
/// (all terms treated as maximally rare — directionally correct but less precise).
///
/// For pure-lexical search at scale (millions of rows, hundreds of files), consider
/// using SQL `LIKE` / `ILIKE` via DuckDB/Trino over the Iceberg-compatible table.
/// This function is best suited for small-medium tables or as a lexical complement
/// to `search()` for tables where the document count per file is manageable.
pub async fn search_text(
    table: &TableIdent,
    query_text: &str,
    text_columns: &[&str],
    top_k: usize,
    catalog: Arc<dyn CatalogProvider>,
    store: Arc<dyn Store>,
    partition_filter: Option<&str>,
) -> AilakeResult<Vec<SearchResult>> {
    use arrow_array::cast::AsArray;

    if text_columns.is_empty() {
        return Err(AilakeError::InvalidArgument(
            "search_text requires at least one text column".into(),
        ));
    }

    let all_files = catalog.list_files(table, None).await?;
    let table_meta = catalog.load_table(table).await?;

    // Partition pruning
    let files: Vec<_> = if let Some(pv) = partition_filter {
        all_files
            .into_iter()
            .filter(|f| f.partition_value.as_deref() == Some(pv))
            .collect()
    } else {
        all_files
    };

    // Load BM25 stats
    let stats_path = table_meta
        .properties
        .get(crate::bm25::BM25_STATS_PATH_PROP)
        .map(String::as_str)
        .unwrap_or(crate::bm25::BM25_STATS_FILE);
    let stats = match store.get(stats_path).await {
        Ok(bytes) => crate::bm25::IdfStats::from_bytes(&bytes)
            .unwrap_or_default(),
        Err(_) => {
            debug!(
                "ailake: BM25 stats not found at '{}' — using empty corpus IDF",
                stats_path
            );
            crate::bm25::IdfStats::default()
        }
    };
    let scorer = crate::bm25::BM25Scorer::new(&stats);

    let mut results: Vec<SearchResult> = Vec::new();

    for file_entry in &files {
        let file_bytes = store.get(&file_entry.path).await?;
        // Use dim=0 — we only read the Parquet columns, not the HNSW.
        let reader = AilakeFileReader::new(file_bytes, "", 0);
        let (batch, _) = reader.read_parquet()?;

        for row_idx in 0..batch.num_rows() {
            let doc_text: String = text_columns
                .iter()
                .filter_map(|&col| {
                    batch.column_by_name(col).and_then(|arr| {
                        arr.as_string_opt::<i32>().and_then(|sa| {
                            if sa.is_valid(row_idx) {
                                Some(sa.value(row_idx).to_string())
                            } else {
                                None
                            }
                        })
                    })
                })
                .collect::<Vec<_>>()
                .join(" ");

            if doc_text.is_empty() {
                continue;
            }

            let bm25 = scorer.score(query_text, &doc_text);
            if bm25 > 0.0 {
                // Negate so that sort-ascending = best-first (lower distance = higher BM25).
                results.push(SearchResult {
                    row_id: RowId::new(row_idx as u64),
                    distance: -bm25,
                    file_path: file_entry.path.clone(),
                });
            }
        }
    }

    results.sort_by(|a, b| a.distance.total_cmp(&b.distance));
    results.truncate(top_k);
    Ok(results)
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

    let combined =
        concat_batches(&base_schema, row_batches).map_err(|e| AilakeError::Arrow(e.to_string()))?;

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
    use crate::writer::MultiVectorBatch;
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
            score_fn: None,
            partition_filter: None,
            hybrid: None,
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
            score_fn: None,
            partition_filter: None,
            hybrid: None,
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
            score_fn: None,
            partition_filter: None,
            hybrid: None,
        };
        let cfg_rerank = SearchConfig {
            top_k: 2,
            ef_search: 50,
            pruning_threshold: f32::INFINITY,
            rerank_factor: Some(2),
            score_fn: None,
            partition_filter: None,
            hybrid: None,
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

    #[tokio::test]
    async fn multimodal_rrf_returns_top_k() {
        let dir = TempDir::new().unwrap();
        let dim = 4usize;
        write_demo_table(&dir, dim, 4).await;

        let store: Arc<dyn Store> = Arc::new(LocalStore::new(dir.path()));
        let catalog: Arc<dyn CatalogProvider> =
            Arc::new(HadoopCatalog::new(store.clone(), "warehouse"));
        let table = TableIdent::new("default", "table");

        // Two modal queries using the same column (single-column table).
        // Different queries to exercise RRF merging.
        let q1 = vec![1.0f32, 0.0, 0.0, 0.0];
        let q2 = vec![0.0f32, 1.0, 0.0, 0.0];

        let queries = vec![
            ModalQuery {
                column: "embedding",
                query: &q1,
                weight: 0.7,
                dim: dim as u32,
            },
            ModalQuery {
                column: "embedding",
                query: &q2,
                weight: 0.3,
                dim: dim as u32,
            },
        ];

        let config = SearchConfig {
            top_k: 2,
            ef_search: 50,
            pruning_threshold: f32::INFINITY,
            rerank_factor: None,
            score_fn: None,
            partition_filter: None,
            hybrid: None,
        };

        let results =
            search_multimodal(&table, &queries, config, catalog, store, FusionMethod::Rrf)
                .await
                .unwrap();

        assert_eq!(results.len(), 2);
        // RRF score stored as -distance; all should be negative
        assert!(results[0].distance <= 0.0);
        // Top result should be one of rows 0 or 1 (nearest to q1 or q2)
        assert!(results[0].row_id.as_u64() < 4);
    }

    /// True cross-modal test: two columns with DIFFERENT dims (4 + 2).
    /// Verifies that search_multimodal correctly routes to each column's HNSW
    /// and that the dim validation in search() handles secondary columns.
    #[tokio::test]
    async fn multimodal_rrf_cross_modal_different_dims() {
        let dir = TempDir::new().unwrap();
        let store: Arc<dyn Store> = Arc::new(LocalStore::new(dir.path()));
        let catalog: Arc<dyn CatalogProvider> =
            Arc::new(HadoopCatalog::new(store.clone(), "warehouse"));
        let table = TableIdent::new("default", "table");

        // Write a 2-column table: "embedding" dim=4, "img_embedding" dim=2
        let schema = Arc::new(Schema::new(vec![Field::new("id", DataType::Int32, false)]));
        let rows = 4usize;
        let ids: Vec<i32> = (0..rows as i32).collect();
        let batch = RecordBatch::try_new(schema, vec![Arc::new(Int32Array::from(ids))]).unwrap();

        let text_embs: Vec<Vec<f32>> = (0..rows)
            .map(|i| {
                let mut v = vec![0.0f32; 4];
                v[i % 4] = 1.0;
                v
            })
            .collect();
        let img_embs: Vec<Vec<f32>> = (0..rows)
            .map(|i| {
                let mut v = vec![0.0f32; 2];
                v[i % 2] = 1.0;
                v
            })
            .collect();

        let text_policy = make_policy(4);
        let img_policy = VectorStoragePolicy {
            column_name: "img_embedding".to_string(),
            dim: 2,
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
        };

        let mut writer = crate::TableWriter::create_or_open(
            catalog.clone(),
            store.clone(),
            text_policy,
            table.clone(),
        )
        .await
        .unwrap();

        let batches = [
            MultiVectorBatch {
                policy: make_policy(4),
                embeddings: &text_embs,
            },
            MultiVectorBatch {
                policy: img_policy,
                embeddings: &img_embs,
            },
        ];
        writer.write_batch_multi(&batch, &batches).await.unwrap();
        writer.commit().await.unwrap();

        // Cross-modal search: text query (dim=4) + image query (dim=2).
        let q_text = vec![1.0f32, 0.0, 0.0, 0.0];
        let q_img = vec![1.0f32, 0.0];

        let queries = vec![
            ModalQuery {
                column: "embedding",
                query: &q_text,
                weight: 0.6,
                dim: 4,
            },
            ModalQuery {
                column: "img_embedding",
                query: &q_img,
                weight: 0.4,
                dim: 2,
            },
        ];
        let config = SearchConfig {
            top_k: 2,
            ef_search: 50,
            pruning_threshold: f32::INFINITY,
            rerank_factor: None,
            score_fn: None,
            partition_filter: None,
            hybrid: None,
        };

        let results =
            search_multimodal(&table, &queries, config, catalog, store, FusionMethod::Rrf)
                .await
                .unwrap();

        assert!(!results.is_empty(), "should return results");
        assert!(results[0].distance <= 0.0, "distance is -rrf_score");
        // Row 0 is nearest to both q_text=[1,0,0,0] and q_img=[1,0]
        assert_eq!(results[0].row_id.as_u64(), 0, "row 0 should rank first");
    }
}
