// SPDX-License-Identifier: MIT OR Apache-2.0
use std::sync::Arc;

use futures::future::try_join_all;
use rayon::prelude::*;
use tracing::{debug, error, warn};

use ailake_catalog::{
    CatalogProvider, DataFileEntry, IndexStatus, SchemaField, TableIdent, TableMetadata,
};
use ailake_core::{AilakeError, AilakeResult, EmbeddingModelInfo, RowId, VectorMetric};
use ailake_file::AilakeFileReader;
use ailake_index::AnyIndex;
use ailake_store::Store;
use ailake_vec::exact_distance;
use arrow_array::{Array, RecordBatch};
use bytes::Bytes;

use crate::equality_delete::EqualityDeleteFilter;
use crate::pruner::{BloomPruner, VectorPruner};
use crate::schema_filler::SchemaFiller;

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
#[allow(clippy::type_complexity)]
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
    /// Column-level predicate pushed down to the Parquet read of each surviving
    /// file, via `AilakeFileReader::read_parquet_filtered` (row-group statistics
    /// skip + exact Arrow `RowFilter` — see `ailake-parquet::ParquetVectorReader`).
    /// Applies only where a Parquet read already happens (flat-scan fallback,
    /// reranking, hybrid, `score_fn`, or equality-delete row checks) — it does
    /// not gate which files survive centroid/partition pruning, and it isn't
    /// applied against columns only materialized later by `SchemaFiller`
    /// (schema-evolution defaults). `None` disables pushdown (default).
    pub column_filter: Option<ailake_core::ColumnFilter>,
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
            column_filter: None,
        }
    }
}

impl SearchConfig {
    pub fn with_column_filter(mut self, filter: ailake_core::ColumnFilter) -> Self {
        self.column_filter = Some(filter);
        self
    }

    pub fn with_pruning(mut self, threshold: f32) -> Self {
        self.pruning_threshold = threshold;
        self
    }

    pub fn with_reranking(mut self, factor: usize) -> Self {
        self.rerank_factor = Some(factor);
        self
    }

    pub fn with_score_fn(
        mut self,
        f: impl Fn(f32, &RecordBatch) -> f32 + Send + Sync + 'static,
    ) -> Self {
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

    // Phase F — Bloom pruning: for hybrid queries, load per-file Bloom filters from
    // the Puffin stats file and skip files where no query term can be present.
    let surviving_files = if let Some(ref h) = config.hybrid {
        let bloom_map = load_bloom_map(&table_meta, store.as_ref()).await;
        if !bloom_map.is_empty() {
            BloomPruner::prune(surviving_files, &h.query_text, &bloom_map)
        } else {
            surviving_files
        }
    } else {
        surviving_files
    };

    // Phase H: load equality delete filter for this snapshot.
    // Reads delete manifests from the catalog and downloads each equality delete Avro file.
    // Empty filter is a no-op. On error: warn and continue with empty filter (data visible).
    let eq_del_filter = match catalog.list_equality_deletes(table, None).await {
        Ok(edfs) if !edfs.is_empty() => {
            match EqualityDeleteFilter::from_files(&store, &edfs).await {
                Ok(f) => f,
                Err(e) => {
                    warn!("ailake: equality delete filter build failed: {e} — rows may appear");
                    EqualityDeleteFilter::empty()
                }
            }
        }
        _ => EqualityDeleteFilter::empty(),
    };

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

    // Observability for the flat-scan fallback below: `deferred` counts files still
    // being indexed by our own deferred-write path (expected, transient). `unexpected`
    // counts files with no AI-Lake index that are NOT in that state — most likely
    // rewritten by a generic Iceberg engine (Spark/Trino OPTIMIZE, DuckDB) with no
    // knowledge of AI-Lake. Those files still return correct results (flat scan is
    // exact), just O(N) instead of O(log N), and silently forever unless recompacted.
    let mut flat_scan_deferred = 0usize;
    let mut flat_scan_unexpected = 0usize;

    // Fetch + search each surviving file concurrently instead of one at a time —
    // the dominant cost per file is a network round-trip (`store.get`), so this
    // overlaps their latencies instead of serializing them. `try_join_all` runs
    // all of them concurrently on the current task (no OS-thread parallelism,
    // no `tokio::spawn`); at the post-pruning scale this operates on (dozens of
    // files, see `VectorPruner` — geometric pruning is designed to cut a
    // 10k-file table down to ~50-100 survivors before this point), that's the
    // right trade-off: no bound needed, and no per-task spawn overhead.
    let outcomes: Vec<FileSearchOutcome> = try_join_all(surviving_files.iter().map(|file_entry| {
        search_one_file(
            file_entry,
            query,
            candidate_k,
            metric,
            &table_meta,
            &config,
            vector_column,
            dim,
            &store,
            &eq_del_filter,
            use_hybrid,
        )
    }))
    .await?;

    for outcome in outcomes {
        match outcome.flat_scan {
            Some(FlatScanKind::Deferred) => flat_scan_deferred += 1,
            Some(FlatScanKind::Unexpected) => flat_scan_unexpected += 1,
            None => {}
        }
        all_results.extend(outcome.results);
        raw_candidates.extend(outcome.candidates);
    }

    if flat_scan_unexpected > 0 {
        warn!(
            "ailake: search degraded — {}/{} files scanned without an AI-Lake index \
             (unexpected — likely external rewrites; {} more in expected deferred-indexing \
             state). Run compaction to restore O(log N) search on affected files",
            flat_scan_unexpected,
            surviving_files.len(),
            flat_scan_deferred
        );
    } else if flat_scan_deferred > 0 {
        debug!(
            "ailake: search — {}/{} files scanned via flat fallback (deferred indexing)",
            flat_scan_deferred,
            surviving_files.len()
        );
    }

    // Hybrid BM25 fusion: applied after all HNSW candidates are collected.
    if let Some(ref h) = config.hybrid {
        let empty_stats = crate::bm25::IdfStats::default();
        let stats = bm25_stats.as_ref().unwrap_or(&empty_stats);
        let scorer = crate::bm25::BM25Scorer::new(stats);

        // Compute BM25 scores before sorting so they stay positionally aligned.
        let bm25_scores_pre: Vec<f32> = raw_candidates
            .iter()
            .map(|(_, _, _, text)| scorer.score(&h.query_text, text))
            .collect();

        // Zip BM25 scores into candidates so they sort together — avoids index mismatch
        // when raw_candidates is reordered by distance below.
        let mut candidates_with_bm25: Vec<((RowId, f32, String, String), f32)> =
            raw_candidates.into_iter().zip(bm25_scores_pre).collect();
        candidates_with_bm25.sort_by(|a, b| a.0 .1.total_cmp(&b.0 .1));
        let n = candidates_with_bm25.len();

        let vec_ranks: Vec<usize> = (0..n).collect();

        // Rank by BM25 score descending using post-sort aligned scores.
        let mut bm25_indexed: Vec<(usize, f32)> = candidates_with_bm25
            .iter()
            .map(|(_, b)| *b)
            .enumerate()
            .collect();
        bm25_indexed.sort_by(|a, b| b.1.total_cmp(&a.1));
        let mut bm25_rank_of = vec![0usize; n];
        for (rank, (idx, _)) in bm25_indexed.iter().enumerate() {
            bm25_rank_of[*idx] = rank;
        }

        use crate::bm25::{linear_score, rrf_score, HybridFusion};

        let fused: Vec<f32> = match h.fusion {
            HybridFusion::Rrf => vec_ranks
                .iter()
                .enumerate()
                .map(|(i, &vr)| rrf_score(vr, bm25_rank_of[i], h.bm25_weight))
                .collect(),
            HybridFusion::Linear => {
                let min_d = candidates_with_bm25
                    .iter()
                    .map(|(r, _)| r.1)
                    .fold(f32::INFINITY, f32::min);
                let max_d = candidates_with_bm25
                    .iter()
                    .map(|(r, _)| r.1)
                    .fold(f32::NEG_INFINITY, f32::max);
                let min_b = candidates_with_bm25
                    .iter()
                    .map(|(_, b)| *b)
                    .fold(f32::INFINITY, f32::min);
                let max_b = candidates_with_bm25
                    .iter()
                    .map(|(_, b)| *b)
                    .fold(f32::NEG_INFINITY, f32::max);
                candidates_with_bm25
                    .iter()
                    .map(|(r, b)| linear_score(r.1, min_d, max_d, *b, min_b, max_b, h.bm25_weight))
                    .collect()
            }
        };

        for (i, ((row_id, _, file_path, _), _)) in candidates_with_bm25.into_iter().enumerate() {
            all_results.push(SearchResult {
                row_id,
                distance: fused[i],
                file_path,
            });
        }

        // For RRF: lower (more negative) = better; for Linear: lower = better. Same convention.
        all_results.sort_by(|a, b| a.distance.total_cmp(&b.distance));
    } else {
        all_results.sort_by(|a, b| a.distance.total_cmp(&b.distance));
    }

    all_results.truncate(config.top_k);
    Ok(all_results)
}

/// Why this file fell back to a flat (exact, O(N)) scan instead of HNSW —
/// carried out of `search_one_file` so the caller can aggregate counts across
/// all concurrently-searched files for the post-loop `warn!`/`debug!` summary.
enum FlatScanKind {
    /// Background index build still in progress (deferred write) — expected, transient.
    Deferred,
    /// No AI-Lake index and NOT in the deferred-indexing state — most likely an
    /// external engine (Spark/Trino OPTIMIZE, DuckDB) rewrote the file, or an
    /// internal inconsistency. Results are still correct, just degraded.
    Unexpected,
}

/// Per-file result of `search_one_file`, folded into `search()`'s accumulators
/// after all files have been searched concurrently.
#[derive(Default)]
struct FileSearchOutcome {
    /// Populated when hybrid search is off.
    results: Vec<SearchResult>,
    /// Populated when hybrid search is on: (row_id, vector distance, file path, text).
    candidates: Vec<(RowId, f32, String, String)>,
    flat_scan: Option<FlatScanKind>,
}

/// Fetches, index-searches (or flat-scans), and filters a single file — the
/// per-file body of `search()`'s main loop, extracted so it can be run
/// concurrently across files via `try_join_all`. Pure with respect to the
/// caller's accumulators: everything it would have mutated in place (result
/// lists, flat-scan counters) comes back in the returned `FileSearchOutcome`
/// instead, so concurrent invocations never share mutable state.
#[allow(clippy::too_many_arguments)]
async fn search_one_file(
    file_entry: &DataFileEntry,
    query: &[f32],
    candidate_k: usize,
    metric: VectorMetric,
    table_meta: &TableMetadata,
    config: &SearchConfig,
    vector_column: &str,
    dim: u32,
    store: &Arc<dyn Store>,
    eq_del_filter: &EqualityDeleteFilter,
    use_hybrid: bool,
) -> AilakeResult<FileSearchOutcome> {
    let mut outcome = FileSearchOutcome::default();

    // V3 Deletion Vector: fetch bitmap once per file (range GET from Puffin .dvd).
    // Independent of the file's own bytes — fetched first so both the range-GET
    // fast path below and the full-file fallback path can use it.
    // None for V2 tables or V3 files with no deletes. On fetch error: warn + continue
    // without mask (surfacing deleted rows is safer than hard-failing the search).
    let dv_bitmap: Option<roaring::RoaringBitmap> = if let Some(ref dv) = file_entry.deletion_vector
    {
        match crate::dv::load_deletion_vector(store, dv).await {
            Ok(bm) => {
                debug!(
                    "ailake: DV loaded ({} deletions) for {}",
                    bm.len(),
                    file_entry.path
                );
                Some(bm)
            }
            Err(e) => {
                warn!(
                    "ailake: DV fetch failed for '{}': {e} — deleted rows may appear",
                    file_entry.path
                );
                None
            }
        }
    } else {
        None
    };

    // Range-GET fast path (Fase 16): when this query needs nothing from the
    // file besides the index itself, skip the whole-file GET and range-GET
    // just the HNSW/IVF-PQ blob (typically 10-20% of the file's total size —
    // see CLAUDE.md §6). Every condition here is knowable from the catalog
    // manifest / query config alone, with no file bytes fetched yet. Limited
    // to the primary vector column and `IndexStatus::Ready` — see
    // `index_loader`'s module doc for why secondary columns can't use it, and
    // `crate::index_loader::load_primary_index`'s contract: any failure there
    // is safe to treat as "use the full-file path", never a hard error.
    let primary_col = table_meta
        .properties
        .get("ailake.vector-column")
        .map(String::as_str)
        .unwrap_or("");
    let fast_path_eligible = config.column_filter.is_none()
        && config.rerank_factor.is_none()
        && config.score_fn.is_none()
        && !use_hybrid
        && eq_del_filter.is_empty()
        && file_entry.index_status == IndexStatus::Ready
        && !file_entry.is_foreign()
        && vector_column == primary_col;

    if fast_path_eligible {
        match crate::index_loader::load_primary_index(store, &file_entry.path).await {
            Ok(index) => {
                let local_results = index.search(query, candidate_k, config.ef_search);
                for (row_id, approx_dist) in local_results {
                    if dv_bitmap
                        .as_ref()
                        .is_some_and(|bm| bm.contains(row_id.as_u64() as u32))
                    {
                        continue;
                    }
                    outcome.results.push(SearchResult {
                        row_id,
                        distance: approx_dist,
                        file_path: file_entry.path.clone(),
                    });
                }
                return Ok(outcome);
            }
            Err(e) => {
                debug!(
                    "ailake: range-GET fast path failed for {} ({e}) — falling back to full-file GET",
                    file_entry.path
                );
            }
        }
    }

    let file_bytes: Bytes = store.get(&file_entry.path).await?;
    let reader = AilakeFileReader::new(file_bytes, vector_column, dim);

    // Predicate pushdown: resolve which original row positions in this file
    // satisfy `config.column_filter`, without disturbing row identity (see
    // `AilakeFileReader::matching_row_ids`). When the set comes back empty,
    // no row in the file can pass — skip HNSW search and any Parquet
    // decode for it entirely (the real perf win: files that don't contain
    // the filtered value are never scanned at all, not just post-filtered).
    let matching_row_ids: Option<std::collections::HashSet<u64>> =
        match config.column_filter.as_ref() {
            Some(filter) => {
                let matching = reader.matching_row_ids(filter)?;
                if matching.is_empty() {
                    return Ok(outcome);
                }
                Some(matching)
            }
            None => None,
        };

    // Parquet read required for: flat scan fallback, exact reranking, score_fn, hybrid,
    // or when equality delete filter must check column values per-row.
    let need_parquet = file_entry.index_status == IndexStatus::Indexing
        || !reader.is_ailake_file()
        || config.rerank_factor.is_some()
        || config.score_fn.is_some()
        || use_hybrid
        || !eq_del_filter.is_empty();

    if file_entry.index_status == IndexStatus::Indexing || !reader.is_ailake_file() {
        match file_entry.index_status {
            IndexStatus::Indexing => {
                outcome.flat_scan = Some(FlatScanKind::Deferred);
                debug!(
                    "ailake: flat scan fallback for {} — index build in progress \
                     (deferred write, expected to resolve once background job completes)",
                    file_entry.path
                );
            }
            IndexStatus::Failed => {
                // An internal indexing failure, not a foreign write — the file still
                // has a real (foreign-write-style) centroid_b64 from make_data_file_entry
                // *_indexing, since only index_status/index_error get patched on failure
                // (see writer.rs::patch_index_failed). Attributing this to an external
                // engine would send on-call looking for a Spark job that never ran.
                outcome.flat_scan = Some(FlatScanKind::Unexpected);
                warn!(
                    "ailake: flat scan fallback for {} — background index build failed \
                     permanently ({}); serving via flat scan until the next compaction \
                     rebuilds the index",
                    file_entry.path,
                    file_entry
                        .index_error
                        .as_deref()
                        .unwrap_or("no error recorded")
                );
            }
            IndexStatus::Ready => {
                // Ready but no loadable index/footer: either a genuine foreign write
                // (no centroid_b64 — see CompactionPlanner::plan's detection) or a rare
                // internally-inconsistent state (Ready without ever getting an index).
                outcome.flat_scan = Some(FlatScanKind::Unexpected);
                if file_entry.is_foreign() {
                    warn!(
                        "ailake: flat scan fallback for {} — file has no AI-Lake index \
                         and no centroid; likely rewritten by a generic Iceberg engine \
                         (OPTIMIZE / rewrite_data_files) with no knowledge of AI-Lake. \
                         Results are still correct (exact O(N) scan), but degraded until \
                         this file is recompacted by the AI-Lake SDK",
                        file_entry.path
                    );
                } else {
                    warn!(
                        "ailake: flat scan fallback for {} — marked Ready but has no \
                         loadable AI-Lake index despite a recorded centroid; internal \
                         inconsistency, not an external rewrite. Run compaction to rebuild",
                        file_entry.path
                    );
                }
            }
        }
        let (raw_batch, raw_vectors) = reader.read_parquet()?;
        // Phase G: inject columns added via schema evolution with initial_default values.
        let batch = SchemaFiller::fill(raw_batch, &table_meta.schema_fields)?;
        for (row_id, distance) in flat_search(&raw_vectors, query, candidate_k, metric) {
            // Skip rows marked as deleted by a V3 Deletion Vector.
            if dv_bitmap
                .as_ref()
                .is_some_and(|bm| bm.contains(row_id.as_u64() as u32))
            {
                continue;
            }
            // Phase H: skip rows matched by an equality delete predicate.
            if eq_del_filter.should_delete_row(&batch, row_id.as_u64() as usize) {
                continue;
            }
            // Predicate pushdown: row survived the file-level check above,
            // but only rows in the matching set itself pass the filter.
            if matching_row_ids
                .as_ref()
                .is_some_and(|ids| !ids.contains(&row_id.as_u64()))
            {
                continue;
            }
            if use_hybrid {
                let text = extract_text_for_row(
                    &batch,
                    row_id.as_u64() as usize,
                    config.hybrid.as_ref().unwrap(),
                );
                outcome
                    .candidates
                    .push((row_id, distance, file_entry.path.clone(), text));
            } else {
                let final_score = apply_score_fn(&config.score_fn, distance, row_id, &batch);
                outcome.results.push(SearchResult {
                    row_id,
                    distance: final_score,
                    file_path: file_entry.path.clone(),
                });
            }
        }
        return Ok(outcome);
    }

    let index = reader.load_any_index_for_column(vector_column)?;
    let local_results = index.search(query, candidate_k, config.ef_search);

    let parquet_data = if need_parquet {
        let (raw_batch, raw_vecs) = reader.read_parquet()?;
        // Phase G: fill missing columns for old files before score_fn / hybrid BM25.
        let filled = SchemaFiller::fill(raw_batch, &table_meta.schema_fields)?;
        Some((filled, raw_vecs))
    } else {
        None
    };

    for (row_id, approx_dist) in local_results {
        // Skip rows marked as deleted by a V3 Deletion Vector.
        if dv_bitmap
            .as_ref()
            .is_some_and(|bm| bm.contains(row_id.as_u64() as u32))
        {
            continue;
        }
        let idx = row_id.as_u64() as usize;
        // Phase H: skip rows matched by an equality delete predicate.
        // parquet_data is always loaded when eq_del_filter is non-empty (see need_parquet).
        if let Some((ref batch, _)) = parquet_data {
            if eq_del_filter.should_delete_row(batch, idx) {
                continue;
            }
        }
        // Predicate pushdown: row survived the file-level check above,
        // but only rows in the matching set itself pass the filter.
        if matching_row_ids
            .as_ref()
            .is_some_and(|ids| !ids.contains(&row_id.as_u64()))
        {
            continue;
        }

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
            outcome
                .candidates
                .push((row_id, distance, file_entry.path.clone(), text));
        } else {
            let final_score = if let Some((ref batch, _)) = parquet_data {
                apply_score_fn(&config.score_fn, distance, row_id, batch)
            } else {
                distance
            };
            outcome.results.push(SearchResult {
                row_id,
                distance: final_score,
                file_path: file_entry.path.clone(),
            });
        }
    }

    Ok(outcome)
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
            column_filter: config.column_filter.clone(),
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
        "normalized_cosine" | "normalizedcosine" => VectorMetric::NormalizedCosine,
        _ => VectorMetric::Cosine,
    }
}

/// Pre-loaded search session: all HNSW indexes loaded into memory once.
///
/// Useful for benchmarks and servers that issue many queries against the same
/// snapshot. Avoids re-loading and re-deserializing indexes on every call.
///
/// **Deleted rows are NOT filtered here**: unlike [`search`]/[`search_text`],
/// this session does not load deletion vectors or equality delete files —
/// rows removed via `delete_rows`/`delete_where` still appear in results.
/// Use [`search`] when delete visibility matters; this type trades that for
/// raw throughput on static snapshots (its benchmark use case).
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
        Ok(bytes) => crate::bm25::IdfStats::from_bytes(&bytes).unwrap_or_default(),
        Err(_) => {
            debug!(
                "ailake: BM25 stats not found at '{}' — using empty corpus IDF",
                stats_path
            );
            crate::bm25::IdfStats::default()
        }
    };
    let scorer = crate::bm25::BM25Scorer::new(&stats);

    // Phase H: equality delete filter for search_text results.
    let eq_del_filter = match catalog.list_equality_deletes(table, None).await {
        Ok(edfs) if !edfs.is_empty() => {
            match EqualityDeleteFilter::from_files(&store, &edfs).await {
                Ok(f) => f,
                Err(e) => {
                    warn!("ailake: equality delete filter build failed in search_text: {e}");
                    EqualityDeleteFilter::empty()
                }
            }
        }
        _ => EqualityDeleteFilter::empty(),
    };

    let mut results: Vec<SearchResult> = Vec::new();

    for file_entry in &files {
        let file_bytes = store.get(&file_entry.path).await?;
        // Use dim=0 — we only read the Parquet columns, not the HNSW.
        let reader = AilakeFileReader::new(file_bytes.clone(), "", 0);

        // Fast path: per-file Tantivy index (O(log N) via inverted index).
        // Falls back to BM25 O(N) brute-force for files without an FTS section.
        if let Ok(Some(fts_blob)) = reader.load_fts_blob() {
            match ailake_fts::FtsSearcher::from_blob(&fts_blob) {
                Ok(fts) => {
                    let hits = fts.search(query_text, top_k * 3).unwrap_or_default();
                    if !hits.is_empty() {
                        // Load batch only for equality delete checking (not for scoring).
                        let reader2 = AilakeFileReader::new(file_bytes, "", 0);
                        let (raw_batch, _) = reader2.read_parquet()?;
                        let batch = SchemaFiller::fill(raw_batch, &table_meta.schema_fields)?;
                        for hit in hits {
                            let row_idx = hit.row_id as usize;
                            if row_idx >= batch.num_rows() {
                                continue;
                            }
                            if eq_del_filter.should_delete_row(&batch, row_idx) {
                                continue;
                            }
                            results.push(SearchResult {
                                row_id: RowId::new(hit.row_id),
                                distance: -hit.score,
                                file_path: file_entry.path.clone(),
                            });
                        }
                    }
                    continue; // skip O(N) BM25 fallback
                }
                Err(e) => {
                    warn!("ailake: FTS blob corrupt for '{}': {e}", file_entry.path);
                    // fall through to BM25 brute-force
                }
            }
        }

        // Fallback: O(N) BM25 brute-force — unchanged from pre-Phase-T behaviour.
        let reader_fb = AilakeFileReader::new(file_bytes, "", 0);
        let (raw_batch, _) = reader_fb.read_parquet()?;
        // Phase G: fill missing columns for old files before BM25 text extraction.
        let batch = SchemaFiller::fill(raw_batch, &table_meta.schema_fields)?;

        for row_idx in 0..batch.num_rows() {
            // Phase H: skip rows matched by equality delete predicate.
            if eq_del_filter.should_delete_row(&batch, row_idx) {
                continue;
            }
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
    schema_fields: &[SchemaField],
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

    // `vector_column` is deliberately excluded from the batch AilakeFileReader::read_parquet()
    // returns — it's decoded separately into `vectors: Vec<Vec<f32>>` and re-appended below as
    // a FixedSizeList<Float32> field. SchemaFiller has no way to know that; left unfiltered it
    // treats the vector column as "missing" (it genuinely isn't in the tabular batch) and
    // injects a synthetic column for it — wrong type (falls through iceberg_type_to_arrow's
    // Utf8 fallback for the vector column's Iceberg type string) and a duplicate field name
    // once the real decoded vector column is appended, breaking pandas' arrow->pandas
    // conversion (`Unsupported cast from fixed_size_list<...> to large_utf8`). Found writing
    // the regression test for the *other* schema-projection bug this function has — filtering
    // it out here is required for schema-filling to be correct at all in fetch_rows.
    let schema_fields_for_fill: Vec<SchemaField> = schema_fields
        .iter()
        .filter(|sf| sf.name != vector_column)
        .cloned()
        .collect();

    // (original_index, distance, single-row RecordBatch, decoded F32 vector)
    let mut collected: Vec<(usize, f32, RecordBatch, Vec<f32>)> = Vec::with_capacity(results.len());

    for (file_path, rows) in &by_file {
        let bytes = store.get(file_path).await?;
        let reader = AilakeFileReader::new(bytes, vector_column, dim);
        let (raw_batch, vectors) = reader.read_parquet()?;
        // Project against the table's *current* Iceberg schema — old files written
        // before a metadata-only evolve_schema/add_column don't physically have the
        // new column. Without this, a file that happens to land first in `collected`
        // silently drives `base_schema` below and the new column never appears in the
        // response at all (not even as null) — confirmed live via Spark's
        // AilakeNative.scan() and ailake.search(fetch_data=True), both of which call
        // this function. Same fix SchemaFiller::fill already applies on the
        // pointer-search path (search()); this was the one full-row-fetch path it
        // never reached.
        let batch = SchemaFiller::fill(raw_batch, &schema_fields_for_fill)?;

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

/// Load per-file BM25 Bloom filters from the Puffin stats file for the current snapshot.
///
/// Returns a map of `file_path → BloomFilter`. Empty map = no stats file available
/// (V2 table, first write, or fetch failure). The scanner applies Bloom pruning only
/// when the map is non-empty.
async fn load_bloom_map(
    table_meta: &ailake_catalog::TableMetadata,
    store: &dyn Store,
) -> std::collections::HashMap<String, crate::bloom::BloomFilter> {
    let stats_path = match &table_meta.current_statistics_path {
        Some(p) => p.clone(),
        None => return std::collections::HashMap::new(),
    };
    let bytes = match store.get(&stats_path).await {
        Ok(b) => b,
        Err(e) => {
            debug!("ailake: Phase F — could not load Puffin stats ({stats_path}): {e}");
            return std::collections::HashMap::new();
        }
    };
    let reader = ailake_catalog::AilakePuffinReader::new(&bytes);
    let bloom_entries = match reader.read_bm25_blooms() {
        Ok(e) => e,
        Err(e) => {
            warn!("ailake: Phase F — Puffin bloom parse error: {e}");
            return std::collections::HashMap::new();
        }
    };
    bloom_entries
        .into_iter()
        .filter_map(|entry| {
            let bf = crate::bloom::BloomFilter::from_bytes(&entry.bloom_bytes)?;
            Some((entry.path, bf))
        })
        .collect()
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
            partition_column_type: None,
            partition_fields: vec![],
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
            crate::TableWriter::create_or_open(catalog, store, make_policy(dim as u32), table, 2)
                .await
                .unwrap();
        writer.write_batch(&batch, &embeddings).await.unwrap();
        writer.commit().await.unwrap();
    }

    /// Same shape as `write_demo_table`, plus a `category` column ("even"/"odd"
    /// by row id) — lets predicate-pushdown tests assert against a real column
    /// while still using the same unit-basis-vector embeddings (row `i`'s
    /// nearest neighbor for query `i` is unambiguous).
    async fn write_demo_table_with_category(dir: &TempDir, dim: usize, rows: usize) {
        use arrow_array::StringArray;
        let store: Arc<dyn Store> = Arc::new(LocalStore::new(dir.path()));
        let catalog = Arc::new(HadoopCatalog::new(store.clone(), "warehouse"));
        let table = TableIdent::new("default", "table");

        let schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int32, false),
            Field::new("category", DataType::Utf8, false),
        ]));
        let ids: Vec<i32> = (0..rows as i32).collect();
        let categories: Vec<&str> = (0..rows)
            .map(|i| if i % 2 == 0 { "even" } else { "odd" })
            .collect();
        let batch = RecordBatch::try_new(
            schema,
            vec![
                Arc::new(Int32Array::from(ids)),
                Arc::new(StringArray::from(categories)),
            ],
        )
        .unwrap();

        let embeddings: Vec<Vec<f32>> = (0..rows)
            .map(|i| {
                let mut v = vec![0.0f32; dim];
                v[i % dim] = 1.0;
                v
            })
            .collect();

        let mut writer =
            crate::TableWriter::create_or_open(catalog, store, make_policy(dim as u32), table, 2)
                .await
                .unwrap();
        writer.write_batch(&batch, &embeddings).await.unwrap();
        writer.commit().await.unwrap();
    }

    #[tokio::test]
    async fn column_filter_preserves_row_identity_on_indexed_path() {
        // The critical correctness property: row_ids returned under a
        // column_filter must still be the TRUE file-relative row ids (usable
        // to fetch the right row), not positions into some internally
        // compacted/filtered batch. Every "even" row's embedding is a unit
        // basis vector at its own dimension, so searching with that exact
        // query and asserting the returned row_id matches would fail loudly
        // if row identity got shifted by the pushdown.
        let dir = TempDir::new().unwrap();
        let dim = 8usize;
        write_demo_table_with_category(&dir, dim, 8).await;

        let store: Arc<dyn Store> = Arc::new(LocalStore::new(dir.path()));
        let catalog: Arc<dyn CatalogProvider> =
            Arc::new(HadoopCatalog::new(store.clone(), "warehouse"));
        let table = TableIdent::new("default", "table");

        // Query matches row 4 exactly ("even" — 4 % 2 == 0).
        let mut query = vec![0.0f32; dim];
        query[4] = 1.0;

        let config = SearchConfig {
            top_k: 1,
            ef_search: 50,
            pruning_threshold: f32::INFINITY,
            rerank_factor: None,
            score_fn: None,
            partition_filter: None,
            hybrid: None,
            column_filter: Some(ailake_core::ColumnFilter::eq(
                "category",
                ailake_core::FilterValue::Str("even".to_string()),
            )),
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
        assert_eq!(
            results[0].row_id.as_u64(),
            4,
            "row_id must be the true file-relative position, not a filtered-batch index"
        );
    }

    #[tokio::test]
    async fn column_filter_excludes_non_matching_rows() {
        let dir = TempDir::new().unwrap();
        let dim = 8usize;
        write_demo_table_with_category(&dir, dim, 8).await;

        let store: Arc<dyn Store> = Arc::new(LocalStore::new(dir.path()));
        let catalog: Arc<dyn CatalogProvider> =
            Arc::new(HadoopCatalog::new(store.clone(), "warehouse"));
        let table = TableIdent::new("default", "table");

        // Query matches row 3 exactly ("odd"); filtering for "even" must exclude it.
        let mut query = vec![0.0f32; dim];
        query[3] = 1.0;

        let config = SearchConfig {
            top_k: 8,
            ef_search: 50,
            pruning_threshold: f32::INFINITY,
            rerank_factor: None,
            score_fn: None,
            partition_filter: None,
            hybrid: None,
            column_filter: Some(ailake_core::ColumnFilter::eq(
                "category",
                ailake_core::FilterValue::Str("even".to_string()),
            )),
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

        assert_eq!(results.len(), 4, "only the 4 even rows should survive");
        for r in &results {
            assert_eq!(
                r.row_id.as_u64() % 2,
                0,
                "row {} is not even",
                r.row_id.as_u64()
            );
        }
    }

    #[tokio::test]
    async fn column_filter_no_match_returns_empty() {
        let dir = TempDir::new().unwrap();
        let dim = 8usize;
        write_demo_table_with_category(&dir, dim, 8).await;

        let store: Arc<dyn Store> = Arc::new(LocalStore::new(dir.path()));
        let catalog: Arc<dyn CatalogProvider> =
            Arc::new(HadoopCatalog::new(store.clone(), "warehouse"));
        let table = TableIdent::new("default", "table");

        let query = vec![1.0f32, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let config = SearchConfig {
            top_k: 8,
            ef_search: 50,
            pruning_threshold: f32::INFINITY,
            rerank_factor: None,
            score_fn: None,
            partition_filter: None,
            hybrid: None,
            column_filter: Some(ailake_core::ColumnFilter::eq(
                "category",
                ailake_core::FilterValue::Str("nonexistent".to_string()),
            )),
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
        assert!(results.is_empty());
    }

    /// Wraps a `Store`, recording every `get`/`get_range` call's byte range
    /// per path — used to prove the Fase 16 range-GET fast path never
    /// touches the row-group tabular/vector data section of a file, not just
    /// to assert it doesn't crash. A whole-file `get` is recorded as
    /// `0..file_size` (its full extent), since that's what it touches even
    /// though the `Store` trait doesn't expose a byte offset for it.
    struct CountingStore {
        inner: Arc<dyn Store>,
        ranges_by_path: std::sync::Mutex<std::collections::HashMap<String, Vec<(u64, u64)>>>,
    }

    impl CountingStore {
        fn new(inner: Arc<dyn Store>) -> Self {
            Self {
                inner,
                ranges_by_path: std::sync::Mutex::new(std::collections::HashMap::new()),
            }
        }

        fn record(&self, path: &str, start: u64, end: u64) {
            self.ranges_by_path
                .lock()
                .unwrap()
                .entry(path.to_string())
                .or_default()
                .push((start, end));
        }

        /// Lowest byte offset fetched for `path` across every recorded call —
        /// the precise, ratio-independent proof that the tabular/vector data
        /// section (everything before the AILK section) was never touched.
        fn min_offset_for(&self, path: &str) -> Option<u64> {
            self.ranges_by_path
                .lock()
                .unwrap()
                .get(path)
                .and_then(|ranges| ranges.iter().map(|(s, _)| *s).min())
        }
    }

    #[async_trait::async_trait]
    impl Store for CountingStore {
        async fn get(&self, path: &str) -> AilakeResult<Bytes> {
            let b = self.inner.get(path).await?;
            self.record(path, 0, b.len() as u64);
            Ok(b)
        }
        async fn get_range(&self, path: &str, range: std::ops::Range<u64>) -> AilakeResult<Bytes> {
            let b = self.inner.get_range(path, range.clone()).await?;
            self.record(path, range.start, range.end);
            Ok(b)
        }
        async fn put(&self, path: &str, data: Bytes) -> AilakeResult<()> {
            self.inner.put(path, data).await
        }
        async fn list(&self, prefix: &str) -> AilakeResult<Vec<String>> {
            self.inner.list(prefix).await
        }
        async fn file_size(&self, path: &str) -> AilakeResult<u64> {
            self.inner.file_size(path).await
        }
        async fn exists(&self, path: &str) -> AilakeResult<bool> {
            self.inner.exists(path).await
        }
        async fn delete(&self, path: &str) -> AilakeResult<()> {
            self.inner.delete(path).await
        }
    }

    #[tokio::test]
    async fn range_get_fast_path_never_touches_tabular_data_section() {
        let dir = TempDir::new().unwrap();
        let dim = 64usize;
        write_demo_table(&dir, dim, 400).await;

        let local: Arc<dyn Store> = Arc::new(LocalStore::new(dir.path()));
        let data_files = local.list("data").await.unwrap();
        assert_eq!(
            data_files.len(),
            1,
            "expect a single part file for this test"
        );
        let data_path = data_files[0].clone();

        // Ground truth: the exact tabular/data-section boundary, from a plain
        // full-file read+parse — independent of the fast path under test.
        let full_bytes = local.get(&data_path).await.unwrap();
        let ailk_offset = ailake_file::AilakeFileReader::new(full_bytes, "embedding", dim as u32)
            .ailk_offset()
            .unwrap();

        let counting = Arc::new(CountingStore::new(local.clone()));
        let store: Arc<dyn Store> = counting.clone();
        let catalog: Arc<dyn CatalogProvider> =
            Arc::new(HadoopCatalog::new(store.clone(), "warehouse"));
        let table = TableIdent::new("default", "table");

        let mut query = vec![0.0f32; dim];
        query[3] = 1.0;
        let config = SearchConfig {
            top_k: 5,
            ef_search: 50,
            pruning_threshold: f32::INFINITY,
            rerank_factor: None,
            score_fn: None,
            partition_filter: None,
            hybrid: None,
            column_filter: None,
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
        assert!(!results.is_empty());

        // The precise, ratio-independent guarantee this feature exists for:
        // regardless of how large the HNSW blob happens to be relative to the
        // tabular data (small synthetic tables can skew that ratio either
        // way — see `range_get_fast_path_matches_full_file_path_results` for
        // the separate correctness-parity proof), the fast path must never
        // fetch a single byte positioned before the AILK section starts.
        let min_offset = counting
            .min_offset_for(&data_path)
            .expect("should have fetched something from the data file");
        assert!(
            min_offset >= ailk_offset,
            "fast path fetched bytes from the tabular/vector data section: \
             min_offset={min_offset} < ailk_offset={ailk_offset}"
        );
    }

    #[tokio::test]
    async fn range_get_fast_path_matches_full_file_path_results() {
        // Forces the full-file fallback via a column_filter that matches
        // every row (id >= 0) — same semantic query and identical distances,
        // just disqualified from the fast path (config.column_filter.is_some()).
        // Comparing the two proves the fast path isn't silently returning
        // different (or wrong-row-identity) results.
        let dir = TempDir::new().unwrap();
        let dim = 16usize;
        write_demo_table(&dir, dim, 20).await;

        let store: Arc<dyn Store> = Arc::new(LocalStore::new(dir.path()));
        let catalog: Arc<dyn CatalogProvider> =
            Arc::new(HadoopCatalog::new(store.clone(), "warehouse"));
        let table = TableIdent::new("default", "table");

        let mut query = vec![0.0f32; dim];
        query[3] = 1.0;

        let fast_config = SearchConfig {
            top_k: 5,
            ef_search: 50,
            pruning_threshold: f32::INFINITY,
            rerank_factor: None,
            score_fn: None,
            partition_filter: None,
            hybrid: None,
            column_filter: None,
        };
        let fast_results = search(
            &table,
            &query,
            fast_config,
            "embedding",
            dim as u32,
            catalog.clone(),
            store.clone(),
        )
        .await
        .unwrap();

        let slow_config = SearchConfig {
            top_k: 5,
            ef_search: 50,
            pruning_threshold: f32::INFINITY,
            rerank_factor: None,
            score_fn: None,
            partition_filter: None,
            hybrid: None,
            column_filter: Some(ailake_core::ColumnFilter::new(
                "id",
                ailake_core::FilterOp::Gte,
                ailake_core::FilterValue::I64(0),
            )),
        };
        let slow_results = search(
            &table,
            &query,
            slow_config,
            "embedding",
            dim as u32,
            catalog,
            store,
        )
        .await
        .unwrap();

        assert_eq!(fast_results.len(), slow_results.len());
        assert!(!fast_results.is_empty());
        for (f, s) in fast_results.iter().zip(slow_results.iter()) {
            assert_eq!(f.row_id, s.row_id, "fast/slow path row_id mismatch");
            assert!(
                (f.distance - s.distance).abs() < 1e-4,
                "fast/slow path distance mismatch: {} vs {}",
                f.distance,
                s.distance
            );
        }
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
            column_filter: None,
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
            column_filter: None,
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
            column_filter: None,
        };
        let cfg_rerank = SearchConfig {
            top_k: 2,
            ef_search: 50,
            pruning_threshold: f32::INFINITY,
            rerank_factor: Some(2),
            score_fn: None,
            partition_filter: None,
            hybrid: None,
            column_filter: None,
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
            column_filter: None,
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
            partition_column_type: None,
            partition_fields: vec![],
        };

        let mut writer = crate::TableWriter::create_or_open(
            catalog.clone(),
            store.clone(),
            text_policy,
            table.clone(),
            2,
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
            column_filter: None,
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

    /// Regression: `fetch_rows` used to build its output schema from whichever file's
    /// physical Parquet schema happened to be read first — a file written before a
    /// metadata-only `evolve_schema`/`add_column` never physically has the new column,
    /// so it was silently absent from the response instead of projected as null.
    /// Confirmed live via Spark's `AilakeNative.scan()` and
    /// `ailake.search(fetch_data=True)`, both backed by this function.
    #[tokio::test]
    async fn fetch_rows_projects_evolved_column_as_null() {
        use ailake_catalog::schema_evolution::{AddColumnRequest, SchemaEvolution};

        let dir = TempDir::new().unwrap();
        let dim = 8usize;
        write_demo_table(&dir, dim, 8).await;

        let store: Arc<dyn Store> = Arc::new(LocalStore::new(dir.path()));
        let catalog: Arc<dyn CatalogProvider> =
            Arc::new(HadoopCatalog::new(store.clone(), "warehouse"));
        let table = TableIdent::new("default", "table");

        // Metadata-only schema evolution — no data files rewritten, so every existing
        // file on disk still physically lacks the "note" column.
        catalog
            .evolve_schema(
                &table,
                SchemaEvolution::new().add_column(AddColumnRequest {
                    name: "note".to_string(),
                    iceberg_type: "string".to_string(),
                    required: false,
                    initial_default: None,
                    write_default: None,
                    doc: None,
                }),
            )
            .await
            .unwrap();

        let query = vec![1.0f32, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let config = SearchConfig {
            top_k: 3,
            ef_search: 50,
            pruning_threshold: f32::INFINITY,
            rerank_factor: None,
            score_fn: None,
            partition_filter: None,
            hybrid: None,
            column_filter: None,
        };
        let results = search(
            &table,
            &query,
            config,
            "embedding",
            dim as u32,
            Arc::clone(&catalog),
            Arc::clone(&store),
        )
        .await
        .unwrap();
        assert_eq!(results.len(), 3);

        let table_meta = catalog.load_table(&table).await.unwrap();
        let batch = fetch_rows(
            &results,
            store,
            "embedding",
            dim as u32,
            &table_meta.schema_fields,
        )
        .await
        .unwrap();

        let note_col = batch
            .column_by_name("note")
            .expect("evolved 'note' column must be present, not silently dropped");
        assert_eq!(note_col.len(), 3);
        assert_eq!(
            note_col.null_count(),
            3,
            "old files predate 'note' — every value must be null, not an error or a missing column"
        );
    }

    /// Regression: the schema-projection fix above (`fetch_rows_projects_evolved_column_as_null`)
    /// initially introduced its own bug — `SchemaFiller::fill` was called with the *unfiltered*
    /// current-schema field list, which includes the vector column itself. Since
    /// `AilakeFileReader::read_parquet()` deliberately returns the vector column out-of-band
    /// (as `vectors: Vec<Vec<f32>>`, not as part of the tabular `RecordBatch`), the filler saw
    /// it as "missing" and injected a synthetic column for it — wrong-typed (`iceberg_type_to_arrow`'s
    /// `Utf8` fallback) and a duplicate of the real decoded vector column appended a few lines
    /// later, breaking pandas' arrow→pandas conversion for *every* `fetch_data=True`/`scan()`
    /// call, not just evolved-schema ones. Caught immediately by testing against real
    /// pandas conversion (not just the Rust arrow API) — this test guards it at the Rust level too.
    #[tokio::test]
    async fn fetch_rows_does_not_duplicate_vector_column() {
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
            rerank_factor: None,
            score_fn: None,
            partition_filter: None,
            hybrid: None,
            column_filter: None,
        };
        let results = search(
            &table,
            &query,
            config,
            "embedding",
            dim as u32,
            Arc::clone(&catalog),
            Arc::clone(&store),
        )
        .await
        .unwrap();

        let table_meta = catalog.load_table(&table).await.unwrap();
        let batch = fetch_rows(
            &results,
            store,
            "embedding",
            dim as u32,
            &table_meta.schema_fields,
        )
        .await
        .unwrap();

        let batch_schema = batch.schema();
        let embedding_fields: Vec<_> = batch_schema
            .fields()
            .iter()
            .filter(|f| f.name() == "embedding")
            .collect();
        assert_eq!(
            embedding_fields.len(),
            1,
            "exactly one 'embedding' field expected, got: {:?}",
            batch_schema
        );
        assert!(
            matches!(embedding_fields[0].data_type(), arrow_schema::DataType::FixedSizeList(_, d) if *d == dim as i32),
            "embedding field must be the decoded FixedSizeList<Float32>, got {:?}",
            embedding_fields[0].data_type()
        );
    }

    // ── Edge cases: dimension mismatch ───────────────────────────────────

    #[tokio::test]
    async fn search_dimension_mismatch_rejected() {
        let dir = TempDir::new().unwrap();
        write_demo_table(&dir, 8, 5).await;

        let store: Arc<dyn Store> = Arc::new(LocalStore::new(dir.path()));
        let catalog: Arc<dyn CatalogProvider> =
            Arc::new(HadoopCatalog::new(store.clone(), "warehouse"));
        let table = TableIdent::new("default", "table");

        // Query with wrong dimension (4 instead of 8)
        let query = vec![1.0f32, 0.0, 0.0, 0.0];
        let config = SearchConfig::default();
        let result = search(
            &table,
            &query,
            config,
            "embedding",
            8,
            Arc::clone(&catalog),
            Arc::clone(&store),
        )
        .await;
        assert!(
            result.is_err(),
            "search with wrong dim should error, got Ok"
        );
    }
}
