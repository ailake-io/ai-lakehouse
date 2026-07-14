// SPDX-License-Identifier: MIT OR Apache-2.0
// HTTP server for AI-Lake — exposes search, write, compact, and info over JSON.
//
// Endpoints:
//   POST /search   {"query":[f32...], "top_k":10, "pruning_threshold":0.8}
//   POST /write    {"texts":["..."], "embeddings":[[f32...]], "batch_id":"..."}
//   POST /compact  {}
//   GET  /info
//
// SECURITY: This server has no authentication. It is designed for trusted-network
// deployments (localhost, VPC-internal, sidecar). Do NOT expose it on a public
// interface without an authenticating reverse proxy (e.g., nginx + mTLS, API gateway).

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;

use axum::{
    extract::{DefaultBodyLimit, State},
    http::StatusCode,
    response::{IntoResponse, Response},
    routing::{get, post},
    Router,
};
use serde::{Deserialize, Serialize};
use tracing::{info, warn};

use ailake_catalog::provider::{
    new_snapshot_id, CatalogProvider, IndexStatus, NewSnapshot, SnapshotOperation, TableIdent,
};
use ailake_catalog::DataFileEntry;
use ailake_core::VectorStoragePolicy;
use ailake_query::{
    CompactionConfig, CompactionExecutor, CompactionPlanner, SearchConfig, TableWriter,
};
use ailake_store::Store;

/// Minimum time between foreign-file probes (see `maybe_probe_auto_compact`).
/// Bounds the extra `list_files` catalog call to at most once per window,
/// server-wide, regardless of query rate.
const AUTO_COMPACT_CHECK_COOLDOWN_MS: u64 = 60_000;

// ---------------------------------------------------------------------------
// Shared state
// ---------------------------------------------------------------------------

pub(crate) struct AppState {
    catalog: Arc<dyn CatalogProvider>,
    store: Arc<dyn Store>,
    table: TableIdent,
    policy: VectorStoragePolicy,
    /// Guards against overlapping auto-compact passes (ADR-018 gap fix: a
    /// long-running server sees the same foreign/externally-rewritten file
    /// (Spark/Trino `OPTIMIZE`, DuckDB) degrade to O(N) flat scan on every
    /// query until someone runs `ailake compact` by hand — see
    /// `maybe_probe_auto_compact`).
    auto_compact_inflight: Arc<AtomicBool>,
    /// Unix-ms timestamp of the last foreign-file probe; rate-limits how
    /// often `handle_search` re-lists files to check for them.
    auto_compact_last_check_ms: Arc<AtomicU64>,
}

// ---------------------------------------------------------------------------
// Error helper
// ---------------------------------------------------------------------------

struct ApiError(String);

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        let body = serde_json::json!({"error": self.0}).to_string();
        (StatusCode::INTERNAL_SERVER_ERROR, body).into_response()
    }
}

impl<E: std::fmt::Display> From<E> for ApiError {
    fn from(e: E) -> Self {
        ApiError(e.to_string())
    }
}

type ApiResult<T> = Result<T, ApiError>;

// ---------------------------------------------------------------------------
// Request / response types
// ---------------------------------------------------------------------------

const MAX_TOP_K: usize = 10_000;
const MAX_BODY_BYTES: usize = 32 * 1024 * 1024; // 32 MB

#[derive(Deserialize)]
struct SearchRequest {
    query: Vec<f32>,
    #[serde(default = "default_top_k")]
    top_k: usize,
    #[serde(default = "default_pruning")]
    pruning_threshold: f32,
}

fn default_top_k() -> usize {
    10
}
fn default_pruning() -> f32 {
    0.8
}

#[derive(Serialize)]
struct SearchResponse {
    results: Vec<SearchResult>,
}

#[derive(Serialize)]
struct SearchResult {
    rank: usize,
    row_id: u64,
    distance: f32,
    file_path: String,
}

#[derive(Deserialize)]
struct WriteRequest {
    texts: Vec<String>,
    embeddings: Vec<Vec<f32>>,
    batch_id: Option<String>,
}

#[derive(Serialize)]
struct WriteResponse {
    snapshot_id: i64,
    rows: usize,
}

#[derive(Deserialize, Default)]
struct CompactRequest {
    #[serde(default = "default_target_size")]
    target_size: u64,
    #[serde(default = "default_min_files")]
    min_files: usize,
}

fn default_target_size() -> u64 {
    536_870_912
}
fn default_min_files() -> usize {
    4
}

#[derive(Serialize)]
struct CompactResponse {
    message: String,
    compacted_files: usize,
}

#[derive(Serialize)]
struct InfoResponse {
    table: String,
    location: String,
    vector_column: String,
    vector_dim: String,
    vector_metric: String,
    files: usize,
    indexed_files: usize,
    failed_files: usize,
    rows: u64,
    size_bytes: u64,
    snapshot_id: Option<i64>,
}

// ---------------------------------------------------------------------------
// Handlers
// ---------------------------------------------------------------------------

async fn handle_search(
    State(state): State<Arc<AppState>>,
    body: String,
) -> ApiResult<impl IntoResponse> {
    let req: SearchRequest =
        serde_json::from_str(&body).map_err(|e| ApiError(format!("invalid JSON: {e}")))?;

    if req.query.is_empty() {
        return Err(ApiError("query must not be empty".into()));
    }
    let top_k = req.top_k.clamp(1, MAX_TOP_K);
    let dim = req.query.len() as u32;
    let config = SearchConfig {
        top_k,
        ef_search: top_k.saturating_mul(5),
        pruning_threshold: req.pruning_threshold,
        rerank_factor: None,
        score_fn: None,
        partition_filter: None,
        hybrid: None,
    };

    let results = ailake_query::search(
        &state.table,
        &req.query,
        config,
        &state.policy.column_name,
        dim,
        Arc::clone(&state.catalog) as Arc<dyn CatalogProvider>,
        Arc::clone(&state.store),
    )
    .await
    .map_err(ApiError::from)?;

    // Self-healing: schedule a (rate-limited, non-blocking) probe for foreign files —
    // see `maybe_probe_auto_compact` — without adding latency to this response.
    maybe_probe_auto_compact(&state);

    let resp = SearchResponse {
        results: results
            .iter()
            .enumerate()
            .map(|(i, r)| SearchResult {
                rank: i + 1,
                row_id: r.row_id.0,
                distance: r.distance,
                file_path: r.file_path.clone(),
            })
            .collect(),
    };
    Ok((StatusCode::OK, serde_json::to_string(&resp).unwrap()))
}

/// Schedule a background foreign-file probe, at most once per
/// `AUTO_COMPACT_CHECK_COOLDOWN_MS`, without blocking the caller.
///
/// `CompactionPlanner::plan()` already prioritizes foreign files (files a
/// generic Iceberg engine rewrote with no knowledge of AI-Lake — see
/// `DataFileEntry::is_foreign`) over the normal size/count thresholds, but
/// only when `ailake compact` actually runs. On a long-running `serve`
/// instance nothing ever calls it unless an operator does so by hand, so a
/// foreign file's O(N) flat-scan degradation (see `flat_scan_unexpected` in
/// `ailake-query/src/scanner.rs`) persists indefinitely. This closes that
/// gap by piggybacking a cheap, rate-limited check on the search path.
fn maybe_probe_auto_compact(state: &Arc<AppState>) {
    let now_ms = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0);
    let last = state.auto_compact_last_check_ms.load(Ordering::Acquire);
    if now_ms.saturating_sub(last) < AUTO_COMPACT_CHECK_COOLDOWN_MS {
        return;
    }
    // CAS claims this check window so concurrent requests don't all re-probe at once;
    // losers just skip — the winner's probe covers them.
    if state
        .auto_compact_last_check_ms
        .compare_exchange(last, now_ms, Ordering::AcqRel, Ordering::Acquire)
        .is_err()
    {
        return;
    }
    let state = Arc::clone(state);
    tokio::spawn(async move {
        if let Err(e) = probe_and_auto_compact(&state).await {
            warn!("ailake: auto-compact (foreign file repair) probe failed: {e}");
        }
    });
}

/// Lists files (metadata only — no data file bytes fetched) and, if any is
/// foreign, runs one blocking compaction pass in this background task.
/// `auto_compact_inflight` prevents a second probe from starting a
/// concurrent pass while one is still running.
async fn probe_and_auto_compact(state: &AppState) -> Result<(), String> {
    let files = state
        .catalog
        .list_files(&state.table, None)
        .await
        .map_err(|e| e.to_string())?;
    if !files.iter().any(DataFileEntry::is_foreign) {
        return Ok(());
    }
    if state
        .auto_compact_inflight
        .compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire)
        .is_err()
    {
        return Ok(()); // a pass triggered by an earlier probe is already running
    }

    let planner = CompactionPlanner::new(CompactionConfig::default());
    let executor = CompactionExecutor::new(Arc::clone(&state.store), state.policy.clone());
    let result = executor
        .run(&planner, &state.table, Arc::clone(&state.catalog), "data")
        .await;
    state.auto_compact_inflight.store(false, Ordering::Release);

    match result {
        Ok(Some(entry)) => {
            info!(
                "ailake: auto-compact repaired foreign file(s) — merged into {}",
                entry.path
            );
            Ok(())
        }
        Ok(None) => Ok(()), // nothing left to compact (raced with a manual compact)
        Err(e) => Err(e.to_string()),
    }
}

async fn handle_write(
    State(state): State<Arc<AppState>>,
    body: String,
) -> ApiResult<impl IntoResponse> {
    let req: WriteRequest =
        serde_json::from_str(&body).map_err(|e| ApiError(format!("invalid JSON: {e}")))?;

    if req.texts.len() != req.embeddings.len() {
        return Err(ApiError(format!(
            "texts length {} != embeddings length {}",
            req.texts.len(),
            req.embeddings.len()
        )));
    }

    let schema = std::sync::Arc::new(arrow_schema::Schema::new(vec![arrow_schema::Field::new(
        "text",
        arrow_schema::DataType::Utf8,
        false,
    )]));
    let text_arr = arrow_array::StringArray::from(req.texts.clone());
    let batch = arrow_array::RecordBatch::try_new(schema, vec![std::sync::Arc::new(text_arr)])
        .map_err(|e| ApiError(format!("RecordBatch error: {e}")))?;

    let mut writer = TableWriter::create_or_open(
        Arc::clone(&state.catalog),
        Arc::clone(&state.store),
        state.policy.clone(),
        state.table.clone(),
        2,
    )
    .await
    .map_err(ApiError::from)?;

    let rows = req.embeddings.len();
    match req.batch_id {
        Some(ref id) => writer
            .write_batch_idempotent(&batch, &req.embeddings, id)
            .await
            .map_err(ApiError::from)?,
        None => writer
            .write_batch(&batch, &req.embeddings)
            .await
            .map_err(ApiError::from)?,
    }
    let snapshot_id = writer.commit().await.map_err(ApiError::from)?;

    let resp = WriteResponse { snapshot_id, rows };
    Ok((StatusCode::OK, serde_json::to_string(&resp).unwrap()))
}

async fn handle_compact(
    State(state): State<Arc<AppState>>,
    body: String,
) -> ApiResult<impl IntoResponse> {
    let req: CompactRequest = if body.trim().is_empty() {
        CompactRequest::default()
    } else {
        serde_json::from_str(&body).map_err(|e| ApiError(format!("invalid JSON: {e}")))?
    };

    let meta = state
        .catalog
        .load_table(&state.table)
        .await
        .map_err(ApiError::from)?;
    let files = state
        .catalog
        .list_files(&state.table, None)
        .await
        .map_err(ApiError::from)?;

    let config = CompactionConfig {
        min_files_to_compact: req.min_files,
        target_file_size_bytes: req.target_size,
        index_strategy: Default::default(),
        max_files_per_pass: 20,
    };
    let planner = CompactionPlanner::new(config);
    let to_compact = planner.plan(&files);

    if to_compact.is_empty() {
        let resp = CompactResponse {
            message: format!("nothing to compact ({} files below threshold)", files.len()),
            compacted_files: 0,
        };
        return Ok((StatusCode::OK, serde_json::to_string(&resp).unwrap()));
    }

    let n = to_compact.len();
    let executor = CompactionExecutor::new(Arc::clone(&state.store), state.policy.clone());
    let output_path = format!(
        "data/compacted-{}.parquet",
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs()
    );
    let new_entry = executor
        .compact(&to_compact, &output_path)
        .await
        .map_err(ApiError::from)?;

    let compacted_paths: std::collections::HashSet<&str> =
        to_compact.iter().map(|f| f.path.as_str()).collect();
    let mut remaining: Vec<_> = files
        .into_iter()
        .filter(|f| !compacted_paths.contains(f.path.as_str()))
        .collect();
    remaining.push(new_entry);

    state
        .catalog
        .commit_snapshot(
            &state.table,
            NewSnapshot {
                snapshot_id: new_snapshot_id(),
                parent_snapshot_id: meta.current_snapshot_id,
                files: remaining,
                operation: SnapshotOperation::Replace,
                iceberg_schema: None,
                extra_properties: std::collections::HashMap::new(),
                bloom_filters: vec![],
                equality_delete_files: vec![],
            },
        )
        .await
        .map_err(ApiError::from)?;

    let resp = CompactResponse {
        message: format!("compacted into {output_path}"),
        compacted_files: n,
    };
    Ok((StatusCode::OK, serde_json::to_string(&resp).unwrap()))
}

async fn handle_info(State(state): State<Arc<AppState>>) -> ApiResult<impl IntoResponse> {
    let meta = state
        .catalog
        .load_table(&state.table)
        .await
        .map_err(ApiError::from)?;
    let files = state
        .catalog
        .list_files(&state.table, None)
        .await
        .map_err(ApiError::from)?;

    let file_count = files.len();
    let row_count: u64 = files.iter().map(|f| f.record_count).sum();
    let size_bytes: u64 = files.iter().map(|f| f.file_size_bytes).sum();
    let ready = files
        .iter()
        .filter(|f| f.index_status == IndexStatus::Ready)
        .count();
    let failed = files
        .iter()
        .filter(|f| f.index_status == IndexStatus::Failed)
        .count();

    let resp = InfoResponse {
        table: format!("{}.{}", state.table.namespace, state.table.name),
        location: meta
            .properties
            .get("ailake.location")
            .cloned()
            .unwrap_or_else(|| meta.location.clone()),
        vector_column: meta
            .properties
            .get("ailake.vector-column")
            .cloned()
            .unwrap_or_else(|| state.policy.column_name.clone()),
        vector_dim: meta
            .properties
            .get("ailake.vector-dim")
            .cloned()
            .unwrap_or_else(|| state.policy.dim.to_string()),
        vector_metric: meta
            .properties
            .get("ailake.vector-metric")
            .cloned()
            .unwrap_or_else(|| "-".to_string()),
        files: file_count,
        indexed_files: ready,
        failed_files: failed,
        rows: row_count,
        size_bytes,
        snapshot_id: meta.current_snapshot_id,
    };
    Ok((StatusCode::OK, serde_json::to_string(&resp).unwrap()))
}

// ---------------------------------------------------------------------------
// Entry point
// ---------------------------------------------------------------------------

pub(crate) async fn run(
    catalog: Arc<dyn CatalogProvider>,
    store: Arc<dyn Store>,
    table: TableIdent,
    policy: VectorStoragePolicy,
    port: u16,
) -> Result<(), String> {
    let state = Arc::new(AppState {
        catalog,
        store,
        table,
        policy,
        auto_compact_inflight: Arc::new(AtomicBool::new(false)),
        auto_compact_last_check_ms: Arc::new(AtomicU64::new(0)),
    });

    let app = Router::new()
        .route("/search", post(handle_search))
        .route("/write", post(handle_write))
        .route("/compact", post(handle_compact))
        .route("/info", get(handle_info))
        .layer(DefaultBodyLimit::max(MAX_BODY_BYTES))
        .with_state(state);

    let addr = format!("0.0.0.0:{port}");
    let listener = tokio::net::TcpListener::bind(&addr)
        .await
        .map_err(|e| format!("bind {addr}: {e}"))?;

    eprintln!("ailake server listening on http://{addr}");
    eprintln!("WARNING: no authentication — expose only on a trusted network or behind an authenticating proxy");
    axum::serve(listener, app).await.map_err(|e| e.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;
    use ailake_catalog::HadoopCatalog;
    use ailake_core::{VectorMetric, VectorPrecision};
    use ailake_query::TableWriter;
    use ailake_store::LocalStore;
    use arrow_array::{Int32Array, RecordBatch};
    use arrow_schema::{DataType, Field, Schema};
    use tempfile::TempDir;

    fn test_policy() -> VectorStoragePolicy {
        VectorStoragePolicy {
            column_name: "embedding".to_string(),
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
        }
    }

    fn test_state(catalog: Arc<dyn CatalogProvider>, store: Arc<dyn Store>) -> AppState {
        AppState {
            catalog,
            store,
            table: TableIdent::new("default", "table"),
            policy: test_policy(),
            auto_compact_inflight: Arc::new(AtomicBool::new(false)),
            auto_compact_last_check_ms: Arc::new(AtomicU64::new(0)),
        }
    }

    /// Real round-trip, no mocks: writes a genuine AI-Lake file via `TableWriter`,
    /// then overwrites its manifest entry to strip `centroid_b64` — the same
    /// after-the-fact state a generic Iceberg engine's `OPTIMIZE`/`rewrite_data_files`
    /// leaves behind (see `DataFileEntry::is_foreign`, `CompactionPlanner::plan`).
    /// `read_parquet()` never touches the AILK footer (see `compaction.rs`
    /// `read_files_parallel`), so this reproduces exactly what `probe_and_auto_compact`
    /// has to detect and repair without needing a second, footerless physical file.
    #[tokio::test]
    async fn probe_and_auto_compact_repairs_foreign_file() {
        let dir = TempDir::new().unwrap();
        let store: Arc<dyn Store> = Arc::new(LocalStore::new(dir.path()));
        let catalog: Arc<dyn CatalogProvider> =
            Arc::new(HadoopCatalog::new(store.clone(), "warehouse"));
        let table = TableIdent::new("default", "table");
        let policy = test_policy();

        let schema = Arc::new(Schema::new(vec![Field::new("id", DataType::Int32, false)]));
        let batch =
            RecordBatch::try_new(schema, vec![Arc::new(Int32Array::from(vec![0i32, 1]))]).unwrap();
        let embeddings = vec![vec![1.0, 0.0, 0.0, 0.0], vec![0.0, 1.0, 0.0, 0.0]];

        let mut writer = TableWriter::create_or_open(
            catalog.clone(),
            store.clone(),
            policy.clone(),
            table.clone(),
            2,
        )
        .await
        .unwrap();
        writer.write_batch(&batch, &embeddings).await.unwrap();
        writer.commit().await.unwrap();

        let files = catalog.list_files(&table, None).await.unwrap();
        assert_eq!(files.len(), 1);
        assert!(
            !files[0].is_foreign(),
            "sanity: TableWriter must produce a native (non-foreign) entry"
        );

        // Simulate the foreign rewrite: same physical bytes, manifest entry stripped
        // of the AI-Lake-only metadata a generic engine's rewrite would never populate.
        let mut foreign_entry = files[0].clone();
        foreign_entry.centroid_b64 = None;
        foreign_entry.hnsw_offset = None;
        foreign_entry.hnsw_len = None;
        let parent_snapshot_id = catalog
            .load_table(&table)
            .await
            .unwrap()
            .current_snapshot_id;
        catalog
            .commit_snapshot(
                &table,
                NewSnapshot {
                    snapshot_id: new_snapshot_id(),
                    parent_snapshot_id,
                    files: vec![foreign_entry],
                    operation: SnapshotOperation::Replace,
                    iceberg_schema: None,
                    extra_properties: std::collections::HashMap::new(),
                    bloom_filters: vec![],
                    equality_delete_files: vec![],
                },
            )
            .await
            .unwrap();

        let files = catalog.list_files(&table, None).await.unwrap();
        assert!(
            files.iter().any(DataFileEntry::is_foreign),
            "setup sanity: file must now read as foreign"
        );

        let state = test_state(catalog.clone(), store.clone());
        probe_and_auto_compact(&state).await.unwrap();

        let files_after = catalog.list_files(&table, None).await.unwrap();
        assert_eq!(
            files_after.len(),
            1,
            "auto-compact should merge the single foreign file into one repaired file"
        );
        assert!(
            !files_after[0].is_foreign(),
            "repaired file must carry a real centroid again"
        );
        assert_eq!(
            files_after[0].record_count, 2,
            "both rows must survive the repair"
        );
    }

    /// No foreign files present — the probe must not touch the catalog beyond the
    /// one `list_files` check (no compaction pass, no snapshot commit).
    #[tokio::test]
    async fn probe_and_auto_compact_is_noop_when_nothing_foreign() {
        let dir = TempDir::new().unwrap();
        let store: Arc<dyn Store> = Arc::new(LocalStore::new(dir.path()));
        let catalog: Arc<dyn CatalogProvider> =
            Arc::new(HadoopCatalog::new(store.clone(), "warehouse"));
        let table = TableIdent::new("default", "table");
        let policy = test_policy();

        let schema = Arc::new(Schema::new(vec![Field::new("id", DataType::Int32, false)]));
        let batch =
            RecordBatch::try_new(schema, vec![Arc::new(Int32Array::from(vec![0i32]))]).unwrap();
        let mut writer =
            TableWriter::create_or_open(catalog.clone(), store.clone(), policy, table.clone(), 2)
                .await
                .unwrap();
        writer
            .write_batch(&batch, &[vec![1.0, 0.0, 0.0, 0.0]])
            .await
            .unwrap();
        writer.commit().await.unwrap();

        let state = test_state(catalog.clone(), store.clone());
        probe_and_auto_compact(&state).await.unwrap();

        let files_after = catalog.list_files(&table, None).await.unwrap();
        assert_eq!(files_after.len(), 1, "no compaction should have run");
        assert!(!state.auto_compact_inflight.load(Ordering::Acquire));
    }

    /// `maybe_probe_auto_compact` must skip re-listing files (and thus skip spawning
    /// a probe task) when called again inside the cooldown window.
    #[tokio::test]
    async fn maybe_probe_auto_compact_respects_cooldown() {
        let dir = TempDir::new().unwrap();
        let store: Arc<dyn Store> = Arc::new(LocalStore::new(dir.path()));
        let catalog: Arc<dyn CatalogProvider> =
            Arc::new(HadoopCatalog::new(store.clone(), "warehouse"));
        let state = Arc::new(test_state(catalog, store));

        // First call claims the window (sets the timestamp to "now").
        maybe_probe_auto_compact(&state);
        let first = state.auto_compact_last_check_ms.load(Ordering::Acquire);
        assert!(first > 0, "first call must claim the cooldown window");

        // Immediate second call must NOT reset the timestamp (still inside cooldown).
        maybe_probe_auto_compact(&state);
        let second = state.auto_compact_last_check_ms.load(Ordering::Acquire);
        assert_eq!(first, second, "second call within cooldown must be a no-op");
    }
}
