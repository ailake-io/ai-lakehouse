// SPDX-License-Identifier: MIT OR Apache-2.0
// HTTP server for AI-Lake — exposes search, write, compact, and info over JSON.
//
// Endpoints:
//   POST /search   {"query":[f32...], "top_k":10, "pruning_threshold":0.8}
//   POST /write    {"texts":["..."], "embeddings":[[f32...]], "batch_id":"..."}
//   POST /compact  {}
//   GET  /info

use std::sync::Arc;

use axum::{
    extract::State,
    http::StatusCode,
    response::{IntoResponse, Response},
    routing::{get, post},
    Router,
};
use serde::{Deserialize, Serialize};

use ailake_catalog::provider::{
    new_snapshot_id, CatalogProvider, IndexStatus, NewSnapshot, SnapshotOperation, TableIdent,
};
use ailake_core::VectorStoragePolicy;
use ailake_query::{
    CompactionConfig, CompactionExecutor, CompactionPlanner, SearchConfig, TableWriter,
};
use ailake_store::Store;

// ---------------------------------------------------------------------------
// Shared state
// ---------------------------------------------------------------------------

pub(crate) struct AppState {
    catalog: Arc<dyn CatalogProvider>,
    store: Arc<dyn Store>,
    table: TableIdent,
    policy: VectorStoragePolicy,
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

    let dim = req.query.len() as u32;
    let config = SearchConfig {
        top_k: req.top_k,
        ef_search: req.top_k * 5,
        pruning_threshold: req.pruning_threshold,
        rerank_factor: None,
        score_fn: None,
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
    });

    let app = Router::new()
        .route("/search", post(handle_search))
        .route("/write", post(handle_write))
        .route("/compact", post(handle_compact))
        .route("/info", get(handle_info))
        .with_state(state);

    let addr = format!("0.0.0.0:{port}");
    let listener = tokio::net::TcpListener::bind(&addr)
        .await
        .map_err(|e| format!("bind {addr}: {e}"))?;

    eprintln!("ailake server listening on http://{addr}");
    axum::serve(listener, app).await.map_err(|e| e.to_string())
}
