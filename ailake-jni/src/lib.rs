// SPDX-License-Identifier: MIT OR Apache-2.0
//! ailake-jni — C-ABI cdylib for Trino / Spark / Flink plugins via JNA.
//!
//! All three plugins call the same `ailake_search_json` / `ailake_write_batch_json` surface.
//!
//! Build: cargo build --release -p ailake-jni
//! The cdylib is loaded by the connector via JNA (System.loadLibrary is not required).

use std::{
    ffi::{c_char, CStr, CString},
    sync::Arc,
};

use ailake_catalog::{HadoopCatalog, TableIdent};
use ailake_core::{EmbeddingModelInfo, VectorMetric};
use ailake_query::{
    fetch_rows as rs_fetch_rows, search as rs_search, search_multimodal as rs_search_multimodal,
    Chunk, ContextAssembler, ContextAssemblerConfig, FusionMethod, ModalQuery, SearchConfig,
    SearchResult,
};
use ailake_store::LocalStore;
use serde::Serialize;
use tracing::{debug, info, warn};

// ── Shared types ──────────────────────────────────────────────────────────────

#[derive(Clone)]
pub struct RowResult {
    pub row_id: u64,
    pub distance: f32,
    pub file_path: String,
}

/// Same data, JSON-serializable for C-ABI surface.
#[derive(Serialize)]
struct RowResultJson {
    row_id: u64,
    distance: f32,
    file_path: String,
}

impl From<SearchResult> for RowResultJson {
    fn from(r: SearchResult) -> Self {
        Self {
            row_id: r.row_id.as_u64(),
            distance: r.distance,
            file_path: r.file_path,
        }
    }
}

// ── Helpers ───────────────────────────────────────────────────────────────────

fn rt() -> &'static tokio::runtime::Runtime {
    use std::sync::OnceLock;
    static RT: OnceLock<tokio::runtime::Runtime> = OnceLock::new();
    RT.get_or_init(|| {
        match tokio::runtime::Builder::new_multi_thread()
            .enable_all()
            .build()
        {
            Ok(rt) => {
                info!("ailake-jni: Tokio multi-thread runtime initialised");
                rt
            }
            Err(e) => {
                warn!(
                    "ailake-jni: multi-thread Tokio runtime failed ({}); \
                     falling back to single-threaded runtime to avoid JVM signal handler conflicts",
                    e
                );
                tokio::runtime::Builder::new_current_thread()
                    .enable_all()
                    .build()
                    .expect("ailake-jni: tokio runtime unavailable")
            }
        }
    })
}

fn parse_metric(s: &str) -> VectorMetric {
    match s {
        "euclidean" => VectorMetric::Euclidean,
        "dot_product" | "dotproduct" => VectorMetric::DotProduct,
        "normalized_cosine" | "normalizedcosine" => VectorMetric::NormalizedCosine,
        _ => VectorMetric::Cosine,
    }
}

/// Core search logic called by C-ABI exports.
#[allow(clippy::too_many_arguments)]
fn do_search(
    warehouse: String,
    namespace: &str,
    table_name: &str,
    vec_col: &str,
    dim: u32,
    query: Vec<f32>,
    top_k: u32,
    ef_search: u32,
    partition_filter: Option<String>,
    hybrid_text: Option<String>,
    text_column: &str,
    bm25_weight: f32,
) -> ailake_core::AilakeResult<Vec<SearchResult>> {
    let store: Arc<dyn ailake_store::Store> = Arc::new(LocalStore::new(&warehouse));
    let catalog = Arc::new(HadoopCatalog::new(store.clone(), &warehouse));
    let table = TableIdent::new(namespace, table_name);
    let hybrid = hybrid_text.map(|qt| {
        ailake_query::HybridConfig::new(qt)
            .with_text_column(text_column)
            .with_bm25_weight(bm25_weight)
    });
    let config = SearchConfig {
        top_k: top_k as usize,
        ef_search: ef_search as usize,
        pruning_threshold: f32::INFINITY,
        rerank_factor: None,
        score_fn: None,
        partition_filter,
        hybrid,
    };
    rt().block_on(rs_search(
        &table, &query, config, vec_col, dim, catalog, store,
    ))
}

// ── Internal helpers ──────────────────────────────────────────────────────────

#[allow(dead_code)]
fn assemble_context(chunk_jsons: Vec<String>, max_tokens: u64) -> String {
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

// ── C-ABI exports (JNA bridge for Trino / Spark / Flink plugins) ─────────────

fn cstr_empty_json() -> *mut c_char {
    CString::new("[]").unwrap().into_raw()
}

fn cstr_err_json(msg: impl std::fmt::Display) -> *mut c_char {
    let s = serde_json::json!({"ok": false, "error": msg.to_string()}).to_string();
    CString::new(s).unwrap_or_default().into_raw()
}

/// ailake-jni version string. Static — do NOT free.
#[no_mangle]
pub extern "C" fn ailake_version() -> *const c_char {
    static V: &str = concat!(env!("CARGO_PKG_VERSION"), "\0");
    V.as_ptr() as *const c_char
}

/// Search a local AI-Lake table. Returns a null-terminated JSON array:
/// `[{"row_id":N,"distance":F,"file_path":"..."}]`
///
/// # Parameters
/// - `table_uri`  — null-terminated UTF-8 path to table root
/// - `query_ptr`  — pointer to f32 array (native byte order)
/// - `query_len`  — number of f32 elements
/// - `top_k`      — nearest neighbors to return
///
/// # Safety
/// Caller must free the returned pointer with `ailake_free_string`.
#[no_mangle]
pub unsafe extern "C" fn ailake_vector_search_json(
    table_uri: *const c_char,
    query_ptr: *const f32,
    query_len: u32,
    top_k: u32,
) -> *mut c_char {
    if table_uri.is_null() || query_ptr.is_null() {
        return cstr_empty_json();
    }
    let uri = match CStr::from_ptr(table_uri).to_str() {
        Ok(s) => s.to_string(),
        Err(_) => return cstr_empty_json(),
    };
    let query = std::slice::from_raw_parts(query_ptr, query_len as usize).to_vec();
    let dim = query.len() as u32;
    let results: Vec<RowResultJson> = match do_search(
        uri,
        "default",
        "table",
        "embedding",
        dim,
        query,
        top_k,
        50,
        None,
        None,
        "chunk_text",
        0.5,
    ) {
        Ok(v) => v.into_iter().map(RowResultJson::from).collect(),
        Err(e) => return cstr_err_json(e),
    };
    let json = serde_json::to_string(&results).unwrap_or_else(|_| "[]".to_string());
    CString::new(json)
        .unwrap_or_else(|_| CString::new("[]").unwrap())
        .into_raw()
}

/// Search via JSON request — preferred for Flink/JVM callers.
///
/// `request_json` must be UTF-8 JSON:
/// `{"warehouse":"...","namespace":"default","table":"mytable","vec_col":"embedding",
///   "dim":128,"query":[0.1,...],"top_k":10,"ef_search":50}`
///
/// Returns JSON: `{"ok":true,"results":[{"row_id":N,"distance":F,"file_path":"..."}]}`
///
/// # Safety
/// Caller must free the returned pointer with `ailake_free_string`.
#[no_mangle]
pub unsafe extern "C" fn ailake_search_json(request_json: *const c_char) -> *mut c_char {
    #[derive(serde::Deserialize)]
    struct Req {
        warehouse: String,
        #[serde(default = "default_ns")]
        namespace: String,
        table: String,
        #[serde(default = "default_col")]
        vec_col: String,
        dim: u32,
        query: Vec<f32>,
        #[serde(default = "default_topk")]
        top_k: u32,
        #[serde(default = "default_ef")]
        ef_search: u32,
        #[serde(default)]
        partition_filter: Option<String>,
        #[serde(default)]
        hybrid_text: Option<String>,
        #[serde(default = "default_text_col")]
        text_column: String,
        #[serde(default = "default_bm25_weight")]
        bm25_weight: f32,
    }
    fn default_ns() -> String {
        "default".into()
    }
    fn default_col() -> String {
        "embedding".into()
    }
    fn default_topk() -> u32 {
        10
    }
    fn default_ef() -> u32 {
        50
    }
    fn default_text_col() -> String {
        "chunk_text".into()
    }
    fn default_bm25_weight() -> f32 {
        0.5
    }

    if request_json.is_null() {
        return cstr_err_json("null request_json");
    }
    let json_str = match CStr::from_ptr(request_json).to_str() {
        Ok(s) => s,
        Err(e) => {
            warn!("ailake_search_json: invalid UTF-8 in request_json: {}", e);
            return cstr_err_json(e);
        }
    };
    let req: Req = match serde_json::from_str(json_str) {
        Ok(r) => r,
        Err(e) => {
            warn!("ailake_search_json: JSON parse error: {}", e);
            return cstr_err_json(e);
        }
    };

    debug!(
        "ailake_search_json: warehouse={} table={}.{} dim={} top_k={}",
        req.warehouse, req.namespace, req.table, req.dim, req.top_k
    );

    let bm25_weight = req.bm25_weight;
    let text_column = req.text_column.clone();
    let results = match do_search(
        req.warehouse,
        &req.namespace,
        &req.table,
        &req.vec_col,
        req.dim,
        req.query,
        req.top_k,
        req.ef_search,
        req.partition_filter,
        req.hybrid_text,
        &text_column,
        bm25_weight,
    ) {
        Ok(v) => v,
        Err(e) => {
            warn!("ailake_search_json: search failed: {}", e);
            return cstr_err_json(e);
        }
    };
    #[derive(serde::Serialize)]
    struct Resp {
        ok: bool,
        results: Vec<RowResultJson>,
    }
    let body = Resp {
        ok: true,
        results: results.into_iter().map(RowResultJson::from).collect(),
    };
    let json = serde_json::to_string(&body)
        .unwrap_or_else(|_| "{\"ok\":false,\"error\":\"serialize\"}".into());
    CString::new(json).unwrap_or_default().into_raw()
}

/// Write a batch of records to an AI-Lake table.
///
/// `request_json` must be UTF-8 JSON:
/// ```json
/// {
///   "warehouse": "/path/to/warehouse",
///   "namespace": "default",
///   "table": "my_table",
///   "vec_col": "embedding",
///   "dim": 128,
///   "metric": "euclidean",
///   "precision": "f16",
///   "ids": [1, 2, 3],
///   "embeddings": [[0.1, 0.2, ...], ...]
/// }
/// ```
///
/// Returns JSON: `{"ok":true,"snapshot_id":N}` or `{"ok":false,"error":"..."}`
///
/// # Safety
/// Caller must free the returned pointer with `ailake_free_string`.
#[no_mangle]
pub unsafe extern "C" fn ailake_write_batch_json(request_json: *const c_char) -> *mut c_char {
    use ailake_core::{PartitionDef, VectorPrecision, VectorStoragePolicy};
    use ailake_query::TableWriter;
    use arrow_array::{Int64Array, RecordBatch};
    use arrow_schema::{DataType, Field, Schema};

    /// One field in a multi-column partition spec.
    /// JSON: `{"column": "agent_id", "transform": "identity", "column_type": "string"}`
    #[derive(serde::Deserialize)]
    struct PartitionFieldReq {
        column: String,
        #[serde(default = "default_transform")]
        transform: String,
        #[serde(default = "default_col_type")]
        column_type: String,
    }
    fn default_transform() -> String {
        "identity".into()
    }
    fn default_col_type() -> String {
        "string".into()
    }

    #[derive(serde::Deserialize)]
    struct Req {
        warehouse: String,
        #[serde(default = "default_ns")]
        namespace: String,
        table: String,
        #[serde(default = "default_col")]
        vec_col: String,
        dim: u32,
        #[serde(default)]
        metric: Option<String>,
        #[serde(default)]
        precision: Option<String>,
        #[serde(default)]
        ivf_residual: bool,
        /// Optional model identifier stored in `ailake.embedding-model` Iceberg property.
        #[serde(default)]
        embedding_model: Option<String>,
        /// Single-column identity partition (legacy). Superseded by `partition_fields`.
        #[serde(default)]
        partition_by: Option<String>,
        /// Runtime partition value. For multi-column specs, use \x1f-separated compound
        /// string or rely on `partition_values` dict (Python SDK converts automatically).
        #[serde(default)]
        partition_value: Option<String>,
        /// Multi-column partition spec (Phase K). When non-empty, takes precedence over
        /// `partition_by`. Supports "identity" and "truncate[W]" transforms.
        #[serde(default)]
        partition_fields: Vec<PartitionFieldReq>,
        /// Iceberg format version: 2 (default, V2) or 3 (V3 opt-in).
        #[serde(default = "default_format_version")]
        format_version: u8,
        ids: Vec<i64>,
        embeddings: Vec<Vec<f32>>,
    }
    fn default_ns() -> String {
        "default".into()
    }
    fn default_col() -> String {
        "embedding".into()
    }
    fn default_format_version() -> u8 {
        2
    }

    if request_json.is_null() {
        return cstr_err_json("null request_json");
    }
    let json_str = match CStr::from_ptr(request_json).to_str() {
        Ok(s) => s,
        Err(e) => {
            warn!(
                "ailake_write_batch_json: invalid UTF-8 in request_json: {}",
                e
            );
            return cstr_err_json(e);
        }
    };
    let req: Req = match serde_json::from_str(json_str) {
        Ok(r) => r,
        Err(e) => {
            warn!("ailake_write_batch_json: JSON parse error: {}", e);
            return cstr_err_json(e);
        }
    };
    if req.ids.len() != req.embeddings.len() {
        warn!(
            "ailake_write_batch_json: ids.len()={} != embeddings.len()={}",
            req.ids.len(),
            req.embeddings.len()
        );
        return cstr_err_json("ids.len() != embeddings.len()");
    }
    debug!(
        "ailake_write_batch_json: warehouse={} table={}.{} rows={}",
        req.warehouse,
        req.namespace,
        req.table,
        req.ids.len()
    );

    let metric = parse_metric(req.metric.as_deref().unwrap_or("euclidean"));
    let precision = match req.precision.as_deref().unwrap_or("f16") {
        "f32" => VectorPrecision::F32,
        "i8" => VectorPrecision::I8,
        _ => VectorPrecision::F16,
    };
    let embedding_model = req
        .embedding_model
        .as_deref()
        .map(EmbeddingModelInfo::from_property_value);
    let partition_fields: Vec<PartitionDef> = req
        .partition_fields
        .into_iter()
        .map(|pf| PartitionDef {
            column: pf.column,
            transform: pf.transform,
            column_type: pf.column_type,
        })
        .collect();

    let policy = VectorStoragePolicy {
        column_name: req.vec_col.clone(),
        dim: req.dim,
        metric,
        precision,
        pq: None,
        keep_raw_for_reranking: true,
        pre_normalize: false,
        hnsw_m: None,
        hnsw_ef_construction: None,
        ivf_residual: req.ivf_residual,
        embedding_model,
        modality: None,
        partition_by: req.partition_by,
        partition_value: req.partition_value,
        partition_column_type: None,
        partition_fields,
    };

    let format_version = req.format_version;
    let table = ailake_catalog::TableIdent::new(&req.namespace, &req.table);
    let store: std::sync::Arc<dyn ailake_store::Store> =
        std::sync::Arc::new(LocalStore::new(&req.warehouse));
    let catalog = std::sync::Arc::new(HadoopCatalog::new(store.clone(), &req.warehouse));

    let schema = std::sync::Arc::new(Schema::new(vec![Field::new("id", DataType::Int64, false)]));
    let batch =
        match RecordBatch::try_new(schema, vec![std::sync::Arc::new(Int64Array::from(req.ids))]) {
            Ok(b) => b,
            Err(e) => return cstr_err_json(e),
        };

    let result = rt().block_on(async {
        let mut writer =
            TableWriter::create_or_open(catalog, store, policy, table, format_version).await?;
        writer.write_batch(&batch, &req.embeddings).await?;
        writer.commit().await
    });

    #[derive(serde::Serialize)]
    struct Resp {
        ok: bool,
        snapshot_id: i64,
    }
    match result {
        Ok(snap) => {
            info!(
                "ailake_write_batch_json: committed snapshot_id={} table={}.{}",
                snap, req.namespace, req.table
            );
            let json = serde_json::to_string(&Resp {
                ok: true,
                snapshot_id: snap,
            })
            .unwrap_or_default();
            CString::new(json).unwrap_or_default().into_raw()
        }
        Err(e) => {
            warn!("ailake_write_batch_json: write failed: {}", e);
            cstr_err_json(e)
        }
    }
}

/// Free a string returned by `ailake_vector_search_json`.
///
/// # Safety
/// `ptr` must be a pointer previously returned by `ailake_vector_search_json`
/// and must not have been freed already.
#[no_mangle]
pub unsafe extern "C" fn ailake_free_string(ptr: *mut c_char) {
    if !ptr.is_null() {
        drop(CString::from_raw(ptr));
    }
}

// ── ailake_search_text_json — pure BM25 text search ──────────────────────────

/// Pure BM25 full-text search — no vector query required.
///
/// `request_json` must be UTF-8 JSON:
/// ```json
/// {
///   "warehouse": "/path/to/warehouse",
///   "namespace": "default",
///   "table": "my_table",
///   "query_text": "rust programming language",
///   "top_k": 10,
///   "text_column": "chunk_text",
///   "partition_filter": null
/// }
/// ```
///
/// Returns JSON: `{"ok":true,"results":[{"row_id":N,"distance":F,"file_path":"..."}]}`
/// where `distance` is the negated BM25 score (lower = more relevant).
///
/// # Safety
/// Caller must free the returned pointer with `ailake_free_string`.
#[no_mangle]
pub unsafe extern "C" fn ailake_search_text_json(request_json: *const c_char) -> *mut c_char {
    use ailake_query::search_text as rs_search_text;

    #[derive(serde::Deserialize)]
    struct Req {
        warehouse: String,
        #[serde(default = "default_ns_st")]
        namespace: String,
        table: String,
        query_text: String,
        #[serde(default = "default_topk_st")]
        top_k: u32,
        #[serde(default = "default_text_col_st")]
        text_column: String,
        #[serde(default)]
        partition_filter: Option<String>,
    }
    fn default_ns_st() -> String {
        "default".into()
    }
    fn default_topk_st() -> u32 {
        10
    }
    fn default_text_col_st() -> String {
        "chunk_text".into()
    }

    if request_json.is_null() {
        return cstr_err_json("null request_json");
    }
    let json_str = match CStr::from_ptr(request_json).to_str() {
        Ok(s) => s,
        Err(e) => return cstr_err_json(e),
    };
    let req: Req = match serde_json::from_str(json_str) {
        Ok(r) => r,
        Err(e) => {
            warn!("ailake_search_text_json: JSON parse error: {}", e);
            return cstr_err_json(e);
        }
    };

    debug!(
        "ailake_search_text_json: warehouse={} table={}.{} query={:?} top_k={}",
        req.warehouse,
        req.namespace,
        req.table,
        &req.query_text[..req.query_text.len().min(60)],
        req.top_k
    );

    let store: Arc<dyn ailake_store::Store> = Arc::new(LocalStore::new(&req.warehouse));
    let catalog = Arc::new(HadoopCatalog::new(store.clone(), &req.warehouse));
    let table = TableIdent::new(&req.namespace, &req.table);
    let pf = req.partition_filter.as_deref();
    let results = match rt().block_on(rs_search_text(
        &table,
        &req.query_text,
        &[req.text_column.as_str()],
        req.top_k as usize,
        catalog,
        store,
        pf,
    )) {
        Ok(v) => v,
        Err(e) => {
            warn!("ailake_search_text_json: search failed: {}", e);
            return cstr_err_json(e);
        }
    };

    #[derive(serde::Serialize)]
    struct Resp {
        ok: bool,
        results: Vec<RowResultJson>,
    }
    let body = Resp {
        ok: true,
        results: results.into_iter().map(RowResultJson::from).collect(),
    };
    let json = serde_json::to_string(&body)
        .unwrap_or_else(|_| "{\"ok\":false,\"error\":\"serialize\"}".into());
    CString::new(json).unwrap_or_default().into_raw()
}

// ── ailake_search_multimodal_json — cross-modal RRF ──────────────────────────

/// Cross-modal vector search via Reciprocal Rank Fusion.
///
/// `request_json` must be UTF-8 JSON:
/// ```json
/// {
///   "warehouse": "/path/to/warehouse",
///   "namespace": "default",
///   "table": "my_table",
///   "queries": [
///     {"col": "embedding",       "query": [0.1, ...], "weight": 0.7, "dim": 0},
///     {"col": "image_embedding", "query": [0.3, ...], "weight": 0.3, "dim": 0}
///   ],
///   "top_k": 10
/// }
/// ```
/// `dim: 0` means auto-detect from `ailake.dim-<col>` / `ailake.vector-dim` in metadata.
///
/// Returns JSON: `{"ok":true,"results":[{"row_id":N,"rrf_score":F,"file_path":"..."}]}`
///
/// # Safety
/// Caller must free the returned pointer with `ailake_free_string`.
#[no_mangle]
pub unsafe extern "C" fn ailake_search_multimodal_json(request_json: *const c_char) -> *mut c_char {
    #[derive(serde::Deserialize)]
    struct ModalQueryReq {
        col: String,
        query: Vec<f32>,
        #[serde(default = "default_weight")]
        weight: f32,
        #[serde(default)]
        dim: u32,
    }
    fn default_weight() -> f32 {
        1.0
    }

    #[derive(serde::Deserialize)]
    struct Req {
        warehouse: String,
        #[serde(default = "default_ns_multi")]
        namespace: String,
        table: String,
        queries: Vec<ModalQueryReq>,
        #[serde(default = "default_topk_multi")]
        top_k: u32,
        #[serde(default)]
        partition_filter: Option<String>,
    }
    fn default_ns_multi() -> String {
        "default".into()
    }
    fn default_topk_multi() -> u32 {
        10
    }

    #[derive(serde::Serialize)]
    struct RrfRow {
        row_id: u64,
        rrf_score: f32,
        file_path: String,
    }
    #[derive(serde::Serialize)]
    struct Resp {
        ok: bool,
        results: Vec<RrfRow>,
    }

    if request_json.is_null() {
        return cstr_err_json("null request_json");
    }
    let json_str = match CStr::from_ptr(request_json).to_str() {
        Ok(s) => s,
        Err(e) => return cstr_err_json(e),
    };
    let req: Req = match serde_json::from_str(json_str) {
        Ok(r) => r,
        Err(e) => {
            warn!("ailake_search_multimodal_json: JSON parse error: {}", e);
            return cstr_err_json(e);
        }
    };
    if req.queries.is_empty() {
        return cstr_err_json("queries array must not be empty");
    }

    let table = TableIdent::new(&req.namespace, &req.table);
    let store: Arc<dyn ailake_store::Store> = Arc::new(LocalStore::new(&req.warehouse));
    let catalog = Arc::new(HadoopCatalog::new(store.clone(), &req.warehouse));

    // Hold owned query vecs alive for the lifetime of ModalQuery borrows
    let modal_queries_owned: Vec<(String, Vec<f32>, f32, u32)> = req
        .queries
        .into_iter()
        .map(|q| (q.col, q.query, q.weight, q.dim))
        .collect();
    let modal_queries: Vec<ModalQuery<'_>> = modal_queries_owned
        .iter()
        .map(|(col, query, weight, dim)| ModalQuery {
            column: col.as_str(),
            query: query.as_slice(),
            weight: *weight,
            dim: *dim,
        })
        .collect();

    let config = SearchConfig {
        top_k: req.top_k as usize,
        partition_filter: req.partition_filter,
        ..Default::default()
    };

    let results = match rt().block_on(rs_search_multimodal(
        &table,
        &modal_queries,
        config,
        catalog,
        store,
        FusionMethod::Rrf,
    )) {
        Ok(v) => v,
        Err(e) => {
            warn!("ailake_search_multimodal_json: search failed: {}", e);
            return cstr_err_json(e);
        }
    };

    // SearchResult.distance stores -rrf_score; negate to expose positive score.
    let rrf_rows: Vec<RrfRow> = results
        .into_iter()
        .map(|r| RrfRow {
            row_id: r.row_id.as_u64(),
            rrf_score: -r.distance,
            file_path: r.file_path,
        })
        .collect();

    let body = Resp {
        ok: true,
        results: rrf_rows,
    };
    let json = serde_json::to_string(&body)
        .unwrap_or_else(|_| "{\"ok\":false,\"error\":\"serialize\"}".into());
    CString::new(json).unwrap_or_default().into_raw()
}

// ── ailake_scan_json — search + fetch full rows ───────────────────────────────

/// Serialize a RecordBatch to the JSON columnar format consumed by the DuckDB extension.
///
/// Response shape:
/// ```json
/// {
///   "ok": true,
///   "schema": [{"name":"id","type":"int64"}, ...],
///   "num_rows": N,
///   "columns": {"id": [...], "text": [...], "_distance": [...]}
/// }
/// ```
/// Supported Arrow types → JSON type tag:
///   Int*/UInt* → "int64"  |  Float32 → "float32"  |  Float64 → "float64"
///   Utf8/LargeUtf8 → "utf8"  |  Boolean → "bool"
///   FixedSizeList<Float32> → "list_float32"   (skipped silently otherwise)
fn record_batch_to_scan_json(batch: &arrow_array::RecordBatch) -> Result<String, String> {
    use arrow_array::{
        Array, BooleanArray, Float32Array, Float64Array, Int16Array, Int32Array, Int64Array,
        Int8Array, LargeStringArray, StringArray, UInt16Array, UInt32Array, UInt64Array,
        UInt8Array,
    };
    use arrow_schema::DataType;
    use serde_json::{Map, Value};

    // Serialize any integer-like array as Vec<Option<i64>>.
    macro_rules! int_vals {
        ($col:expr, $T:ty, $num_rows:expr) => {{
            let arr = $col.as_any().downcast_ref::<$T>().ok_or(concat!(
                "downcast ",
                stringify!($T),
                " failed"
            ))?;
            (0..$num_rows)
                .map(|i| {
                    if arr.is_null(i) {
                        Value::Null
                    } else {
                        Value::Number((arr.value(i) as i64).into())
                    }
                })
                .collect::<Vec<_>>()
        }};
    }

    let num_rows = batch.num_rows();
    let mut schema_arr: Vec<serde_json::Value> = Vec::new();
    let mut columns_map: Map<String, Value> = Map::new();

    for (field, col) in batch.schema().fields().iter().zip(batch.columns()) {
        let name = field.name().clone();

        match field.data_type() {
            DataType::Int8 => {
                let vals = int_vals!(col, Int8Array, num_rows);
                schema_arr.push(serde_json::json!({"name": name, "type": "int64"}));
                columns_map.insert(name, Value::Array(vals));
            }
            DataType::Int16 => {
                let vals = int_vals!(col, Int16Array, num_rows);
                schema_arr.push(serde_json::json!({"name": name, "type": "int64"}));
                columns_map.insert(name, Value::Array(vals));
            }
            DataType::Int32 | DataType::Date32 => {
                let vals = int_vals!(col, Int32Array, num_rows);
                schema_arr.push(serde_json::json!({"name": name, "type": "int64"}));
                columns_map.insert(name, Value::Array(vals));
            }
            DataType::Int64
            | DataType::Date64
            | DataType::Timestamp(_, _)
            | DataType::Duration(_) => {
                let vals = int_vals!(col, Int64Array, num_rows);
                schema_arr.push(serde_json::json!({"name": name, "type": "int64"}));
                columns_map.insert(name, Value::Array(vals));
            }
            DataType::UInt8 => {
                let vals = int_vals!(col, UInt8Array, num_rows);
                schema_arr.push(serde_json::json!({"name": name, "type": "int64"}));
                columns_map.insert(name, Value::Array(vals));
            }
            DataType::UInt16 => {
                let vals = int_vals!(col, UInt16Array, num_rows);
                schema_arr.push(serde_json::json!({"name": name, "type": "int64"}));
                columns_map.insert(name, Value::Array(vals));
            }
            DataType::UInt32 => {
                let vals = int_vals!(col, UInt32Array, num_rows);
                schema_arr.push(serde_json::json!({"name": name, "type": "int64"}));
                columns_map.insert(name, Value::Array(vals));
            }
            DataType::UInt64 => {
                let vals = int_vals!(col, UInt64Array, num_rows);
                schema_arr.push(serde_json::json!({"name": name, "type": "int64"}));
                columns_map.insert(name, Value::Array(vals));
            }

            DataType::Float32 => {
                let arr = col
                    .as_any()
                    .downcast_ref::<Float32Array>()
                    .ok_or("downcast Float32Array")?;
                let vals: Vec<Value> = (0..num_rows)
                    .map(|i| {
                        if arr.is_null(i) {
                            Value::Null
                        } else {
                            let v = arr.value(i);
                            serde_json::Number::from_f64(v as f64)
                                .map(Value::Number)
                                .unwrap_or(Value::Null)
                        }
                    })
                    .collect();
                schema_arr.push(serde_json::json!({"name": name, "type": "float32"}));
                columns_map.insert(name, Value::Array(vals));
            }

            DataType::Float64 => {
                let arr = col
                    .as_any()
                    .downcast_ref::<Float64Array>()
                    .ok_or("downcast Float64Array")?;
                let vals: Vec<Value> = (0..num_rows)
                    .map(|i| {
                        if arr.is_null(i) {
                            Value::Null
                        } else {
                            serde_json::Number::from_f64(arr.value(i))
                                .map(Value::Number)
                                .unwrap_or(Value::Null)
                        }
                    })
                    .collect();
                schema_arr.push(serde_json::json!({"name": name, "type": "float64"}));
                columns_map.insert(name, Value::Array(vals));
            }

            DataType::Utf8 => {
                let arr = col
                    .as_any()
                    .downcast_ref::<StringArray>()
                    .ok_or("downcast StringArray")?;
                let vals: Vec<Value> = (0..num_rows)
                    .map(|i| {
                        if arr.is_null(i) {
                            Value::Null
                        } else {
                            Value::String(arr.value(i).to_string())
                        }
                    })
                    .collect();
                schema_arr.push(serde_json::json!({"name": name, "type": "utf8"}));
                columns_map.insert(name, Value::Array(vals));
            }

            DataType::LargeUtf8 => {
                let arr = col
                    .as_any()
                    .downcast_ref::<LargeStringArray>()
                    .ok_or("downcast LargeStringArray")?;
                let vals: Vec<Value> = (0..num_rows)
                    .map(|i| {
                        if arr.is_null(i) {
                            Value::Null
                        } else {
                            Value::String(arr.value(i).to_string())
                        }
                    })
                    .collect();
                schema_arr.push(serde_json::json!({"name": name, "type": "utf8"}));
                columns_map.insert(name, Value::Array(vals));
            }

            DataType::Boolean => {
                let arr = col
                    .as_any()
                    .downcast_ref::<BooleanArray>()
                    .ok_or("downcast BooleanArray")?;
                let vals: Vec<Value> = (0..num_rows)
                    .map(|i| {
                        if arr.is_null(i) {
                            Value::Null
                        } else {
                            Value::Bool(arr.value(i))
                        }
                    })
                    .collect();
                schema_arr.push(serde_json::json!({"name": name, "type": "bool"}));
                columns_map.insert(name, Value::Array(vals));
            }

            DataType::FixedSizeList(inner_field, _)
                if matches!(inner_field.data_type(), DataType::Float32) =>
            {
                use arrow_array::FixedSizeListArray;
                let arr = col
                    .as_any()
                    .downcast_ref::<FixedSizeListArray>()
                    .ok_or("downcast FixedSizeListArray")?;
                let vals: Vec<Value> = (0..num_rows)
                    .map(|i| {
                        if arr.is_null(i) {
                            return Value::Null;
                        }
                        let list_val = arr.value(i);
                        let fa = list_val
                            .as_any()
                            .downcast_ref::<Float32Array>()
                            .map(|fa| {
                                (0..fa.len())
                                    .map(|j| {
                                        serde_json::Number::from_f64(fa.value(j) as f64)
                                            .map(Value::Number)
                                            .unwrap_or(Value::Null)
                                    })
                                    .collect::<Vec<_>>()
                            })
                            .unwrap_or_default();
                        Value::Array(fa)
                    })
                    .collect();
                schema_arr.push(serde_json::json!({"name": name, "type": "list_float32"}));
                columns_map.insert(name, Value::Array(vals));
            }

            // Unsupported types silently skipped — downstream sees fewer columns.
            _ => {}
        }
    }

    let resp = serde_json::json!({
        "ok": true,
        "schema": schema_arr,
        "num_rows": num_rows,
        "columns": Value::Object(columns_map),
    });
    serde_json::to_string(&resp).map_err(|e| e.to_string())
}

/// Scan an AI-Lake table: vector search + full row fetch in one call.
///
/// `request_json` — same format as `ailake_search_json` (warehouse/namespace/table/
/// vec_col/dim/query/top_k/ef_search).
///
/// Returns JSON:
/// ```json
/// {
///   "ok": true,
///   "schema": [{"name":"id","type":"int64"}, ...],
///   "num_rows": N,
///   "columns": {"id": [...], "_distance": [...], ...}
/// }
/// ```
///
/// The vector column is included as `list_float32` (F32-decoded values, not raw F16 bytes).
/// `_distance` is always the last column.
///
/// # Safety
/// Caller must free the returned pointer with `ailake_free_string`.
#[no_mangle]
pub unsafe extern "C" fn ailake_scan_json(request_json: *const c_char) -> *mut c_char {
    #[derive(serde::Deserialize)]
    struct Req {
        warehouse: String,
        #[serde(default = "scan_default_ns")]
        namespace: String,
        table: String,
        #[serde(default = "scan_default_col")]
        vec_col: String,
        dim: u32,
        query: Vec<f32>,
        #[serde(default = "scan_default_topk")]
        top_k: u32,
        #[serde(default = "scan_default_ef")]
        ef_search: u32,
        #[serde(default)]
        partition_filter: Option<String>,
    }
    fn scan_default_ns() -> String {
        "default".into()
    }
    fn scan_default_col() -> String {
        "embedding".into()
    }
    fn scan_default_topk() -> u32 {
        10
    }
    fn scan_default_ef() -> u32 {
        50
    }

    if request_json.is_null() {
        return cstr_err_json("null request_json");
    }
    let json_str = match CStr::from_ptr(request_json).to_str() {
        Ok(s) => s,
        Err(e) => return cstr_err_json(e),
    };
    let req: Req = match serde_json::from_str(json_str) {
        Ok(r) => r,
        Err(e) => {
            warn!("ailake_scan_json: JSON parse error: {}", e);
            return cstr_err_json(e);
        }
    };

    debug!(
        "ailake_scan_json: warehouse={} table={}.{} dim={} top_k={}",
        req.warehouse, req.namespace, req.table, req.dim, req.top_k
    );

    let results = match do_search(
        req.warehouse.clone(),
        &req.namespace,
        &req.table,
        &req.vec_col,
        req.dim,
        req.query,
        req.top_k,
        req.ef_search,
        req.partition_filter,
        None,
        "",
        0.0,
    ) {
        Ok(v) => v,
        Err(e) => {
            warn!("ailake_scan_json: search failed: {}", e);
            return cstr_err_json(e);
        }
    };

    // Separate store for fetching row data (do_search owns its own store internally).
    let store: Arc<dyn ailake_store::Store> = Arc::new(LocalStore::new(&req.warehouse));

    let batch = match rt().block_on(rs_fetch_rows(&results, store, &req.vec_col, req.dim)) {
        Ok(b) => b,
        Err(e) => {
            warn!("ailake_scan_json: fetch_rows failed: {}", e);
            return cstr_err_json(e);
        }
    };

    match record_batch_to_scan_json(&batch) {
        Ok(json) => CString::new(json).unwrap_or_default().into_raw(),
        Err(e) => {
            warn!("ailake_scan_json: serialization failed: {}", e);
            cstr_err_json(e)
        }
    }
}

// ── ailake_delete_where_json — Phase H: logical row deletion ──────────────────

/// Delete all rows where `column` equals any value in `values`.
///
/// Writes an Iceberg equality delete file + delete manifest. No data files are
/// rewritten. Deleted rows are filtered out by the scanner on every subsequent
/// read. Compatible with both V2 and V3 tables.
///
/// `request_json`:
/// ```json
/// {
///   "warehouse": "/data/my_table",
///   "namespace": "default",
///   "table": "my_table",
///   "column": "document_id",
///   "values": ["doc-a", "doc-b", "doc-c"]
/// }
/// ```
///
/// Returns `{"ok":true}` on success, `{"error":"..."}` on failure.
///
/// # Safety
/// Caller must free the returned pointer with `ailake_free_string`.
#[no_mangle]
pub unsafe extern "C" fn ailake_delete_where_json(request_json: *const c_char) -> *mut c_char {
    use ailake_query::delete_where;

    #[derive(serde::Deserialize)]
    struct Req {
        warehouse: String,
        #[serde(default = "dw_default_ns")]
        namespace: String,
        table: String,
        column: String,
        values: Vec<String>,
    }
    fn dw_default_ns() -> String {
        "default".into()
    }

    if request_json.is_null() {
        return cstr_err_json("null request_json");
    }
    let json_str = match CStr::from_ptr(request_json).to_str() {
        Ok(s) => s,
        Err(e) => return cstr_err_json(e),
    };
    let req: Req = match serde_json::from_str(json_str) {
        Ok(r) => r,
        Err(e) => {
            warn!("ailake_delete_where_json: JSON parse error: {}", e);
            return cstr_err_json(e);
        }
    };

    debug!(
        "ailake_delete_where_json: warehouse={} table={}.{} column={} values={}",
        req.warehouse,
        req.namespace,
        req.table,
        req.column,
        req.values.len()
    );

    let store: Arc<dyn ailake_store::Store> = Arc::new(LocalStore::new(&req.warehouse));
    let catalog = Arc::new(HadoopCatalog::new(store.clone(), &req.warehouse));
    let table = TableIdent::new(&req.namespace, &req.table);
    let values_ref: Vec<&str> = req.values.iter().map(String::as_str).collect();

    let result = rt().block_on(delete_where(
        catalog,
        store,
        &table,
        &req.column,
        &values_ref,
    ));

    #[derive(serde::Serialize)]
    struct Resp {
        ok: bool,
    }
    match result {
        Ok(()) => {
            info!(
                "ailake_delete_where_json: deleted {} values from column '{}' in {}.{}",
                req.values.len(),
                req.column,
                req.namespace,
                req.table
            );
            CString::new(serde_json::to_string(&Resp { ok: true }).unwrap_or_default())
                .unwrap_or_default()
                .into_raw()
        }
        Err(e) => {
            warn!("ailake_delete_where_json: failed: {}", e);
            cstr_err_json(e)
        }
    }
}

// ── ailake_evolve_schema_json — Phase G: metadata-only schema evolution ───────

/// Add or rename columns without rewriting data files.
///
/// Old files automatically return `initial_default` (or null) for new columns.
/// Field IDs remain stable across renames — engines that rely on field-id
/// (Spark, Trino) continue to work after a rename.
///
/// `request_json`:
/// ```json
/// {
///   "warehouse": "/data/my_table",
///   "namespace": "default",
///   "table": "my_table",
///   "add_columns": [
///     { "name": "score", "type": "float", "initial_default": 0.0 }
///   ],
///   "rename_columns": [
///     { "from": "old_name", "to": "new_name" }
///   ]
/// }
/// ```
///
/// Returns `{"ok":true}` on success, `{"error":"..."}` on failure.
///
/// # Safety
/// Caller must free the returned pointer with `ailake_free_string`.
#[no_mangle]
pub unsafe extern "C" fn ailake_evolve_schema_json(request_json: *const c_char) -> *mut c_char {
    use ailake_catalog::provider::CatalogProvider;
    use ailake_catalog::{AddColumnRequest, RenameColumnRequest, SchemaEvolution};

    #[derive(serde::Deserialize)]
    struct AddColReq {
        name: String,
        #[serde(rename = "type")]
        iceberg_type: String,
        #[serde(default)]
        initial_default: Option<serde_json::Value>,
        #[serde(default)]
        write_default: Option<serde_json::Value>,
        #[serde(default)]
        doc: Option<String>,
    }
    #[derive(serde::Deserialize)]
    struct RenameColReq {
        from: String,
        to: String,
    }
    #[derive(serde::Deserialize)]
    struct Req {
        warehouse: String,
        #[serde(default = "es_default_ns")]
        namespace: String,
        table: String,
        #[serde(default)]
        add_columns: Vec<AddColReq>,
        #[serde(default)]
        rename_columns: Vec<RenameColReq>,
    }
    fn es_default_ns() -> String {
        "default".into()
    }

    if request_json.is_null() {
        return cstr_err_json("null request_json");
    }
    let json_str = match CStr::from_ptr(request_json).to_str() {
        Ok(s) => s,
        Err(e) => return cstr_err_json(e),
    };
    let req: Req = match serde_json::from_str(json_str) {
        Ok(r) => r,
        Err(e) => {
            warn!("ailake_evolve_schema_json: JSON parse error: {}", e);
            return cstr_err_json(e);
        }
    };

    debug!(
        "ailake_evolve_schema_json: warehouse={} table={}.{} add={} rename={}",
        req.warehouse,
        req.namespace,
        req.table,
        req.add_columns.len(),
        req.rename_columns.len()
    );

    let mut evolution = SchemaEvolution::new();
    for r in req.rename_columns {
        evolution = evolution.rename_column(r.from, r.to);
    }
    for a in req.add_columns {
        evolution = evolution.add_column(AddColumnRequest {
            name: a.name,
            iceberg_type: a.iceberg_type,
            required: false,
            initial_default: a.initial_default,
            write_default: a.write_default,
            doc: a.doc,
        });
    }

    let store: Arc<dyn ailake_store::Store> = Arc::new(LocalStore::new(&req.warehouse));
    let catalog = Arc::new(HadoopCatalog::new(store.clone(), &req.warehouse));
    let table = TableIdent::new(&req.namespace, &req.table);

    let result = rt().block_on(catalog.evolve_schema(&table, evolution));

    #[derive(serde::Serialize)]
    struct Resp {
        ok: bool,
        new_schema_id: i32,
    }
    match result {
        Ok(schema_id) => {
            info!(
                "ailake_evolve_schema_json: schema evolved for {}.{} new_schema_id={}",
                req.namespace, req.table, schema_id
            );
            CString::new(
                serde_json::to_string(&Resp {
                    ok: true,
                    new_schema_id: schema_id,
                })
                .unwrap_or_default(),
            )
            .unwrap_or_default()
            .into_raw()
        }
        Err(e) => {
            warn!("ailake_evolve_schema_json: failed: {}", e);
            cstr_err_json(e)
        }
    }
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

    #[test]
    fn cabi_null_guard() {
        let ptr = unsafe { ailake_vector_search_json(std::ptr::null(), std::ptr::null(), 0, 10) };
        assert!(!ptr.is_null());
        let json = unsafe { CStr::from_ptr(ptr).to_str().unwrap().to_string() };
        assert_eq!(json, "[]");
        unsafe { ailake_free_string(ptr) };
    }

    // ── L1: partition_fields + format_version in write_batch_json ────────────

    #[test]
    fn write_batch_json_null_guard() {
        let ptr = unsafe { ailake_write_batch_json(std::ptr::null()) };
        assert!(!ptr.is_null());
        let json = unsafe { CStr::from_ptr(ptr).to_str().unwrap().to_string() };
        assert!(json.contains("error") || json.contains("null"));
        unsafe { ailake_free_string(ptr) };
    }

    #[test]
    fn write_batch_json_partition_fields_parses() {
        // JSON with partition_fields + format_version=3 must parse without panic.
        // We pass a non-existent warehouse so it fails at I/O, not at JSON parse.
        let req = r#"{
            "warehouse": "/nonexistent/path",
            "namespace": "default",
            "table": "test",
            "dim": 4,
            "format_version": 3,
            "partition_fields": [
                {"column": "agent_id", "transform": "identity", "column_type": "string"},
                {"column": "ts", "transform": "truncate[4]", "column_type": "string"}
            ],
            "ids": [1],
            "embeddings": [[0.1, 0.2, 0.3, 0.4]]
        }"#;
        let c = std::ffi::CString::new(req).unwrap();
        let ptr = unsafe { ailake_write_batch_json(c.as_ptr()) };
        assert!(!ptr.is_null());
        let json = unsafe { CStr::from_ptr(ptr).to_str().unwrap().to_string() };
        // Expect I/O error (not JSON parse error)
        assert!(json.contains("error"), "expected error json, got: {}", json);
        assert!(
            !json.contains("JSON parse"),
            "unexpected JSON parse error: {}",
            json
        );
        unsafe { ailake_free_string(ptr) };
    }

    // ── L2: ailake_delete_where_json ─────────────────────────────────────────

    #[test]
    fn delete_where_json_null_guard() {
        let ptr = unsafe { ailake_delete_where_json(std::ptr::null()) };
        assert!(!ptr.is_null());
        let json = unsafe { CStr::from_ptr(ptr).to_str().unwrap().to_string() };
        assert!(json.contains("error"));
        unsafe { ailake_free_string(ptr) };
    }

    #[test]
    fn delete_where_json_bad_warehouse_returns_error() {
        let req = r#"{
            "warehouse": "/nonexistent/warehouse",
            "table": "my_table",
            "column": "document_id",
            "values": ["doc-a", "doc-b"]
        }"#;
        let c = std::ffi::CString::new(req).unwrap();
        let ptr = unsafe { ailake_delete_where_json(c.as_ptr()) };
        assert!(!ptr.is_null());
        let json = unsafe { CStr::from_ptr(ptr).to_str().unwrap().to_string() };
        assert!(json.contains("error"), "expected error, got: {}", json);
        unsafe { ailake_free_string(ptr) };
    }

    // ── L3: ailake_evolve_schema_json ────────────────────────────────────────

    #[test]
    fn evolve_schema_json_null_guard() {
        let ptr = unsafe { ailake_evolve_schema_json(std::ptr::null()) };
        assert!(!ptr.is_null());
        let json = unsafe { CStr::from_ptr(ptr).to_str().unwrap().to_string() };
        assert!(json.contains("error"));
        unsafe { ailake_free_string(ptr) };
    }

    #[test]
    fn evolve_schema_json_bad_warehouse_returns_error() {
        let req = r#"{
            "warehouse": "/nonexistent/warehouse",
            "table": "my_table",
            "add_columns": [{"name": "score", "type": "float"}],
            "rename_columns": [{"from": "old", "to": "new"}]
        }"#;
        let c = std::ffi::CString::new(req).unwrap();
        let ptr = unsafe { ailake_evolve_schema_json(c.as_ptr()) };
        assert!(!ptr.is_null());
        let json = unsafe { CStr::from_ptr(ptr).to_str().unwrap().to_string() };
        assert!(json.contains("error"), "expected error, got: {}", json);
        unsafe { ailake_free_string(ptr) };
    }
}
