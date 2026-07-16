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

use ailake_catalog::{
    CatalogProvider, HadoopCatalog, RestCatalog, RestCatalogAuth, RestCatalogConfig, TableIdent,
};
use ailake_core::{EmbeddingModelInfo, VectorMetric};
use ailake_query::{
    fetch_rows as rs_fetch_rows, search as rs_search, search_multimodal as rs_search_multimodal,
    Chunk, CompactionConfig, CompactionExecutor, CompactionPlanner, ContextAssembler,
    ContextAssemblerConfig, FusionMethod, ModalQuery, SearchConfig, SearchResult,
};
use ailake_store::LocalStore;
use serde::Serialize;
use tracing::{debug, error, info, warn};

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

/// Catalog-selection fields shared by every JSON `Req`/`Opts` struct via
/// `#[serde(flatten)]`. `catalog` defaults to `"hadoop"` (unchanged behavior
/// for every existing caller — Spark/Trino/Flink plugins that never set this
/// field keep working exactly as before). `"rest"` talks to any Iceberg REST
/// Catalog spec server — see `docs/guides/REST_CATALOG.md`.
///
/// Deliberately NOT threaded into `do_search`/`ailake_vector_search_json`
/// (the raw-pointer legacy entry point with no JSON body to carry this
/// config) — that path stays Hadoop-only, matching its existing behavior.
#[derive(serde::Deserialize, Default)]
struct CatalogOpts {
    #[serde(default)]
    catalog: Option<String>,
    #[serde(default)]
    rest_uri: Option<String>,
    #[serde(default)]
    rest_prefix: Option<String>,
    #[serde(default)]
    rest_warehouse: Option<String>,
    #[serde(default)]
    rest_auth: Option<String>,
    #[serde(default)]
    rest_token: Option<String>,
    #[serde(default)]
    rest_oauth_token_endpoint: Option<String>,
    #[serde(default)]
    rest_oauth_client_id: Option<String>,
    #[serde(default)]
    rest_oauth_client_secret: Option<String>,
    #[serde(default)]
    rest_oauth_scope: Option<String>,
}

/// Builds the `CatalogProvider` a JSON request asked for. `warehouse` is
/// always used for `Store` resolution and as the Hadoop catalog root (see
/// `LocalStore::new(&warehouse)` at each call site) — unrelated to `opts`,
/// which only selects/configures the catalog *metadata* backend.
fn resolve_catalog(
    warehouse: &str,
    store: Arc<dyn ailake_store::Store>,
    opts: &CatalogOpts,
) -> Result<Arc<dyn CatalogProvider>, String> {
    match opts.catalog.as_deref().unwrap_or("hadoop") {
        "hadoop" => Ok(Arc::new(HadoopCatalog::new(store, warehouse))),
        "rest" => {
            let uri = opts
                .rest_uri
                .clone()
                .ok_or("catalog=\"rest\" requires \"rest_uri\"")?;
            let auth = match opts.rest_auth.as_deref().unwrap_or("none") {
                "none" => RestCatalogAuth::None,
                "bearer" => {
                    let token = opts
                        .rest_token
                        .clone()
                        .ok_or("rest_auth=\"bearer\" requires \"rest_token\"")?;
                    RestCatalogAuth::Bearer(token)
                }
                "oauth2" => {
                    let token_endpoint = opts
                        .rest_oauth_token_endpoint
                        .clone()
                        .ok_or("rest_auth=\"oauth2\" requires \"rest_oauth_token_endpoint\"")?;
                    let client_id = opts
                        .rest_oauth_client_id
                        .clone()
                        .ok_or("rest_auth=\"oauth2\" requires \"rest_oauth_client_id\"")?;
                    let client_secret = opts
                        .rest_oauth_client_secret
                        .clone()
                        .ok_or("rest_auth=\"oauth2\" requires \"rest_oauth_client_secret\"")?;
                    RestCatalogAuth::OAuth2 {
                        token_endpoint,
                        client_id,
                        client_secret,
                        scope: opts.rest_oauth_scope.clone(),
                    }
                }
                other => return Err(format!("unknown rest_auth: {other}")),
            };
            let config = RestCatalogConfig {
                uri,
                prefix: opts.rest_prefix.clone(),
                warehouse: opts.rest_warehouse.clone(),
                auth,
            };
            Ok(Arc::new(RestCatalog::new(config, store)))
        }
        other => Err(format!(
            "unknown catalog backend: {other} (supported: \"hadoop\", \"rest\")"
        )),
    }
}

/// Returns a per-table `Mutex` shared across all JNI calls in this process.
///
/// `HadoopCatalog` serializes commits via a per-instance mutex, but each JNI call
/// creates a fresh catalog instance — so concurrent calls for the same table race
/// on `metadata.json`. This static map provides the missing cross-call serialization,
/// preserving the "single-process" catalog isolation guarantee documented on HadoopCatalog.
fn jni_table_lock(warehouse: &str, namespace: &str, table: &str) -> Arc<std::sync::Mutex<()>> {
    use std::collections::HashMap;
    use std::sync::{Mutex, OnceLock};
    static LOCKS: OnceLock<Mutex<HashMap<String, Arc<std::sync::Mutex<()>>>>> = OnceLock::new();
    let map = LOCKS.get_or_init(|| Mutex::new(HashMap::new()));
    let mut guard = map.lock().unwrap_or_else(|e| e.into_inner());
    guard
        .entry(format!("{warehouse}/{namespace}/{table}"))
        .or_insert_with(|| Arc::new(std::sync::Mutex::new(())))
        .clone()
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
    pruning_threshold: f32,
    catalog_opts: &CatalogOpts,
) -> ailake_core::AilakeResult<Vec<SearchResult>> {
    let store: Arc<dyn ailake_store::Store> = Arc::new(LocalStore::new(&warehouse));
    let catalog = resolve_catalog(&warehouse, store.clone(), catalog_opts)
        .map_err(ailake_core::AilakeError::InvalidArgument)?;
    let table = TableIdent::new(namespace, table_name);
    let hybrid = hybrid_text.map(|qt| {
        ailake_query::HybridConfig::new(qt)
            .with_text_column(text_column)
            .with_bm25_weight(bm25_weight)
    });
    let config = SearchConfig {
        top_k: top_k as usize,
        ef_search: ef_search as usize,
        pruning_threshold,
        rerank_factor: None,
        score_fn: None,
        partition_filter,
        hybrid,
        column_filter: None,
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
    CString::new(s)
        .unwrap_or_else(|_| {
            CString::new(r#"{"ok":false,"error":"internal: error message contained null byte"}"#)
                .unwrap()
        })
        .into_raw()
}

fn cstr_json(s: String) -> *mut c_char {
    CString::new(s)
        .unwrap_or_else(|_| {
            CString::new(r#"{"ok":false,"error":"internal: JSON output contained null byte"}"#)
                .unwrap()
        })
        .into_raw()
}

fn panic_payload_message(payload: &(dyn std::any::Any + Send)) -> String {
    if let Some(s) = payload.downcast_ref::<&str>() {
        s.to_string()
    } else if let Some(s) = payload.downcast_ref::<String>() {
        s.clone()
    } else {
        "non-string panic payload".to_string()
    }
}

/// Runs `f` (a C-ABI entry point's body) inside `catch_unwind`, converting a panic
/// anywhere in the call chain (ailake-query/ailake-catalog included) into the same
/// `{"ok":false,"error":...}` envelope normal errors already use, instead of letting
/// it unwind across the FFI boundary — undefined behavior per the Rust reference, and
/// in practice an abort of the whole host process (JVM/DuckDB/Python) for a single
/// bad request.
fn catch_ffi_panic<F>(fn_name: &str, f: F) -> *mut c_char
where
    F: FnOnce() -> *mut c_char + std::panic::UnwindSafe,
{
    match std::panic::catch_unwind(f) {
        Ok(ptr) => ptr,
        Err(payload) => {
            let msg = panic_payload_message(&*payload);
            error!("ailake: panic caught in {fn_name}: {msg}");
            cstr_err_json(format!("internal panic in {fn_name}: {msg}"))
        }
    }
}

/// ailake-jni version string. Static — do NOT free.
#[no_mangle]
pub extern "C" fn ailake_version() -> *const c_char {
    static V: &str = concat!(env!("CARGO_PKG_VERSION"), "\0");
    V.as_ptr() as *const c_char
}

/// Search a local AI-Lake table.
///
/// Returns `{"ok":true,"results":[{"row_id":N,"distance":F,"file_path":"..."}]}` on success,
/// or `{"ok":false,"error":"..."}` on failure.
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
    catch_ffi_panic("ailake_vector_search_json", move || {
        if table_uri.is_null() || query_ptr.is_null() {
            return cstr_empty_json();
        }
        if query_len == 0 {
            return cstr_err_json("query_len must be > 0");
        }
        // u32::MAX as usize = 4B f32s = ~16 GB; guard prevents OOM from a buggy or malicious JNA caller.
        if query_len > 65_536 {
            return cstr_err_json(format!(
                "query_len {query_len} exceeds maximum supported dimension (65536)"
            ));
        }
        let uri = match unsafe { CStr::from_ptr(table_uri) }.to_str() {
            Ok(s) => s.to_string(),
            Err(e) => return cstr_err_json(format!("invalid UTF-8 in table_uri: {e}")),
        };
        let query = unsafe { std::slice::from_raw_parts(query_ptr, query_len as usize) }.to_vec();
        let dim = query.len() as u32;
        let results: Vec<RowResultJson> = match do_search(
            uri.clone(),
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
            f32::INFINITY,
            &CatalogOpts::default(),
        ) {
            Ok(v) => v.into_iter().map(RowResultJson::from).collect(),
            Err(e) => {
                warn!(
                    "ailake_vector_search_json: search failed table_uri={}: {}",
                    uri, e
                );
                return cstr_err_json(e);
            }
        };
        #[derive(serde::Serialize)]
        struct Resp {
            ok: bool,
            results: Vec<RowResultJson>,
        }
        let body = Resp { ok: true, results };
        let json = serde_json::to_string(&body)
            .unwrap_or_else(|_| r#"{"ok":false,"error":"serialize"}"#.into());
        CString::new(json)
            .unwrap_or_else(|_| {
                CString::new(r#"{"ok":false,"error":"null byte in output"}"#).unwrap()
            })
            .into_raw()
    })
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
    catch_ffi_panic("ailake_search_json", move || {
        #[derive(serde::Deserialize)]
        struct Req {
            warehouse: String,
            #[serde(flatten)]
            catalog_opts: CatalogOpts,
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
            #[serde(default)]
            pruning_threshold: Option<f32>,
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
        let json_str = match unsafe { CStr::from_ptr(request_json) }.to_str() {
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
        let pruning_threshold = req.pruning_threshold.unwrap_or(f32::INFINITY);
        let results = match do_search(
            req.warehouse.clone(),
            &req.namespace,
            &req.table,
            &req.vec_col,
            req.dim,
            req.query,
            req.top_k,
            req.ef_search.min(100_000),
            req.partition_filter,
            req.hybrid_text,
            &text_column,
            bm25_weight,
            pruning_threshold,
            &req.catalog_opts,
        ) {
            Ok(v) => v,
            Err(e) => {
                warn!(
                    "ailake_search_json: search failed warehouse={} table={}.{}: {}",
                    req.warehouse, req.namespace, req.table, e
                );
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
        cstr_json(json)
    })
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
    catch_ffi_panic("ailake_write_batch_json", move || {
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
            #[serde(flatten)]
            catalog_opts: CatalogOpts,
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
            /// Text columns to embed as Tantivy FTS index in the AILK_FTS section.
            /// Empty = no FTS (default, zero overhead).
            #[serde(default)]
            fts_columns: Vec<String>,
            /// Tantivy tokenizer for FTS (default: "default").
            #[serde(default = "default_fts_tokenizer")]
            fts_tokenizer: String,
            /// Per-table HNSW M parameter (graph connectivity). None = use default from HnswConfig.
            #[serde(default)]
            hnsw_m: Option<u32>,
            /// Per-table HNSW ef_construction parameter. None = use default from HnswConfig.
            #[serde(default)]
            hnsw_ef_construction: Option<u32>,
            /// Normalize vectors to unit L2 at write time (recommended for cosine).
            #[serde(default)]
            pre_normalize: bool,
            /// Build index in a background Tokio task (write_batch_auto_deferred).
            /// Parquet is committed immediately; index is appended asynchronously.
            #[serde(default)]
            deferred: bool,
            /// Extra string columns included in the Parquet batch (required for FTS).
            /// JSON: `{"text": ["row0 text", "row1 text", ...], "title": [...]}`
            #[serde(default)]
            columns: std::collections::HashMap<String, Vec<String>>,
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
        fn default_fts_tokenizer() -> String {
            "default".into()
        }

        if request_json.is_null() {
            return cstr_err_json("null request_json");
        }
        let json_str = match unsafe { CStr::from_ptr(request_json) }.to_str() {
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
            "ailake_write_batch_json: ids.len()={} != embeddings.len()={} warehouse={} table={}.{}",
            req.ids.len(),
            req.embeddings.len(),
            req.warehouse,
            req.namespace,
            req.table,
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
            pre_normalize: req.pre_normalize,
            hnsw_m: req.hnsw_m,
            hnsw_ef_construction: req.hnsw_ef_construction,
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
        let catalog = match resolve_catalog(&req.warehouse, store.clone(), &req.catalog_opts) {
            Ok(c) => c,
            Err(e) => return cstr_err_json(e),
        };

        use arrow_array::StringArray;
        let mut fields = vec![Field::new("id", DataType::Int64, false)];
        let mut arrays: Vec<std::sync::Arc<dyn arrow_array::Array>> =
            vec![std::sync::Arc::new(Int64Array::from(req.ids))];
        let mut ordered_cols: Vec<(String, Vec<String>)> = req.columns.into_iter().collect();
        ordered_cols.sort_by(|a, b| a.0.cmp(&b.0));
        for (col_name, values) in ordered_cols {
            fields.push(Field::new(&col_name, DataType::Utf8, true));
            arrays.push(std::sync::Arc::new(StringArray::from(values)));
        }
        let schema = std::sync::Arc::new(Schema::new(fields));
        let batch = match RecordBatch::try_new(schema, arrays) {
            Ok(b) => b,
            Err(e) => return cstr_err_json(e),
        };

        let fts_cfg: Option<ailake_fts::FtsConfig> = if req.fts_columns.is_empty() {
            None
        } else {
            Some(ailake_fts::FtsConfig {
                text_columns: req.fts_columns,
                tokenizer: req.fts_tokenizer,
                writer_heap_bytes: 50 * 1024 * 1024,
            })
        };

        let deferred = req.deferred;
        let _table_lock = jni_table_lock(&req.warehouse, &req.namespace, &req.table);
        let _commit_guard = _table_lock.lock().unwrap_or_else(|e| e.into_inner());
        let result = rt().block_on(async {
            let base =
                TableWriter::create_or_open(catalog, store, policy, table, format_version).await?;
            let mut writer = if let Some(cfg) = fts_cfg {
                base.with_fts_config(cfg)
            } else {
                base
            };
            if deferred {
                writer
                    .write_batch_auto_deferred(&batch, &req.embeddings)
                    .await?;
            } else {
                writer.write_batch_auto(&batch, &req.embeddings).await?;
            }
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
                serde_json::to_string(&Resp {
                    ok: true,
                    snapshot_id: snap,
                })
                .map(cstr_json)
                .unwrap_or_else(cstr_err_json)
            }
            Err(e) => {
                warn!("ailake_write_batch_json: write failed: {}", e);
                cstr_err_json(e)
            }
        }
    })
}

/// Extracts one row per top-level list element as `Vec<f32>` from an Arrow
/// `List<Float32>` or `FixedSizeList<Float32>` array — the two on-wire shapes
/// `ailake_write_batch_ipc` accepts for the vector column. `dim` is the
/// declared table dimension; every row's element count must match it exactly.
fn extract_embeddings_f32(
    arr: &dyn arrow_array::Array,
    dim: usize,
) -> Result<Vec<Vec<f32>>, String> {
    use arrow_array::{Array, FixedSizeListArray, Float32Array, ListArray};
    use arrow_schema::DataType;
    match arr.data_type() {
        DataType::List(_) => {
            let list = arr
                .as_any()
                .downcast_ref::<ListArray>()
                .ok_or_else(|| "vector column: expected ListArray".to_string())?;
            let values = list
                .values()
                .as_any()
                .downcast_ref::<Float32Array>()
                .ok_or_else(|| "vector column: List child must be Float32".to_string())?;
            let offsets = list.value_offsets();
            let mut out = Vec::with_capacity(list.len());
            for i in 0..list.len() {
                let start = offsets[i] as usize;
                let end = offsets[i + 1] as usize;
                if end - start != dim {
                    return Err(format!(
                        "vector column: row {i} has {} dims, expected {dim}",
                        end - start
                    ));
                }
                out.push(values.values()[start..end].to_vec());
            }
            Ok(out)
        }
        DataType::FixedSizeList(_, list_dim) => {
            if *list_dim as usize != dim {
                return Err(format!(
                    "vector column: FixedSizeList dim {list_dim} != declared dim {dim}"
                ));
            }
            let list = arr
                .as_any()
                .downcast_ref::<FixedSizeListArray>()
                .ok_or_else(|| "vector column: expected FixedSizeListArray".to_string())?;
            let values = list
                .values()
                .as_any()
                .downcast_ref::<Float32Array>()
                .ok_or_else(|| "vector column: FixedSizeList child must be Float32".to_string())?;
            let mut out = Vec::with_capacity(list.len());
            for i in 0..list.len() {
                let start = i * dim;
                out.push(values.values()[start..start + dim].to_vec());
            }
            Ok(out)
        }
        other => Err(format!(
            "vector column: unsupported Arrow type {other:?}, expected List<Float32> or FixedSizeList<Float32>"
        )),
    }
}

/// Projects `batch` onto every column except `drop_idx` — used to strip the
/// vector column back out of the decoded IPC batch before handing the
/// remaining (id + extra text columns) batch to `TableWriter`, which expects
/// the vector column supplied separately as `&[Vec<f32>]`.
fn project_dropping(
    batch: &arrow_array::RecordBatch,
    drop_idx: usize,
) -> Result<arrow_array::RecordBatch, arrow_schema::ArrowError> {
    let keep: Vec<usize> = (0..batch.num_columns())
        .filter(|&i| i != drop_idx)
        .collect();
    batch.project(&keep)
}

/// Write a batch of records to an AI-Lake table — Arrow IPC variant of
/// `ailake_write_batch_json` (Fase 10, ADR-017).
///
/// Replaces the JSON `"embeddings": [[...], ...]` payload (measured ~150ms of
/// JVM-side `Float.toString` formatting + UTF-8 encode per 1k×1536-dim batch)
/// with a single Arrow IPC **stream** RecordBatch carrying `id` (Int64), the
/// vector column (`List<Float32>` or `FixedSizeList<Float32>`, named to match
/// `vec_col` in `opts_json`), and any extra text columns (Utf8) — the same
/// shape `ailake_write_batch_json` builds internally from `ids`/`embeddings`/
/// `columns`, just arriving pre-built instead of hand-assembled from JSON.
///
/// `opts_json` carries every field `ailake_write_batch_json`'s `Req` has
/// *except* `ids`/`embeddings`/`columns` (those three live in the IPC batch
/// instead):
/// ```json
/// {
///   "warehouse": "/path/to/warehouse", "namespace": "default", "table": "docs",
///   "vec_col": "embedding", "dim": 1536, "metric": "cosine", "precision": "f16"
/// }
/// ```
///
/// Returns JSON: `{"ok":true,"snapshot_id":N}` or `{"ok":false,"error":"..."}`.
///
/// # Safety
/// `ipc_bytes` must point to `ipc_len` valid bytes (an Arrow IPC stream with
/// exactly one RecordBatch). `ipc_len` is `i64` rather than `usize` deliberately —
/// a fixed 8-byte width avoids any ambiguity against JNA's `long` marshalling
/// (Java `long` → native `long`, 8 bytes on the 64-bit Linux targets this
/// project runs on; `usize` would carry the same width here today, but pinning
/// the FFI-facing type to a fixed-width integer keeps that an implementation
/// detail rather than a load-bearing assumption of the C-ABI contract).
/// Caller must free the returned pointer with `ailake_free_string`.
#[no_mangle]
pub unsafe extern "C" fn ailake_write_batch_ipc(
    ipc_bytes: *const u8,
    ipc_len: i64,
    opts_json: *const c_char,
) -> *mut c_char {
    catch_ffi_panic("ailake_write_batch_ipc", move || {
        use ailake_core::{PartitionDef, VectorPrecision, VectorStoragePolicy};
        use ailake_query::TableWriter;

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
        struct Opts {
            warehouse: String,
            #[serde(flatten)]
            catalog_opts: CatalogOpts,
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
            #[serde(default)]
            embedding_model: Option<String>,
            #[serde(default)]
            partition_by: Option<String>,
            #[serde(default)]
            partition_value: Option<String>,
            #[serde(default)]
            partition_fields: Vec<PartitionFieldReq>,
            #[serde(default = "default_format_version")]
            format_version: u8,
            #[serde(default)]
            fts_columns: Vec<String>,
            #[serde(default = "default_fts_tokenizer")]
            fts_tokenizer: String,
            #[serde(default)]
            hnsw_m: Option<u32>,
            #[serde(default)]
            hnsw_ef_construction: Option<u32>,
            #[serde(default)]
            pre_normalize: bool,
            #[serde(default)]
            deferred: bool,
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
        fn default_fts_tokenizer() -> String {
            "default".into()
        }

        if opts_json.is_null() {
            return cstr_err_json("null opts_json");
        }
        let json_str = match unsafe { CStr::from_ptr(opts_json) }.to_str() {
            Ok(s) => s,
            Err(e) => {
                warn!("ailake_write_batch_ipc: invalid UTF-8 in opts_json: {}", e);
                return cstr_err_json(e);
            }
        };
        let opts: Opts = match serde_json::from_str(json_str) {
            Ok(o) => o,
            Err(e) => {
                warn!("ailake_write_batch_ipc: opts_json parse error: {}", e);
                return cstr_err_json(e);
            }
        };

        if ipc_bytes.is_null() {
            return cstr_err_json("null ipc_bytes");
        }
        if ipc_len < 0 {
            return cstr_err_json("negative ipc_len");
        }
        let raw: &[u8] = unsafe { std::slice::from_raw_parts(ipc_bytes, ipc_len as usize) };
        let mut reader =
            match arrow_ipc::reader::StreamReader::try_new(std::io::Cursor::new(raw), None) {
                Ok(r) => r,
                Err(e) => {
                    warn!("ailake_write_batch_ipc: IPC stream header error: {}", e);
                    return cstr_err_json(e);
                }
            };
        let batch = match reader.next() {
            Some(Ok(b)) => b,
            Some(Err(e)) => {
                warn!("ailake_write_batch_ipc: IPC batch decode error: {}", e);
                return cstr_err_json(e);
            }
            None => return cstr_err_json("empty IPC stream: no RecordBatch"),
        };

        let vec_idx = match batch.schema().index_of(&opts.vec_col) {
            Ok(i) => i,
            Err(_) => {
                return cstr_err_json(format!(
                    "vector column '{}' not found in IPC batch schema",
                    opts.vec_col
                ))
            }
        };
        let embeddings =
            match extract_embeddings_f32(batch.column(vec_idx).as_ref(), opts.dim as usize) {
                Ok(v) => v,
                Err(e) => return cstr_err_json(e),
            };
        let reduced = match project_dropping(&batch, vec_idx) {
            Ok(b) => b,
            Err(e) => return cstr_err_json(e),
        };

        debug!(
            "ailake_write_batch_ipc: warehouse={} table={}.{} rows={}",
            opts.warehouse,
            opts.namespace,
            opts.table,
            embeddings.len()
        );

        let metric = parse_metric(opts.metric.as_deref().unwrap_or("euclidean"));
        let precision = match opts.precision.as_deref().unwrap_or("f16") {
            "f32" => VectorPrecision::F32,
            "i8" => VectorPrecision::I8,
            _ => VectorPrecision::F16,
        };
        let embedding_model = opts
            .embedding_model
            .as_deref()
            .map(EmbeddingModelInfo::from_property_value);
        let partition_fields: Vec<PartitionDef> = opts
            .partition_fields
            .into_iter()
            .map(|pf| PartitionDef {
                column: pf.column,
                transform: pf.transform,
                column_type: pf.column_type,
            })
            .collect();

        let policy = VectorStoragePolicy {
            column_name: opts.vec_col.clone(),
            dim: opts.dim,
            metric,
            precision,
            pq: None,
            keep_raw_for_reranking: true,
            pre_normalize: opts.pre_normalize,
            hnsw_m: opts.hnsw_m,
            hnsw_ef_construction: opts.hnsw_ef_construction,
            ivf_residual: opts.ivf_residual,
            embedding_model,
            modality: None,
            partition_by: opts.partition_by,
            partition_value: opts.partition_value,
            partition_column_type: None,
            partition_fields,
        };

        let format_version = opts.format_version;
        let table = ailake_catalog::TableIdent::new(&opts.namespace, &opts.table);
        let store: std::sync::Arc<dyn ailake_store::Store> =
            std::sync::Arc::new(LocalStore::new(&opts.warehouse));
        let catalog = match resolve_catalog(&opts.warehouse, store.clone(), &opts.catalog_opts) {
            Ok(c) => c,
            Err(e) => return cstr_err_json(e),
        };

        let fts_cfg: Option<ailake_fts::FtsConfig> = if opts.fts_columns.is_empty() {
            None
        } else {
            Some(ailake_fts::FtsConfig {
                text_columns: opts.fts_columns,
                tokenizer: opts.fts_tokenizer,
                writer_heap_bytes: 50 * 1024 * 1024,
            })
        };

        let deferred = opts.deferred;
        let _table_lock = jni_table_lock(&opts.warehouse, &opts.namespace, &opts.table);
        let _commit_guard = _table_lock.lock().unwrap_or_else(|e| e.into_inner());
        let result = rt().block_on(async {
            let base =
                TableWriter::create_or_open(catalog, store, policy, table, format_version).await?;
            let mut writer = if let Some(cfg) = fts_cfg {
                base.with_fts_config(cfg)
            } else {
                base
            };
            if deferred {
                writer
                    .write_batch_auto_deferred(&reduced, &embeddings)
                    .await?;
            } else {
                writer.write_batch_auto(&reduced, &embeddings).await?;
            }
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
                    "ailake_write_batch_ipc: committed snapshot_id={} table={}.{}",
                    snap, opts.namespace, opts.table
                );
                serde_json::to_string(&Resp {
                    ok: true,
                    snapshot_id: snap,
                })
                .map(cstr_json)
                .unwrap_or_else(cstr_err_json)
            }
            Err(e) => {
                warn!("ailake_write_batch_ipc: write failed: {}", e);
                cstr_err_json(e)
            }
        }
    })
}

/// Write a batch with N independent vector columns into a single AI-Lake file.
///
/// Each column gets its own HNSW section in the file footer (Phase 8 multimodal
/// tables — e.g. text + image embeddings on the same row). The first entry in
/// `vector_columns` is the primary column, used for geometric pruning in the
/// manifest. This is the JNI-surface equivalent of `ailake-py`'s
/// `TableWriter.write_batch_multi` — previously only reachable from Python via
/// PyO3, with no C-ABI path for Trino/Spark/Flink to populate a multi-vector
/// table (they could call `ailake_search_multimodal_json` but never write the
/// data it searches).
///
/// `request_json`:
/// ```json
/// {
///   "warehouse": "...", "namespace": "default", "table": "docs",
///   "ids": [1, 2, 3],
///   "vector_columns": [
///     {"col": "embedding", "dim": 1536, "metric": "cosine", "precision": "f16",
///      "modality": "text", "embeddings": [[...], [...], [...]]},
///     {"col": "image_embedding", "dim": 512, "metric": "cosine",
///      "modality": "image", "embeddings": [[...], [...], [...]]}
///   ],
///   "columns": {"chunk_text": ["...", "...", "..."]},
///   "format_version": 2, "deferred": false
/// }
/// ```
/// Returns `{"ok":true,"snapshot_id":N}` on success, or `{"ok":false,"error":"..."}`.
///
/// # Safety
/// Caller must free the returned pointer with `ailake_free_string`.
#[no_mangle]
pub unsafe extern "C" fn ailake_write_batch_multi_json(request_json: *const c_char) -> *mut c_char {
    catch_ffi_panic("ailake_write_batch_multi_json", move || {
        use ailake_core::{VectorModality, VectorPrecision, VectorStoragePolicy};
        use ailake_query::{MultiVectorBatch, TableWriter};
        use arrow_array::{Int64Array, RecordBatch};
        use arrow_schema::{DataType, Field, Schema};

        #[derive(serde::Deserialize)]
        struct VecColReq {
            col: String,
            dim: u32,
            #[serde(default)]
            metric: Option<String>,
            #[serde(default)]
            precision: Option<String>,
            #[serde(default)]
            modality: Option<String>,
            embeddings: Vec<Vec<f32>>,
        }

        #[derive(serde::Deserialize)]
        struct Req {
            warehouse: String,
            #[serde(flatten)]
            catalog_opts: CatalogOpts,
            #[serde(default = "default_ns")]
            namespace: String,
            table: String,
            ids: Vec<i64>,
            vector_columns: Vec<VecColReq>,
            #[serde(default)]
            embedding_model: Option<String>,
            #[serde(default = "default_format_version")]
            format_version: u8,
            #[serde(default)]
            fts_columns: Vec<String>,
            #[serde(default = "default_fts_tokenizer")]
            fts_tokenizer: String,
            #[serde(default)]
            deferred: bool,
            #[serde(default)]
            columns: std::collections::HashMap<String, Vec<String>>,
        }
        fn default_ns() -> String {
            "default".into()
        }
        fn default_format_version() -> u8 {
            2
        }
        fn default_fts_tokenizer() -> String {
            "default".into()
        }

        if request_json.is_null() {
            return cstr_err_json("null request_json");
        }
        let json_str = match unsafe { CStr::from_ptr(request_json) }.to_str() {
            Ok(s) => s,
            Err(e) => {
                warn!(
                    "ailake_write_batch_multi_json: invalid UTF-8 in request_json: {}",
                    e
                );
                return cstr_err_json(e);
            }
        };
        let req: Req = match serde_json::from_str(json_str) {
            Ok(r) => r,
            Err(e) => {
                warn!("ailake_write_batch_multi_json: JSON parse error: {}", e);
                return cstr_err_json(e);
            }
        };
        if req.vector_columns.is_empty() {
            return cstr_err_json("vector_columns must not be empty");
        }
        for vc in &req.vector_columns {
            if vc.embeddings.len() != req.ids.len() {
                warn!(
                    "ailake_write_batch_multi_json: ids.len()={} != embeddings.len()={} for column='{}' warehouse={} table={}.{}",
                    req.ids.len(), vc.embeddings.len(), vc.col, req.warehouse, req.namespace, req.table,
                );
                return cstr_err_json(format!(
                    "ids.len() != embeddings.len() for vector column '{}'",
                    vc.col
                ));
            }
        }
        debug!(
            "ailake_write_batch_multi_json: warehouse={} table={}.{} rows={} vector_columns={}",
            req.warehouse,
            req.namespace,
            req.table,
            req.ids.len(),
            req.vector_columns.len(),
        );

        let embedding_model = req
            .embedding_model
            .as_deref()
            .map(EmbeddingModelInfo::from_property_value);

        let policies: Vec<VectorStoragePolicy> = req
            .vector_columns
            .iter()
            .map(|vc| {
                let metric = parse_metric(vc.metric.as_deref().unwrap_or("euclidean"));
                let precision = match vc.precision.as_deref().unwrap_or("f16") {
                    "f32" => VectorPrecision::F32,
                    "i8" => VectorPrecision::I8,
                    _ => VectorPrecision::F16,
                };
                let modality = vc
                    .modality
                    .as_deref()
                    .and_then(|s| s.parse::<VectorModality>().ok());
                VectorStoragePolicy {
                    column_name: vc.col.clone(),
                    dim: vc.dim,
                    metric,
                    precision,
                    pq: None,
                    keep_raw_for_reranking: true,
                    pre_normalize: false,
                    hnsw_m: None,
                    hnsw_ef_construction: None,
                    ivf_residual: false,
                    embedding_model: embedding_model.clone(),
                    modality,
                    partition_by: None,
                    partition_value: None,
                    partition_column_type: None,
                    partition_fields: vec![],
                }
            })
            .collect();

        let mv_batches: Vec<MultiVectorBatch<'_>> = policies
            .iter()
            .zip(req.vector_columns.iter())
            .map(|(policy, vc)| MultiVectorBatch {
                policy: policy.clone(),
                embeddings: &vc.embeddings,
            })
            .collect();

        let format_version = req.format_version;
        let table = ailake_catalog::TableIdent::new(&req.namespace, &req.table);
        let store: std::sync::Arc<dyn ailake_store::Store> =
            std::sync::Arc::new(LocalStore::new(&req.warehouse));
        let catalog = match resolve_catalog(&req.warehouse, store.clone(), &req.catalog_opts) {
            Ok(c) => c,
            Err(e) => return cstr_err_json(e),
        };

        use arrow_array::StringArray;
        let mut fields = vec![Field::new("id", DataType::Int64, false)];
        let mut arrays: Vec<std::sync::Arc<dyn arrow_array::Array>> =
            vec![std::sync::Arc::new(Int64Array::from(req.ids))];
        let mut ordered_cols: Vec<(String, Vec<String>)> = req.columns.into_iter().collect();
        ordered_cols.sort_by(|a, b| a.0.cmp(&b.0));
        for (col_name, values) in ordered_cols {
            fields.push(Field::new(&col_name, DataType::Utf8, true));
            arrays.push(std::sync::Arc::new(StringArray::from(values)));
        }
        let schema = std::sync::Arc::new(Schema::new(fields));
        let batch = match RecordBatch::try_new(schema, arrays) {
            Ok(b) => b,
            Err(e) => return cstr_err_json(e),
        };

        let fts_cfg: Option<ailake_fts::FtsConfig> = if req.fts_columns.is_empty() {
            None
        } else {
            Some(ailake_fts::FtsConfig {
                text_columns: req.fts_columns,
                tokenizer: req.fts_tokenizer,
                writer_heap_bytes: 50 * 1024 * 1024,
            })
        };

        let deferred = req.deferred;
        let primary_policy = policies[0].clone();
        let _table_lock = jni_table_lock(&req.warehouse, &req.namespace, &req.table);
        let _commit_guard = _table_lock.lock().unwrap_or_else(|e| e.into_inner());
        let result = rt().block_on(async {
            let base =
                TableWriter::create_or_open(catalog, store, primary_policy, table, format_version)
                    .await?;
            let mut writer = if let Some(cfg) = fts_cfg {
                base.with_fts_config(cfg)
            } else {
                base
            };
            if deferred {
                writer
                    .write_batch_multi_deferred(&batch, &mv_batches)
                    .await?;
            } else {
                writer.write_batch_multi(&batch, &mv_batches).await?;
            }
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
                    "ailake_write_batch_multi_json: committed snapshot_id={} table={}.{} columns={}",
                    snap, req.namespace, req.table, req.vector_columns.len()
                );
                serde_json::to_string(&Resp {
                    ok: true,
                    snapshot_id: snap,
                })
                .map(cstr_json)
                .unwrap_or_else(cstr_err_json)
            }
            Err(e) => {
                warn!("ailake_write_batch_multi_json: write failed: {}", e);
                cstr_err_json(e)
            }
        }
    })
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
    catch_ffi_panic("ailake_search_text_json", move || {
        use ailake_query::search_text as rs_search_text;

        #[derive(serde::Deserialize)]
        struct Req {
            warehouse: String,
            #[serde(flatten)]
            catalog_opts: CatalogOpts,
            #[serde(default = "default_ns_st")]
            namespace: String,
            table: String,
            query_text: String,
            #[serde(default = "default_topk_st")]
            top_k: u32,
            /// Legacy single-column field. Ignored when `text_columns` is non-empty.
            #[serde(default = "default_text_col_st")]
            text_column: String,
            /// Multi-column text search (preferred). Falls back to `text_column` when empty.
            #[serde(default)]
            text_columns: Vec<String>,
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
        let json_str = match unsafe { CStr::from_ptr(request_json) }.to_str() {
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
            req.query_text.chars().take(60).collect::<String>(),
            req.top_k
        );

        let store: Arc<dyn ailake_store::Store> = Arc::new(LocalStore::new(&req.warehouse));
        let catalog = match resolve_catalog(&req.warehouse, store.clone(), &req.catalog_opts) {
            Ok(c) => c,
            Err(e) => return cstr_err_json(e),
        };
        let table = TableIdent::new(&req.namespace, &req.table);
        let pf = req.partition_filter.as_deref();
        // Prefer multi-column spec; fall back to legacy single-column field.
        let cols_owned: Vec<String> = if req.text_columns.is_empty() {
            vec![req.text_column]
        } else {
            req.text_columns
        };
        let cols_refs: Vec<&str> = cols_owned.iter().map(String::as_str).collect();
        let results = match rt().block_on(rs_search_text(
            &table,
            &req.query_text,
            &cols_refs,
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
        cstr_json(json)
    })
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
    catch_ffi_panic("ailake_search_multimodal_json", move || {
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
            #[serde(flatten)]
            catalog_opts: CatalogOpts,
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
        let json_str = match unsafe { CStr::from_ptr(request_json) }.to_str() {
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
        let catalog = match resolve_catalog(&req.warehouse, store.clone(), &req.catalog_opts) {
            Ok(c) => c,
            Err(e) => return cstr_err_json(e),
        };

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
        cstr_json(json)
    })
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
    catch_ffi_panic("ailake_scan_json", move || {
        #[derive(serde::Deserialize)]
        struct Req {
            warehouse: String,
            #[serde(flatten)]
            catalog_opts: CatalogOpts,
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
        let json_str = match unsafe { CStr::from_ptr(request_json) }.to_str() {
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
            req.ef_search.min(100_000),
            req.partition_filter,
            None,
            "",
            0.0,
            f32::INFINITY,
            &req.catalog_opts,
        ) {
            Ok(v) => v,
            Err(e) => {
                warn!("ailake_scan_json: search failed: {}", e);
                return cstr_err_json(e);
            }
        };

        // Separate store for fetching row data (do_search owns its own store internally).
        let store: Arc<dyn ailake_store::Store> = Arc::new(LocalStore::new(&req.warehouse));

        // Current Iceberg schema — so a file written before a metadata-only
        // evolve_schema/add_column still gets the new column projected in as null
        // instead of silently omitted (see fetch_rows's doc comment).
        let scan_catalog = match resolve_catalog(&req.warehouse, store.clone(), &req.catalog_opts) {
            Ok(c) => c,
            Err(e) => return cstr_err_json(e),
        };
        let scan_table = TableIdent::new(&req.namespace, &req.table);
        let schema_fields = match rt().block_on(scan_catalog.load_table(&scan_table)) {
            Ok(meta) => meta.schema_fields,
            Err(e) => {
                warn!(
                    "ailake_scan_json: load_table for schema fields failed: {}",
                    e
                );
                Vec::new()
            }
        };

        let batch = match rt().block_on(rs_fetch_rows(
            &results,
            store,
            &req.vec_col,
            req.dim,
            &schema_fields,
        )) {
            Ok(b) => b,
            Err(e) => {
                warn!("ailake_scan_json: fetch_rows failed: {}", e);
                return cstr_err_json(e);
            }
        };

        match record_batch_to_scan_json(&batch) {
            Ok(json) => cstr_json(json),
            Err(e) => {
                warn!("ailake_scan_json: serialization failed: {}", e);
                cstr_err_json(e)
            }
        }
    })
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
    catch_ffi_panic("ailake_delete_where_json", move || {
        use ailake_query::delete_where;

        #[derive(serde::Deserialize)]
        struct Req {
            warehouse: String,
            #[serde(flatten)]
            catalog_opts: CatalogOpts,
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
        let json_str = match unsafe { CStr::from_ptr(request_json) }.to_str() {
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
        let catalog = match resolve_catalog(&req.warehouse, store.clone(), &req.catalog_opts) {
            Ok(c) => c,
            Err(e) => return cstr_err_json(e),
        };
        let table = TableIdent::new(&req.namespace, &req.table);
        let values_ref: Vec<&str> = req.values.iter().map(String::as_str).collect();

        let _table_lock = jni_table_lock(&req.warehouse, &req.namespace, &req.table);
        let _commit_guard = _table_lock.lock().unwrap_or_else(|e| e.into_inner());
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
                serde_json::to_string(&Resp { ok: true })
                    .map(cstr_json)
                    .unwrap_or_else(cstr_err_json)
            }
            Err(e) => {
                warn!("ailake_delete_where_json: failed: {}", e);
                cstr_err_json(e)
            }
        }
    })
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
    catch_ffi_panic("ailake_evolve_schema_json", move || {
        use ailake_catalog::{AddColumnRequest, SchemaEvolution};

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
            #[serde(flatten)]
            catalog_opts: CatalogOpts,
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
        let json_str = match unsafe { CStr::from_ptr(request_json) }.to_str() {
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
        let catalog = match resolve_catalog(&req.warehouse, store.clone(), &req.catalog_opts) {
            Ok(c) => c,
            Err(e) => return cstr_err_json(e),
        };
        let table = TableIdent::new(&req.namespace, &req.table);

        let _table_lock = jni_table_lock(&req.warehouse, &req.namespace, &req.table);
        let _commit_guard = _table_lock.lock().unwrap_or_else(|e| e.into_inner());
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
                serde_json::to_string(&Resp {
                    ok: true,
                    new_schema_id: schema_id,
                })
                .map(cstr_json)
                .unwrap_or_else(cstr_err_json)
            }
            Err(e) => {
                warn!("ailake_evolve_schema_json: failed: {}", e);
                cstr_err_json(e)
            }
        }
    })
}

/// Compact small files in an AI-Lake table into a larger merged file.
///
/// `request_json` must be UTF-8 JSON:
/// ```json
/// {
///   "warehouse": "/path/to/warehouse",
///   "namespace": "default",
///   "table": "my_table",
///   "min_files": 4,
///   "target_size_bytes": 134217728,
///   "max_files_per_pass": 20,
///   "deferred": false
/// }
/// ```
///
/// Returns `{"ok":true,"files_compacted":N,"output_path":"..."}` or
/// `{"ok":true,"files_compacted":0}` when nothing to compact.
///
/// # Safety
/// Caller must free the returned pointer with `ailake_free_string`.
#[no_mangle]
pub unsafe extern "C" fn ailake_compact_json(request_json: *const c_char) -> *mut c_char {
    catch_ffi_panic("ailake_compact_json", move || {
        #[derive(serde::Deserialize)]
        struct Req {
            warehouse: String,
            #[serde(flatten)]
            catalog_opts: CatalogOpts,
            #[serde(default = "compact_default_ns")]
            namespace: String,
            table: String,
            #[serde(default)]
            min_files: Option<usize>,
            #[serde(default)]
            target_size_bytes: Option<u64>,
            #[serde(default)]
            max_files_per_pass: Option<usize>,
            #[serde(default)]
            deferred: bool,
        }
        fn compact_default_ns() -> String {
            "default".into()
        }

        if request_json.is_null() {
            return cstr_err_json("null request_json");
        }
        let json_str = match unsafe { CStr::from_ptr(request_json) }.to_str() {
            Ok(s) => s,
            Err(e) => return cstr_err_json(e),
        };
        let req: Req = match serde_json::from_str(json_str) {
            Ok(r) => r,
            Err(e) => {
                warn!("ailake_compact_json: JSON parse error: {}", e);
                return cstr_err_json(e);
            }
        };

        debug!(
            "ailake_compact_json: warehouse={} table={}.{} deferred={}",
            req.warehouse, req.namespace, req.table, req.deferred
        );

        let store: Arc<dyn ailake_store::Store> = Arc::new(LocalStore::new(&req.warehouse));
        let catalog = match resolve_catalog(&req.warehouse, store.clone(), &req.catalog_opts) {
            Ok(c) => c,
            Err(e) => return cstr_err_json(e),
        };
        let table = TableIdent::new(&req.namespace, &req.table);

        let _table_lock = jni_table_lock(&req.warehouse, &req.namespace, &req.table);
        let _commit_guard = _table_lock.lock().unwrap_or_else(|e| e.into_inner());
        let result = rt().block_on(async {
            let meta = catalog.load_table(&table).await?;

            let dim: u32 = meta
                .properties
                .get("ailake.vector-dim")
                .and_then(|v| v.parse().ok())
                .unwrap_or(128);
            let vec_col = meta
                .properties
                .get("ailake.vector-column")
                .cloned()
                .unwrap_or_else(|| "embedding".into());
            let metric = parse_metric(
                meta.properties
                    .get("ailake.vector-metric")
                    .map(|s| s.as_str())
                    .unwrap_or("cosine"),
            );
            let pre_normalize = meta
                .properties
                .get("ailake.pre-normalize")
                .map(|s| s == "true")
                .unwrap_or(false);
            let hnsw_m = meta
                .properties
                .get("ailake.hnsw-m")
                .and_then(|s| s.parse().ok());
            let hnsw_ef_construction = meta
                .properties
                .get("ailake.hnsw-ef-construction")
                .and_then(|s| s.parse().ok());

            let policy = ailake_core::VectorStoragePolicy {
                column_name: vec_col,
                dim,
                metric,
                precision: ailake_core::VectorPrecision::F16,
                pq: None,
                keep_raw_for_reranking: true,
                pre_normalize,
                hnsw_m,
                hnsw_ef_construction,
                ivf_residual: false,
                embedding_model: None,
                modality: None,
                partition_by: None,
                partition_value: None,
                partition_column_type: None,
                partition_fields: vec![],
            };

            let config = CompactionConfig {
                min_files_to_compact: req.min_files.unwrap_or(4),
                target_file_size_bytes: req.target_size_bytes.unwrap_or(128 * 1024 * 1024),
                index_strategy: ailake_query::compaction::CompactionIndexStrategy::Auto,
                max_files_per_pass: req.max_files_per_pass.unwrap_or(20),
            };
            let planner = CompactionPlanner::new(config);
            let executor = CompactionExecutor::new(store.clone(), policy);

            let output_prefix = "data";
            if req.deferred {
                executor
                    .run_deferred(&planner, &table, catalog, output_prefix)
                    .await
            } else {
                executor.run(&planner, &table, catalog, output_prefix).await
            }
        });

        #[derive(serde::Serialize)]
        struct Resp {
            ok: bool,
            files_compacted: usize,
            #[serde(skip_serializing_if = "Option::is_none")]
            output_path: Option<String>,
        }
        match result {
            Ok(Some(entry)) => {
                info!(
                    "ailake_compact_json: compaction done table={}.{} output={}",
                    req.namespace, req.table, entry.path
                );
                serde_json::to_string(&Resp {
                    ok: true,
                    files_compacted: 1,
                    output_path: Some(entry.path),
                })
                .map(cstr_json)
                .unwrap_or_else(cstr_err_json)
            }
            Ok(None) => {
                info!(
                    "ailake_compact_json: nothing to compact for table={}.{}",
                    req.namespace, req.table
                );
                serde_json::to_string(&Resp {
                    ok: true,
                    files_compacted: 0,
                    output_path: None,
                })
                .map(cstr_json)
                .unwrap_or_else(cstr_err_json)
            }
            Err(e) => {
                warn!("ailake_compact_json: compaction failed: {}", e);
                cstr_err_json(e)
            }
        }
    })
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── catch_ffi_panic ────────────────────────────────────────────────────────
    //
    // Regression: none of the extern "C" entry points caught a panic from the called
    // ailake-query/ailake-catalog stack — it would unwind across the FFI boundary
    // (undefined behavior per the Rust reference) instead of becoming a normal
    // {"ok":false,"error":...} response. Every ailake_*_json fn now wraps its body in
    // catch_ffi_panic; these tests cover the wrapper itself directly.

    #[test]
    fn catch_ffi_panic_converts_str_panic_to_error_json() {
        let ptr = catch_ffi_panic("test_fn", || panic!("boom"));
        assert!(!ptr.is_null());
        let json = unsafe { CStr::from_ptr(ptr).to_str().unwrap().to_string() };
        unsafe { ailake_free_string(ptr) };
        assert!(json.contains("\"ok\":false"), "got: {json}");
        assert!(json.contains("boom"), "got: {json}");
        assert!(json.contains("test_fn"), "got: {json}");
    }

    #[test]
    fn catch_ffi_panic_converts_string_panic_to_error_json() {
        let ptr = catch_ffi_panic("test_fn2", || panic!("{}", format!("dyn {}", "boom")));
        assert!(!ptr.is_null());
        let json = unsafe { CStr::from_ptr(ptr).to_str().unwrap().to_string() };
        unsafe { ailake_free_string(ptr) };
        assert!(json.contains("dyn boom"), "got: {json}");
    }

    #[test]
    fn catch_ffi_panic_passes_through_normal_return() {
        let ptr = catch_ffi_panic("test_fn3", || cstr_json(r#"{"ok":true}"#.to_string()));
        assert!(!ptr.is_null());
        let json = unsafe { CStr::from_ptr(ptr).to_str().unwrap().to_string() };
        unsafe { ailake_free_string(ptr) };
        assert_eq!(json, r#"{"ok":true}"#);
    }

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
    fn write_batch_json_new_params_parse() {
        // hnsw_m, hnsw_ef_construction, pre_normalize, deferred must parse without panic.
        let req = r#"{
            "warehouse": "/nonexistent/path",
            "namespace": "default",
            "table": "test",
            "dim": 4,
            "hnsw_m": 32,
            "hnsw_ef_construction": 200,
            "pre_normalize": true,
            "deferred": true,
            "ids": [1],
            "embeddings": [[0.1, 0.2, 0.3, 0.4]]
        }"#;
        let c = std::ffi::CString::new(req).unwrap();
        let ptr = unsafe { ailake_write_batch_json(c.as_ptr()) };
        assert!(!ptr.is_null());
        let json = unsafe { CStr::from_ptr(ptr).to_str().unwrap().to_string() };
        assert!(json.contains("error"), "expected error json, got: {}", json);
        assert!(
            !json.contains("JSON parse"),
            "unexpected JSON parse error: {}",
            json
        );
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

    // ── write_batch_multi_json ────────────────────────────────────────────────

    #[test]
    fn write_batch_multi_json_null_guard() {
        let ptr = unsafe { ailake_write_batch_multi_json(std::ptr::null()) };
        assert!(!ptr.is_null());
        let json = unsafe { CStr::from_ptr(ptr).to_str().unwrap().to_string() };
        assert!(json.contains("error") || json.contains("null"));
        unsafe { ailake_free_string(ptr) };
    }

    #[test]
    fn write_batch_multi_json_rejects_empty_vector_columns() {
        let req = r#"{
            "warehouse": "/nonexistent/path",
            "namespace": "default",
            "table": "test",
            "ids": [1],
            "vector_columns": []
        }"#;
        let c = std::ffi::CString::new(req).unwrap();
        let ptr = unsafe { ailake_write_batch_multi_json(c.as_ptr()) };
        assert!(!ptr.is_null());
        let json = unsafe { CStr::from_ptr(ptr).to_str().unwrap().to_string() };
        assert!(json.contains("vector_columns"), "got: {json}");
        unsafe { ailake_free_string(ptr) };
    }

    #[test]
    fn write_batch_multi_json_rejects_mismatched_embedding_length() {
        let req = r#"{
            "warehouse": "/nonexistent/path",
            "namespace": "default",
            "table": "test",
            "ids": [1, 2],
            "vector_columns": [
                {"col": "embedding", "dim": 4, "embeddings": [[0.1, 0.2, 0.3, 0.4]]}
            ]
        }"#;
        let c = std::ffi::CString::new(req).unwrap();
        let ptr = unsafe { ailake_write_batch_multi_json(c.as_ptr()) };
        assert!(!ptr.is_null());
        let json = unsafe { CStr::from_ptr(ptr).to_str().unwrap().to_string() };
        assert!(json.contains("ids.len()"), "got: {json}");
        unsafe { ailake_free_string(ptr) };
    }

    #[test]
    fn write_batch_multi_json_two_columns_parses() {
        // Two vector columns (text + image), extra string column, fts_columns,
        // deferred — all new fields must parse without panic. Nonexistent
        // warehouse so it fails at I/O, not at JSON parsing.
        let req = r#"{
            "warehouse": "/nonexistent/path",
            "namespace": "default",
            "table": "test",
            "ids": [1, 2],
            "vector_columns": [
                {"col": "embedding", "dim": 4, "metric": "cosine", "modality": "text",
                 "embeddings": [[0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.8]]},
                {"col": "image_embedding", "dim": 2, "metric": "cosine", "modality": "image",
                 "embeddings": [[0.1, 0.2], [0.3, 0.4]]}
            ],
            "columns": {"chunk_text": ["row0", "row1"]},
            "fts_columns": ["chunk_text"],
            "format_version": 3,
            "deferred": true
        }"#;
        let c = std::ffi::CString::new(req).unwrap();
        let ptr = unsafe { ailake_write_batch_multi_json(c.as_ptr()) };
        assert!(!ptr.is_null());
        let json = unsafe { CStr::from_ptr(ptr).to_str().unwrap().to_string() };
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

    // ── write_batch_ipc (Fase 10, ADR-017) ──────────────────────────────────────
    //
    // Real (non-mocked) round-trips against a tempdir warehouse: build an Arrow
    // IPC stream the same shape a JVM caller would send, hand it to
    // `ailake_write_batch_ipc`, then confirm via `ailake_search_json` that the
    // rows actually landed and are searchable — not just "did not crash".

    fn build_ipc_stream_list_f32(ids: &[i64], embeddings: &[Vec<f32>], texts: &[&str]) -> Vec<u8> {
        use arrow_array::builder::{Float32Builder, ListBuilder};
        use arrow_array::{Int64Array, StringArray};
        use arrow_schema::{DataType, Field, Schema};

        let schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int64, false),
            Field::new(
                "embedding",
                DataType::List(Arc::new(Field::new("item", DataType::Float32, true))),
                false,
            ),
            Field::new("chunk_text", DataType::Utf8, true),
        ]));

        let mut list_builder = ListBuilder::new(Float32Builder::new());
        for row in embeddings {
            for v in row {
                list_builder.values().append_value(*v);
            }
            list_builder.append(true);
        }
        let embedding_array = list_builder.finish();

        let batch = arrow_array::RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(Int64Array::from(ids.to_vec())),
                Arc::new(embedding_array),
                Arc::new(StringArray::from(texts.to_vec())),
            ],
        )
        .unwrap();

        let mut buf = Vec::new();
        {
            let mut writer = arrow_ipc::writer::StreamWriter::try_new(&mut buf, &schema).unwrap();
            writer.write(&batch).unwrap();
            writer.finish().unwrap();
        }
        buf
    }

    fn build_ipc_stream_fixed_size_list_f32(
        ids: &[i64],
        embeddings: &[Vec<f32>],
        dim: i32,
    ) -> Vec<u8> {
        use arrow_array::builder::{FixedSizeListBuilder, Float32Builder};
        use arrow_array::Int64Array;
        use arrow_schema::{DataType, Field, Schema};

        let schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int64, false),
            Field::new(
                "embedding",
                DataType::FixedSizeList(Arc::new(Field::new("item", DataType::Float32, true)), dim),
                false,
            ),
        ]));

        let mut list_builder = FixedSizeListBuilder::new(Float32Builder::new(), dim);
        for row in embeddings {
            for v in row {
                list_builder.values().append_value(*v);
            }
            list_builder.append(true);
        }
        let embedding_array = list_builder.finish();

        let batch = arrow_array::RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(Int64Array::from(ids.to_vec())),
                Arc::new(embedding_array),
            ],
        )
        .unwrap();

        let mut buf = Vec::new();
        {
            let mut writer = arrow_ipc::writer::StreamWriter::try_new(&mut buf, &schema).unwrap();
            writer.write(&batch).unwrap();
            writer.finish().unwrap();
        }
        buf
    }

    fn call_write_batch_ipc(buf: &[u8], opts_json: &str) -> String {
        let c_opts = std::ffi::CString::new(opts_json).unwrap();
        let ptr =
            unsafe { ailake_write_batch_ipc(buf.as_ptr(), buf.len() as i64, c_opts.as_ptr()) };
        assert!(!ptr.is_null());
        let json = unsafe { CStr::from_ptr(ptr).to_str().unwrap().to_string() };
        unsafe { ailake_free_string(ptr) };
        json
    }

    #[test]
    fn write_batch_ipc_real_round_trip_list_f32() {
        let dir = tempfile::TempDir::new().unwrap();
        let warehouse = dir.path().to_str().unwrap();

        let ids = vec![1i64, 2, 3];
        let embeddings = vec![
            vec![1.0f32, 0.0, 0.0, 0.0],
            vec![0.0, 1.0, 0.0, 0.0],
            vec![0.0, 0.0, 1.0, 0.0],
        ];
        let texts = vec!["row one", "row two", "row three"];
        let buf = build_ipc_stream_list_f32(&ids, &embeddings, &texts);

        let opts = serde_json::json!({
            "warehouse": warehouse,
            "namespace": "default",
            "table": "docs",
            "vec_col": "embedding",
            "dim": 4,
            "metric": "euclidean",
        })
        .to_string();
        let write_json = call_write_batch_ipc(&buf, &opts);
        assert!(
            write_json.contains("\"ok\":true"),
            "expected success, got: {write_json}"
        );
        assert!(write_json.contains("snapshot_id"), "got: {write_json}");

        // Confirm the rows really landed and are searchable — full parity with
        // what ailake_write_batch_json would have produced.
        let search_req = serde_json::json!({
            "warehouse": warehouse,
            "namespace": "default",
            "table": "docs",
            "vec_col": "embedding",
            "dim": 4,
            "query": [1.0, 0.0, 0.0, 0.0],
            "top_k": 3,
        })
        .to_string();
        let c_search = std::ffi::CString::new(search_req).unwrap();
        let ptr = unsafe { ailake_search_json(c_search.as_ptr()) };
        let search_json = unsafe { CStr::from_ptr(ptr).to_str().unwrap().to_string() };
        unsafe { ailake_free_string(ptr) };
        assert!(
            search_json.contains("\"ok\":true"),
            "search failed: {search_json}"
        );
        let parsed: serde_json::Value = serde_json::from_str(&search_json).unwrap();
        let results = parsed["results"].as_array().unwrap();
        assert_eq!(
            results.len(),
            3,
            "expected all 3 rows back, got: {search_json}"
        );
    }

    #[test]
    fn write_batch_ipc_real_round_trip_fixed_size_list_f32() {
        let dir = tempfile::TempDir::new().unwrap();
        let warehouse = dir.path().to_str().unwrap();

        let ids = vec![10i64, 20];
        let embeddings = vec![vec![0.5f32, 0.5, 0.5, 0.5], vec![0.1, 0.2, 0.3, 0.4]];
        let buf = build_ipc_stream_fixed_size_list_f32(&ids, &embeddings, 4);

        let opts = serde_json::json!({
            "warehouse": warehouse,
            "namespace": "default",
            "table": "docs_fixed",
            "vec_col": "embedding",
            "dim": 4,
        })
        .to_string();
        let write_json = call_write_batch_ipc(&buf, &opts);
        assert!(
            write_json.contains("\"ok\":true"),
            "expected success, got: {write_json}"
        );
    }

    #[test]
    fn write_batch_ipc_dim_mismatch_returns_error() {
        let dir = tempfile::TempDir::new().unwrap();
        let warehouse = dir.path().to_str().unwrap();

        let ids = vec![1i64];
        let embeddings = vec![vec![1.0f32, 2.0, 3.0]]; // 3 dims
        let texts = vec!["x"];
        let buf = build_ipc_stream_list_f32(&ids, &embeddings, &texts);

        let opts = serde_json::json!({
            "warehouse": warehouse,
            "namespace": "default",
            "table": "docs",
            "vec_col": "embedding",
            "dim": 4, // declared dim mismatches actual row width
        })
        .to_string();
        let write_json = call_write_batch_ipc(&buf, &opts);
        assert!(write_json.contains("\"ok\":false"), "got: {write_json}");
        assert!(write_json.contains("dims"), "got: {write_json}");
    }

    #[test]
    fn write_batch_ipc_missing_vec_col_returns_error() {
        let dir = tempfile::TempDir::new().unwrap();
        let warehouse = dir.path().to_str().unwrap();

        let ids = vec![1i64];
        let embeddings = vec![vec![1.0f32, 2.0, 3.0, 4.0]];
        let texts = vec!["x"];
        let buf = build_ipc_stream_list_f32(&ids, &embeddings, &texts);

        let opts = serde_json::json!({
            "warehouse": warehouse,
            "namespace": "default",
            "table": "docs",
            "vec_col": "does_not_exist",
            "dim": 4,
        })
        .to_string();
        let write_json = call_write_batch_ipc(&buf, &opts);
        assert!(write_json.contains("\"ok\":false"), "got: {write_json}");
        assert!(write_json.contains("not found"), "got: {write_json}");
    }

    #[test]
    fn write_batch_ipc_null_guards() {
        let opts = std::ffi::CString::new(r#"{"warehouse":"/x","table":"t","dim":4}"#).unwrap();
        let ptr = unsafe { ailake_write_batch_ipc(std::ptr::null(), 0, opts.as_ptr()) };
        assert!(!ptr.is_null());
        let json = unsafe { CStr::from_ptr(ptr).to_str().unwrap().to_string() };
        unsafe { ailake_free_string(ptr) };
        assert!(json.contains("null"), "got: {json}");

        let buf = [0u8; 4];
        let ptr =
            unsafe { ailake_write_batch_ipc(buf.as_ptr(), buf.len() as i64, std::ptr::null()) };
        assert!(!ptr.is_null());
        let json = unsafe { CStr::from_ptr(ptr).to_str().unwrap().to_string() };
        unsafe { ailake_free_string(ptr) };
        assert!(json.contains("null"), "got: {json}");
    }

    #[test]
    fn write_batch_ipc_negative_len_guard() {
        let opts = std::ffi::CString::new(r#"{"warehouse":"/x","table":"t","dim":4}"#).unwrap();
        let buf = [0u8; 4];
        let ptr = unsafe { ailake_write_batch_ipc(buf.as_ptr(), -1, opts.as_ptr()) };
        assert!(!ptr.is_null());
        let json = unsafe { CStr::from_ptr(ptr).to_str().unwrap().to_string() };
        unsafe { ailake_free_string(ptr) };
        assert!(json.contains("negative"), "got: {json}");
    }

    // ── Proptest: FFI fuzzing ─────────────────────────────────────
    //
    // 1. Garbage strings — every single-param JSON export must survive
    //    arbitrary non-null byte sequences and return valid JSON.
    // 2. Extreme numeric values at the legacy binary C-ABI boundary.
    // 3. Random IPC buffers — must not crash.
    // 4. Round-trip: write random valid data, search it back.

    use proptest::prelude::*;
    use proptest::proptest;

    // ── 1. FFI string layer fuzzing ────────────────────────────────
    //
    // Generate byte sequences without interior nulls (valid UTF-8 or not).
    // Every export must return `{...}` or `[...]` JSON, never crash.

    proptest! {
        #[test]
        fn ffi_search_json_arbitrary_string(
            bytes in proptest::collection::vec(1u8..=255u8, 0..200),
        ) {
            let c_input = CString::new(bytes).unwrap();
            let ptr = unsafe { ailake_search_json(c_input.as_ptr()) };
            assert!(!ptr.is_null());
            let s = unsafe { CStr::from_ptr(ptr) }.to_str().unwrap().to_string();
            unsafe { ailake_free_string(ptr) };
            assert!(s.starts_with('{') || s.starts_with('['), "non-JSON: {s}");
        }

        #[test]
        fn ffi_write_batch_json_arbitrary_string(
            bytes in proptest::collection::vec(1u8..=255u8, 0..200),
        ) {
            let c_input = CString::new(bytes).unwrap();
            let ptr = unsafe { ailake_write_batch_json(c_input.as_ptr()) };
            assert!(!ptr.is_null());
            let s = unsafe { CStr::from_ptr(ptr) }.to_str().unwrap().to_string();
            unsafe { ailake_free_string(ptr) };
            assert!(s.starts_with('{') || s.starts_with('['), "non-JSON: {s}");
        }

        #[test]
        fn ffi_search_text_json_arbitrary_string(
            bytes in proptest::collection::vec(1u8..=255u8, 0..200),
        ) {
            let c_input = CString::new(bytes).unwrap();
            let ptr = unsafe { ailake_search_text_json(c_input.as_ptr()) };
            assert!(!ptr.is_null());
            let s = unsafe { CStr::from_ptr(ptr) }.to_str().unwrap().to_string();
            unsafe { ailake_free_string(ptr) };
            assert!(s.starts_with('{') || s.starts_with('['), "non-JSON: {s}");
        }

        #[test]
        fn ffi_scan_json_arbitrary_string(
            bytes in proptest::collection::vec(1u8..=255u8, 0..200),
        ) {
            let c_input = CString::new(bytes).unwrap();
            let ptr = unsafe { ailake_scan_json(c_input.as_ptr()) };
            assert!(!ptr.is_null());
            let s = unsafe { CStr::from_ptr(ptr) }.to_str().unwrap().to_string();
            unsafe { ailake_free_string(ptr) };
            assert!(s.starts_with('{') || s.starts_with('['), "non-JSON: {s}");
        }

        #[test]
        fn ffi_compact_json_arbitrary_string(
            bytes in proptest::collection::vec(1u8..=255u8, 0..200),
        ) {
            let c_input = CString::new(bytes).unwrap();
            let ptr = unsafe { ailake_compact_json(c_input.as_ptr()) };
            assert!(!ptr.is_null());
            let s = unsafe { CStr::from_ptr(ptr) }.to_str().unwrap().to_string();
            unsafe { ailake_free_string(ptr) };
            assert!(s.starts_with('{') || s.starts_with('['), "non-JSON: {s}");
        }
    }

    // ── 2. Legacy binary API: boundary values ────────────────────
    //
    // ailake_vector_search_json takes raw f32 pointer + length + top_k.
    // Must not crash for boundary dims (0, 1, max allocatable), zero top_k,
    // or large top_k. query_len always matches the actual allocation.

    proptest! {
        #[test]
        fn ffi_legacy_api_boundary_values(
            table_uri_bytes in proptest::collection::vec(1u8..=255u8, 0..100),
            dim in 0u32..4097,  // covers 0..4096 inclusive
            top_k in prop::num::u32::ANY,
        ) {
            let c_uri = CString::new(table_uri_bytes).unwrap();
            // Cap allocation at 4096 f32s = 16 KB, keeps test fast
            let safe_dim = dim.min(4096);
            let query = vec![0.0f32; safe_dim as usize];
            let ptr = unsafe { ailake_vector_search_json(
                c_uri.as_ptr(),
                query.as_ptr(),
                safe_dim, // always matches allocated buffer
                top_k,
            )};
            assert!(!ptr.is_null());
            let s = unsafe { CStr::from_ptr(ptr) }.to_str().unwrap().to_string();
            unsafe { ailake_free_string(ptr) };
            assert!(s.starts_with('{') || s.starts_with('['), "non-JSON: {s}");
        }
    }

    // ── 3. IPC API: random byte buffers ───────────────────────────
    //
    // ailake_write_batch_ipc takes raw bytes + i64 length + opts JSON.
    // Random byte buffers + random opts must return error JSON.

    proptest! {
        #[test]
        fn ffi_ipc_random_buffers(
            buf_bytes in proptest::collection::vec(proptest::num::u8::ANY, 0..100),
            opts_bytes in proptest::collection::vec(1u8..=255u8, 0..200),
        ) {
            let c_opts = CString::new(opts_bytes).unwrap();
            let ptr = unsafe { ailake_write_batch_ipc(
                buf_bytes.as_ptr(),
                buf_bytes.len() as i64,
                c_opts.as_ptr(),
            )};
            assert!(!ptr.is_null());
            let s = unsafe { CStr::from_ptr(ptr) }.to_str().unwrap().to_string();
            unsafe { ailake_free_string(ptr) };
            assert!(s.starts_with('{') || s.starts_with('['), "non-JSON: {s}");
        }
    }

    // ── 4. Round-trip: write random valid data → search it back ──

    proptest! {
        #![proptest_config(ProptestConfig {
            cases: 20, .. ProptestConfig::default()
        })]

        #[test]
        fn ffi_write_search_roundtrip(
            dim in 1u32..8,
            num_rows in 1usize..6,
        ) {
            let dir = tempfile::TempDir::new().unwrap();
            let warehouse = dir.path().to_str().unwrap().to_string();

            // Build deterministic embeddings from dim/num_rows
            let ids: Vec<i64> = (0..num_rows as i64).collect();
            let embeddings: Vec<Vec<f32>> = (0..num_rows)
                .map(|i| {
                    (0..dim as usize)
                        .map(|j| ((i * dim as usize + j) as f32) / 100.0)
                        .collect()
                })
                .collect();
            let texts: Vec<String> = (0..num_rows)
                .map(|i| format!("row {i}"))
                .collect();
            let text_refs: Vec<&str> = texts.iter().map(|s| s.as_str()).collect();

            let buf = build_ipc_stream_list_f32(&ids, &embeddings, &text_refs);

            let opts = serde_json::json!({
                "warehouse": warehouse,
                "namespace": "default",
                "table": "proptest_t",
                "vec_col": "embedding",
                "dim": dim,
                "metric": "euclidean",
            }).to_string();

            let write_json = call_write_batch_ipc(&buf, &opts);
            assert!(
                write_json.contains("\"ok\":true"),
                "write failed: {write_json}"
            );

            // Search with the first embedding as query
            let query_vec: Vec<f32> = embeddings[0].clone();
            let search_req = serde_json::json!({
                "warehouse": warehouse,
                "namespace": "default",
                "table": "proptest_t",
                "vec_col": "embedding",
                "dim": dim,
                "query": query_vec,
                "top_k": num_rows as u32,
            }).to_string();
            let c_search = CString::new(search_req).unwrap();
            let ptr = unsafe { ailake_search_json(c_search.as_ptr()) };
            assert!(!ptr.is_null());
            let search_json = unsafe { CStr::from_ptr(ptr) }.to_str().unwrap().to_string();
            unsafe { ailake_free_string(ptr) };
            let parsed: serde_json::Value = serde_json::from_str(&search_json).unwrap();
            assert!(
                parsed["ok"] == serde_json::Value::Bool(true),
                "search failed: {search_json}"
            );
            let results = parsed["results"].as_array();
            assert!(
                results.is_some() && results.unwrap().len() == num_rows,
                "expected {num_rows} results, got: {search_json}"
            );
        }
    }

    #[cfg(miri)]
    mod miri_tests {
        use super::*;
        use std::ffi::{CStr, CString};

        /// CStr::from_ptr com string UTF-8 válida sob Miri.
        #[test]
        fn miri_cstr_from_ptr_valid() {
            let s = CString::new(r#"{"ok":true}"#).unwrap();
            let cstr = unsafe { CStr::from_ptr(s.as_ptr()) };
            let parsed: serde_json::Value =
                serde_json::from_str(cstr.to_str().unwrap()).unwrap();
            assert!(parsed["ok"] == serde_json::Value::Bool(true));
        }

        /// CStr::from_ptr com string vazia.
        #[test]
        fn miri_cstr_from_ptr_empty() {
            let s = CString::new("").unwrap();
            let cstr = unsafe { CStr::from_ptr(s.as_ptr()) };
            assert!(cstr.to_bytes().is_empty());
        }
    }
}
