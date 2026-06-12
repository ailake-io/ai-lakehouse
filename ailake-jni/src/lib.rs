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
use ailake_core::VectorMetric;
use ailake_query::{
    fetch_rows as rs_fetch_rows, search as rs_search, Chunk, ContextAssembler,
    ContextAssemblerConfig, SearchConfig, SearchResult,
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
) -> ailake_core::AilakeResult<Vec<SearchResult>> {
    let store: Arc<dyn ailake_store::Store> = Arc::new(LocalStore::new(&warehouse));
    let catalog = Arc::new(HadoopCatalog::new(store.clone(), &warehouse));
    let table = TableIdent::new(namespace, table_name);
    let config = SearchConfig {
        top_k: top_k as usize,
        ef_search: ef_search as usize,
        pruning_threshold: f32::INFINITY,
        rerank_factor: None,
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
    let s = format!("{{\"ok\":false,\"error\":\"{msg}\"}}");
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
    let results: Vec<RowResultJson> =
        match do_search(uri, "default", "table", "embedding", dim, query, top_k, 50) {
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

    let results = match do_search(
        req.warehouse,
        &req.namespace,
        &req.table,
        &req.vec_col,
        req.dim,
        req.query,
        req.top_k,
        req.ef_search,
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
    use ailake_core::{VectorPrecision, VectorStoragePolicy};
    use ailake_query::TableWriter;
    use arrow_array::{Int64Array, RecordBatch};
    use arrow_schema::{DataType, Field, Schema};

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
        ids: Vec<i64>,
        embeddings: Vec<Vec<f32>>,
    }
    fn default_ns() -> String {
        "default".into()
    }
    fn default_col() -> String {
        "embedding".into()
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
        embedding_model: None,
    };

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
        let mut writer = TableWriter::create_or_open(catalog, store, policy, table).await?;
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
}
