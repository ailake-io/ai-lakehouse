// SPDX-License-Identifier: MIT OR Apache-2.0
//! ailake-py — PyO3 Python bindings
//!
//! Thin async-to-sync bridge. All logic lives in ailake-query and friends.
//! Build with: maturin develop --release

// PyO3 proc-macros emit implicit Into<PyErr> conversions that clippy
// flags as useless_conversion. Suppress it for the whole crate.
#![allow(clippy::useless_conversion)]

use std::sync::Arc;

use arrow_array::{RecordBatch, StringArray};
use arrow_ipc::writer::FileWriter;
use arrow_schema::{DataType, Schema};
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyBytes, PyDict, PyList};
use tracing::{debug, warn};

use ailake_catalog::{
    hadoop::HadoopCatalog,
    provider::{CatalogProvider, TableIdent, TableProperties},
    RestCatalog, RestCatalogAuth, RestCatalogConfig,
};
use ailake_core::{
    EmbeddingModelInfo, PartitionDef, VectorMetric, VectorModality, VectorPrecision,
    VectorStoragePolicy,
};
use ailake_query::{
    delete_rows as rs_delete_rows, fetch_rows as rs_fetch_rows, search as rs_search,
    search_multimodal as rs_search_multimodal, Chunk, ContextAssembler, ContextAssemblerConfig,
    EmbedFn, FusionMethod, MigrationJob, MigrationProgress, MigrationStrategy, ModalQuery,
    MultiVectorBatch, ProgressFn, SearchConfig, TableWriter as RsTableWriter,
};
use ailake_store::{store::Store, LocalStore};

fn rt() -> PyResult<tokio::runtime::Runtime> {
    tokio::runtime::Runtime::new().map_err(|e| {
        PyRuntimeError::new_err(format!("ailake: failed to create Tokio runtime: {e}"))
    })
}

/// Wraps an i64 Unix epoch nanosecond value so `build_batch_with_extra` can
/// tell it apart from a plain `int` extra column and emit a real
/// `Timestamp(Nanosecond, UTC)` Arrow column instead of `Int64`.
///
/// Required for `last_accessed_at`/`created_at` columns consumed by
/// `decay_memories()` (`ailake_query::memory_decay::days_old_vec` only
/// accepts `Timestamp(ns/us)` or `Utf8`, never `Int64`).
///
/// Example::
///
///     writer.write_batch(texts, embeddings, extra_columns={
///         "last_accessed_at": [ailake.TimestampNs(ailake.now_ns())] * len(texts),
///     })
#[pyclass(name = "TimestampNs", from_py_object)]
#[derive(Clone, Copy)]
pub struct TimestampNs {
    #[pyo3(get)]
    pub ns: i64,
}

#[pymethods]
impl TimestampNs {
    #[new]
    fn new(ns: i64) -> Self {
        Self { ns }
    }
    fn __repr__(&self) -> String {
        format!("TimestampNs({})", self.ns)
    }
    fn __int__(&self) -> i64 {
        self.ns
    }
}

/// Build a RecordBatch from texts and optional extra columns.
///
/// Texts become the `"text"` column (Utf8). Extra columns are inferred from
/// the Python value type of the first element:
///   - `TimestampNs` → Timestamp(Nanosecond, UTC)
///   - `bool` → Boolean
///   - `float` → Float32
///   - `int` → Int64
///   - `str`/other → Utf8
fn build_batch_with_extra(
    py: Python<'_>,
    texts: Vec<String>,
    extra_columns: Option<&Bound<'_, PyDict>>,
) -> PyResult<RecordBatch> {
    use arrow_array::{BooleanArray, Float32Array, Int64Array, TimestampNanosecondArray};
    use arrow_schema::{Field, TimeUnit};

    let mut fields = vec![Field::new("text", DataType::Utf8, false)];
    let text_arr: Arc<dyn arrow_array::Array> = Arc::new(StringArray::from(texts));
    let mut arrays: Vec<Arc<dyn arrow_array::Array>> = vec![text_arr];

    if let Some(extra) = extra_columns {
        for (k, v) in extra.iter() {
            let col_name: String = k.extract()?;
            let values: Vec<Py<PyAny>> = v.extract()?;

            // Infer type from first element
            let first = values.first().map(|x| x.bind(py));
            if first
                .as_ref()
                .map(|x| x.is_instance_of::<TimestampNs>())
                .unwrap_or(false)
            {
                let arr: Vec<Option<i64>> = values
                    .iter()
                    .map(|x| x.bind(py).extract::<TimestampNs>().ok().map(|t| t.ns))
                    .collect();
                fields.push(Field::new(
                    &col_name,
                    DataType::Timestamp(TimeUnit::Nanosecond, Some("UTC".into())),
                    true,
                ));
                arrays.push(Arc::new(
                    TimestampNanosecondArray::from(arr).with_timezone("UTC"),
                ));
            } else if first
                .as_ref()
                .map(|x| x.is_instance_of::<pyo3::types::PyBool>())
                .unwrap_or(false)
            {
                let arr: Vec<Option<bool>> = values
                    .iter()
                    .map(|x| x.bind(py).extract::<bool>().ok())
                    .collect();
                fields.push(Field::new(&col_name, DataType::Boolean, true));
                arrays.push(Arc::new(BooleanArray::from(arr)));
            } else if first
                .as_ref()
                .map(|x| x.is_instance_of::<pyo3::types::PyFloat>())
                .unwrap_or(false)
            {
                let arr: Vec<Option<f32>> = values
                    .iter()
                    .map(|x| x.bind(py).extract::<f32>().ok())
                    .collect();
                fields.push(Field::new(&col_name, DataType::Float32, true));
                arrays.push(Arc::new(Float32Array::from(arr)));
            } else if first
                .as_ref()
                .map(|x| x.is_instance_of::<pyo3::types::PyInt>())
                .unwrap_or(false)
            {
                let arr: Vec<Option<i64>> = values
                    .iter()
                    .map(|x| x.bind(py).extract::<i64>().ok())
                    .collect();
                fields.push(Field::new(&col_name, DataType::Int64, true));
                arrays.push(Arc::new(Int64Array::from(arr)));
            } else {
                // Default: string column
                let arr: Vec<Option<String>> = values
                    .iter()
                    .map(|x| x.bind(py).extract::<String>().ok())
                    .collect();
                fields.push(Field::new(&col_name, DataType::Utf8, true));
                arrays.push(Arc::new(StringArray::from(arr)));
            }
        }
    }

    let schema = Arc::new(Schema::new(fields));
    RecordBatch::try_new(schema, arrays).map_err(|e| PyValueError::new_err(e.to_string()))
}

/// `catalog_opts` selects/configures the catalog *metadata* backend; `path`
/// still always resolves to a local `Store` (ailake-py has no `store_from_url`
/// equivalent yet — S3/GCS/Azure aren't reachable from any binding today, a
/// separate, pre-existing gap not closed here). `catalog_opts["catalog"]`:
/// `"hadoop"` (default, unchanged behavior for every caller that omits it) or
/// `"rest"` — talks to any Iceberg REST Catalog spec server, keys mirror
/// `ailake-cli`'s `--rest-*` flags (`rest_uri`, `rest_prefix`,
/// `rest_warehouse`, `rest_auth`: `"none"`/`"bearer"`/`"oauth2"`, `rest_token`,
/// `rest_oauth_token_endpoint`, `rest_oauth_client_id`,
/// `rest_oauth_client_secret`, `rest_oauth_scope`). See
/// docs/guides/REST_CATALOG.md.
fn local_catalog_store(
    path: &str,
    catalog_opts: Option<&std::collections::HashMap<String, String>>,
) -> PyResult<(Arc<dyn CatalogProvider>, Arc<dyn Store>)> {
    let store: Arc<dyn Store> = Arc::new(LocalStore::new(path));
    // Use a file:// URI as warehouse so that Iceberg metadata.json and manifest
    // files contain absolute file:// paths. Required for Trino's Iceberg
    // connector and any reader that resolves location URIs strictly.
    // LocalStore::full_path strips the file:// prefix before I/O.
    //
    // std::path::absolute resolves relative paths without requiring the directory
    // to exist (unlike canonicalize, which fails on new table paths).
    let absolute = std::path::absolute(path).unwrap_or_else(|_| std::path::PathBuf::from(path));
    let warehouse_uri = format!("file://{}", absolute.display());

    let backend = catalog_opts
        .and_then(|o| o.get("catalog"))
        .map(String::as_str)
        .unwrap_or("hadoop");
    let catalog: Arc<dyn CatalogProvider> = match backend {
        "hadoop" => Arc::new(HadoopCatalog::new(Arc::clone(&store), &warehouse_uri)),
        "rest" => {
            // Only reachable when `backend == "rest"`, which only comes from
            // `catalog_opts["catalog"]`, so `catalog_opts` itself is `Some` here.
            let opts = catalog_opts.expect("catalog=\"rest\" implies catalog_opts is Some");
            let uri = opts.get("rest_uri").cloned().ok_or_else(|| {
                PyValueError::new_err("catalog_opts['catalog']=\"rest\" requires \"rest_uri\"")
            })?;
            let auth = match opts.get("rest_auth").map(String::as_str).unwrap_or("none") {
                "none" => RestCatalogAuth::None,
                "bearer" => {
                    let token = opts.get("rest_token").cloned().ok_or_else(|| {
                        PyValueError::new_err("rest_auth=\"bearer\" requires \"rest_token\"")
                    })?;
                    RestCatalogAuth::Bearer(token)
                }
                "oauth2" => {
                    let token_endpoint = opts
                        .get("rest_oauth_token_endpoint")
                        .cloned()
                        .ok_or_else(|| {
                            PyValueError::new_err(
                                "rest_auth=\"oauth2\" requires \"rest_oauth_token_endpoint\"",
                            )
                        })?;
                    let client_id = opts.get("rest_oauth_client_id").cloned().ok_or_else(|| {
                        PyValueError::new_err(
                            "rest_auth=\"oauth2\" requires \"rest_oauth_client_id\"",
                        )
                    })?;
                    let client_secret =
                        opts.get("rest_oauth_client_secret")
                            .cloned()
                            .ok_or_else(|| {
                                PyValueError::new_err(
                                    "rest_auth=\"oauth2\" requires \"rest_oauth_client_secret\"",
                                )
                            })?;
                    RestCatalogAuth::OAuth2 {
                        token_endpoint,
                        client_id,
                        client_secret,
                        scope: opts.get("rest_oauth_scope").cloned(),
                    }
                }
                other => return Err(PyValueError::new_err(format!("unknown rest_auth: {other}"))),
            };
            let config = RestCatalogConfig {
                uri,
                prefix: opts.get("rest_prefix").cloned(),
                warehouse: opts.get("rest_warehouse").cloned(),
                auth,
            };
            Arc::new(RestCatalog::new(config, Arc::clone(&store)))
        }
        other => {
            return Err(PyValueError::new_err(format!(
                "unknown catalog backend: {other} (supported: \"hadoop\", \"rest\")"
            )))
        }
    };
    Ok((catalog, store))
}

/// Python-facing table writer. Wraps ailake_query::TableWriter.
#[pyclass]
pub struct TableWriter {
    inner: Option<RsTableWriter>,
    runtime: tokio::runtime::Runtime,
    /// Optional Python callable used for Pattern B: embed_fn(texts) -> list[list[float]]
    embed_fn: Option<Py<PyAny>>,
    /// Mirrors the primary column's dim/ivf_residual — RsTableWriter's `policy`
    /// field is private, so write_batch_ivf_pq(_deferred) needs its own copy to
    /// build an IvfPqConfig without a new Rust-side accessor.
    dim: u32,
    ivf_residual: bool,
}

#[pymethods]
impl TableWriter {
    /// Open (or create) an AI-Lake table at `path` on the local filesystem.
    #[new]
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (path, vector_column="embedding", dim=1536, metric="cosine", precision="f16", pre_normalize=false, hnsw_m=None, hnsw_ef_construction=None, pq_only=false, ivf_residual=false, embedding_model=None, embedding_model_version=None, embed_fn=None, partition_by=None, partition_value=None, partition_column_type=None, partition_fields=None, partition_values=None, bm25_text_column=None, format_version=2, fts_text_columns=None, fts_tokenizer="default", catalog_opts=None))]
    fn new(
        py: Python<'_>,
        path: &str,
        vector_column: &str,
        dim: u32,
        metric: &str,
        precision: &str,
        pre_normalize: bool,
        hnsw_m: Option<u32>,
        hnsw_ef_construction: Option<u32>,
        pq_only: bool,
        ivf_residual: bool,
        embedding_model: Option<&str>,
        embedding_model_version: Option<&str>,
        embed_fn: Option<Py<PyAny>>,
        partition_by: Option<String>,
        partition_value: Option<String>,
        partition_column_type: Option<String>,
        // Multi-column partition spec (Phase K).
        // List of (column, transform, column_type) tuples.
        // Example: [("agent_id", "identity", "string"), ("ts", "truncate[4]", "string")]
        partition_fields: Option<Vec<(String, String, String)>>,
        // Dict of {column: raw_value} for multi-column partition value at write time.
        // Converted to \x1f-separated compound string in partition_fields order.
        // Ignored when partition_value is also set (partition_value takes priority).
        partition_values: Option<std::collections::HashMap<String, String>>,
        bm25_text_column: Option<String>,
        format_version: u8,
        fts_text_columns: Option<Vec<String>>,
        fts_tokenizer: &str,
        catalog_opts: Option<std::collections::HashMap<String, String>>,
    ) -> PyResult<Self> {
        let rt = rt()?;
        debug!(
            "ailake-py: TableWriter::new path={} dim={} metric={} precision={} pre_normalize={} hnsw_m={:?} hnsw_ef={:?} pq_only={} ivf_residual={} embedding_model={:?} partition_by={:?}",
            path, dim, metric, precision, pre_normalize, hnsw_m, hnsw_ef_construction, pq_only, ivf_residual, embedding_model, partition_by
        );
        let mut policy =
            VectorStoragePolicy::default_f16(vector_column, dim, parse_metric(metric)?);
        policy.precision = parse_precision(precision);
        policy.pre_normalize = pre_normalize;
        policy.hnsw_m = hnsw_m;
        policy.hnsw_ef_construction = hnsw_ef_construction;
        policy.keep_raw_for_reranking = !pq_only;
        policy.ivf_residual = ivf_residual;
        policy.partition_by = partition_by;
        policy.partition_column_type = partition_column_type;

        // Phase K: multi-column partition spec.
        if let Some(fields) = partition_fields {
            policy.partition_fields = fields
                .into_iter()
                .map(|(col, tr, ct)| ailake_core::PartitionDef {
                    column: col,
                    transform: tr,
                    column_type: ct,
                })
                .collect();
        }

        // Resolve partition_value: explicit string wins; else build from dict in field order.
        policy.partition_value = if let Some(pv) = partition_value {
            Some(pv)
        } else if let Some(pv_map) = partition_values {
            if policy.partition_fields.is_empty() {
                None
            } else {
                let parts: Vec<String> = policy
                    .partition_fields
                    .iter()
                    .map(|pf| pv_map.get(&pf.column).cloned().unwrap_or_default())
                    .collect();
                Some(parts.join("\x1f"))
            }
        } else {
            None
        };
        if let Some(model_name) = embedding_model {
            let mut model_info = EmbeddingModelInfo::new(model_name).with_dim(dim);
            if let Some(version) = embedding_model_version {
                model_info = model_info.with_version(version);
            }
            policy.embedding_model = Some(model_info);
        }
        let (catalog, store) = local_catalog_store(path, catalog_opts.as_ref())?;
        let table = TableIdent::new("default", "table");

        let stored_embed_fn = embed_fn.map(|f| f.clone_ref(py));
        let mut writer = rt
            .block_on(RsTableWriter::create_or_open(
                catalog,
                store,
                policy,
                table,
                format_version,
            ))
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        if let Some(col) = bm25_text_column {
            writer = writer.with_bm25(col);
        }
        if let Some(cols) = fts_text_columns {
            let cfg = ailake_fts::FtsConfig {
                text_columns: cols,
                tokenizer: fts_tokenizer.to_string(),
                writer_heap_bytes: 50 * 1024 * 1024,
            };
            writer = writer.with_fts_config(cfg);
        }

        Ok(Self {
            inner: Some(writer),
            runtime: rt,
            embed_fn: stored_embed_fn,
            dim,
            ivf_residual,
        })
    }

    /// Write a batch of rows.
    ///
    /// Args:
    ///   texts: list[str] — text content for each row
    ///   embeddings: list[list[float]] or None — one embedding per row.
    ///               When None, embed_fn set at construction time is called automatically
    ///               (Pattern B). Raises ValueError when embeddings is None and no embed_fn.
    #[pyo3(signature = (texts, embeddings=None, extra_columns=None))]
    fn write_batch(
        &mut self,
        py: Python<'_>,
        texts: Vec<String>,
        embeddings: Option<Vec<Vec<f32>>>,
        extra_columns: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<()> {
        let embs: Vec<Vec<f32>> = match embeddings {
            Some(e) => e,
            None => {
                let embed_fn = self.embed_fn.as_ref().ok_or_else(|| {
                    PyValueError::new_err(
                        "embeddings is required when embed_fn was not set on TableWriter",
                    )
                })?;
                let py_texts =
                    PyList::new(py, &texts).map_err(|e| PyValueError::new_err(e.to_string()))?;
                let result = embed_fn
                    .call1(py, (py_texts,))
                    .map_err(|e| PyValueError::new_err(format!("embed_fn error: {e}")))?;
                result.bind(py).extract::<Vec<Vec<f32>>>().map_err(|e| {
                    PyValueError::new_err(format!("embed_fn must return list[list[float]]: {e}"))
                })?
            }
        };

        let batch = build_batch_with_extra(py, texts, extra_columns)?;

        let writer = self
            .inner
            .as_mut()
            .ok_or_else(|| PyValueError::new_err("TableWriter already committed"))?;

        self.runtime
            .block_on(writer.write_batch(&batch, &embs))
            .map_err(|e| {
                warn!("ailake-py: write_batch failed: {}", e);
                PyValueError::new_err(e.to_string())
            })
    }

    /// Write a batch with auto index selection and deferred (background) index build.
    ///
    /// Persists Parquet immediately (~200k vec/s). Selects IVF-PQ when a GPU or ≥8 CPU
    /// cores are present and batch ≥5 000 vectors; falls back to HNSW. Index built in a
    /// background task — shard is served via flat scan until the index is ready.
    ///
    /// Args:
    ///   texts: list[str] — text content for each row
    ///   embeddings: list[list[float]] — one embedding per row
    #[pyo3(signature = (texts, embeddings, extra_columns=None))]
    fn write_batch_auto_deferred(
        &mut self,
        py: Python<'_>,
        texts: Vec<String>,
        embeddings: Vec<Vec<f32>>,
        extra_columns: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<()> {
        let batch = build_batch_with_extra(py, texts, extra_columns)?;

        let writer = self
            .inner
            .as_mut()
            .ok_or_else(|| PyValueError::new_err("TableWriter already committed"))?;

        self.runtime
            .block_on(writer.write_batch_auto_deferred(&batch, &embeddings))
            .map_err(|e| {
                warn!("ailake-py: write_batch_auto_deferred failed: {}", e);
                PyValueError::new_err(e.to_string())
            })
    }

    /// Write a batch, forcing IVF-PQ indexing (synchronous build).
    ///
    /// Unlike `write_batch_auto_deferred`, which only picks IVF-PQ when its
    /// hardware/batch-size heuristic (GPU or ≥8 cores and ≥5 000 vectors)
    /// says so, this always builds IVF-PQ — smaller index, better for S3
    /// sequential-scan workloads. Blocks until the index is fully built.
    ///
    /// Args:
    ///   texts: list[str] — text content for each row
    ///   embeddings: list[list[float]] — one embedding per row
    #[pyo3(signature = (texts, embeddings, extra_columns=None))]
    fn write_batch_ivf_pq(
        &mut self,
        py: Python<'_>,
        texts: Vec<String>,
        embeddings: Vec<Vec<f32>>,
        extra_columns: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<()> {
        let batch = build_batch_with_extra(py, texts, extra_columns)?;

        let writer = self
            .inner
            .as_mut()
            .ok_or_else(|| PyValueError::new_err("TableWriter already committed"))?;

        let mut ivf_config =
            ailake_query::IvfPqConfig::for_dataset(self.dim as usize, embeddings.len());
        if self.ivf_residual {
            ivf_config = ivf_config.with_residual();
        }

        self.runtime
            .block_on(writer.write_batch_ivf_pq(&batch, &embeddings, ivf_config))
            .map_err(|e| {
                warn!("ailake-py: write_batch_ivf_pq failed: {}", e);
                PyValueError::new_err(e.to_string())
            })
    }

    /// Deferred variant of `write_batch_ivf_pq` — persists Parquet immediately
    /// (~200k vec/s) and builds the IVF-PQ index in a background task.
    ///
    /// Args:
    ///   texts: list[str] — text content for each row
    ///   embeddings: list[list[float]] — one embedding per row
    #[pyo3(signature = (texts, embeddings, extra_columns=None))]
    fn write_batch_ivf_pq_deferred(
        &mut self,
        py: Python<'_>,
        texts: Vec<String>,
        embeddings: Vec<Vec<f32>>,
        extra_columns: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<()> {
        let batch = build_batch_with_extra(py, texts, extra_columns)?;

        let writer = self
            .inner
            .as_mut()
            .ok_or_else(|| PyValueError::new_err("TableWriter already committed"))?;

        let mut ivf_config =
            ailake_query::IvfPqConfig::for_dataset(self.dim as usize, embeddings.len());
        if self.ivf_residual {
            ivf_config = ivf_config.with_residual();
        }

        self.runtime
            .block_on(writer.write_batch_ivf_pq_deferred(&batch, &embeddings, ivf_config))
            .map_err(|e| {
                warn!("ailake-py: write_batch_ivf_pq_deferred failed: {}", e);
                PyValueError::new_err(e.to_string())
            })
    }

    /// Idempotent write — no-op if `batch_id` was already committed.
    ///
    /// Args:
    ///   texts: list[str] — text content for each row
    ///   embeddings: list[list[float]] — one embedding per row
    ///   batch_id: str — unique key for this batch (e.g. Airflow run_id + task_id)
    #[pyo3(signature = (texts, embeddings, batch_id, extra_columns=None))]
    fn write_batch_idempotent(
        &mut self,
        py: Python<'_>,
        texts: Vec<String>,
        embeddings: Vec<Vec<f32>>,
        batch_id: String,
        extra_columns: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<()> {
        let batch = build_batch_with_extra(py, texts, extra_columns)?;

        let writer = self
            .inner
            .as_mut()
            .ok_or_else(|| PyValueError::new_err("TableWriter already committed"))?;

        self.runtime
            .block_on(writer.write_batch_idempotent(&batch, &embeddings, &batch_id))
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    /// Write a batch with N independent vector columns.
    ///
    /// Each column gets its own HNSW index in the file footer. Use for multimodal tables
    /// where the same row has embeddings from different modalities (text + image, etc.).
    ///
    /// Args:
    ///   texts: list[str] — text content for each row (primary tabular column)
    ///   columns: list[tuple[VectorColSpec, list[list[float]]]]
    ///            Each tuple: (column_spec, embeddings_for_that_column)
    ///            Length of each embedding list must equal len(texts).
    #[pyo3(signature = (texts, columns, extra_columns=None))]
    fn write_batch_multi(
        &mut self,
        py: Python<'_>,
        texts: Vec<String>,
        columns: Vec<(Py<VectorColSpec>, Vec<Vec<f32>>)>,
        extra_columns: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<()> {
        if columns.is_empty() {
            return Err(PyValueError::new_err(
                "write_batch_multi requires at least one column",
            ));
        }

        let batch = build_batch_with_extra(py, texts, extra_columns)?;

        // Build owned policies + embedding vecs.
        let mut mv_batches: Vec<(VectorStoragePolicy, Vec<Vec<f32>>)> =
            Vec::with_capacity(columns.len());
        for (spec_py, embs) in &columns {
            let spec = spec_py.borrow(py);
            let metric = parse_metric(&spec.metric)?;
            let modality = spec
                .modality
                .as_deref()
                .and_then(|s| s.parse::<VectorModality>().ok());
            let mut policy = VectorStoragePolicy::default_f16(&spec.column, spec.dim, metric);
            policy.modality = modality;
            policy.precision = parse_precision(&spec.precision);
            policy.pre_normalize = spec.pre_normalize;
            policy.hnsw_m = spec.hnsw_m;
            policy.hnsw_ef_construction = spec.hnsw_ef_construction;
            mv_batches.push((policy, embs.clone()));
        }

        let batches: Vec<MultiVectorBatch<'_>> = mv_batches
            .iter()
            .map(|(policy, embs)| MultiVectorBatch {
                policy: policy.clone(),
                embeddings: embs.as_slice(),
            })
            .collect();

        let writer = self
            .inner
            .as_mut()
            .ok_or_else(|| PyValueError::new_err("TableWriter already committed"))?;

        self.runtime
            .block_on(writer.write_batch_multi(&batch, &batches))
            .map_err(|e| {
                warn!("ailake-py: write_batch_multi failed: {}", e);
                PyValueError::new_err(e.to_string())
            })
    }

    /// Deferred variant of `write_batch_multi`.
    ///
    /// Persists Parquet immediately and builds all N column HNSW indexes in a
    /// background task. During the build window, search is served via flat scan
    /// (exact, GPU-accelerated when available). Transitions to HNSW-indexed search
    /// automatically when `IndexStatus::Ready`. On permanent build failure the file
    /// is marked `IndexStatus::Failed` (visible via `ailake info`) and compaction
    /// will rebuild the index on the next run.
    ///
    /// Use when ingest throughput matters more than immediate HNSW availability.
    ///
    /// Args:
    ///   texts: list[str]
    ///   columns: list[tuple[VectorColSpec, list[list[float]]]]
    #[pyo3(signature = (texts, columns, extra_columns=None))]
    fn write_batch_multi_deferred(
        &mut self,
        py: Python<'_>,
        texts: Vec<String>,
        columns: Vec<(Py<VectorColSpec>, Vec<Vec<f32>>)>,
        extra_columns: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<()> {
        if columns.is_empty() {
            return Err(PyValueError::new_err(
                "write_batch_multi_deferred requires at least one column",
            ));
        }

        let batch = build_batch_with_extra(py, texts, extra_columns)?;

        let mut mv_batches: Vec<(VectorStoragePolicy, Vec<Vec<f32>>)> =
            Vec::with_capacity(columns.len());
        for (spec_py, embs) in &columns {
            let spec = spec_py.borrow(py);
            let metric = parse_metric(&spec.metric)?;
            let modality = spec
                .modality
                .as_deref()
                .and_then(|s| s.parse::<VectorModality>().ok());
            let mut policy = VectorStoragePolicy::default_f16(&spec.column, spec.dim, metric);
            policy.modality = modality;
            policy.precision = parse_precision(&spec.precision);
            policy.pre_normalize = spec.pre_normalize;
            policy.hnsw_m = spec.hnsw_m;
            policy.hnsw_ef_construction = spec.hnsw_ef_construction;
            mv_batches.push((policy, embs.clone()));
        }

        let batches: Vec<MultiVectorBatch<'_>> = mv_batches
            .iter()
            .map(|(policy, embs)| MultiVectorBatch {
                policy: policy.clone(),
                embeddings: embs.as_slice(),
            })
            .collect();

        let writer = self
            .inner
            .as_mut()
            .ok_or_else(|| PyValueError::new_err("TableWriter already committed"))?;

        self.runtime
            .block_on(writer.write_batch_multi_deferred(&batch, &batches))
            .map_err(|e| {
                warn!("ailake-py: write_batch_multi_deferred failed: {}", e);
                PyValueError::new_err(e.to_string())
            })
    }

    /// Commit all staged batches as a new Iceberg snapshot.
    fn commit(&mut self) -> PyResult<i64> {
        let writer = self
            .inner
            .take()
            .ok_or_else(|| PyValueError::new_err("TableWriter already committed"))?;
        self.runtime
            .block_on(writer.commit())
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }
}

/// Search a table for the top-k nearest vectors to `query`.
///
/// When `hybrid_text` is provided, hybrid BM25+vector search is performed:
/// HNSW retrieves a larger candidate pool, then BM25 scores are computed
/// against `hybrid_text` and fused via RRF. Requires BM25 stats to be
/// accumulated at write time via `TableWriter(bm25_text_column=...)`.
///
/// Returns a list of dicts: [{"row_id": int, "distance": float, "file": str}, ...]
#[pyfunction]
#[allow(clippy::too_many_arguments)]
#[pyo3(signature = (path, query, top_k=10, partition_filter=None, hybrid_text=None, text_column="chunk_text", bm25_weight=0.5, pruning_threshold=None, ef_search=None, rerank_factor=None, catalog_opts=None))]
fn search(
    py: Python<'_>,
    path: &str,
    query: Vec<f32>,
    top_k: usize,
    partition_filter: Option<String>,
    hybrid_text: Option<String>,
    text_column: &str,
    bm25_weight: f32,
    pruning_threshold: Option<f32>,
    ef_search: Option<usize>,
    rerank_factor: Option<usize>,
    catalog_opts: Option<std::collections::HashMap<String, String>>,
) -> PyResult<Py<PyAny>> {
    let rt = rt()?;
    debug!(
        "ailake-py: search path={} dim={} top_k={} partition={:?} hybrid={:?}",
        path,
        query.len(),
        top_k,
        partition_filter,
        hybrid_text.as_deref().map(|t| &t[..t.len().min(50)])
    );
    let (catalog, store) = local_catalog_store(path, catalog_opts.as_ref())?;
    let table = TableIdent::new("default", "table");

    let meta = rt
        .block_on(catalog.load_table(&table))
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    let dim: u32 = meta
        .properties
        .get("ailake.vector-dim")
        .and_then(|s| s.parse().ok())
        .unwrap_or(query.len() as u32);
    let vector_column = meta
        .properties
        .get("ailake.vector-column")
        .cloned()
        .unwrap_or_else(|| "embedding".into());

    let hybrid = hybrid_text.map(|qt| {
        ailake_query::HybridConfig::new(qt)
            .with_text_column(text_column)
            .with_bm25_weight(bm25_weight)
    });

    let config = SearchConfig {
        top_k,
        ef_search: ef_search.unwrap_or(50),
        pruning_threshold: pruning_threshold.unwrap_or(f32::INFINITY),
        rerank_factor,
        score_fn: None,
        partition_filter,
        hybrid,
        column_filter: None,
    };

    let results = rt
        .block_on(rs_search(
            &table,
            &query,
            config,
            &vector_column,
            dim,
            catalog,
            store,
        ))
        .map_err(|e| PyValueError::new_err(e.to_string()))?;

    let list = PyList::empty(py);
    for r in results {
        let d = PyDict::new(py);
        d.set_item("row_id", r.row_id.as_u64())?;
        d.set_item("distance", r.distance)?;
        d.set_item("file", r.file_path)?;
        list.append(d)?;
    }
    Ok(list.into())
}

/// Pure BM25 full-text search (no vector query required).
///
/// Scans all Parquet files and ranks rows by BM25 score against `query_text`.
/// IDF stats must be accumulated at write time via `TableWriter(bm25_text_column=...)`.
/// O(N) complexity per call — best for small/medium tables or offline ranking.
///
/// Returns a list of dicts: [{"row_id": int, "distance": float, "file": str}, ...]
/// where `distance` is the negated BM25 score (lower = more relevant, for consistency
/// with the vector search convention).
#[pyfunction]
#[pyo3(signature = (path, query_text, top_k=10, text_column="chunk_text", partition_filter=None, catalog_opts=None))]
fn search_text(
    py: Python<'_>,
    path: &str,
    query_text: &str,
    top_k: usize,
    text_column: &str,
    partition_filter: Option<String>,
    catalog_opts: Option<std::collections::HashMap<String, String>>,
) -> PyResult<Py<PyAny>> {
    use ailake_query::search_text as rs_search_text;
    let rt = rt()?;
    let (catalog, store) = local_catalog_store(path, catalog_opts.as_ref())?;
    let table = TableIdent::new("default", "table");
    let pf = partition_filter.as_deref();
    let results = rt
        .block_on(rs_search_text(
            &table,
            query_text,
            &[text_column],
            top_k,
            catalog,
            store,
            pf,
        ))
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    let list = PyList::empty(py);
    for r in results {
        let d = PyDict::new(py);
        d.set_item("row_id", r.row_id.as_u64())?;
        d.set_item("distance", r.distance)?;
        d.set_item("file", r.file_path)?;
        list.append(d)?;
    }
    Ok(list.into())
}

/// Search a table and fetch the full row data for top-k results.
///
/// Returns IPC-serialized bytes of a RecordBatch containing all Parquet columns for
/// the matched rows, plus a `_distance: float32` column appended at the end.
/// Rows are ordered by ascending distance (nearest first).
///
/// Python side deserializes with: `pyarrow.ipc.open_file(io.BytesIO(bytes)).read_all()`
#[pyfunction]
#[allow(clippy::too_many_arguments)]
#[pyo3(signature = (path, query, top_k=10, partition_filter=None, hybrid_text=None, text_column="chunk_text", bm25_weight=0.5, pruning_threshold=None, ef_search=None, rerank_factor=None, catalog_opts=None))]
fn search_with_data(
    py: Python<'_>,
    path: &str,
    query: Vec<f32>,
    top_k: usize,
    partition_filter: Option<String>,
    hybrid_text: Option<String>,
    text_column: &str,
    bm25_weight: f32,
    pruning_threshold: Option<f32>,
    ef_search: Option<usize>,
    rerank_factor: Option<usize>,
    catalog_opts: Option<std::collections::HashMap<String, String>>,
) -> PyResult<Py<PyAny>> {
    let rt = rt()?;
    debug!(
        "ailake-py: search_with_data path={} dim={} top_k={} partition={:?} hybrid={:?}",
        path,
        query.len(),
        top_k,
        partition_filter,
        hybrid_text.as_deref().map(|t| &t[..t.len().min(50)])
    );

    let (catalog, store) = local_catalog_store(path, catalog_opts.as_ref())?;
    let table = TableIdent::new("default", "table");

    let meta = rt
        .block_on(catalog.load_table(&table))
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    let dim: u32 = meta
        .properties
        .get("ailake.vector-dim")
        .and_then(|s| s.parse().ok())
        .unwrap_or(query.len() as u32);
    let vector_column = meta
        .properties
        .get("ailake.vector-column")
        .cloned()
        .unwrap_or_else(|| "embedding".into());

    let hybrid = hybrid_text.map(|qt| {
        ailake_query::HybridConfig::new(qt)
            .with_text_column(text_column)
            .with_bm25_weight(bm25_weight)
    });

    let config = SearchConfig {
        top_k,
        ef_search: ef_search.unwrap_or(50),
        pruning_threshold: pruning_threshold.unwrap_or(f32::INFINITY),
        rerank_factor,
        score_fn: None,
        partition_filter,
        hybrid,
        column_filter: None,
    };

    let results = rt
        .block_on(rs_search(
            &table,
            &query,
            config,
            &vector_column,
            dim,
            Arc::clone(&catalog),
            Arc::clone(&store),
        ))
        .map_err(|e| PyValueError::new_err(e.to_string()))?;

    let batch = rt
        .block_on(rs_fetch_rows(
            &results,
            store,
            &vector_column,
            dim,
            &meta.schema_fields,
        ))
        .map_err(|e| PyValueError::new_err(e.to_string()))?;

    let ipc_bytes = record_batch_to_ipc(&batch)?;
    Ok(PyBytes::new(py, &ipc_bytes).into())
}

fn record_batch_to_ipc(batch: &RecordBatch) -> PyResult<Vec<u8>> {
    let mut buf = Vec::new();
    {
        let mut writer = FileWriter::try_new(&mut buf, batch.schema_ref())
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        writer
            .write(batch)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        writer
            .finish()
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
    }
    Ok(buf)
}

/// Assemble a list of text chunks into structured XML context for LLM input.
///
/// Args:
///   chunks: list[dict] with keys: document_id, chunk_index, chunk_text,
///           and optional: document_title, section_path, source_uri, distance,
///           embedding (list[float] — enables cosine-distance dedup; chunks
///           without an "embedding" key are never deduplicated against anything,
///           matching the Rust default when no embedding is available).
///   max_tokens: int — token budget (4 chars ≈ 1 token)
///   dedup_threshold: float — cosine distance below which chunks are deduplicated.
///                    Only takes effect for chunks carrying an "embedding" key.
///   group_by_document: bool — group and sort chunks by document_id/chunk_index
///                      before rendering (default True).
///   max_chunks_per_document: int — cap chunks per document group (default 10).
///
/// Returns a dict: {"text": str, "chunk_count": int, "token_estimate": int}.
#[pyfunction]
#[pyo3(signature = (chunks, max_tokens=4096, dedup_threshold=0.05, group_by_document=true, max_chunks_per_document=10))]
fn assemble_context(
    py: Python<'_>,
    chunks: Vec<Bound<'_, PyDict>>,
    max_tokens: usize,
    dedup_threshold: f32,
    group_by_document: bool,
    max_chunks_per_document: usize,
) -> PyResult<Py<PyAny>> {
    let config = ContextAssemblerConfig {
        max_tokens,
        dedup_threshold,
        group_by_document,
        max_chunks_per_document,
    };
    let ca = ContextAssembler::new(config);

    let rust_chunks: Vec<Chunk> = chunks
        .iter()
        .map(|d| {
            let get_str = |key: &str| -> String {
                d.get_item(key)
                    .ok()
                    .flatten()
                    .and_then(|v| v.extract::<String>().ok())
                    .unwrap_or_default()
            };
            let get_opt = |key: &str| -> Option<String> {
                d.get_item(key)
                    .ok()
                    .flatten()
                    .and_then(|v| v.extract::<String>().ok())
            };
            let embedding: Option<Vec<f32>> = d
                .get_item("embedding")
                .ok()
                .flatten()
                .and_then(|v| v.extract::<Vec<f32>>().ok());
            Chunk {
                document_id: get_str("document_id"),
                chunk_index: d
                    .get_item("chunk_index")
                    .ok()
                    .flatten()
                    .and_then(|v| v.extract::<u32>().ok())
                    .unwrap_or(0),
                chunk_text: get_str("chunk_text"),
                document_title: get_opt("document_title"),
                section_path: get_opt("section_path"),
                source_uri: get_opt("source_uri"),
                distance: d
                    .get_item("distance")
                    .ok()
                    .flatten()
                    .and_then(|v| v.extract::<f32>().ok())
                    .unwrap_or(0.0),
                embedding,
            }
        })
        .collect();

    let ctx = ca.assemble_chunks(rust_chunks);
    let d = PyDict::new(py);
    d.set_item("text", ctx.text)?;
    d.set_item("chunk_count", ctx.chunk_count)?;
    d.set_item("token_estimate", ctx.token_estimate)?;
    Ok(d.into())
}

/// Migrate an AI-Lake table from one embedding model to another.
///
/// Args:
///   path: filesystem path to the table root (same as TableWriter path)
///   old_column: name of the existing embedding column (e.g. "embedding")
///   new_column: name for the migrated embedding column (e.g. "embedding_v2")
///   text_column: Parquet column that holds the raw text to re-embed (default "chunk_text")
///   embed_fn: callable(list[str]) -> list[list[float]] — your embedding model
///   strategy: "atomic_replace" (lower storage) or "dual_write_then_cutover" (zero downtime)
///   batch_size: number of texts per embed_fn call (default 512)
///   new_model: optional model name stored in Iceberg properties after migration
///   new_model_version: optional version tag for the new model
#[pyfunction]
#[allow(clippy::too_many_arguments)]
#[pyo3(signature = (path, old_column, new_column, embed_fn, text_column="chunk_text", strategy="dual_write_then_cutover", batch_size=512, new_model=None, new_model_version=None, on_progress=None, catalog_opts=None))]
fn migrate_embeddings(
    py: Python<'_>,
    path: &str,
    old_column: &str,
    new_column: &str,
    embed_fn: Py<PyAny>,
    text_column: &str,
    strategy: &str,
    batch_size: usize,
    new_model: Option<&str>,
    new_model_version: Option<&str>,
    on_progress: Option<Py<PyAny>>,
    catalog_opts: Option<std::collections::HashMap<String, String>>,
) -> PyResult<()> {
    let migration_strategy = match strategy {
        "atomic_replace" => MigrationStrategy::AtomicReplace,
        "dual_write_then_cutover" => MigrationStrategy::DualWriteThenCutover,
        other => {
            return Err(PyValueError::new_err(format!(
                "unknown strategy '{other}' — use atomic_replace or dual_write_then_cutover"
            )))
        }
    };

    let new_model_info = new_model.map(|name| {
        let mut info = EmbeddingModelInfo::new(name);
        if let Some(v) = new_model_version {
            info = info.with_version(v);
        }
        info
    });

    // Wrap Python callable as a Rust closure.
    // Migration runs via block_on on the current thread — the GIL is held throughout.
    // We use Python::attach (pyo3 0.29 rename of with_gil) to re-acquire the GIL
    // token inside the closure; this is safe because block_on does not release the GIL.
    let embed_fn_arc: EmbedFn = {
        let embed_fn = embed_fn.clone_ref(py);
        Arc::new(move |texts: &[String]| {
            Python::attach(|py| {
                let py_texts = PyList::new(py, texts)
                    .map_err(|e| ailake_core::AilakeError::InvalidArgument(e.to_string()))?;
                let result = embed_fn
                    .call1(py, (py_texts,))
                    .map_err(|e| ailake_core::AilakeError::InvalidArgument(e.to_string()))?;
                // extract Vec<Vec<f32>> directly — works for list[list[float]] and np.ndarray
                let vecs: Vec<Vec<f32>> = result.bind(py).extract().map_err(|e| {
                    ailake_core::AilakeError::InvalidArgument(format!(
                        "embed_fn must return list[list[float]]: {e}"
                    ))
                })?;
                Ok(vecs)
            })
        })
    };

    let progress_arc: Option<ProgressFn> = on_progress.map(|cb| {
        let cb = cb.clone_ref(py);
        let arc: ProgressFn = Arc::new(move |p: MigrationProgress| {
            Python::attach(|py| {
                let kwargs = PyDict::new(py);
                let _ = kwargs.set_item("files_done", p.files_done);
                let _ = kwargs.set_item("files_total", p.files_total);
                let _ = kwargs.set_item("rows_migrated", p.rows_migrated);
                let _ = cb.call(py, (), Some(&kwargs));
            });
        });
        arc
    });

    let (catalog, store) = local_catalog_store(path, catalog_opts.as_ref())?;
    let table = TableIdent::new("default", "table");

    let job = MigrationJob {
        table,
        old_column: old_column.to_string(),
        new_column: new_column.to_string(),
        text_column: text_column.to_string(),
        embed_fn: embed_fn_arc,
        strategy: migration_strategy,
        batch_size,
        new_model: new_model_info,
        on_progress: progress_arc,
    };

    let rt = rt()?;
    rt.block_on(job.run(catalog, store))
        .map_err(|e| PyValueError::new_err(e.to_string()))
}

fn parse_metric(s: &str) -> PyResult<VectorMetric> {
    match s {
        "cosine" => Ok(VectorMetric::Cosine),
        "euclidean" => Ok(VectorMetric::Euclidean),
        "dot_product" | "dotproduct" => Ok(VectorMetric::DotProduct),
        "normalized_cosine" => Ok(VectorMetric::NormalizedCosine),
        other => Err(PyValueError::new_err(format!(
            "unknown metric '{other}' — use cosine, euclidean, dot_product, or normalized_cosine"
        ))),
    }
}

/// Specification for one vector column in a multimodal write or search.
///
/// Args:
///   column: str — vector column name (e.g. "embedding", "image_embedding")
///   dim: int — dimensionality of vectors in this column
///   metric: str — "cosine" | "euclidean" | "dot_product" | "normalized_cosine"
///   modality: str | None — "text" | "image" | "audio" | "video" (optional tag)
///   precision: str — "f16" (default) | "f32" | "i8"
///   pre_normalize: bool — normalize this column's vectors to unit L2 at write time
///   hnsw_m: int | None — HNSW M for this column's index (None = table/lib default)
///   hnsw_ef_construction: int | None — HNSW ef_construction for this column's index
#[pyclass]
pub struct VectorColSpec {
    #[pyo3(get, set)]
    pub column: String,
    #[pyo3(get, set)]
    pub dim: u32,
    #[pyo3(get, set)]
    pub metric: String,
    #[pyo3(get, set)]
    pub modality: Option<String>,
    #[pyo3(get, set)]
    pub precision: String,
    #[pyo3(get, set)]
    pub pre_normalize: bool,
    #[pyo3(get, set)]
    pub hnsw_m: Option<u32>,
    #[pyo3(get, set)]
    pub hnsw_ef_construction: Option<u32>,
}

#[pymethods]
impl VectorColSpec {
    #[new]
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (column, dim, metric="cosine", modality=None, precision="f16", pre_normalize=false, hnsw_m=None, hnsw_ef_construction=None))]
    fn new(
        column: String,
        dim: u32,
        metric: &str,
        modality: Option<String>,
        precision: &str,
        pre_normalize: bool,
        hnsw_m: Option<u32>,
        hnsw_ef_construction: Option<u32>,
    ) -> Self {
        Self {
            column,
            dim,
            metric: metric.to_string(),
            modality,
            precision: precision.to_string(),
            pre_normalize,
            hnsw_m,
            hnsw_ef_construction,
        }
    }
}

fn parse_precision(s: &str) -> ailake_core::VectorPrecision {
    match s {
        "f32" => ailake_core::VectorPrecision::F32,
        "i8" => ailake_core::VectorPrecision::I8,
        _ => ailake_core::VectorPrecision::F16,
    }
}

/// Cross-modal search: run independent searches across N vector columns and
/// fuse results using Reciprocal Rank Fusion (RRF).
///
/// Args:
///   path: str — table directory (same as search())
///   queries: list[tuple[str, list[float], float]] — (column_name, query_vec, weight)
///            weight is relative importance (1.0 = equal). Typical: 0.7 for text, 0.3 for image.
///   top_k: int — number of final fused results to return
///   dim: int | None — vector dimension (auto-detected from table metadata when None)
///
/// Returns a list of dicts: [{"row_id": int, "rrf_score": float, "file": str}, ...]
/// rrf_score is higher for better results.
#[pyfunction]
#[allow(clippy::too_many_arguments)]
#[pyo3(signature = (path, queries, top_k=10, dim=None, partition_filter=None, ef_search=None, pruning_threshold=None, rerank_factor=None, catalog_opts=None))]
fn search_multimodal(
    py: Python<'_>,
    path: &str,
    queries: Vec<(String, Vec<f32>, f32)>,
    top_k: usize,
    dim: Option<u32>,
    partition_filter: Option<String>,
    ef_search: Option<usize>,
    pruning_threshold: Option<f32>,
    rerank_factor: Option<usize>,
    catalog_opts: Option<std::collections::HashMap<String, String>>,
) -> PyResult<Py<PyAny>> {
    let rt = rt()?;
    let (catalog, store) = local_catalog_store(path, catalog_opts.as_ref())?;
    let table = TableIdent::new("default", "table");

    // Load metadata once to resolve per-column dims.
    // `dim` arg (if given) overrides for the primary column only; secondary
    // columns resolve via `ailake.dim-<col>` properties written at write time.
    let table_meta = rt
        .block_on(catalog.load_table(&table))
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    let primary_col = table_meta
        .properties
        .get("ailake.vector-column")
        .cloned()
        .unwrap_or_default();
    let primary_dim: u32 = dim.unwrap_or_else(|| {
        table_meta
            .properties
            .get("ailake.vector-dim")
            .and_then(|s| s.parse().ok())
            .unwrap_or_else(|| queries.first().map(|(_, q, _)| q.len() as u32).unwrap_or(0))
    });

    let modal_queries: Vec<ModalQuery<'_>> = queries
        .iter()
        .map(|(col, q, w)| {
            let col_dim = if col == &primary_col {
                primary_dim
            } else {
                table_meta
                    .properties
                    .get(&format!("ailake.dim-{col}"))
                    .and_then(|s| s.parse().ok())
                    .unwrap_or(q.len() as u32)
            };
            ModalQuery {
                column: col.as_str(),
                query: q.as_slice(),
                weight: *w,
                dim: col_dim,
            }
        })
        .collect();

    let config = SearchConfig {
        top_k,
        ef_search: ef_search.unwrap_or(50),
        pruning_threshold: pruning_threshold.unwrap_or(f32::INFINITY),
        rerank_factor,
        score_fn: None,
        partition_filter,
        hybrid: None,
        column_filter: None,
    };

    let results = rt
        .block_on(rs_search_multimodal(
            &table,
            &modal_queries,
            config,
            catalog,
            store,
            FusionMethod::Rrf,
        ))
        .map_err(|e| PyValueError::new_err(e.to_string()))?;

    let list = PyList::empty(py);
    for r in results {
        let d = PyDict::new(py);
        d.set_item("row_id", r.row_id.as_u64())?;
        // distance stores -rrf_score; expose rrf_score (higher = better) to caller
        d.set_item("rrf_score", -r.distance)?;
        d.set_item("file", r.file_path)?;
        list.append(d)?;
    }
    Ok(list.into())
}

/// In-memory bounded ring buffer for agent short-term memory.
///
/// Stores the N most recent (text, embedding) pairs. On overflow the oldest entry
/// is evicted. Supports brute-force cosine search and draining to an AI-Lake table.
#[pyclass(name = "WorkingMemoryBuffer")]
pub struct PyWorkingMemoryBuffer {
    inner: ailake_query::WorkingMemoryBuffer,
    runtime: tokio::runtime::Runtime,
}

#[pymethods]
impl PyWorkingMemoryBuffer {
    #[new]
    #[pyo3(signature = (max_rows=1000))]
    fn new(max_rows: usize) -> PyResult<Self> {
        Ok(Self {
            inner: ailake_query::WorkingMemoryBuffer::new(max_rows),
            runtime: rt()?,
        })
    }

    /// Add an entry. Evicts the oldest entry when at capacity.
    #[pyo3(signature = (text, embedding, importance=1.0))]
    fn push(&mut self, text: String, embedding: Vec<f32>, importance: f32) {
        self.inner.push(text, embedding, importance);
    }

    /// Brute-force cosine search over the buffer.
    ///
    /// Returns list of dicts: [{"text": str, "distance": float, "importance": float}, ...]
    /// sorted by ascending distance (most similar first).
    fn search(&self, py: Python<'_>, query: Vec<f32>, top_k: usize) -> PyResult<Py<PyAny>> {
        let results = self.inner.search(&query, top_k);
        let list = PyList::empty(py);
        for (dist, entry) in results {
            let d = PyDict::new(py);
            d.set_item("text", &entry.text)?;
            d.set_item("distance", dist)?;
            d.set_item("importance", entry.importance)?;
            list.append(d)?;
        }
        Ok(list.into())
    }

    /// Write all buffered entries to an AI-Lake table and clear the buffer.
    ///
    /// Does NOT commit — call `writer.commit()` afterwards to persist the snapshot.
    fn drain_to_table(&mut self, writer: &mut TableWriter) -> PyResult<()> {
        let rs_writer = writer
            .inner
            .as_mut()
            .ok_or_else(|| PyValueError::new_err("TableWriter already committed"))?;
        self.runtime
            .block_on(self.inner.drain_to_table(rs_writer))
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    fn __len__(&self) -> usize {
        self.inner.len()
    }

    fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }

    fn is_full(&self) -> bool {
        self.inner.is_full()
    }
}

/// Recompute `recency_weight` for all records in a table using exponential decay.
///
/// Reads the `last_accessed_at` column (ISO 8601 string) from each data file,
/// applies `recency_weight = exp(-decay_lambda * days_since_access)`, rewrites
/// the column, and commits a new Iceberg snapshot.
///
/// Args:
///   path: str — table directory (same as TableWriter)
///   decay_lambda: float — decay rate. Higher = faster decay. Default 0.1.
///
/// Returns the number of files updated.
#[pyfunction]
#[pyo3(signature = (path, decay_lambda=0.1, catalog_opts=None))]
fn decay_memories(
    path: &str,
    decay_lambda: f32,
    catalog_opts: Option<std::collections::HashMap<String, String>>,
) -> PyResult<usize> {
    use ailake_core::VectorMetric;
    use ailake_query::MemoryDecayJob;

    let rt = rt()?;
    let (catalog, store) = local_catalog_store(path, catalog_opts.as_ref())?;
    let table = TableIdent::new("default", "table");

    // Load stored policy from table metadata
    let table_meta = rt
        .block_on(catalog.load_table(&table))
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    let col = table_meta
        .properties
        .get("ailake.vector-column")
        .cloned()
        .unwrap_or_else(|| "embedding".to_string());
    let dim: u32 = table_meta
        .properties
        .get("ailake.vector-dim")
        .and_then(|s| s.parse().ok())
        .unwrap_or(1536);
    let metric = table_meta
        .properties
        .get("ailake.vector-metric")
        .map(|s| match s.as_str() {
            "euclidean" | "l2" => VectorMetric::Euclidean,
            "dot" | "inner_product" | "dot_product" => VectorMetric::DotProduct,
            _ => VectorMetric::Cosine,
        })
        .unwrap_or(VectorMetric::Cosine);

    let policy = ailake_core::VectorStoragePolicy::default_f16(&col, dim, metric);
    let job = MemoryDecayJob::new(catalog, store, policy, decay_lambda);

    rt.block_on(job.run(&table))
        .map_err(|e| PyValueError::new_err(e.to_string()))
}

/// Compact small files in an AI-Lake table into a larger merged file.
///
/// Native binding over `ailake_query::compaction` (CompactionPlanner +
/// CompactionExecutor) — no external `ailake` CLI binary required, unlike the
/// pure-Python `compact()` this replaces, which silently no-op'd
/// (`{"ok": true, "files_compacted": 0}`, indistinguishable from "nothing to
/// compact") whenever the CLI binary wasn't on PATH.
///
/// Args:
///   path: table root path or URI (same value passed to TableWriter)
///   min_files: minimum eligible files required to trigger compaction (default 4)
///   target_size_bytes: files smaller than this are merge candidates (default 512 MiB,
///                      matching `ailake compact`'s own CLI default)
///   max_files_per_pass: maximum files merged in one pass (default 20)
///   deferred: when True, persists the merged Parquet immediately and builds the
///             index in the background instead of blocking until it's ready
///
/// Returns a dict: {"ok": True, "files_compacted": int, "output_path": str | None}.
#[pyfunction]
#[pyo3(signature = (path, min_files=4, target_size_bytes=536_870_912, max_files_per_pass=20, deferred=false, catalog_opts=None))]
fn compact(
    py: Python<'_>,
    path: &str,
    min_files: usize,
    target_size_bytes: u64,
    max_files_per_pass: usize,
    deferred: bool,
    catalog_opts: Option<std::collections::HashMap<String, String>>,
) -> PyResult<Py<PyAny>> {
    use ailake_core::VectorPrecision;
    use ailake_query::compaction::{CompactionConfig, CompactionExecutor, CompactionPlanner};

    let rt = rt()?;
    let (catalog, store) = local_catalog_store(path, catalog_opts.as_ref())?;
    let table = TableIdent::new("default", "table");

    let meta = rt
        .block_on(catalog.load_table(&table))
        .map_err(|e| PyValueError::new_err(e.to_string()))?;

    let dim: u32 = meta
        .properties
        .get("ailake.vector-dim")
        .and_then(|s| s.parse().ok())
        .ok_or_else(|| PyValueError::new_err("table missing ailake.vector-dim property"))?;
    let column = meta
        .properties
        .get("ailake.vector-column")
        .cloned()
        .unwrap_or_else(|| "embedding".to_string());
    let metric = parse_metric(
        meta.properties
            .get("ailake.vector-metric")
            .map(|s| s.as_str())
            .unwrap_or("cosine"),
    )
    .unwrap_or(VectorMetric::Cosine);
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

    let policy = VectorStoragePolicy {
        column_name: column,
        dim,
        metric,
        precision: VectorPrecision::F16,
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
        min_files_to_compact: min_files,
        target_file_size_bytes: target_size_bytes,
        index_strategy: Default::default(),
        max_files_per_pass,
    };
    let planner = CompactionPlanner::new(config);
    let executor = CompactionExecutor::new(Arc::clone(&store), policy);

    let result = if deferred {
        rt.block_on(executor.run_deferred(&planner, &table, Arc::clone(&catalog), "data"))
    } else {
        rt.block_on(executor.run(&planner, &table, Arc::clone(&catalog), "data"))
    }
    .map_err(|e| PyValueError::new_err(e.to_string()))?;

    let d = PyDict::new(py);
    d.set_item("ok", true)?;
    match result {
        Some(entry) => {
            d.set_item("files_compacted", 1)?;
            d.set_item("output_path", entry.path)?;
        }
        None => {
            d.set_item("files_compacted", 0)?;
            d.set_item("output_path", py.None())?;
        }
    }
    Ok(d.into())
}

/// Estimate storage usage before writing a table (pure math, no I/O).
///
/// Mirrors `ailake estimate`'s math exactly (ailake-cli/src/main.rs
/// run_estimate) — F32/F16/I8 raw vector bytes, an HNSW index-size
/// approximation, and IVF-PQ code bytes, across 6 storage-precision modes.
///
/// Args:
///   rows: number of vectors
///   dim: vector dimensionality
///   hnsw_m: HNSW M parameter — connections per node (default 16)
///   pq_m: PQ sub-vectors M for the IVF-PQ/PQ-only rows (default: dim/32, clamped [8, dim])
///
/// Returns a list of dicts, one per storage mode:
///   {"mode": str, "vectors_bytes": int, "index_bytes": int, "total_bytes": int,
///    "reduction_vs_f32_hnsw": float, "recall": str, "note": str}
#[pyfunction]
#[pyo3(signature = (rows, dim, hnsw_m=16, pq_m=None))]
fn estimate(
    py: Python<'_>,
    rows: u64,
    dim: u32,
    hnsw_m: u32,
    pq_m: Option<u32>,
) -> PyResult<Py<PyAny>> {
    let dim = dim as u64;
    let pq_m = pq_m
        .map(|m| m as u64)
        .unwrap_or_else(|| (dim / 32).max(8).min(dim));

    let vec_f32 = rows.saturating_mul(dim).saturating_mul(4);
    let vec_f16 = rows.saturating_mul(dim).saturating_mul(2);
    let vec_i8 = rows.saturating_mul(dim);

    // HNSW index: ~M×2 neighbors × 9 bytes avg (bincode overhead-adjusted).
    let hnsw_bytes = rows
        .saturating_mul(hnsw_m as u64)
        .saturating_mul(2)
        .saturating_mul(9);
    // IVF-PQ codes: 1 byte per sub-quantizer code per row.
    let pq_bytes = rows.saturating_mul(pq_m);

    let baseline_total = vec_f32 + hnsw_bytes;

    let rows_table: [(&str, u64, u64, &str, &str); 6] = [
        ("F32 (baseline)", vec_f32, hnsw_bytes, "~99%", ""),
        ("F16 (default)", vec_f16, hnsw_bytes, "~99%", ""),
        ("I8", vec_i8, hnsw_bytes, "~97%", ""),
        (
            "F16 + IVF-PQ index",
            vec_f16,
            pq_bytes,
            "~99%",
            "reranks with raw F16",
        ),
        (
            "I8  + IVF-PQ index",
            vec_i8,
            pq_bytes,
            "~97%",
            "reranks with raw I8",
        ),
        (
            "PQ-only (pq_only=True)",
            0,
            pq_bytes,
            "~94%",
            "no reranking",
        ),
    ];

    let list = PyList::empty(py);
    for (mode, vectors_bytes, index_bytes, recall, note) in rows_table {
        let total = vectors_bytes + index_bytes;
        let reduction = baseline_total as f64 / total.max(1) as f64;
        let d = PyDict::new(py);
        d.set_item("mode", mode)?;
        d.set_item("vectors_bytes", vectors_bytes)?;
        d.set_item("index_bytes", index_bytes)?;
        d.set_item("total_bytes", total)?;
        d.set_item("reduction_vs_f32_hnsw", reduction)?;
        d.set_item("recall", recall)?;
        d.set_item("note", note)?;
        list.append(d)?;
    }
    Ok(list.into())
}

/// Mark rows as deleted in a V3 AI-Lake table using Iceberg Deletion Vectors.
///
/// Writes a Roaring Bitmap blob into a Puffin `.dvd` file and commits a new
/// snapshot so the deleted rows are invisible to all subsequent searches.
/// The data file itself is not modified; DVs are incremental and mergeable.
///
/// Args:
///     table_path: path to the table directory (local or object-store URL).
///     file_path: path of the Parquet data file (as shown by `ailake.info()`).
///     row_ids: list of 0-based row positions to delete.
///
/// Raises:
///     ValueError: if the table is format-version < 3, or file_path not found.
///
/// Example::
///
///     ailake.delete_rows(
///         "s3://my-lake/docs",
///         "data/part-00001.parquet",
///         [5, 10, 42],
///     )
#[pyfunction]
#[pyo3(signature = (table_path, file_path, row_ids, catalog_opts=None))]
fn delete_rows(
    table_path: &str,
    file_path: &str,
    row_ids: Vec<u32>,
    catalog_opts: Option<std::collections::HashMap<String, String>>,
) -> PyResult<()> {
    let rt = rt()?;
    let (catalog, store) = local_catalog_store(table_path, catalog_opts.as_ref())?;
    let table = TableIdent::new("default", "table");

    rt.block_on(rs_delete_rows(catalog, store, &table, file_path, &row_ids))
        .map_err(|e| PyValueError::new_err(e.to_string()))
}

/// Current UTC time as Unix epoch nanoseconds.
///
/// Use to populate `created_at` and `last_accessed_at` columns in
/// `LlmContextSchema` / `EpisodicMemorySchema` tables. The matching Arrow
/// type is `pa.timestamp('ns', tz='UTC')`.
#[pyfunction]
fn now_ns() -> i64 {
    ailake_core::now_ns()
}

/// Add a column to the table schema without rewriting data files (Phase G).
///
/// Old files missing the column will return `initial_default` (or null if omitted)
/// at read time — no compaction needed.
///
/// Args:
///     table_path: path to the table (local dir or s3://... URI)
///     name: column name
///     iceberg_type: Iceberg type string — "int", "long", "float", "double",
///         "boolean", "string", "date", "timestamp", "timestamptz", "binary"
///     required: if True, the field is marked non-nullable (use False for additions)
///     initial_default: JSON-serialisable scalar returned for old files, e.g. 0, 0.0,
///         "unknown", True, None → null
///     write_default: default written to new files when no value is supplied
///     doc: optional field documentation stored in the schema
///
/// Returns:
///     new schema-id (int)
#[pyfunction]
#[allow(clippy::too_many_arguments)]
#[pyo3(signature = (table_path, name, iceberg_type, required=false, initial_default=None, write_default=None, doc=None, catalog_opts=None))]
fn add_column(
    table_path: &str,
    name: &str,
    iceberg_type: &str,
    required: bool,
    initial_default: Option<pyo3::Bound<'_, PyAny>>,
    write_default: Option<pyo3::Bound<'_, PyAny>>,
    doc: Option<&str>,
    catalog_opts: Option<std::collections::HashMap<String, String>>,
) -> PyResult<i32> {
    let rt = rt()?;
    let (catalog, _store) = local_catalog_store(table_path, catalog_opts.as_ref())?;
    let table = TableIdent::new("default", "table");

    let py_to_json = |v: Option<pyo3::Bound<'_, PyAny>>| -> Option<serde_json::Value> {
        v.and_then(|py| {
            if let Ok(b) = py.extract::<bool>() {
                return Some(serde_json::Value::Bool(b));
            }
            if let Ok(i) = py.extract::<i64>() {
                return Some(serde_json::json!(i));
            }
            if let Ok(f) = py.extract::<f64>() {
                return Some(serde_json::json!(f));
            }
            if let Ok(s) = py.extract::<String>() {
                return Some(serde_json::Value::String(s));
            }
            None
        })
    };

    use ailake_catalog::{AddColumnRequest, SchemaEvolution};
    let req = AddColumnRequest {
        name: name.to_string(),
        iceberg_type: iceberg_type.to_string(),
        required,
        initial_default: py_to_json(initial_default),
        write_default: py_to_json(write_default),
        doc: doc.map(str::to_string),
    };
    let evolution = SchemaEvolution::new().add_column(req);
    rt.block_on(catalog.evolve_schema(&table, evolution))
        .map_err(|e| PyValueError::new_err(e.to_string()))
}

/// Rename a column in the table schema without rewriting data files (Phase G).
///
/// Field IDs are stable — Iceberg and the AI-Lake SDK identify columns by ID,
/// not name. Old files are not affected (reads continue to work).
///
/// Returns:
///     new schema-id (int)
#[pyfunction]
#[pyo3(signature = (table_path, old_name, new_name, catalog_opts=None))]
fn rename_column(
    table_path: &str,
    old_name: &str,
    new_name: &str,
    catalog_opts: Option<std::collections::HashMap<String, String>>,
) -> PyResult<i32> {
    let rt = rt()?;
    let (catalog, _store) = local_catalog_store(table_path, catalog_opts.as_ref())?;
    let table = TableIdent::new("default", "table");

    use ailake_catalog::SchemaEvolution;
    let evolution = SchemaEvolution::new().rename_column(old_name, new_name);
    rt.block_on(catalog.evolve_schema(&table, evolution))
        .map_err(|e| PyValueError::new_err(e.to_string()))
}

/// Logically delete all rows where `column` equals any value in `values` (Phase H).
///
/// Writes an Iceberg equality delete file, then commits a Delete snapshot that
/// inherits existing data files. Scanners will mask matching rows at read time
/// without rewriting data files.
///
/// Args:
///     table_path: path to the table (local dir or s3://... URI)
///     column: name of the equality column (e.g. "document_id", "agent_id")
///     values: list of string values identifying rows to delete
///
/// Example::
///
///     ailake.delete_where(
///         "s3://my-lake/docs",
///         "document_id",
///         ["doc-abc", "doc-def"],
///     )
#[pyfunction]
#[pyo3(signature = (table_path, column, values, catalog_opts=None))]
fn delete_where(
    table_path: &str,
    column: &str,
    values: Vec<String>,
    catalog_opts: Option<std::collections::HashMap<String, String>>,
) -> PyResult<()> {
    let rt = rt()?;
    let (catalog, store) = local_catalog_store(table_path, catalog_opts.as_ref())?;
    let table = TableIdent::new("default", "table");
    let value_refs: Vec<&str> = values.iter().map(String::as_str).collect();
    rt.block_on(ailake_query::delete_where(
        catalog,
        store,
        &table,
        column,
        &value_refs,
    ))
    .map_err(|e| PyValueError::new_err(e.to_string()))
}

/// Add a new vector column to an existing table schema without rewriting data files.
///
/// Registers the new column in `metadata.json` (Iceberg schema + ailake properties).
/// Old files return `null` for this column until `backfill_vector_column` is run.
///
/// Args:
///     table_path: path to the table (local dir or s3://... URI)
///     column: new vector column name
///     dim: vector dimensionality
///     metric: distance metric — "cosine" (default), "euclidean", "dot_product"
///     precision: storage precision — "f16" (default), "f32", "i8"
///     pre_normalize: if True, vectors are normalized to unit L2 at write time
///     hnsw_m: HNSW M parameter (connections per node, default 16)
///     hnsw_ef_construction: HNSW ef_construction (default 150)
///
/// Returns:
///     new schema-id (int)
#[pyfunction]
#[allow(clippy::too_many_arguments)]
#[pyo3(signature = (table_path, column, dim, metric="cosine", precision="f16", pre_normalize=false, hnsw_m=None, hnsw_ef_construction=None, catalog_opts=None))]
fn add_vector_column(
    table_path: &str,
    column: &str,
    dim: u32,
    metric: &str,
    precision: &str,
    pre_normalize: bool,
    hnsw_m: Option<u32>,
    hnsw_ef_construction: Option<u32>,
    catalog_opts: Option<std::collections::HashMap<String, String>>,
) -> PyResult<i32> {
    use ailake_core::{VectorColSpec as RsVectorColSpec, VectorMetric, VectorPrecision};

    let rt = rt()?;
    let (catalog, _store) = local_catalog_store(table_path, catalog_opts.as_ref())?;
    let table = TableIdent::new("default", "table");

    let metric_val = match metric {
        "euclidean" => VectorMetric::Euclidean,
        "dotproduct" | "dot_product" => VectorMetric::DotProduct,
        "normalizedcosine" | "normalized_cosine" => VectorMetric::NormalizedCosine,
        _ => VectorMetric::Cosine,
    };
    let precision_val = match precision {
        "f32" => VectorPrecision::F32,
        "i8" => VectorPrecision::I8,
        _ => VectorPrecision::F16,
    };

    let spec = RsVectorColSpec {
        column_name: column.to_string(),
        dim,
        metric: metric_val,
        precision: precision_val,
        pre_normalize,
        hnsw_m,
        hnsw_ef_construction,
    };

    rt.block_on(catalog.add_vector_column(&table, &spec))
        .map_err(|e| PyValueError::new_err(e.to_string()))
}

/// Backfill a new vector column in all existing files of a table.
///
/// Reads each file, calls `embed_fn` on `text_column`, and rewrites the file
/// with both the original vector column and the new one. Idempotent: files that
/// already contain the new column are skipped.
///
/// Args:
///     table_path: path to the table (local dir or s3://... URI)
///     column: name of the new vector column (must already exist via add_vector_column)
///     text_column: Parquet column with text to embed (default: "chunk_text")
///     embed_fn: callable(list[str]) -> list[list[float]] — your embedding function
///     batch_size: number of texts per embed_fn call (default: 512)
///
/// Example::
///
///     from openai import OpenAI
///     client = OpenAI()
///
///     def embed(texts):
///         resp = client.embeddings.create(model="text-embedding-3-small", input=texts)
///         return [e.embedding for e in resp.data]
///
///     ailake.backfill_vector_column(
///         "s3://my-lake/docs",
///         column="embedding_v2",
///         text_column="chunk_text",
///         embed_fn=embed,
///     )
#[pyfunction]
#[pyo3(signature = (table_path, column, embed_fn, text_column="chunk_text", batch_size=512, catalog_opts=None))]
fn backfill_vector_column(
    py: Python<'_>,
    table_path: &str,
    column: &str,
    embed_fn: Py<PyAny>,
    text_column: &str,
    batch_size: usize,
    catalog_opts: Option<std::collections::HashMap<String, String>>,
) -> PyResult<()> {
    use ailake_core::{VectorColSpec as RsVectorColSpec, VectorMetric, VectorPrecision};
    use ailake_query::BackfillJob;

    let rt = rt()?;
    let (catalog, store) = local_catalog_store(table_path, catalog_opts.as_ref())?;
    let table_ident = TableIdent::new("default", "table");

    // Load new column properties from the catalog to build VectorColSpec.
    let table_meta = rt
        .block_on(catalog.load_table(&table_ident))
        .map_err(|e| PyValueError::new_err(e.to_string()))?;

    let dim_key = format!("ailake.dim-{column}");
    let metric_key = format!("ailake.metric-{column}");
    let precision_key = format!("ailake.precision-{column}");

    let dim: u32 = table_meta
        .properties
        .get(&dim_key)
        .and_then(|s| s.parse().ok())
        .ok_or_else(|| {
            PyValueError::new_err(format!(
                "column '{column}' not found — call add_vector_column first"
            ))
        })?;

    let metric = match table_meta
        .properties
        .get(&metric_key)
        .map(|s| s.as_str())
        .unwrap_or("cosine")
    {
        "euclidean" => VectorMetric::Euclidean,
        "dotproduct" | "dot_product" => VectorMetric::DotProduct,
        "normalizedcosine" | "normalized_cosine" => VectorMetric::NormalizedCosine,
        _ => VectorMetric::Cosine,
    };

    let precision = match table_meta
        .properties
        .get(&precision_key)
        .map(|s| s.as_str())
        .unwrap_or("f16")
    {
        "f32" => VectorPrecision::F32,
        "i8" => VectorPrecision::I8,
        _ => VectorPrecision::F16,
    };

    let new_col = RsVectorColSpec {
        column_name: column.to_string(),
        dim,
        metric,
        precision,
        pre_normalize: false,
        hnsw_m: None,
        hnsw_ef_construction: None,
    };

    // Wrap Python callable as EmbedFn (same pattern as migrate_embeddings).
    let embed_fn_arc: ailake_query::EmbedFn = {
        let embed_fn = embed_fn.clone_ref(py);
        Arc::new(move |texts: &[String]| {
            Python::attach(|py| {
                let py_texts = PyList::new(py, texts)
                    .map_err(|e| ailake_core::AilakeError::InvalidArgument(e.to_string()))?;
                let result = embed_fn
                    .call1(py, (py_texts,))
                    .map_err(|e| ailake_core::AilakeError::InvalidArgument(e.to_string()))?;
                let vecs: Vec<Vec<f32>> = result.bind(py).extract().map_err(|e| {
                    ailake_core::AilakeError::InvalidArgument(format!(
                        "embed_fn must return list[list[float]]: {e}"
                    ))
                })?;
                Ok(vecs)
            })
        })
    };

    let job = BackfillJob {
        table: table_ident,
        text_column: text_column.to_string(),
        new_col,
        embed_fn: embed_fn_arc,
        batch_size,
        on_progress: None,
    };

    rt.block_on(job.run(catalog, store))
        .map_err(|e| PyValueError::new_err(e.to_string()))
}

#[pymodule]
fn _ailake(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<TableWriter>()?;
    m.add_class::<VectorColSpec>()?;
    m.add_class::<PyWorkingMemoryBuffer>()?;
    m.add_class::<TimestampNs>()?;
    m.add_function(wrap_pyfunction!(search, m)?)?;
    m.add_function(wrap_pyfunction!(search_text, m)?)?;
    m.add_function(wrap_pyfunction!(search_multimodal, m)?)?;
    m.add_function(wrap_pyfunction!(search_with_data, m)?)?;
    m.add_function(wrap_pyfunction!(assemble_context, m)?)?;
    m.add_function(wrap_pyfunction!(migrate_embeddings, m)?)?;
    m.add_function(wrap_pyfunction!(decay_memories, m)?)?;
    m.add_function(wrap_pyfunction!(delete_rows, m)?)?;
    m.add_function(wrap_pyfunction!(now_ns, m)?)?;
    m.add_function(wrap_pyfunction!(add_column, m)?)?;
    m.add_function(wrap_pyfunction!(rename_column, m)?)?;
    m.add_function(wrap_pyfunction!(delete_where, m)?)?;
    m.add_function(wrap_pyfunction!(hardware_info, m)?)?;
    m.add_function(wrap_pyfunction!(add_vector_column, m)?)?;
    m.add_function(wrap_pyfunction!(backfill_vector_column, m)?)?;
    m.add_function(wrap_pyfunction!(compact, m)?)?;
    m.add_function(wrap_pyfunction!(estimate, m)?)?;
    m.add_function(wrap_pyfunction!(create_table, m)?)?;
    Ok(())
}

/// Create an empty AI-Lake/Iceberg table with the given schema and policy.
///
/// Args:
///     path: table root path (local dir or s3://...)
///     dim: vector dimension
///     vector_column: vector column name (default "embedding")
///     metric: distance metric (default "cosine")
///     precision: storage precision (default "f16")
///     format_version: Iceberg format version (2 or 3, default 2)
///     hnsw_m: HNSW M parameter (default None = use native default)
///     hnsw_ef_construction: HNSW ef_construction (default None = use native default)
///     pre_normalize: normalize vectors to unit L2 (default False)
///     modality: vector modality ("text", "image", "audio", "video", default "")
///     partition_by: column to partition by (default "")
///     partition_value: runtime partition value (default "")
///     partition_column_type: Iceberg partition column type (default "")
///     partition_fields_json: JSON array of partition fields (default "")
///     fts_columns: comma-separated FTS text columns (default "")
///     fts_tokenizer: FTS tokenizer (default "")
///     embedding_model: embedding model name (default "")
///     namespace: Iceberg namespace (default "default")
///     table_name: table name (default "table")
///     catalog_opts: optional dict with "catalog", "rest_uri", etc.
///
/// Returns:
///     True on success, raises ValueError on error.
#[pyfunction]
#[allow(clippy::too_many_arguments)]
#[pyo3(signature = (path, dim, vector_column="embedding", metric="cosine", precision="f16",
    format_version=2, hnsw_m=None, hnsw_ef_construction=None, pre_normalize=false,
    modality="", partition_by="", partition_value="", partition_column_type="",
    partition_fields_json="", fts_columns="", fts_tokenizer="", embedding_model="",
    namespace="default", table_name="table", catalog_opts=None))]
fn create_table(
    path: &str,
    dim: u32,
    vector_column: &str,
    metric: &str,
    precision: &str,
    format_version: u8,
    hnsw_m: Option<u32>,
    hnsw_ef_construction: Option<u32>,
    pre_normalize: bool,
    modality: &str,
    partition_by: &str,
    partition_value: &str,
    partition_column_type: &str,
    partition_fields_json: &str,
    fts_columns: &str,
    fts_tokenizer: &str,
    embedding_model: &str,
    namespace: &str,
    table_name: &str,
    catalog_opts: Option<std::collections::HashMap<String, String>>,
) -> PyResult<bool> {
    let rt = rt()?;
    let (catalog, _store) = local_catalog_store(path, catalog_opts.as_ref())?;
    let table = TableIdent::new(namespace, table_name);

    let metric = parse_metric(metric)?;
    let precision = match precision {
        "f32" => VectorPrecision::F32,
        "i8" => VectorPrecision::I8,
        _ => VectorPrecision::F16,
    };
    let modality = if modality.is_empty() {
        None
    } else {
        match modality.parse::<VectorModality>() {
            Ok(m) => Some(m),
            Err(_) => {
                return Err(PyValueError::new_err(format!(
                    "unknown modality: {modality}"
                )))
            }
        }
    };

    let partition_fields: Vec<PartitionDef> = if partition_fields_json.is_empty() {
        vec![]
    } else {
        serde_json::from_str(partition_fields_json)
            .map_err(|e| PyValueError::new_err(format!("invalid partition_fields_json: {e}")))?
    };

    let mut extra = std::collections::HashMap::new();
    if !fts_columns.is_empty() {
        extra.insert("ailake.fts.enabled".to_string(), "true".to_string());
        extra.insert(
            "ailake.fts.text-columns".to_string(),
            fts_columns.to_string(),
        );
        if !fts_tokenizer.is_empty() {
            extra.insert(
                "ailake.fts.tokenizer".to_string(),
                fts_tokenizer.to_string(),
            );
        }
    }

    let policy = VectorStoragePolicy {
        column_name: vector_column.to_string(),
        dim,
        metric,
        precision,
        pq: None,
        keep_raw_for_reranking: true,
        pre_normalize,
        hnsw_m,
        hnsw_ef_construction,
        ivf_residual: false,
        embedding_model: if embedding_model.is_empty() {
            None
        } else {
            Some(EmbeddingModelInfo::new(embedding_model.to_string()))
        },
        modality,
        partition_by: if partition_by.is_empty() {
            None
        } else {
            Some(partition_by.to_string())
        },
        partition_value: if partition_value.is_empty() {
            None
        } else {
            Some(partition_value.to_string())
        },
        partition_column_type: if partition_column_type.is_empty() {
            None
        } else {
            Some(partition_column_type.to_string())
        },
        partition_fields,
    };

    let props = TableProperties {
        policy,
        extra,
        format_version,
        partition_column_type: None,
    };

    rt.block_on(catalog.create_table(&table, &props))
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok(true)
}

/// Return detected hardware capabilities as a plain dict.
///
/// Keys:
///   backend           — "nvidia-cuda" | "amd-rocm" | "cpu-simd"
///   has_cuda          — "true" / "false"
///   has_rocm          — "true" / "false"
///   cpu_logical_cores — number of logical cores available to rayon
///   has_avx2          — "true" / "false" (x86_64 only)
///   has_avx512        — "true" / "false" (x86_64 AVX512F only)
///   recommend_ivf_pq  — "true" / "false" for a 5 000-vector batch (threshold probe)
///
/// Example::
///
///     info = ailake.hardware_info()
///     print(info["backend"])   # "cpu-simd" or "nvidia-cuda" / "amd-rocm"
#[pyfunction]
fn hardware_info() -> std::collections::HashMap<String, String> {
    use ailake_index::hardware::HardwareBackend;
    use ailake_index::HardwareProfile;

    let p = HardwareProfile::detect();
    let backend = match p.backend {
        HardwareBackend::NvidiaCuda => "nvidia-cuda",
        HardwareBackend::AmdRocm => "amd-rocm",
        HardwareBackend::CpuSimd => "cpu-simd",
    };
    let mut m = std::collections::HashMap::new();
    m.insert("backend".into(), backend.into());
    m.insert("has_cuda".into(), p.has_cuda.to_string());
    m.insert("has_rocm".into(), p.has_rocm.to_string());
    m.insert("cpu_logical_cores".into(), p.cpu_logical_cores.to_string());
    m.insert("has_avx2".into(), p.has_avx2.to_string());
    m.insert("has_avx512".into(), p.has_avx512.to_string());
    m.insert(
        "recommend_ivf_pq".into(),
        p.recommend_ivf_pq(5_000).to_string(),
    );
    m
}
