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
use arrow_schema::{DataType, Field, Schema};
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyBytes, PyDict, PyList};
use tracing::{debug, warn};

use ailake_catalog::{
    hadoop::HadoopCatalog,
    provider::{CatalogProvider, TableIdent},
};
use ailake_core::{EmbeddingModelInfo, VectorMetric, VectorModality, VectorStoragePolicy};
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

/// Build a RecordBatch from texts and optional extra columns.
///
/// Texts become the `"text"` column (Utf8). Extra columns are inferred from
/// the Python value type of the first element:
///   - `int` → Int64
///   - `float` → Float32
///   - `bool` → Boolean
///   - `str`/other → Utf8
fn build_batch_with_extra(
    py: Python<'_>,
    texts: Vec<String>,
    extra_columns: Option<&Bound<'_, PyDict>>,
) -> PyResult<RecordBatch> {
    use arrow_array::{BooleanArray, Float32Array, Int64Array};
    use arrow_schema::Field;

    let mut fields = vec![Field::new("text", DataType::Utf8, false)];
    let text_arr: Arc<dyn arrow_array::Array> = Arc::new(StringArray::from(texts));
    let mut arrays: Vec<Arc<dyn arrow_array::Array>> = vec![text_arr];

    if let Some(extra) = extra_columns {
        for (k, v) in extra.iter() {
            let col_name: String = k.extract()?;
            let values: Vec<Py<PyAny>> = v.extract()?;

            // Infer type from first element
            let first = values.first().map(|x| x.bind(py));
            if first.as_ref().map(|x| x.is_instance_of::<pyo3::types::PyBool>()).unwrap_or(false) {
                let arr: Vec<Option<bool>> = values.iter()
                    .map(|x| x.bind(py).extract::<bool>().ok())
                    .collect();
                fields.push(Field::new(&col_name, DataType::Boolean, true));
                arrays.push(Arc::new(BooleanArray::from(arr)));
            } else if first.as_ref().map(|x| x.is_instance_of::<pyo3::types::PyFloat>()).unwrap_or(false) {
                let arr: Vec<Option<f32>> = values.iter()
                    .map(|x| x.bind(py).extract::<f32>().ok())
                    .collect();
                fields.push(Field::new(&col_name, DataType::Float32, true));
                arrays.push(Arc::new(Float32Array::from(arr)));
            } else if first.as_ref().map(|x| x.is_instance_of::<pyo3::types::PyInt>()).unwrap_or(false) {
                let arr: Vec<Option<i64>> = values.iter()
                    .map(|x| x.bind(py).extract::<i64>().ok())
                    .collect();
                fields.push(Field::new(&col_name, DataType::Int64, true));
                arrays.push(Arc::new(Int64Array::from(arr)));
            } else {
                // Default: string column
                let arr: Vec<Option<String>> = values.iter()
                    .map(|x| x.bind(py).extract::<String>().ok())
                    .collect();
                fields.push(Field::new(&col_name, DataType::Utf8, true));
                arrays.push(Arc::new(StringArray::from(arr)));
            }
        }
    }

    let schema = Arc::new(Schema::new(fields));
    RecordBatch::try_new(schema, arrays)
        .map_err(|e| PyValueError::new_err(e.to_string()))
}

fn local_catalog_store(path: &str) -> (Arc<dyn CatalogProvider>, Arc<dyn Store>) {
    let store: Arc<dyn Store> = Arc::new(LocalStore::new(path));
    // Use a file:// URI as warehouse so that Iceberg metadata.json and manifest
    // files contain absolute file:// paths. Required for Trino's Iceberg
    // connector and any reader that resolves location URIs strictly.
    // LocalStore::full_path strips the file:// prefix before I/O.
    let canonical = std::fs::canonicalize(path).unwrap_or_else(|_| std::path::PathBuf::from(path));
    let warehouse_uri = format!("file://{}", canonical.display());
    let catalog: Arc<dyn CatalogProvider> =
        Arc::new(HadoopCatalog::new(Arc::clone(&store), &warehouse_uri));
    (catalog, store)
}

/// Python-facing table writer. Wraps ailake_query::TableWriter.
#[pyclass]
pub struct TableWriter {
    inner: Option<RsTableWriter>,
    runtime: tokio::runtime::Runtime,
    /// Optional Python callable used for Pattern B: embed_fn(texts) -> list[list[float]]
    embed_fn: Option<Py<PyAny>>,
}

#[pymethods]
impl TableWriter {
    /// Open (or create) an AI-Lake table at `path` on the local filesystem.
    #[new]
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (path, vector_column="embedding", dim=1536, metric="cosine", pre_normalize=false, hnsw_m=None, hnsw_ef_construction=None, pq_only=false, ivf_residual=false, embedding_model=None, embedding_model_version=None, embed_fn=None, partition_by=None, partition_value=None, bm25_text_column=None, format_version=2))]
    fn new(
        py: Python<'_>,
        path: &str,
        vector_column: &str,
        dim: u32,
        metric: &str,
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
        bm25_text_column: Option<String>,
        format_version: u8,
    ) -> PyResult<Self> {
        let rt = rt()?;
        debug!(
            "ailake-py: TableWriter::new path={} dim={} metric={} pre_normalize={} hnsw_m={:?} hnsw_ef={:?} pq_only={} ivf_residual={} embedding_model={:?} partition_by={:?}",
            path, dim, metric, pre_normalize, hnsw_m, hnsw_ef_construction, pq_only, ivf_residual, embedding_model, partition_by
        );
        let mut policy =
            VectorStoragePolicy::default_f16(vector_column, dim, parse_metric(metric)?);
        policy.pre_normalize = pre_normalize;
        policy.hnsw_m = hnsw_m;
        policy.hnsw_ef_construction = hnsw_ef_construction;
        policy.keep_raw_for_reranking = !pq_only;
        policy.ivf_residual = ivf_residual;
        policy.partition_by = partition_by;
        policy.partition_value = partition_value;
        if let Some(model_name) = embedding_model {
            let mut model_info = EmbeddingModelInfo::new(model_name).with_dim(dim);
            if let Some(version) = embedding_model_version {
                model_info = model_info.with_version(version);
            }
            policy.embedding_model = Some(model_info);
        }
        let (catalog, store) = local_catalog_store(path);
        let table = TableIdent::new("default", "table");

        let stored_embed_fn = embed_fn.map(|f| f.clone_ref(py));
        let mut writer = rt
            .block_on(RsTableWriter::create_or_open(catalog, store, policy, table, format_version))
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        if let Some(col) = bm25_text_column {
            writer = writer.with_bm25(col);
        }

        Ok(Self {
            inner: Some(writer),
            runtime: rt,
            embed_fn: stored_embed_fn,
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
    fn write_batch_multi(
        &mut self,
        py: Python<'_>,
        texts: Vec<String>,
        columns: Vec<(Py<VectorColSpec>, Vec<Vec<f32>>)>,
    ) -> PyResult<()> {
        if columns.is_empty() {
            return Err(PyValueError::new_err(
                "write_batch_multi requires at least one column",
            ));
        }

        let schema = Arc::new(Schema::new(vec![Field::new("text", DataType::Utf8, false)]));
        let text_arr = StringArray::from(texts);
        let batch = RecordBatch::try_new(schema, vec![Arc::new(text_arr)])
            .map_err(|e| PyValueError::new_err(e.to_string()))?;

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
    /// automatically when `IndexStatus::Ready`.
    ///
    /// Use when ingest throughput matters more than immediate HNSW availability.
    ///
    /// Args:
    ///   texts: list[str]
    ///   columns: list[tuple[VectorColSpec, list[list[float]]]]
    fn write_batch_multi_deferred(
        &mut self,
        py: Python<'_>,
        texts: Vec<String>,
        columns: Vec<(Py<VectorColSpec>, Vec<Vec<f32>>)>,
    ) -> PyResult<()> {
        if columns.is_empty() {
            return Err(PyValueError::new_err(
                "write_batch_multi_deferred requires at least one column",
            ));
        }

        let schema = Arc::new(Schema::new(vec![Field::new("text", DataType::Utf8, false)]));
        let text_arr = StringArray::from(texts);
        let batch = RecordBatch::try_new(schema, vec![Arc::new(text_arr)])
            .map_err(|e| PyValueError::new_err(e.to_string()))?;

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
#[pyo3(signature = (path, query, top_k=10, partition_filter=None, hybrid_text=None, text_column="chunk_text", bm25_weight=0.5))]
fn search(
    py: Python<'_>,
    path: &str,
    query: Vec<f32>,
    top_k: usize,
    partition_filter: Option<String>,
    hybrid_text: Option<String>,
    text_column: &str,
    bm25_weight: f32,
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
    let (catalog, store) = local_catalog_store(path);
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
        ef_search: 50,
        pruning_threshold: f32::INFINITY,
        rerank_factor: None,
        score_fn: None,
        partition_filter,
        hybrid,
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
#[pyo3(signature = (path, query_text, top_k=10, text_column="chunk_text", partition_filter=None))]
fn search_text(
    py: Python<'_>,
    path: &str,
    query_text: &str,
    top_k: usize,
    text_column: &str,
    partition_filter: Option<String>,
) -> PyResult<Py<PyAny>> {
    use ailake_query::search_text as rs_search_text;
    let rt = rt()?;
    let (catalog, store) = local_catalog_store(path);
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
#[pyo3(signature = (path, query, top_k=10, partition_filter=None))]
fn search_with_data(
    py: Python<'_>,
    path: &str,
    query: Vec<f32>,
    top_k: usize,
    partition_filter: Option<String>,
) -> PyResult<Py<PyAny>> {
    let rt = rt()?;
    debug!(
        "ailake-py: search_with_data path={} dim={} top_k={} partition={:?}",
        path,
        query.len(),
        top_k,
        partition_filter
    );

    let (catalog, store) = local_catalog_store(path);
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

    let config = SearchConfig {
        top_k,
        ef_search: 50,
        pruning_threshold: f32::INFINITY,
        rerank_factor: None,
        score_fn: None,
        partition_filter,
        hybrid: None,
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
        .block_on(rs_fetch_rows(&results, store, &vector_column, dim))
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
///           and optional: document_title, section_path, source_uri, distance
///   max_tokens: int — token budget (4 chars ≈ 1 token)
///   dedup_threshold: float — cosine distance below which chunks are deduplicated
#[pyfunction]
#[pyo3(signature = (chunks, max_tokens=4096, dedup_threshold=0.05))]
fn assemble_context(
    chunks: Vec<Bound<'_, PyDict>>,
    max_tokens: usize,
    dedup_threshold: f32,
) -> PyResult<String> {
    let config = ContextAssemblerConfig {
        max_tokens,
        dedup_threshold,
        ..Default::default()
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
                embedding: None,
            }
        })
        .collect();

    let ctx = ca.assemble_chunks(rust_chunks);
    Ok(ctx.text)
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
#[pyo3(signature = (path, old_column, new_column, embed_fn, text_column="chunk_text", strategy="dual_write_then_cutover", batch_size=512, new_model=None, new_model_version=None, on_progress=None))]
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

    let (catalog, store) = local_catalog_store(path);
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
}

#[pymethods]
impl VectorColSpec {
    #[new]
    #[pyo3(signature = (column, dim, metric="cosine", modality=None))]
    fn new(column: String, dim: u32, metric: &str, modality: Option<String>) -> Self {
        Self {
            column,
            dim,
            metric: metric.to_string(),
            modality,
        }
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
#[pyo3(signature = (path, queries, top_k=10, dim=None, partition_filter=None))]
fn search_multimodal(
    py: Python<'_>,
    path: &str,
    queries: Vec<(String, Vec<f32>, f32)>,
    top_k: usize,
    dim: Option<u32>,
    partition_filter: Option<String>,
) -> PyResult<Py<PyAny>> {
    let rt = rt()?;
    let (catalog, store) = local_catalog_store(path);
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
        ef_search: 50,
        pruning_threshold: f32::INFINITY,
        rerank_factor: None,
        score_fn: None,
        partition_filter,
        hybrid: None,
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
#[pyo3(signature = (path, decay_lambda=0.1))]
fn decay_memories(path: &str, decay_lambda: f32) -> PyResult<usize> {
    use ailake_core::VectorMetric;
    use ailake_query::MemoryDecayJob;

    let rt = rt()?;
    let (catalog, store) = local_catalog_store(path);
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
        .and_then(|s| match s.as_str() {
            "euclidean" | "l2" => Some(VectorMetric::Euclidean),
            "dot" | "inner_product" | "dot_product" => Some(VectorMetric::DotProduct),
            _ => Some(VectorMetric::Cosine),
        })
        .unwrap_or(VectorMetric::Cosine);

    let policy = ailake_core::VectorStoragePolicy::default_f16(&col, dim, metric);
    let job = MemoryDecayJob::new(catalog, store, policy, decay_lambda);

    rt.block_on(job.run(&table))
        .map_err(|e| PyValueError::new_err(e.to_string()))
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
#[pyo3(signature = (table_path, file_path, row_ids))]
fn delete_rows(table_path: &str, file_path: &str, row_ids: Vec<u32>) -> PyResult<()> {
    let rt = rt()?;
    let (catalog, store) = local_catalog_store(table_path);
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

#[pymodule]
fn _ailake(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<TableWriter>()?;
    m.add_class::<VectorColSpec>()?;
    m.add_class::<PyWorkingMemoryBuffer>()?;
    m.add_function(wrap_pyfunction!(search, m)?)?;
    m.add_function(wrap_pyfunction!(search_text, m)?)?;
    m.add_function(wrap_pyfunction!(search_multimodal, m)?)?;
    m.add_function(wrap_pyfunction!(search_with_data, m)?)?;
    m.add_function(wrap_pyfunction!(assemble_context, m)?)?;
    m.add_function(wrap_pyfunction!(migrate_embeddings, m)?)?;
    m.add_function(wrap_pyfunction!(decay_memories, m)?)?;
    m.add_function(wrap_pyfunction!(delete_rows, m)?)?;
    m.add_function(wrap_pyfunction!(now_ns, m)?)?;
    Ok(())
}
