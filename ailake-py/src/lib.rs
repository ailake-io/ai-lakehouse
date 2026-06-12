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
use ailake_core::{AilakeResult, EmbeddingModelInfo, VectorMetric, VectorStoragePolicy};
use ailake_query::{
    fetch_rows as rs_fetch_rows, search as rs_search, Chunk, ContextAssembler,
    ContextAssemblerConfig, MigrationJob, MigrationStrategy, SearchConfig,
    TableWriter as RsTableWriter,
};
use ailake_store::{store::Store, LocalStore};

fn rt() -> PyResult<tokio::runtime::Runtime> {
    tokio::runtime::Runtime::new().map_err(|e| {
        PyRuntimeError::new_err(format!("ailake: failed to create Tokio runtime: {e}"))
    })
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
}

#[pymethods]
impl TableWriter {
    /// Open (or create) an AI-Lake table at `path` on the local filesystem.
    #[new]
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (path, vector_column="embedding", dim=1536, metric="cosine", pre_normalize=false, hnsw_m=None, hnsw_ef_construction=None, pq_only=false, ivf_residual=false, embedding_model=None, embedding_model_version=None))]
    fn new(
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
    ) -> PyResult<Self> {
        let rt = rt()?;
        debug!(
            "ailake-py: TableWriter::new path={} dim={} metric={} pre_normalize={} hnsw_m={:?} hnsw_ef={:?} pq_only={} ivf_residual={} embedding_model={:?}",
            path, dim, metric, pre_normalize, hnsw_m, hnsw_ef_construction, pq_only, ivf_residual, embedding_model
        );
        let mut policy =
            VectorStoragePolicy::default_f16(vector_column, dim, parse_metric(metric)?);
        policy.pre_normalize = pre_normalize;
        policy.hnsw_m = hnsw_m;
        policy.hnsw_ef_construction = hnsw_ef_construction;
        policy.keep_raw_for_reranking = !pq_only;
        policy.ivf_residual = ivf_residual;
        if let Some(model_name) = embedding_model {
            let mut model_info = EmbeddingModelInfo::new(model_name);
            if let Some(version) = embedding_model_version {
                model_info = model_info.with_version(version);
            }
            policy.embedding_model = Some(model_info);
        }
        let (catalog, store) = local_catalog_store(path);
        let table = TableIdent::new("default", "table");

        let writer = rt
            .block_on(RsTableWriter::create_or_open(catalog, store, policy, table))
            .map_err(|e| PyValueError::new_err(e.to_string()))?;

        Ok(Self {
            inner: Some(writer),
            runtime: rt,
        })
    }

    /// Write a batch of rows.
    ///
    /// Args:
    ///   texts: list[str] — text content for each row
    ///   embeddings: list[list[float]] — one embedding per row
    fn write_batch(&mut self, texts: Vec<String>, embeddings: Vec<Vec<f32>>) -> PyResult<()> {
        let schema = Arc::new(Schema::new(vec![Field::new("text", DataType::Utf8, false)]));
        let text_arr = StringArray::from(texts);
        let batch = RecordBatch::try_new(schema, vec![Arc::new(text_arr)])
            .map_err(|e| PyValueError::new_err(e.to_string()))?;

        let writer = self
            .inner
            .as_mut()
            .ok_or_else(|| PyValueError::new_err("TableWriter already committed"))?;

        self.runtime
            .block_on(writer.write_batch(&batch, &embeddings))
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
    fn write_batch_auto_deferred(
        &mut self,
        texts: Vec<String>,
        embeddings: Vec<Vec<f32>>,
    ) -> PyResult<()> {
        let schema = Arc::new(Schema::new(vec![Field::new("text", DataType::Utf8, false)]));
        let text_arr = StringArray::from(texts);
        let batch = RecordBatch::try_new(schema, vec![Arc::new(text_arr)])
            .map_err(|e| PyValueError::new_err(e.to_string()))?;

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
    fn write_batch_idempotent(
        &mut self,
        texts: Vec<String>,
        embeddings: Vec<Vec<f32>>,
        batch_id: String,
    ) -> PyResult<()> {
        let schema = Arc::new(Schema::new(vec![Field::new("text", DataType::Utf8, false)]));
        let text_arr = StringArray::from(texts);
        let batch = RecordBatch::try_new(schema, vec![Arc::new(text_arr)])
            .map_err(|e| PyValueError::new_err(e.to_string()))?;

        let writer = self
            .inner
            .as_mut()
            .ok_or_else(|| PyValueError::new_err("TableWriter already committed"))?;

        self.runtime
            .block_on(writer.write_batch_idempotent(&batch, &embeddings, &batch_id))
            .map_err(|e| PyValueError::new_err(e.to_string()))
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
/// Returns a list of dicts: [{"row_id": int, "distance": float, "file": str}, ...]
#[pyfunction]
#[pyo3(signature = (path, query, top_k=10))]
fn search(py: Python<'_>, path: &str, query: Vec<f32>, top_k: usize) -> PyResult<Py<PyAny>> {
    let rt = rt()?;
    debug!(
        "ailake-py: search path={} dim={} top_k={}",
        path,
        query.len(),
        top_k
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

/// Search a table and fetch the full row data for top-k results.
///
/// Returns IPC-serialized bytes of a RecordBatch containing all Parquet columns for
/// the matched rows, plus a `_distance: float32` column appended at the end.
/// Rows are ordered by ascending distance (nearest first).
///
/// Python side deserializes with: `pyarrow.ipc.open_file(io.BytesIO(bytes)).read_all()`
#[pyfunction]
#[pyo3(signature = (path, query, top_k=10))]
fn search_with_data(
    py: Python<'_>,
    path: &str,
    query: Vec<f32>,
    top_k: usize,
) -> PyResult<Py<PyAny>> {
    let rt = rt()?;
    debug!(
        "ailake-py: search_with_data path={} dim={} top_k={}",
        path,
        query.len(),
        top_k
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
#[pyo3(signature = (path, old_column, new_column, embed_fn, text_column="chunk_text", strategy="dual_write_then_cutover", batch_size=512, new_model=None, new_model_version=None))]
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
    let embed_fn_arc: Arc<dyn Fn(&[String]) -> AilakeResult<Vec<Vec<f32>>> + Send + Sync> = {
        let embed_fn = embed_fn.clone_ref(py);
        Arc::new(move |texts: &[String]| {
            Python::attach(|py| {
                let py_texts = PyList::new(py, texts)
                    .map_err(|e| ailake_core::AilakeError::InvalidArgument(e.to_string()))?;
                let result = embed_fn
                    .call1(py, (py_texts,))
                    .map_err(|e| ailake_core::AilakeError::InvalidArgument(e.to_string()))?;
                // extract Vec<Vec<f32>> directly — works for list[list[float]] and np.ndarray
                let vecs: Vec<Vec<f32>> = result
                    .bind(py)
                    .extract()
                    .map_err(|e| ailake_core::AilakeError::InvalidArgument(
                        format!("embed_fn must return list[list[float]]: {e}")
                    ))?;
                Ok(vecs)
            })
        })
    };

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
        on_progress: None,
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

#[pymodule]
fn _ailake(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<TableWriter>()?;
    m.add_function(wrap_pyfunction!(search, m)?)?;
    m.add_function(wrap_pyfunction!(search_with_data, m)?)?;
    m.add_function(wrap_pyfunction!(assemble_context, m)?)?;
    m.add_function(wrap_pyfunction!(migrate_embeddings, m)?)?;
    Ok(())
}
