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
use arrow_schema::{DataType, Field, Schema};
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use tracing::{debug, warn};

use ailake_catalog::{
    hadoop::HadoopCatalog,
    provider::{CatalogProvider, TableIdent},
};
use ailake_core::{VectorMetric, VectorStoragePolicy};
use ailake_query::{
    search as rs_search, Chunk, ContextAssembler, ContextAssemblerConfig, SearchConfig,
    TableWriter as RsTableWriter,
};
use ailake_store::{store::Store, LocalStore};

fn rt() -> PyResult<tokio::runtime::Runtime> {
    tokio::runtime::Runtime::new()
        .map_err(|e| PyRuntimeError::new_err(format!("ailake: failed to create Tokio runtime: {e}")))
}

fn local_catalog_store(path: &str) -> (Arc<dyn CatalogProvider>, Arc<dyn Store>) {
    let store: Arc<dyn Store> = Arc::new(LocalStore::new(path));
    let catalog: Arc<dyn CatalogProvider> = Arc::new(HadoopCatalog::new(Arc::clone(&store), ""));
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
    #[pyo3(signature = (path, vector_column="embedding", dim=1536, metric="cosine"))]
    fn new(path: &str, vector_column: &str, dim: u32, metric: &str) -> PyResult<Self> {
        let rt = rt()?;
        debug!("ailake-py: TableWriter::new path={} dim={} metric={}", path, dim, metric);
        let policy = VectorStoragePolicy::default_f16(vector_column, dim, parse_metric(metric)?);
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
fn search(py: Python<'_>, path: &str, query: Vec<f32>, top_k: usize) -> PyResult<PyObject> {
    let rt = rt()?;
    debug!("ailake-py: search path={} dim={} top_k={}", path, query.len(), top_k);
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

fn parse_metric(s: &str) -> PyResult<VectorMetric> {
    match s {
        "cosine" => Ok(VectorMetric::Cosine),
        "euclidean" => Ok(VectorMetric::Euclidean),
        "dot_product" | "dotproduct" => Ok(VectorMetric::DotProduct),
        other => Err(PyValueError::new_err(format!(
            "unknown metric '{other}' — use cosine, euclidean, or dot_product"
        ))),
    }
}

#[pymodule]
fn ailake(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<TableWriter>()?;
    m.add_function(wrap_pyfunction!(search, m)?)?;
    m.add_function(wrap_pyfunction!(assemble_context, m)?)?;
    Ok(())
}
