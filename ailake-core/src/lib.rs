// SPDX-License-Identifier: MIT OR Apache-2.0
//! ailake-core — shared type system
//!
//! No I/O, no async, no external deps beyond serde/uuid/thiserror/half.
//! Every other crate depends on this one. This crate depends on nothing internal.

pub mod episodic;
pub mod error;
pub mod filter;
pub mod schema;
pub mod types;

/// Upper bound on `top_k` accepted anywhere a search request enters the system (JNI C-ABI,
/// `ailake-query::scanner`, CLI, Python bindings). `top_k` flows into `ef.max(top_k * 10)` in
/// `ailake-index`'s HNSW search and then a `BinaryHeap::with_capacity` sized off that — an
/// unvalidated huge `top_k` requests a multi-GB allocation whose failure aborts the process.
/// Single source of truth so every entry point enforces the same limit.
pub const MAX_TOP_K: usize = 100_000;

pub use episodic::{
    episodic_columns, hybrid_score, recency_weight, EpisodicMemorySchema, RecencyConfig,
};
pub use error::{AilakeError, AilakeResult};
pub use filter::{ColumnFilter, FilterOp, FilterValue};
pub use schema::{
    llm_columns, multimodal_columns, now_ns, tool_call_columns, LlmContextSchema,
    MultimodalContextSchema, PQConfig, PartitionDef, ToolCallOutcome, ToolCallSchema,
    VectorStoragePolicy,
};
pub use types::{
    ByteLen, ByteOffset, Centroid, Dim, EmbeddingModelInfo, RowId, VectorColSpec, VectorMetric,
    VectorModality, VectorPrecision,
};
