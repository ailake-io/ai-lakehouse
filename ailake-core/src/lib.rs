// SPDX-License-Identifier: MIT OR Apache-2.0
//! ailake-core — shared type system
//!
//! No I/O, no async, no external deps beyond serde/uuid/thiserror/half.
//! Every other crate depends on this one. This crate depends on nothing internal.

pub mod episodic;
pub mod error;
pub mod schema;
pub mod types;

pub use error::{AilakeError, AilakeResult};
pub use episodic::{
    episodic_columns, hybrid_score, recency_weight, EpisodicMemorySchema, RecencyConfig,
};
pub use schema::{
    llm_columns, multimodal_columns, tool_call_columns, LlmContextSchema, MultimodalContextSchema,
    PQConfig, ToolCallOutcome, ToolCallSchema, VectorStoragePolicy,
};
pub use types::{
    ByteLen, ByteOffset, Centroid, Dim, EmbeddingModelInfo, RowId, VectorMetric, VectorModality,
    VectorPrecision,
};
