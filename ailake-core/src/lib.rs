//! ailake-core — shared type system
//!
//! No I/O, no async, no external deps beyond serde/uuid/thiserror/half.
//! Every other crate depends on this one. This crate depends on nothing internal.

pub mod error;
pub mod types;
pub mod schema;

pub use error::{AilakeError, AilakeResult};
pub use types::{RowId, VectorMetric, VectorPrecision, Dim, ByteOffset, ByteLen, Centroid};
pub use schema::{LlmContextSchema, VectorStoragePolicy, PQConfig, llm_columns};
