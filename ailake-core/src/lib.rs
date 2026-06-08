// SPDX-License-Identifier: MIT OR Apache-2.0
//! ailake-core — shared type system
//!
//! No I/O, no async, no external deps beyond serde/uuid/thiserror/half.
//! Every other crate depends on this one. This crate depends on nothing internal.

pub mod error;
pub mod schema;
pub mod types;

pub use error::{AilakeError, AilakeResult};
pub use schema::{
    llm_columns, BinaryConfig, LlmContextSchema, PQConfig, RaBitQConfig, VectorStoragePolicy,
};
pub use types::{ByteLen, ByteOffset, Centroid, Dim, RowId, VectorMetric, VectorPrecision};
