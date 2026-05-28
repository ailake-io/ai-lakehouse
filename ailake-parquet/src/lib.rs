// SPDX-License-Identifier: MIT OR Apache-2.0
//! ailake-parquet — Parquet I/O with VECTOR column type
//!
//! Reads and writes the Parquet section of AI-Lake files.
//! Does NOT touch the AI-Lake footer — that is ailake-file's responsibility.

pub mod reader;
pub mod schema;
pub mod writer;

pub use reader::ParquetVectorReader;
pub use writer::ParquetVectorWriter;
