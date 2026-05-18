//! ailake-parquet — Parquet I/O with VECTOR column type
//!
//! Reads and writes the Parquet section of AI-Lake files.
//! Does NOT touch the AI-Lake footer — that is ailake-file's responsibility.

pub mod writer;
pub mod reader;
pub mod schema;

pub use writer::ParquetVectorWriter;
pub use reader::ParquetVectorReader;
