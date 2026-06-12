// SPDX-License-Identifier: MIT OR Apache-2.0
use thiserror::Error;

#[derive(Error, Debug)]
pub enum AilakeError {
    #[error("unsupported format version: {0}")]
    UnsupportedFormatVersion(u16),

    #[error("AI-Lake footer magic mismatch: expected AILK, got {0:?}")]
    InvalidAilakeMagic([u8; 4]),

    #[error("Parquet footer magic mismatch: expected PAR1, got {0:?}")]
    InvalidParquetMagic([u8; 4]),

    #[error("positional invariant violated: parquet rows {parquet} != HNSW nodes {hnsw}")]
    RowCountMismatch { parquet: u64, hnsw: u64 },

    #[error("vector dimension mismatch: expected {expected}, got {actual}")]
    DimensionMismatch { expected: u32, actual: u32 },

    #[error(
        "embedding model mismatch: table uses '{table_model}' (dim={table_dim}), \
         batch uses '{batch_model}' (dim={batch_dim})"
    )]
    ModelMismatch {
        table_model: String,
        table_dim: u32,
        batch_model: String,
        batch_dim: u32,
    },

    #[error("centroid length mismatch: expected dim={expected_dim}, got {actual} bytes")]
    InvalidCentroidLength { expected_dim: u32, actual: usize },

    #[error("file is not a valid AI-Lake file (no AILK trailer)")]
    NotAnAilakeFile,

    #[error("invalid argument: {0}")]
    InvalidArgument(String),

    #[error("catalog error: {0}")]
    Catalog(String),

    #[error("store error: {0}")]
    Store(String),

    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    #[error("Parquet error: {0}")]
    Parquet(String),

    #[error("bincode error: {0}")]
    Bincode(String),

    #[error("Arrow error: {0}")]
    Arrow(String),

    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),
}

pub type AilakeResult<T> = Result<T, AilakeError>;
