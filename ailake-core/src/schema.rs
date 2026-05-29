// SPDX-License-Identifier: MIT OR Apache-2.0
use crate::types::{VectorMetric, VectorPrecision};
use serde::{Deserialize, Serialize};

/// Canonical column names for LLM-context tables.
/// ContextAssembler reads columns by these names.
pub mod llm_columns {
    pub const CHUNK_ID: &str = "chunk_id";
    pub const DOCUMENT_ID: &str = "document_id";
    pub const CHUNK_INDEX: &str = "chunk_index";
    pub const TOTAL_CHUNKS: &str = "total_chunks";
    pub const CHUNK_TEXT: &str = "chunk_text";
    pub const DOCUMENT_TITLE: &str = "document_title";
    pub const SECTION_PATH: &str = "section_path";
    pub const PRECEDING_CONTEXT: &str = "preceding_context";
    pub const FOLLOWING_CONTEXT: &str = "following_context";
    pub const DOCUMENT_SUMMARY: &str = "document_summary";
    pub const CHUNK_SUMMARY: &str = "chunk_summary";
    pub const SOURCE_URI: &str = "source_uri";
    pub const PAGE_NUMBER: &str = "page_number";
    pub const CREATED_AT: &str = "created_at";
    pub const DOCUMENT_DATE: &str = "document_date";
    pub const EMBEDDING: &str = "embedding";
    pub const CONTEXT_EMBEDDING: &str = "context_embedding";
}

/// Vector storage configuration applied at table creation time.
/// Stored in Iceberg metadata.json properties.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VectorStoragePolicy {
    pub column_name: String,
    pub dim: u32,
    pub metric: VectorMetric,
    pub precision: VectorPrecision,
    pub pq: Option<PQConfig>,
    pub keep_raw_for_reranking: bool,
}

impl VectorStoragePolicy {
    pub fn default_f16(column: &str, dim: u32, metric: VectorMetric) -> Self {
        Self {
            column_name: column.to_string(),
            dim,
            metric,
            precision: VectorPrecision::F16,
            pq: None,
            keep_raw_for_reranking: true,
        }
    }
}

/// Product Quantization configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PQConfig {
    /// Number of sub-vectors M (dim must be divisible by M)
    pub num_subvectors: usize,
    /// Bits per code (8 = 256 centroids per sub-vector)
    pub bits_per_code: u8,
    /// Number of training samples for codebook
    pub train_sample_size: usize,
}

/// Marker struct for documentation purposes — actual schema is enforced by
/// column names in llm_columns module.
pub struct LlmContextSchema;
