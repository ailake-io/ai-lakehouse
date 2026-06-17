// SPDX-License-Identifier: MIT OR Apache-2.0
use crate::error::{AilakeError, AilakeResult};
use crate::types::{EmbeddingModelInfo, VectorMetric, VectorModality, VectorPrecision};
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
    /// Normalize each input vector to unit L2 length before indexing.
    /// Enables the NormalizedCosine fast path in HNSW: distance = 1 - dot(a, b),
    /// no sqrt, ~2× faster distance computation. Semantics unchanged — same top-k
    /// results as Cosine. Most embedding models (OpenAI, Cohere, etc.) produce
    /// nearly-unit vectors; enabling this adds negligible write overhead.
    #[serde(default)]
    pub pre_normalize: bool,
    /// HNSW M parameter — connections per node. `None` = default (16).
    /// Higher M → better recall, more memory, slower build.
    /// Recommended values: 8 (low-memory), 16 (default), 32 (high-recall), 64 (max).
    #[serde(default)]
    pub hnsw_m: Option<u32>,
    /// HNSW ef_construction — candidate pool size during build. `None` = default (150).
    /// Higher ef_construction → better graph quality, slower build.
    /// Recommended values: 100 (fast), 150 (default), 200 (quality), 400 (max quality).
    #[serde(default)]
    pub hnsw_ef_construction: Option<u32>,
    /// IVF-PQ residual encoding — train PQ on per-cluster residuals (vec - coarse_centroid).
    /// Same bytes/vector, ~2-4pp better recall@10. Only applies when IVF-PQ index is used.
    #[serde(default)]
    pub ivf_residual: bool,
    /// Optional embedding model metadata. When set:
    /// - Stored as `ailake.embedding-model` in Iceberg table properties.
    /// - Validated on every `write_batch`: dim mismatch → hard error; name mismatch → warning.
    /// - Required for `migrate_embeddings` to track the model transition.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub embedding_model: Option<EmbeddingModelInfo>,
    /// Modality tag for this vector column (text / image / audio / video).
    /// Stored as `ailake.modality-<col>` in Iceberg properties and Parquet KV metadata.
    /// Allows readers to select the correct HNSW by modality without reading data.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub modality: Option<VectorModality>,
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
            pre_normalize: false,
            hnsw_m: None,
            hnsw_ef_construction: None,
            ivf_residual: false,
            embedding_model: None,
            modality: None,
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

/// Canonical column names for multimodal LLM-context tables.
/// Extends `LlmContextSchema` with media and cross-modal embedding columns.
///
/// Usage: write tables whose Parquet schema includes these column names alongside
/// `llm_columns::*`. The AI-Lake SDK reads them by name — no code-gen required.
///
/// Typical multimodal row:
/// - chunk_text  + embedding (text)
/// - image_embedding (CLIP/SigLIP dim=512)
/// - media_uri pointing to the source image/audio/video in object storage
/// - audio_transcript when the source is audio/video
/// - media_caption from a captioning model
pub mod multimodal_columns {
    /// URI of the raw media asset in object storage (s3://, gs://, az://, https://).
    /// AI-Lake is NOT a blob store — store media externally; only the URI lives here.
    pub const MEDIA_URI: &str = "media_uri";
    /// MIME type of the media asset (e.g. "image/jpeg", "audio/mpeg", "video/mp4").
    pub const MEDIA_MIME: &str = "media_mime";
    /// Human-readable caption generated by a vision/audio model (e.g. BLIP-2, Whisper).
    pub const MEDIA_CAPTION: &str = "media_caption";
    /// Image embedding column (e.g. CLIP ViT-B/32, SigLIP dim=512).
    /// Physical type: FIXED_LEN_BYTE_ARRAY (F16) — same as text `embedding`.
    pub const IMAGE_EMBEDDING: &str = "image_embedding";
    /// Transcription of spoken content from audio or video assets (Whisper output).
    pub const AUDIO_TRANSCRIPT: &str = "audio_transcript";
    /// Base64-encoded thumbnail (JPEG, ≤ 64×64 px) for inline LLM context.
    /// Allows multimodal LLMs to receive a visual preview without fetching media_uri.
    pub const THUMBNAIL_B64: &str = "thumbnail_b64";
}

/// Marker struct for multimodal LLM-context tables.
/// Actual schema is enforced by column names in `multimodal_columns` module.
///
/// A multimodal table combines all `llm_columns::*` fields (text + embeddings)
/// with `multimodal_columns::*` (media URI, MIME, caption, image_embedding,
/// audio_transcript, thumbnail_b64).
///
/// Example Arrow schema (abridged):
/// ```text
/// chunk_id:          Utf8
/// chunk_text:        Utf8
/// embedding:         FixedSizeBinary(3072)   -- text, F16, dim=1536
/// image_embedding:   FixedSizeBinary(1024)   -- image, F16, dim=512
/// media_uri:         Utf8
/// media_mime:        Utf8
/// media_caption:     Utf8
/// audio_transcript:  Utf8
/// thumbnail_b64:     Utf8
/// ```
pub struct MultimodalContextSchema;

// ── Phase 9 — Agent / Episodic Memory ────────────────────────────────────────

/// Outcome of a tool call recorded in a `ToolCallSchema` table.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ToolCallOutcome {
    Success,
    Failure,
    Timeout,
}

impl ToolCallOutcome {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Success => "success",
            Self::Failure => "failure",
            Self::Timeout => "timeout",
        }
    }
}

impl std::fmt::Display for ToolCallOutcome {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

impl std::str::FromStr for ToolCallOutcome {
    type Err = AilakeError;
    fn from_str(s: &str) -> AilakeResult<Self> {
        match s {
            "success" => Ok(Self::Success),
            "failure" => Ok(Self::Failure),
            "timeout" => Ok(Self::Timeout),
            other => Err(AilakeError::InvalidArgument(format!(
                "unknown ToolCallOutcome '{other}' (valid: success, failure, timeout)"
            ))),
        }
    }
}

/// Canonical column names for agent tool-call history tables.
///
/// Each row records one tool invocation: agent identity, session context,
/// inputs/outputs as JSON, outcome, and latency. The `embedding` column
/// (from `llm_columns::EMBEDDING`) holds a vector over the concatenated
/// `tool_name + tool_input_json` text, enabling semantic search over past
/// tool calls ("when did the agent call X in contexts similar to Y?").
///
/// Usage: include these columns alongside `llm_columns::*` in the Arrow
/// schema of a `ToolCallSchema` table.
pub mod tool_call_columns {
    /// UUID of the agent instance (identifies which agent performed the call).
    pub const AGENT_ID: &str = "agent_id";
    /// UUID of the conversation / task session.
    pub const SESSION_ID: &str = "session_id";
    /// Zero-based index of this tool call within the session.
    pub const STEP_INDEX: &str = "step_index";
    /// Name of the tool that was invoked (e.g. "web_search", "code_exec").
    pub const TOOL_NAME: &str = "tool_name";
    /// JSON-serialized input arguments passed to the tool.
    pub const TOOL_INPUT_JSON: &str = "tool_input_json";
    /// JSON-serialized output returned by the tool (or error message on failure).
    pub const TOOL_OUTPUT_JSON: &str = "tool_output_json";
    /// Outcome of the call: "success" | "failure" | "timeout".
    /// Use `ToolCallOutcome` enum for typed access.
    pub const OUTCOME: &str = "outcome";
    /// Wall-clock latency of the tool call in milliseconds.
    pub const LATENCY_MS: &str = "latency_ms";
}

/// Marker struct for agent tool-call history tables (Phase 9).
/// Actual schema is enforced by column names in `tool_call_columns` module.
///
/// A tool-call table extends `LlmContextSchema` with agent identity and
/// invocation metadata, enabling semantic search over an agent's history:
///
/// ```text
/// agent_id:         Utf8          -- UUID string
/// session_id:       Utf8          -- UUID string
/// step_index:       UInt32
/// tool_name:        Utf8
/// tool_input_json:  Utf8
/// tool_output_json: Utf8
/// outcome:          Utf8          -- "success" | "failure" | "timeout"
/// latency_ms:       UInt32
/// embedding:        FixedSizeBinary(N)  -- F16, over tool_name+tool_input_json
/// ```
///
/// Recommended index: one HNSW over `embedding` (text, cosine).
/// Partition by `agent_id` via `VectorStoragePolicy` for isolated per-agent search.
pub struct ToolCallSchema;
