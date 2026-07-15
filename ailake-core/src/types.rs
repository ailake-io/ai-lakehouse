// SPDX-License-Identifier: MIT OR Apache-2.0
use serde::{Deserialize, Serialize};

/// Positional index of a row in a Parquet file and its paired HNSW node.
/// Row N in Parquet == HNSW node with key RowId(N). This invariant is sacred.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub struct RowId(pub u64);

impl RowId {
    pub fn new(n: u64) -> Self {
        Self(n)
    }
    pub fn as_u64(self) -> u64 {
        self.0
    }
    pub fn as_usize(self) -> usize {
        self.0 as usize
    }
}

impl From<u64> for RowId {
    fn from(n: u64) -> Self {
        Self(n)
    }
}

/// Vector dimensionality
pub type Dim = u32;
/// Byte offset within a file
pub type ByteOffset = u64;
/// Byte length
pub type ByteLen = u64;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[repr(u8)]
pub enum VectorPrecision {
    F32 = 0,
    F16 = 1,
    I8 = 2,
    // 3 was `Binary` (1-bit packed quantization for the now-removed Binary Hamming
    // index, v0.0.14) — removed: no encoder ever implemented real bit-packing (the
    // write path silently fell back to F16 while the field metadata still
    // labeled the column "binary", a size mismatch), and no public API
    // (CLI/Python/JNI) ever exposed it. Reserved, not reassigned — see
    // `ailake-file::footer::Precision` and `docs/specs/FILE_FORMAT.md` §3.1.
}

impl VectorPrecision {
    /// Bytes per vector element
    pub fn bytes_per_element(self) -> usize {
        match self {
            Self::F32 => 4,
            Self::F16 => 2,
            Self::I8 => 1,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[repr(u8)]
pub enum VectorMetric {
    Cosine = 0,
    Euclidean = 1,
    DotProduct = 2,
    /// Cosine on pre-normalized unit vectors: distance = 1 - dot(a, b).
    /// No sqrt in the hot loop — ~2× faster than Cosine for the same recall.
    /// Set VectorStoragePolicy::pre_normalize = true to enable automatically.
    NormalizedCosine = 3,
}

/// Identifies the embedding model used to produce vectors in a table or file.
/// Stored in Iceberg properties so any reader can detect model changes before
/// mixing incompatible vectors.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct EmbeddingModelInfo {
    /// Human-readable model identifier, e.g. "text-embedding-3-small" or "my-model-v2".
    pub name: String,
    /// Optional model version or checkpoint tag, e.g. "2024-01".
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub version: Option<String>,
    /// Expected embedding dimension — used to detect model/table mismatches per-file.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub dim: Option<u32>,
    /// Expected distance metric — used to detect model/table mismatches per-file.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub metric: Option<VectorMetric>,
}

impl EmbeddingModelInfo {
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            version: None,
            dim: None,
            metric: None,
        }
    }

    pub fn with_version(mut self, version: impl Into<String>) -> Self {
        self.version = Some(version.into());
        self
    }

    pub fn with_dim(mut self, dim: u32) -> Self {
        self.dim = Some(dim);
        self
    }

    pub fn with_metric(mut self, metric: VectorMetric) -> Self {
        self.metric = Some(metric);
        self
    }

    /// Canonical key stored in Iceberg properties.
    pub fn property_key() -> &'static str {
        "ailake.embedding-model"
    }

    /// Returns "<name>" or "<name>@<version>" for display / property value.
    pub fn to_property_value(&self) -> String {
        match &self.version {
            Some(v) => format!("{}@{}", self.name, v),
            None => self.name.clone(),
        }
    }

    /// Parse back from a property value written by `to_property_value`.
    pub fn from_property_value(s: &str) -> Self {
        if let Some((name, version)) = s.split_once('@') {
            Self {
                name: name.to_string(),
                version: Some(version.to_string()),
                dim: None,
                metric: None,
            }
        } else {
            Self {
                name: s.to_string(),
                version: None,
                dim: None,
                metric: None,
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn embedding_model_info_roundtrip_with_version() {
        let info = EmbeddingModelInfo::new("text-embedding-3-small").with_version("2024-01");
        assert_eq!(info.to_property_value(), "text-embedding-3-small@2024-01");
        let parsed = EmbeddingModelInfo::from_property_value("text-embedding-3-small@2024-01");
        // from_property_value only restores name+version; dim/metric are not in the property string
        assert_eq!(parsed.name, info.name);
        assert_eq!(parsed.version, info.version);
    }

    #[test]
    fn embedding_model_info_with_dim_and_metric() {
        use super::VectorMetric;
        let info = EmbeddingModelInfo::new("my-model")
            .with_dim(1536)
            .with_metric(VectorMetric::Cosine);
        assert_eq!(info.dim, Some(1536));
        assert_eq!(info.metric, Some(VectorMetric::Cosine));
        // property_value only encodes name (no dim/metric)
        assert_eq!(info.to_property_value(), "my-model");
    }

    #[test]
    fn embedding_model_info_roundtrip_no_version() {
        let info = EmbeddingModelInfo::new("my-model");
        assert_eq!(info.to_property_value(), "my-model");
        assert_eq!(EmbeddingModelInfo::from_property_value("my-model"), info);
    }

    #[test]
    fn embedding_model_info_property_key() {
        assert_eq!(EmbeddingModelInfo::property_key(), "ailake.embedding-model");
    }

    #[test]
    fn embedding_model_info_fixture_value() {
        // Exact value used by write_fixture.py → Go integration test.
        let parsed = EmbeddingModelInfo::from_property_value("fixture-model@v1");
        assert_eq!(parsed.name, "fixture-model");
        assert_eq!(parsed.version.as_deref(), Some("v1"));
        assert_eq!(parsed.to_property_value(), "fixture-model@v1");
    }

    #[test]
    fn embedding_model_info_first_at_only() {
        // split_once('@') splits at first '@'; remainder goes into version.
        let parsed = EmbeddingModelInfo::from_property_value("model@v1@extra");
        assert_eq!(parsed.name, "model");
        assert_eq!(parsed.version.as_deref(), Some("v1@extra"));
    }
}

/// Modality tag for a vector column.
///
/// Stored in Iceberg table properties as `ailake.modality-<col>` and in Parquet
/// field key-value metadata as `ailake.modality-<col>`. Readers use this to
/// select the correct HNSW index without inspecting raw data.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum VectorModality {
    Text,
    Image,
    Audio,
    Video,
}

impl VectorModality {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Text => "text",
            Self::Image => "image",
            Self::Audio => "audio",
            Self::Video => "video",
        }
    }
}

impl std::str::FromStr for VectorModality {
    type Err = ();

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "text" => Ok(Self::Text),
            "image" => Ok(Self::Image),
            "audio" => Ok(Self::Audio),
            "video" => Ok(Self::Video),
            _ => Err(()),
        }
    }
}

/// Per-file geometric statistics used for pruning
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Centroid {
    /// Mean vector of all vectors in the file (always F32, not quantized)
    pub values: Vec<f32>,
    /// Maximum distance from any vector in the file to the centroid
    pub radius: f32,
    pub metric: VectorMetric,
}

/// Specification for a vector column to add to an existing table.
///
/// Used by `CatalogProvider::add_vector_column` and `BackfillJob`.
/// Converts to `VectorStoragePolicy` for writing.
#[derive(Debug, Clone)]
pub struct VectorColSpec {
    pub column_name: String,
    pub dim: u32,
    pub metric: VectorMetric,
    pub precision: VectorPrecision,
    pub pre_normalize: bool,
    pub hnsw_m: Option<u32>,
    pub hnsw_ef_construction: Option<u32>,
}
