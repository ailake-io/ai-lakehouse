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
    Binary = 3,
}

impl VectorPrecision {
    /// Bytes per vector element
    pub fn bytes_per_element(self) -> usize {
        match self {
            Self::F32 => 4,
            Self::F16 => 2,
            Self::I8 => 1,
            Self::Binary => 1, // ceil(dim/8) handled at call site
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[repr(u8)]
pub enum VectorMetric {
    Cosine = 0,
    Euclidean = 1,
    DotProduct = 2,
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
