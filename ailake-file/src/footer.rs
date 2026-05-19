// Binary layout for the AI-Lake footer extension.
// See docs/specs/FILE_FORMAT.md for field-by-field spec.

use ailake_core::{AilakeError, AilakeResult, VectorMetric, VectorPrecision};

pub const AILAKE_MAGIC: [u8; 4] = *b"AILK";
pub const AILAKE_FORMAT_VERSION: u16 = 1;
pub const TRAILER_SIZE: usize = 24;
pub const HEADER_SIZE: usize = 64;

#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Precision {
    F32 = 0,
    F16 = 1,
    I8 = 2,
    Binary = 3,
}

#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum DistanceMetric {
    Cosine = 0,
    Euclidean = 1,
    DotProduct = 2,
}

impl From<VectorPrecision> for Precision {
    fn from(p: VectorPrecision) -> Self {
        match p {
            VectorPrecision::F32 => Precision::F32,
            VectorPrecision::F16 => Precision::F16,
            VectorPrecision::I8 => Precision::I8,
            VectorPrecision::Binary => Precision::Binary,
        }
    }
}

impl From<VectorMetric> for DistanceMetric {
    fn from(m: VectorMetric) -> Self {
        match m {
            VectorMetric::Cosine => DistanceMetric::Cosine,
            VectorMetric::Euclidean => DistanceMetric::Euclidean,
            VectorMetric::DotProduct => DistanceMetric::DotProduct,
        }
    }
}

impl TryFrom<u8> for Precision {
    type Error = AilakeError;
    fn try_from(v: u8) -> AilakeResult<Self> {
        match v {
            0 => Ok(Precision::F32),
            1 => Ok(Precision::F16),
            2 => Ok(Precision::I8),
            3 => Ok(Precision::Binary),
            _ => Err(AilakeError::UnsupportedFormatVersion(v as u16)),
        }
    }
}

impl TryFrom<u8> for DistanceMetric {
    type Error = AilakeError;
    fn try_from(v: u8) -> AilakeResult<Self> {
        match v {
            0 => Ok(DistanceMetric::Cosine),
            1 => Ok(DistanceMetric::Euclidean),
            2 => Ok(DistanceMetric::DotProduct),
            _ => Err(AilakeError::UnsupportedFormatVersion(v as u16)),
        }
    }
}

/// 64-byte header at the start of the AI-Lake footer extension.
#[derive(Debug, Clone)]
pub struct AilakeHeader {
    pub format_version: u16,
    pub flags: u16,
    pub dim: u32,
    pub precision: Precision,
    pub distance_metric: DistanceMetric,
    pub record_count: u64,
    pub centroid_offset: u64,
    pub centroid_len: u64,
    pub hnsw_offset: u64,
    pub hnsw_len: u64,
}

impl AilakeHeader {
    pub fn to_bytes(&self) -> [u8; HEADER_SIZE] {
        let mut b = [0u8; HEADER_SIZE];
        b[0..4].copy_from_slice(&AILAKE_MAGIC);
        b[4..6].copy_from_slice(&self.format_version.to_le_bytes());
        b[6..8].copy_from_slice(&self.flags.to_le_bytes());
        b[8..12].copy_from_slice(&self.dim.to_le_bytes());
        b[12] = self.precision as u8;
        b[13] = self.distance_metric as u8;
        // b[14..16] reserved = 0
        b[16..24].copy_from_slice(&self.record_count.to_le_bytes());
        b[24..32].copy_from_slice(&self.centroid_offset.to_le_bytes());
        b[32..40].copy_from_slice(&self.centroid_len.to_le_bytes());
        b[40..48].copy_from_slice(&self.hnsw_offset.to_le_bytes());
        b[48..56].copy_from_slice(&self.hnsw_len.to_le_bytes());
        // b[56..64] reserved = 0
        b
    }

    pub fn from_bytes(b: &[u8; HEADER_SIZE]) -> AilakeResult<Self> {
        if b[0..4] != AILAKE_MAGIC {
            return Err(AilakeError::InvalidAilakeMagic(b[0..4].try_into().unwrap()));
        }
        let format_version = u16::from_le_bytes(b[4..6].try_into().unwrap());
        if format_version != AILAKE_FORMAT_VERSION {
            return Err(AilakeError::UnsupportedFormatVersion(format_version));
        }
        Ok(AilakeHeader {
            format_version,
            flags: u16::from_le_bytes(b[6..8].try_into().unwrap()),
            dim: u32::from_le_bytes(b[8..12].try_into().unwrap()),
            precision: Precision::try_from(b[12])?,
            distance_metric: DistanceMetric::try_from(b[13])?,
            record_count: u64::from_le_bytes(b[16..24].try_into().unwrap()),
            centroid_offset: u64::from_le_bytes(b[24..32].try_into().unwrap()),
            centroid_len: u64::from_le_bytes(b[32..40].try_into().unwrap()),
            hnsw_offset: u64::from_le_bytes(b[40..48].try_into().unwrap()),
            hnsw_len: u64::from_le_bytes(b[48..56].try_into().unwrap()),
        })
    }
}

/// 24-byte trailer — the last bytes of every AI-Lake file.
#[derive(Debug, Clone)]
pub struct AilakeTrailer {
    pub footer_offset: u64,
    pub footer_len: u64,
    pub format_version: u16,
    pub flags: u16,
}

impl AilakeTrailer {
    pub fn to_bytes(&self) -> [u8; TRAILER_SIZE] {
        let mut b = [0u8; TRAILER_SIZE];
        b[0..8].copy_from_slice(&self.footer_offset.to_le_bytes());
        b[8..16].copy_from_slice(&self.footer_len.to_le_bytes());
        b[16..18].copy_from_slice(&self.format_version.to_le_bytes());
        b[18..20].copy_from_slice(&self.flags.to_le_bytes());
        b[20..24].copy_from_slice(&AILAKE_MAGIC);
        b
    }

    pub fn from_bytes(b: &[u8; TRAILER_SIZE]) -> AilakeResult<Self> {
        if b[20..24] != AILAKE_MAGIC {
            return Err(AilakeError::InvalidAilakeMagic(
                b[20..24].try_into().unwrap(),
            ));
        }
        Ok(AilakeTrailer {
            footer_offset: u64::from_le_bytes(b[0..8].try_into().unwrap()),
            footer_len: u64::from_le_bytes(b[8..16].try_into().unwrap()),
            format_version: u16::from_le_bytes(b[16..18].try_into().unwrap()),
            flags: u16::from_le_bytes(b[18..20].try_into().unwrap()),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn header_roundtrip() {
        let h = AilakeHeader {
            format_version: 1,
            flags: 0,
            dim: 1536,
            precision: Precision::F16,
            distance_metric: DistanceMetric::Cosine,
            record_count: 50_000,
            centroid_offset: 64,
            centroid_len: 1536 * 4 + 4,
            hnsw_offset: 64 + 1536 * 4 + 4,
            hnsw_len: 4_194_304,
        };
        let bytes = h.to_bytes();
        let h2 = AilakeHeader::from_bytes(&bytes).unwrap();
        assert_eq!(h2.dim, 1536);
        assert_eq!(h2.precision, Precision::F16);
        assert_eq!(h2.distance_metric, DistanceMetric::Cosine);
        assert_eq!(h2.record_count, 50_000);
    }

    #[test]
    fn trailer_roundtrip() {
        let t = AilakeTrailer {
            footer_offset: 12_582_912,
            footer_len: 4_194_304,
            format_version: 1,
            flags: 0,
        };
        let bytes = t.to_bytes();
        let t2 = AilakeTrailer::from_bytes(&bytes).unwrap();
        assert_eq!(t2.footer_offset, 12_582_912);
        assert_eq!(&bytes[20..24], b"AILK");
    }

    #[test]
    fn invalid_magic_rejected() {
        let mut bytes = [0u8; HEADER_SIZE];
        bytes[0..4].copy_from_slice(b"BLAH");
        assert!(AilakeHeader::from_bytes(&bytes).is_err());
    }
}
