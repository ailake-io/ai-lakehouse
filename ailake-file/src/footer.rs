// SPDX-License-Identifier: MIT OR Apache-2.0
// Binary layout for the AI-Lake footer extension.
// See docs/specs/FILE_FORMAT.md for field-by-field spec.

use ailake_core::{AilakeError, AilakeResult, VectorMetric, VectorPrecision};

pub const AILAKE_MAGIC: [u8; 4] = *b"AILK";
pub const AILAKE_FORMAT_VERSION: u16 = 1;
pub const TRAILER_SIZE: usize = 24;
pub const HEADER_SIZE: usize = 64;

/// `flags` bit 0 = 1: IVF-PQ index. Default (flags = 0): HNSW index.
pub const FLAG_INDEX_IVF_PQ: u16 = 0x0001;

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
    NormalizedCosine = 3,
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
            VectorMetric::NormalizedCosine => DistanceMetric::NormalizedCosine,
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
            _ => Err(AilakeError::InvalidArgument(format!(
                "invalid precision byte: {v} (valid: 0=F32, 1=F16, 2=I8, 3=Binary)"
            ))),
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
            3 => Ok(DistanceMetric::NormalizedCosine),
            _ => Err(AilakeError::InvalidArgument(format!(
                "invalid distance metric byte: {v} (valid: 0=Cosine, 1=Euclidean, 2=DotProduct, 3=NormalizedCosine)"
            ))),
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
            return Err(AilakeError::InvalidAilakeMagic([b[0], b[1], b[2], b[3]]));
        }
        let format_version = u16::from_le_bytes([b[4], b[5]]);
        if format_version != AILAKE_FORMAT_VERSION {
            return Err(AilakeError::UnsupportedFormatVersion(format_version));
        }
        Ok(AilakeHeader {
            format_version,
            flags: u16::from_le_bytes([b[6], b[7]]),
            dim: u32::from_le_bytes([b[8], b[9], b[10], b[11]]),
            precision: Precision::try_from(b[12])?,
            distance_metric: DistanceMetric::try_from(b[13])?,
            record_count: u64::from_le_bytes([
                b[16], b[17], b[18], b[19], b[20], b[21], b[22], b[23],
            ]),
            centroid_offset: u64::from_le_bytes([
                b[24], b[25], b[26], b[27], b[28], b[29], b[30], b[31],
            ]),
            centroid_len: u64::from_le_bytes([
                b[32], b[33], b[34], b[35], b[36], b[37], b[38], b[39],
            ]),
            hnsw_offset: u64::from_le_bytes([
                b[40], b[41], b[42], b[43], b[44], b[45], b[46], b[47],
            ]),
            hnsw_len: u64::from_le_bytes([b[48], b[49], b[50], b[51], b[52], b[53], b[54], b[55]]),
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
            return Err(AilakeError::InvalidAilakeMagic([
                b[20], b[21], b[22], b[23],
            ]));
        }
        Ok(AilakeTrailer {
            footer_offset: u64::from_le_bytes([b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7]]),
            footer_len: u64::from_le_bytes([b[8], b[9], b[10], b[11], b[12], b[13], b[14], b[15]]),
            format_version: u16::from_le_bytes([b[16], b[17]]),
            flags: u16::from_le_bytes([b[18], b[19]]),
        })
    }
}

/// Returns the byte offset in `buf` where the Parquet footer thrift starts.
///
/// Parquet tail layout: `[...footer_thrift...][footer_len: u32 LE][PAR1: 4 bytes]`
///
/// Used by both the writer (to know where to splice AILK sections) and the reader
/// (to locate the AILK trailer for KV-less bootstrap).
pub fn parquet_footer_start(buf: &[u8]) -> AilakeResult<usize> {
    let len = buf.len();
    if len < 8 {
        return Err(AilakeError::Parquet("file too small".into()));
    }
    if &buf[len - 4..] != b"PAR1" {
        return Err(AilakeError::Parquet("missing PAR1 footer magic".into()));
    }
    let footer_thrift_len =
        u32::from_le_bytes(buf[len - 8..len - 4].try_into().unwrap()) as usize;
    len.checked_sub(8 + footer_thrift_len)
        .ok_or_else(|| AilakeError::Parquet("footer length overflow".into()))
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
