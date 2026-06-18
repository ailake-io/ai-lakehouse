// SPDX-License-Identifier: MIT OR Apache-2.0
//! AI-Lake Puffin statistics files (Phase F).
//!
//! Extends the existing Puffin DV support (`delete.rs`) with two new blob types:
//!
//! - `ailake-vector-stats-v1`: all data-file centroids + radii for a snapshot,
//!   stored as bincode-encoded `Vec<VectorStatEntry>`. Enables readers to fetch
//!   geometric pruning data with a single GET instead of scanning manifest KV.
//!
//! - `ailake-bm25-bloom-v1`: per-file Bloom filters over BM25 indexed terms,
//!   stored as bincode-encoded `Vec<BM25BloomEntry>`. Readers skip files where
//!   no query term passes the filter (zero false negatives).
//!
//! Puffin file layout (reuses the Iceberg Puffin spec §4 format):
//! ```text
//! PFAc  (4 bytes magic)
//! [blob bytes — vector stats]
//! [blob bytes — BM25 bloom, optional]
//! footer_JSON
//! footer_len (4 bytes LE)
//! PFAc  (4 bytes magic)
//! ```

use ailake_core::{AilakeError, AilakeResult};
use bytes::Bytes;
use serde::{Deserialize, Serialize};

// ── Blob type tags ────────────────────────────────────────────────────────────

/// Puffin blob type for per-snapshot vector statistics (centroid + radius per file).
pub const BLOB_TYPE_VECTOR_STATS: &str = "ailake-vector-stats-v1";
/// Puffin blob type for per-file BM25 Bloom filters.
pub const BLOB_TYPE_BM25_BLOOM: &str = "ailake-bm25-bloom-v1";

// ── Blob entry types ──────────────────────────────────────────────────────────

/// Per-file vector statistics stored in the `ailake-vector-stats-v1` blob.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VectorStatEntry {
    /// Data file path (relative within the table root).
    pub path: String,
    /// Centroid vector (F32, length = column dim).
    pub centroid: Vec<f32>,
    /// Maximum distance from any vector in the file to the centroid.
    pub radius: f32,
}

/// Per-file BM25 Bloom filter stored in the `ailake-bm25-bloom-v1` blob.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BM25BloomEntry {
    /// Data file path (relative within the table root).
    pub path: String,
    /// Serialized `BloomFilter::to_bytes()` output.
    pub bloom_bytes: Vec<u8>,
}

// ── Puffin writer ─────────────────────────────────────────────────────────────

const PUFFIN_MAGIC: &[u8] = b"PFAc";

/// Result of `AilakePuffinWriter::write_stats`.
pub struct PuffinStatsResult {
    pub bytes: Bytes,
    /// Byte length of the Puffin footer JSON (for `file-footer-size-in-bytes`).
    pub footer_size: usize,
    /// (offset, length) of the `ailake-vector-stats-v1` blob.
    pub vector_stats_blob: (u64, u64),
    /// (offset, length) of the `ailake-bm25-bloom-v1` blob, if present.
    pub bm25_bloom_blob: Option<(u64, u64)>,
}

pub struct AilakePuffinWriter;

impl AilakePuffinWriter {
    /// Build a Puffin stats file containing vector-stats and optional BM25 bloom blobs.
    ///
    /// Returns `PuffinStatsResult` with the raw bytes and blob-location metadata,
    /// suitable for constructing `IcebergStatisticsRef`.
    pub fn write_stats(
        vector_stats: &[VectorStatEntry],
        bm25_blooms: &[BM25BloomEntry],
        snap_id: i64,
    ) -> AilakeResult<PuffinStatsResult> {
        // Blob 1: vector stats (bincode, no compression — small random-access data)
        let vec_blob = bincode::serialize(vector_stats)
            .map_err(|e| AilakeError::Bincode(e.to_string()))?;

        // Blob 2: BM25 bloom (optional)
        let bm25_blob: Option<Vec<u8>> = if !bm25_blooms.is_empty() {
            Some(
                bincode::serialize(bm25_blooms)
                    .map_err(|e| AilakeError::Bincode(e.to_string()))?,
            )
        } else {
            None
        };

        // Compute blob offsets (magic is 4 bytes at position 0)
        let vec_offset = PUFFIN_MAGIC.len() as u64;
        let vec_len = vec_blob.len() as u64;

        let bm25_offset = vec_offset + vec_len;
        let bm25_len = bm25_blob.as_ref().map_or(0, |b| b.len() as u64);

        // Puffin footer JSON
        let mut blobs_json = vec![serde_json::json!({
            "type": BLOB_TYPE_VECTOR_STATS,
            "snapshot-id": snap_id,
            "sequence-number": 0,
            "offset": vec_offset,
            "length": vec_len,
            "fields": []
        })];
        if bm25_blob.is_some() {
            blobs_json.push(serde_json::json!({
                "type": BLOB_TYPE_BM25_BLOOM,
                "snapshot-id": snap_id,
                "sequence-number": 0,
                "offset": bm25_offset,
                "length": bm25_len,
                "fields": []
            }));
        }
        let footer_json = serde_json::json!({
            "blobs": blobs_json,
            "properties": { "ailake.stats-version": "1" }
        })
        .to_string();
        let footer_bytes = footer_json.as_bytes();
        let footer_size = footer_bytes.len();
        let footer_len_le = (footer_size as u32).to_le_bytes();

        let mut out = Vec::with_capacity(
            PUFFIN_MAGIC.len() * 2
                + vec_blob.len()
                + bm25_len as usize
                + footer_size
                + 4,
        );
        out.extend_from_slice(PUFFIN_MAGIC);
        out.extend_from_slice(&vec_blob);
        if let Some(ref b) = bm25_blob {
            out.extend_from_slice(b);
        }
        out.extend_from_slice(footer_bytes);
        out.extend_from_slice(&footer_len_le);
        out.extend_from_slice(PUFFIN_MAGIC);

        Ok(PuffinStatsResult {
            bytes: Bytes::from(out),
            footer_size,
            vector_stats_blob: (vec_offset, vec_len),
            bm25_bloom_blob: bm25_blob.map(|_| (bm25_offset, bm25_len)),
        })
    }
}

// ── Puffin reader ─────────────────────────────────────────────────────────────

pub struct AilakePuffinReader<'a> {
    data: &'a [u8],
}

impl<'a> AilakePuffinReader<'a> {
    pub fn new(data: &'a [u8]) -> Self {
        Self { data }
    }

    /// Parse the Puffin footer JSON.
    fn footer(&self) -> AilakeResult<serde_json::Value> {
        let n = self.data.len();
        // Minimum: magic(4) + footer_len(4) + magic(4) = 12
        if n < 12 {
            return Err(AilakeError::Catalog("Puffin file too short".into()));
        }
        let footer_len = u32::from_le_bytes(
            self.data[n - 8..n - 4].try_into().unwrap(),
        ) as usize;
        let footer_start = n - 8 - footer_len;
        serde_json::from_slice(&self.data[footer_start..footer_start + footer_len])
            .map_err(|e| AilakeError::Catalog(format!("Puffin footer parse: {e}")))
    }

    fn blob_slice(&self, blob: &serde_json::Value) -> AilakeResult<&[u8]> {
        let offset = blob["offset"].as_u64().unwrap_or(0) as usize;
        let length = blob["length"].as_u64().unwrap_or(0) as usize;
        if offset + length > self.data.len() {
            return Err(AilakeError::Catalog("Puffin blob offset out of range".into()));
        }
        Ok(&self.data[offset..offset + length])
    }

    /// Read the `ailake-vector-stats-v1` blob. Returns empty `Vec` if not present.
    pub fn read_vector_stats(&self) -> AilakeResult<Vec<VectorStatEntry>> {
        let footer = self.footer()?;
        let blobs = match footer["blobs"].as_array() {
            Some(b) => b,
            None => return Ok(vec![]),
        };
        for blob in blobs {
            if blob["type"].as_str() == Some(BLOB_TYPE_VECTOR_STATS) {
                let slice = self.blob_slice(blob)?;
                return bincode::deserialize(slice)
                    .map_err(|e| AilakeError::Bincode(e.to_string()));
            }
        }
        Ok(vec![])
    }

    /// Read the `ailake-bm25-bloom-v1` blob. Returns empty `Vec` if not present.
    pub fn read_bm25_blooms(&self) -> AilakeResult<Vec<BM25BloomEntry>> {
        let footer = self.footer()?;
        let blobs = match footer["blobs"].as_array() {
            Some(b) => b,
            None => return Ok(vec![]),
        };
        for blob in blobs {
            if blob["type"].as_str() == Some(BLOB_TYPE_BM25_BLOOM) {
                let slice = self.blob_slice(blob)?;
                return bincode::deserialize(slice)
                    .map_err(|e| AilakeError::Bincode(e.to_string()));
            }
        }
        Ok(vec![])
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // Inline FNV-64a for test-only bloom construction (avoids circular dep on ailake-query).
    fn fnv64a(data: &[u8], seed: u64) -> u64 {
        let mut h = seed ^ 14695981039346656037u64;
        for &b in data {
            h ^= b as u64;
            h = h.wrapping_mul(1099511628211u64);
        }
        h
    }

    fn sample_vector_stats() -> Vec<VectorStatEntry> {
        vec![
            VectorStatEntry {
                path: "data/part-00001.parquet".into(),
                centroid: vec![0.1, 0.2, 0.3],
                radius: 0.5,
            },
            VectorStatEntry {
                path: "data/part-00002.parquet".into(),
                centroid: vec![0.9, 0.8, 0.7],
                radius: 0.3,
            },
        ]
    }

    fn sample_bloom() -> Vec<BM25BloomEntry> {
        // Build minimal bloom bytes inline (format: u64_le(num_bits) || u64 words).
        // Use same FNV double-hashing as BloomFilter but here inline for test isolation.
        let num_bits: usize = 1024;
        let mut words = vec![0u64; num_bits / 64];
        for term in &["rust", "iceberg"] {
            let h1 = fnv64a(term.as_bytes(), 0);
            let h2 = fnv64a(term.as_bytes(), h1);
            for k in 0..4u64 {
                let bit = ((h1.wrapping_add(k.wrapping_mul(h2))) as usize) % num_bits;
                words[bit / 64] |= 1u64 << (bit % 64);
            }
        }
        let mut bytes = (num_bits as u64).to_le_bytes().to_vec();
        for w in &words {
            bytes.extend_from_slice(&w.to_le_bytes());
        }
        vec![BM25BloomEntry {
            path: "data/part-00001.parquet".into(),
            bloom_bytes: bytes,
        }]
    }

    #[test]
    fn vector_stats_roundtrip() {
        let stats = sample_vector_stats();
        let result = AilakePuffinWriter::write_stats(&stats, &[], 42).unwrap();
        assert!(result.bytes.starts_with(PUFFIN_MAGIC));
        assert!(result.bytes.ends_with(PUFFIN_MAGIC));
        assert!(result.bm25_bloom_blob.is_none());

        let reader = AilakePuffinReader::new(&result.bytes);
        let recovered = reader.read_vector_stats().unwrap();
        assert_eq!(recovered.len(), 2);
        assert_eq!(recovered[0].path, "data/part-00001.parquet");
        assert!((recovered[0].radius - 0.5).abs() < 1e-6);
        assert_eq!(recovered[1].centroid, vec![0.9f32, 0.8, 0.7]);
    }

    #[test]
    fn bm25_bloom_roundtrip() {
        let stats = sample_vector_stats();
        let blooms = sample_bloom();
        let result = AilakePuffinWriter::write_stats(&stats, &blooms, 99).unwrap();
        assert!(result.bm25_bloom_blob.is_some());

        let reader = AilakePuffinReader::new(&result.bytes);
        let recovered_blooms = reader.read_bm25_blooms().unwrap();
        assert_eq!(recovered_blooms.len(), 1);
        assert_eq!(recovered_blooms[0].path, "data/part-00001.parquet");

        // Verify bloom bytes roundtrip intact (content-check done in ailake-query bloom tests).
        assert!(!recovered_blooms[0].bloom_bytes.is_empty());
        // num_bits header must parse
        assert!(recovered_blooms[0].bloom_bytes.len() >= 8);
        let nb = u64::from_le_bytes(recovered_blooms[0].bloom_bytes[..8].try_into().unwrap());
        assert_eq!(nb, 1024);
    }

    #[test]
    fn empty_bloom_produces_no_bloom_blob() {
        let stats = sample_vector_stats();
        let result = AilakePuffinWriter::write_stats(&stats, &[], 1).unwrap();
        let reader = AilakePuffinReader::new(&result.bytes);
        let blooms = reader.read_bm25_blooms().unwrap();
        assert!(blooms.is_empty());
    }

    #[test]
    fn footer_size_matches_actual() {
        let stats = sample_vector_stats();
        let result = AilakePuffinWriter::write_stats(&stats, &[], 7).unwrap();
        // The declared footer_size must match what the reader parses.
        let n = result.bytes.len();
        let footer_len_from_file = u32::from_le_bytes(result.bytes[n-8..n-4].try_into().unwrap()) as usize;
        assert_eq!(footer_len_from_file, result.footer_size);
    }
}
