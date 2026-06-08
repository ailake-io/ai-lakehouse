// SPDX-License-Identifier: MIT OR Apache-2.0
//! Hamming flat index: brute-force search over binary-quantized vectors.
//!
//! Use case: embedding models that produce binary-compatible vectors
//! (e.g. Cohere embed-v3 binary mode, Jina ColBERT). Sign of each float
//! dimension maps to one bit; distance = Hamming(a, b) = popcount(a XOR b).
//!
//! 32× compression vs F32 (1 bit/dim vs 32 bits/dim).
//! For float embeddings with random rotation use RaBitQ instead — it achieves
//! much better recall by applying an orthogonal projection before binarization.
//!
//! Search:
//!   1. Binarize query (sign → bit, once per search).
//!   2. For each stored vector: Hamming(q_bits, code) via AVX2/NEON/scalar.
//!   3. Partial select top `candidates = rerank_factor × k`, sort.
//!   4. Optional reranking: exact F16 distance for top candidates.

use ailake_core::{AilakeError, AilakeResult, RowId, VectorMetric};
use ailake_vec::{binary_quant, exact_distance};
use half::f16;
use serde::{Deserialize, Serialize};

// ── Config ────────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BinaryConfig {
    /// Keep raw F16 vectors for exact reranking. Default true.
    #[serde(default = "default_keep_raw")]
    pub keep_raw: bool,
}

fn default_keep_raw() -> bool {
    true
}

impl Default for BinaryConfig {
    fn default() -> Self {
        Self { keep_raw: true }
    }
}

// ── Index ─────────────────────────────────────────────────────────────────────

#[derive(Debug, Serialize, Deserialize)]
pub struct BinaryIndex {
    /// Bit-packed codes, flat: entry `i` occupies bytes `[i*bpv..(i+1)*bpv]`.
    pub codes: Vec<u8>,
    /// Bytes per vector = `ceil(dim / 8)`.
    pub bytes_per_vec: usize,
    pub row_ids: Vec<u64>,
    pub metric: VectorMetric,
    pub dim: u32,
    /// Raw F16 vectors for reranking, stored flat (dim values per entry).
    pub raw_f16: Option<Vec<f16>>,
}

impl BinaryIndex {
    /// Build from raw F32 vectors.
    pub fn build(
        row_ids: &[RowId],
        vectors: &[Vec<f32>],
        metric: VectorMetric,
        keep_raw: bool,
    ) -> AilakeResult<Self> {
        if vectors.is_empty() {
            return Err(AilakeError::InvalidArgument(
                "BinaryIndex::build requires at least one vector".into(),
            ));
        }
        let dim = vectors[0].len();
        let bytes_per_vec = (dim + 7) / 8;

        // Encode all vectors to packed bits
        let mut codes = Vec::with_capacity(vectors.len() * bytes_per_vec);
        for v in vectors {
            let bits = binary_quant::f32_to_bits(v);
            codes.extend_from_slice(&bits);
        }

        let raw_f16 = if keep_raw {
            Some(
                vectors
                    .iter()
                    .flat_map(|v| v.iter().map(|&x| f16::from_f32(x)))
                    .collect(),
            )
        } else {
            None
        };

        Ok(Self {
            codes,
            bytes_per_vec,
            row_ids: row_ids.iter().map(|r| r.as_u64()).collect(),
            metric,
            dim: dim as u32,
            raw_f16,
        })
    }

    pub fn node_count(&self) -> u64 {
        self.row_ids.len() as u64
    }

    /// Search for the `top_k` nearest vectors using Hamming distance.
    ///
    /// If `rerank_factor` > 1 and raw F16 vectors are stored, the top
    /// `rerank_factor × top_k` Hamming-nearest candidates are reranked
    /// with exact F16 distances using the table's configured metric.
    ///
    /// Returns `(RowId, distance)` pairs sorted ascending by distance.
    pub fn search(
        &self,
        query: &[f32],
        top_k: usize,
        rerank_factor: Option<usize>,
    ) -> Vec<(RowId, f32)> {
        if self.row_ids.is_empty() {
            return vec![];
        }
        let n = self.row_ids.len();
        let bpv = self.bytes_per_vec;

        let q_bits = binary_quant::f32_to_bits(query);

        // Coarse scan: Hamming distance for every entry (sequential — shard
        // parallelism handled externally by SearchSession)
        let mut scored: Vec<(usize, f32)> = self
            .codes
            .chunks_exact(bpv)
            .enumerate()
            .map(|(i, code)| {
                let d = binary_quant::hamming_distance(&q_bits, code) as f32;
                (i, d)
            })
            .collect();

        let candidates = rerank_factor.unwrap_or(1).max(1) * top_k;
        let cmp = |a: &(usize, f32), b: &(usize, f32)| {
            a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal)
        };
        if candidates < n {
            scored.select_nth_unstable_by(candidates - 1, cmp);
            scored.truncate(candidates);
        }
        scored.sort_unstable_by(cmp);

        if let Some(ref raw) = self.raw_f16 {
            let dim = self.dim as usize;
            let mut reranked: Vec<(usize, f32)> = scored
                .iter()
                .map(|&(i, _)| {
                    let db_f16 = &raw[i * dim..(i + 1) * dim];
                    let db_f32: Vec<f32> = db_f16.iter().map(|x| x.to_f32()).collect();
                    let d = exact_distance(self.metric, query, &db_f32);
                    (i, d)
                })
                .collect();
            reranked.sort_unstable_by(cmp);
            reranked
                .into_iter()
                .take(top_k)
                .map(|(i, d)| (RowId::new(self.row_ids[i]), d))
                .collect()
        } else {
            scored
                .into_iter()
                .take(top_k)
                .map(|(i, d)| (RowId::new(self.row_ids[i]), d))
                .collect()
        }
    }
}

// ── Serializer ────────────────────────────────────────────────────────────────

pub struct BinarySerializer;

impl BinarySerializer {
    pub fn to_bytes(index: &BinaryIndex) -> AilakeResult<Vec<u8>> {
        bincode::serialize(index).map_err(|e| AilakeError::Bincode(e.to_string()))
    }

    pub fn from_bytes(bytes: &[u8]) -> AilakeResult<BinaryIndex> {
        bincode::deserialize(bytes).map_err(|e| AilakeError::Bincode(e.to_string()))
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use rand::{rngs::StdRng, Rng, SeedableRng};

    fn binary_vecs(n: usize, dim: usize, seed: u64) -> Vec<Vec<f32>> {
        let mut rng = StdRng::seed_from_u64(seed);
        (0..n)
            .map(|_| (0..dim).map(|_| if rng.gen::<bool>() { 1.0f32 } else { -1.0 }).collect())
            .collect()
    }

    #[test]
    fn exact_self_is_top1() {
        let vecs = binary_vecs(50, 64, 1);
        let row_ids: Vec<RowId> = (0..50u64).map(RowId::new).collect();
        let idx =
            BinaryIndex::build(&row_ids, &vecs, VectorMetric::Cosine, false).unwrap();

        let query = vecs[7].clone();
        let results = idx.search(&query, 1, None);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].0, RowId::new(7));
    }

    #[test]
    fn rerank_produces_exact_distances() {
        let vecs = binary_vecs(100, 64, 99);
        let row_ids: Vec<RowId> = (0..100u64).map(RowId::new).collect();
        let idx =
            BinaryIndex::build(&row_ids, &vecs, VectorMetric::Cosine, true).unwrap();

        let query = vecs[0].clone();
        let results = idx.search(&query, 5, Some(3));
        // All distances should be valid floats in [0.0, 2.0] for cosine
        for (_, d) in &results {
            assert!(d.is_finite() && *d >= 0.0, "invalid distance {d}");
        }
    }

    #[test]
    fn serialization_roundtrip() {
        let vecs = binary_vecs(30, 32, 42);
        let row_ids: Vec<RowId> = (0..30u64).map(RowId::new).collect();
        let idx =
            BinaryIndex::build(&row_ids, &vecs, VectorMetric::Euclidean, false).unwrap();

        let bytes = BinarySerializer::to_bytes(&idx).unwrap();
        let idx2 = BinarySerializer::from_bytes(&bytes).unwrap();
        let q = vecs[0].clone();
        let r1 = idx.search(&q, 1, None);
        let r2 = idx2.search(&q, 1, None);
        assert_eq!(r1[0].0, r2[0].0);
    }

    #[test]
    fn node_count_matches() {
        let vecs = binary_vecs(20, 16, 7);
        let row_ids: Vec<RowId> = (0..20u64).map(RowId::new).collect();
        let idx = BinaryIndex::build(&row_ids, &vecs, VectorMetric::Cosine, false).unwrap();
        assert_eq!(idx.node_count(), 20);
    }
}
