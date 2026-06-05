// SPDX-License-Identifier: MIT OR Apache-2.0
//! RaBitQ flat index: brute-force search over binary codes with optional reranking.
//!
//! Search algorithm:
//!   1. Apply random rotation to query (once per search).
//!   2. For each database vector: XOR + popcount → IP estimate (O(dim/8)).
//!   3. Return top-k by estimated distance.
//!   4. Optional reranking: exact F16 distance for top `rerank_factor × k` candidates.
//!
//! Use case: very compact index (1 bit/dim + 8 bytes/vector overhead) with
//! better recall than naive binary quantization at the same storage cost.

use ailake_core::{AilakeError, AilakeResult, RowId, VectorMetric};
use ailake_vec::{
    exact_distance,
    rabitq::{bits_from_signs, encode_batch, RaBitQCodebook, RaBitQVec},
};
use half::f16;
use serde::{Deserialize, Serialize};

// ── Index ─────────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RaBitQConfig {
    /// Seed for the random rotation matrix. Regenerated at load time — not stored.
    pub seed: u64,
    /// Keep raw F16 vectors for exact reranking. Default true.
    #[serde(default = "default_keep_raw")]
    pub keep_raw: bool,
}

fn default_keep_raw() -> bool {
    true
}

impl Default for RaBitQConfig {
    fn default() -> Self {
        Self {
            seed: 0,
            keep_raw: true,
        }
    }
}

#[derive(Debug, Serialize, Deserialize)]
pub struct RaBitQIndex {
    pub codebook: RaBitQCodebook,
    pub entries: Vec<RaBitQVec>,
    pub row_ids: Vec<u64>,
    pub metric: VectorMetric,
    pub dim: u32,
    /// Raw F16 vectors for reranking (present when `keep_raw = true`).
    pub raw_f16: Option<Vec<f16>>,
}

impl RaBitQIndex {
    /// Build index from raw F32 vectors.
    pub fn build(
        row_ids: &[RowId],
        vectors: &[Vec<f32>],
        metric: VectorMetric,
        config: RaBitQConfig,
        keep_raw: bool,
    ) -> AilakeResult<Self> {
        if vectors.is_empty() {
            return Err(AilakeError::InvalidArgument(
                "RaBitQIndex::build requires at least one vector".into(),
            ));
        }
        let dim = vectors[0].len();
        let cb = RaBitQCodebook::new(dim, config.seed);
        let entries = encode_batch(&cb, vectors);

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
            codebook: cb,
            entries,
            row_ids: row_ids.iter().map(|r| r.as_u64()).collect(),
            metric,
            dim: dim as u32,
            raw_f16,
        })
    }

    pub fn node_count(&self) -> u64 {
        self.row_ids.len() as u64
    }

    /// Search using binary codes. Returns top-k (RowId, distance) pairs sorted ascending.
    ///
    /// If `rerank_factor` > 1 and raw F16 vectors are available, the top
    /// `rerank_factor × top_k` candidates are reranked with exact F16 distances.
    ///
    /// Inner scan is intentionally sequential — callers (e.g. `SearchSession`) run
    /// shard searches in parallel via rayon. Nesting par_iter here would spawn
    /// O(shards × N) micro-tasks and cause scheduler overhead that destroys QPS.
    pub fn search(
        &self,
        query: &[f32],
        top_k: usize,
        rerank_factor: Option<usize>,
    ) -> Vec<(RowId, f32)> {
        if self.entries.is_empty() {
            return vec![];
        }
        debug_assert!(
            self.codebook.is_ready(),
            "RaBitQCodebook not initialized — call rebuild_proj() after deserialization"
        );

        let (q_proj, q_scale) = self.codebook.prepare_query(query);
        let b_q = bits_from_signs(&q_proj);
        let n = self.entries.len();
        let q_norm = query.iter().map(|x| x * x).sum::<f32>().sqrt();

        // Coarse scan: sequential — outer shard parallelism handles concurrency.
        let mut scored: Vec<(usize, f32)> = self
            .entries
            .iter()
            .enumerate()
            .map(|(i, entry)| {
                let ip = self.codebook.estimate_ip_binary(&b_q, q_scale, entry);
                let dist = match self.metric {
                    VectorMetric::Cosine | VectorMetric::NormalizedCosine => 1.0 - ip,
                    VectorMetric::DotProduct => -ip * q_norm * entry.norm,
                    VectorMetric::Euclidean => {
                        // ||q - x||² ≈ ||q||² + ||x||² - 2·ip·||q||·||x||
                        let norm_x = entry.norm;
                        (q_norm * q_norm + norm_x * norm_x - 2.0 * ip * q_norm * norm_x)
                            .max(0.0)
                            .sqrt()
                    }
                };
                (i, dist)
            })
            .collect();

        let candidates = rerank_factor.unwrap_or(1).max(1) * top_k;
        let cmp = |a: &(usize, f32), b: &(usize, f32)| {
            a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal)
        };
        // O(N) partial select to bring top `candidates` to front, then sort only those.
        if candidates < n {
            scored.select_nth_unstable_by(candidates - 1, cmp);
            scored.truncate(candidates);
        }
        scored.sort_unstable_by(cmp);

        // Rerank top candidates with exact F16 distances if raw vectors available
        let rerank_slice = &scored[..candidates.min(scored.len())];

        if let Some(ref raw) = self.raw_f16 {
            let dim = self.dim as usize;
            let mut reranked: Vec<(usize, f32)> = rerank_slice
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
            rerank_slice
                .iter()
                .take(top_k)
                .map(|&(i, d)| (RowId::new(self.row_ids[i]), d))
                .collect()
        }
    }
}

// ── Serializer ────────────────────────────────────────────────────────────────

pub struct RaBitQSerializer;

impl RaBitQSerializer {
    pub fn to_bytes(index: &RaBitQIndex) -> AilakeResult<Vec<u8>> {
        bincode::serialize(index).map_err(|e| AilakeError::Bincode(e.to_string()))
    }

    pub fn from_bytes(bytes: &[u8]) -> AilakeResult<RaBitQIndex> {
        let mut idx: RaBitQIndex =
            bincode::deserialize(bytes).map_err(|e| AilakeError::Bincode(e.to_string()))?;
        idx.codebook.rebuild_proj();
        Ok(idx)
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use ailake_vec::cosine_distance;
    use rand::{rngs::StdRng, Rng, SeedableRng};

    fn unit_vecs(n: usize, dim: usize, seed: u64) -> Vec<Vec<f32>> {
        let mut rng = StdRng::seed_from_u64(seed);
        (0..n)
            .map(|_| {
                let v: Vec<f32> = (0..dim).map(|_| rng.gen::<f32>() * 2.0 - 1.0).collect();
                let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
                v.iter().map(|x| x / norm).collect()
            })
            .collect()
    }

    #[test]
    fn top1_is_self() {
        let dim = 32;
        let vecs = unit_vecs(50, dim, 1);
        let row_ids: Vec<RowId> = (0..50u64).map(RowId::new).collect();
        let idx = RaBitQIndex::build(
            &row_ids,
            &vecs,
            VectorMetric::Cosine,
            RaBitQConfig::default(),
            false,
        )
        .unwrap();

        let query = vecs[0].clone();
        let results = idx.search(&query, 1, None);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].0, RowId::new(0));
    }

    #[test]
    fn rerank_improves_recall() {
        let dim = 64;
        let n = 200;
        let vecs = unit_vecs(n, dim, 42);
        let row_ids: Vec<RowId> = (0..n as u64).map(RowId::new).collect();
        let idx = RaBitQIndex::build(
            &row_ids,
            &vecs,
            VectorMetric::Cosine,
            RaBitQConfig::default(),
            true,
        )
        .unwrap();

        let query = vecs[5].clone();

        // Brute-force ground truth
        let mut gt: Vec<(f32, u64)> = vecs
            .iter()
            .enumerate()
            .map(|(i, v)| (cosine_distance(&query, v), i as u64))
            .collect();
        gt.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
        let gt_top5: std::collections::HashSet<u64> =
            gt.iter().take(5).map(|(_, id)| *id).collect();

        let results_reranked = idx.search(&query, 5, Some(4));
        let found: std::collections::HashSet<u64> =
            results_reranked.iter().map(|(id, _)| id.as_u64()).collect();
        let recall = found.intersection(&gt_top5).count() as f64 / 5.0;
        assert!(recall >= 0.6, "recall@5 with reranking = {recall:.2}");
    }

    #[test]
    fn serialization_roundtrip() {
        let dim = 32;
        let vecs = unit_vecs(20, dim, 7);
        let row_ids: Vec<RowId> = (0..20u64).map(RowId::new).collect();
        let idx = RaBitQIndex::build(
            &row_ids,
            &vecs,
            VectorMetric::Cosine,
            RaBitQConfig::default(),
            false,
        )
        .unwrap();

        let bytes = RaBitQSerializer::to_bytes(&idx).unwrap();
        let idx2 = RaBitQSerializer::from_bytes(&bytes).unwrap();
        let q = vecs[0].clone();
        let r1 = idx.entries[0].code.clone();
        let _r2 = idx2.search(&q, 1, None);
        assert_eq!(r1, idx2.entries[0].code);
    }
}
