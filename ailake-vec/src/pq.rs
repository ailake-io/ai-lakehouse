// SPDX-License-Identifier: MIT OR Apache-2.0
// Product Quantization — reduces per-vector storage from dim*4 bytes to num_subvectors bytes.
// At dim=1536, M=48: 6144 bytes → 48 bytes per vector (128x reduction, ~93-95% recall@10).

use ailake_core::AilakeError;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PQCodebook {
    /// Number of sub-vectors (M)
    pub num_subvectors: usize,
    /// Number of centroids per sub-space (K, typically 256 so codes fit in u8)
    pub num_centroids: usize,
    /// Dimensionality of each sub-vector = dim / num_subvectors
    pub sub_dim: usize,
    /// Centroids: [num_subvectors][num_centroids][sub_dim]
    pub centroids: Vec<Vec<Vec<f32>>>,
}

impl PQCodebook {
    /// Train PQ codebook via k-means on each sub-space independently.
    pub fn train(
        vectors: &[Vec<f32>],
        num_subvectors: usize,
        num_centroids: usize,
        max_iter: usize,
    ) -> Result<Self, AilakeError> {
        Self::train_with_kmeans(vectors, num_subvectors, num_centroids, max_iter, kmeans)
    }

    /// Train PQ codebook with a custom k-means backend (e.g. GPU-accelerated).
    ///
    /// `kmeans_fn(vecs, k, max_iter)` must return exactly `k` centroids of the
    /// same dimensionality as `vecs`.  The built-in CPU path passes `kmeans`.
    pub fn train_with_kmeans<F>(
        vectors: &[Vec<f32>],
        num_subvectors: usize,
        num_centroids: usize,
        max_iter: usize,
        kmeans_fn: F,
    ) -> Result<Self, AilakeError>
    where
        F: Fn(&[Vec<f32>], usize, usize) -> Vec<Vec<f32>>,
    {
        if vectors.is_empty() {
            return Err(AilakeError::Catalog(
                "PQ training requires at least 1 vector".into(),
            ));
        }
        let dim = vectors[0].len();
        if !dim.is_multiple_of(num_subvectors) {
            return Err(AilakeError::Catalog(format!(
                "dim {dim} not divisible by num_subvectors {num_subvectors}"
            )));
        }
        let sub_dim = dim / num_subvectors;
        let n_train = num_centroids.min(vectors.len());

        let mut centroids = Vec::with_capacity(num_subvectors);
        for m in 0..num_subvectors {
            let start = m * sub_dim;
            let end = start + sub_dim;
            let sub_vecs: Vec<Vec<f32>> = vectors.iter().map(|v| v[start..end].to_vec()).collect();
            let sub_centroids = kmeans_fn(&sub_vecs, n_train, max_iter);
            centroids.push(sub_centroids);
        }

        Ok(Self {
            num_subvectors,
            num_centroids,
            sub_dim,
            centroids,
        })
    }

    /// Encode a single vector into `num_subvectors` u8 codes.
    pub fn encode(&self, vector: &[f32]) -> Vec<u8> {
        let mut codes = Vec::with_capacity(self.num_subvectors);
        for m in 0..self.num_subvectors {
            let start = m * self.sub_dim;
            let sub = &vector[start..start + self.sub_dim];
            let best = self.centroids[m]
                .iter()
                .enumerate()
                .map(|(k, c)| (k, l2_sq(sub, c)))
                .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
                .map(|(k, _)| k)
                .unwrap_or(0);
            codes.push(best as u8);
        }
        codes
    }

    /// Decode codes back into an approximate vector (centroid reconstruction).
    pub fn decode(&self, codes: &[u8]) -> Vec<f32> {
        let mut out = Vec::with_capacity(self.num_subvectors * self.sub_dim);
        for (m, &code) in codes.iter().enumerate() {
            out.extend_from_slice(&self.centroids[m][code as usize]);
        }
        out
    }

    /// Precompute query-to-centroid L2 distances for Asymmetric Distance Computation.
    /// Returns [num_subvectors][num_centroids] distance table.
    /// ADC is O(M*K) per query precomputation, then O(M) per encoded vector — much faster
    /// than symmetric distance which would require decoding each vector first.
    pub fn compute_adc_table(&self, query: &[f32]) -> Vec<Vec<f32>> {
        (0..self.num_subvectors)
            .map(|m| {
                let start = m * self.sub_dim;
                let q_sub = &query[start..start + self.sub_dim];
                self.centroids[m].iter().map(|c| l2_sq(q_sub, c)).collect()
            })
            .collect()
    }

    /// Compute approximate L2 distance using the precomputed ADC table.
    pub fn adc_distance(&self, codes: &[u8], table: &[Vec<f32>]) -> f32 {
        codes
            .iter()
            .enumerate()
            .map(|(m, &c)| table[m][c as usize])
            .sum()
    }
}

/// K-means clustering (k-means++ init, up to `max_iter` iterations).
fn kmeans(points: &[Vec<f32>], k: usize, max_iter: usize) -> Vec<Vec<f32>> {
    let dim = points[0].len();
    let mut centroids = kmeans_pp_init(points, k);

    for _ in 0..max_iter {
        // Parallel assignment: each point finds its nearest centroid independently.
        let assignments: Vec<usize> = points
            .par_iter()
            .map(|p| nearest_centroid(p, &centroids))
            .collect();

        // Update centroids (serial reduction — n×dim is cache-friendly enough here)
        let mut new_centroids = vec![vec![0.0f32; dim]; k];
        let mut counts = vec![0usize; k];
        for (point, &assigned) in points.iter().zip(assignments.iter()) {
            for (d, &v) in point.iter().enumerate() {
                new_centroids[assigned][d] += v;
            }
            counts[assigned] += 1;
        }
        let mut converged = true;
        for (i, count) in counts.iter().enumerate() {
            if *count > 0 {
                let scale = *count as f32;
                for x in new_centroids[i].iter_mut() {
                    *x /= scale;
                }
            } else {
                // Empty cluster: keep old centroid
                new_centroids[i] = centroids[i].clone();
            }
            if l2_sq(&new_centroids[i], &centroids[i]) > 1e-8 {
                converged = false;
            }
        }
        centroids = new_centroids;
        if converged {
            break;
        }
    }
    centroids
}

/// K-means++ centroid initialization — O(n × k) via incremental min-dist update.
fn kmeans_pp_init(points: &[Vec<f32>], k: usize) -> Vec<Vec<f32>> {
    let mut centroids = Vec::with_capacity(k);
    let mut rng_state = 0x123456789u64;

    centroids.push(points[0].clone());
    // Track min distance from each point to the nearest centroid chosen so far.
    let mut min_dists: Vec<f32> = points.par_iter().map(|p| l2_sq(p, &centroids[0])).collect();

    while centroids.len() < k {
        let total: f32 = min_dists.iter().sum();
        rng_state = rng_state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let r = (rng_state >> 33) as f32 / (u32::MAX as f32);
        let target = r * total;
        let mut cumsum = 0.0f32;
        let mut chosen = points.len() - 1;
        for (i, &d) in min_dists.iter().enumerate() {
            cumsum += d;
            if cumsum >= target {
                chosen = i;
                break;
            }
        }
        let new_centroid = points[chosen].clone();
        // Incremental update: only recompute distance to the newly added centroid.
        points
            .par_iter()
            .zip(min_dists.par_iter_mut())
            .for_each(|(p, min_d)| {
                let d = l2_sq(p, &new_centroid);
                if d < *min_d {
                    *min_d = d;
                }
            });
        centroids.push(new_centroid);
    }
    centroids
}

fn nearest_centroid(point: &[f32], centroids: &[Vec<f32>]) -> usize {
    centroids
        .iter()
        .enumerate()
        .map(|(i, c)| (i, l2_sq(point, c)))
        .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
        .map(|(i, _)| i)
        .unwrap_or(0)
}

fn l2_sq(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| (x - y).powi(2)).sum()
}

/// Train k-means centroids on `vectors`. Returns `k` centroids of same dimensionality.
/// Exposed for IVF coarse quantizer training.
pub fn kmeans_centroids(vectors: &[Vec<f32>], k: usize, max_iter: usize) -> Vec<Vec<f32>> {
    let k_eff = k.min(vectors.len());
    kmeans(vectors, k_eff, max_iter)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn unit_vecs(n: usize, dim: usize) -> Vec<Vec<f32>> {
        (0..n)
            .map(|i| {
                let mut v = vec![0.0f32; dim];
                v[i % dim] = 1.0;
                v
            })
            .collect()
    }

    #[test]
    fn encode_decode_roundtrip_approx() {
        let dim = 8;
        let vecs = unit_vecs(64, dim);
        let cb = PQCodebook::train(&vecs, 2, 4, 50).unwrap();
        for v in &vecs {
            let codes = cb.encode(v);
            assert_eq!(codes.len(), 2);
            let decoded = cb.decode(&codes);
            assert_eq!(decoded.len(), dim);
        }
    }

    #[test]
    fn adc_distance_non_negative() {
        let dim = 8;
        let vecs = unit_vecs(32, dim);
        let cb = PQCodebook::train(&vecs, 2, 4, 50).unwrap();
        let query = vec![1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let table = cb.compute_adc_table(&query);
        for v in &vecs {
            let codes = cb.encode(v);
            let dist = cb.adc_distance(&codes, &table);
            assert!(dist >= 0.0, "ADC distance must be non-negative");
        }
    }

    #[test]
    fn dim_not_divisible_errors() {
        let vecs = unit_vecs(16, 9);
        assert!(PQCodebook::train(&vecs, 4, 4, 10).is_err());
    }

    #[test]
    fn nearest_neighbor_rank_preserved() {
        // Two clusters: vecs around [1,0,...,0] and [0,...,0,1]
        let dim = 8;
        let mut vecs: Vec<Vec<f32>> = Vec::new();
        for _ in 0..20 {
            let mut v = vec![0.0f32; dim];
            v[0] = 1.0;
            vecs.push(v);
        }
        for _ in 0..20 {
            let mut v = vec![0.0f32; dim];
            v[7] = 1.0;
            vecs.push(v);
        }
        let cb = PQCodebook::train(&vecs, 2, 4, 100).unwrap();
        let q1 = vec![1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let q2 = vec![0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0];
        let t1 = cb.compute_adc_table(&q1);
        let t2 = cb.compute_adc_table(&q2);
        let code1 = cb.encode(&vecs[0]);
        let code2 = cb.encode(&vecs[39]);
        // q1 closer to vecs[0] than to vecs[39]
        assert!(cb.adc_distance(&code1, &t1) < cb.adc_distance(&code2, &t1));
        // q2 closer to vecs[39] than to vecs[0]
        assert!(cb.adc_distance(&code2, &t2) < cb.adc_distance(&code1, &t2));
    }
}
