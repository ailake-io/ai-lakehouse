// SPDX-License-Identifier: MIT OR Apache-2.0
// IVF-PQ index: Inverted File Index with Product Quantization.
//
// vs HNSW tradeoffs:
//   - Index size: ~100x smaller (PQ codes vs raw vectors + graph pointers)
//   - S3 reads: sequential inverted-list scan vs random graph traversal
//   - Recall: slightly lower at same memory, tunable via nprobe
//   - Build: O(n * nlist) k-means vs O(n log n) HNSW insertions
//
// Non-residual variant: global PQ codebook trained on all vectors.
// Simpler than per-cluster residual PQ, adequate for dim >= 64.

use serde::{Deserialize, Serialize};
use tracing::{debug, info, warn};

use ailake_core::{AilakeError, AilakeResult, RowId, VectorMetric};
use ailake_vec::{kmeans_centroids, PQCodebook};

/// K-means dispatch: NVIDIA CUDA → AMD ROCm → CPU rayon fallback.
fn kmeans_dispatch(vecs: &[Vec<f32>], k: usize, max_iter: usize) -> Vec<Vec<f32>> {
    if let Some(result) = crate::gpu::try_nvidia_kmeans(vecs, k, max_iter) {
        debug!(
            "ailake: IVF-PQ k-means used NVIDIA CUDA (n={} k={} max_iter={})",
            vecs.len(),
            k,
            max_iter
        );
        return result;
    }
    if let Some(result) = crate::gpu::try_rocm_kmeans(vecs, k, max_iter) {
        debug!(
            "ailake: IVF-PQ k-means used AMD ROCm (n={} k={} max_iter={})",
            vecs.len(),
            k,
            max_iter
        );
        return result;
    }
    debug!(
        "ailake: IVF-PQ k-means using CPU rayon (n={} k={} max_iter={})",
        vecs.len(),
        k,
        max_iter
    );
    kmeans_centroids(vecs, k, max_iter)
}

/// Configuration for IVF-PQ index construction and search.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IvfPqConfig {
    /// Number of coarse Voronoi cells (inverted lists).
    /// Rule of thumb: sqrt(n) for balanced recall/speed. Default 256.
    pub nlist: usize,
    /// Cells probed per query. Higher = better recall, more compute.
    /// nprobe=1 is ANN; nprobe=nlist is exact. Default 8.
    pub nprobe: usize,
    /// PQ sub-vector count M. Must divide dim. Default 8.
    pub pq_m: usize,
    /// PQ centroids per sub-space K. Must be ≤ 256 (u8 codes). Default 256.
    pub pq_k: usize,
    /// k-means max iterations for both coarse and PQ training. Default 25.
    pub max_iter: usize,
}

impl Default for IvfPqConfig {
    fn default() -> Self {
        Self {
            nlist: 256,
            nprobe: 8,
            pq_m: 8,
            pq_k: 256,
            max_iter: 25,
        }
    }
}

impl IvfPqConfig {
    /// Derive sensible defaults from vector dimensionality.
    pub fn for_dim(dim: usize) -> Self {
        let pq_m = (dim / 16).clamp(4, 64);
        Self {
            pq_m: find_valid_pq_m(pq_m, dim),
            ..Self::default()
        }
    }

    /// Derive sensible defaults from both dimensionality and dataset size.
    ///
    /// `nlist` scales with sqrt(n_vectors) so each cluster gets ~4 vectors on average.
    /// Clamped to [16, 1024] to avoid degenerate configs on tiny or huge datasets.
    pub fn for_dataset(dim: usize, n_vectors: usize) -> Self {
        let nlist = ((n_vectors as f64).sqrt() as usize).clamp(16, 1024);
        let nprobe = (nlist / 8).max(1);
        let pq_m_hint = (dim / 16).clamp(4, 64);
        Self {
            nlist,
            nprobe,
            pq_m: find_valid_pq_m(pq_m_hint, dim),
            pq_k: 256,
            max_iter: 25,
        }
    }
}

pub struct IvfPqIndex {
    pub config: IvfPqConfig,
    pub metric: VectorMetric,
    pub dim: usize,
    /// Coarse cluster centroids: [nlist × dim]
    coarse_centroids: Vec<Vec<f32>>,
    /// Global PQ codebook trained on all vectors
    pq: PQCodebook,
    /// Inverted lists: row IDs per cluster
    inv_row_ids: Vec<Vec<u64>>,
    /// PQ codes per cluster, flat: inv_codes[i].len() == inv_row_ids[i].len() * pq_m
    inv_codes: Vec<Vec<u8>>,
}

impl IvfPqIndex {
    /// Train IVF-PQ index.
    pub fn train(
        row_ids: &[RowId],
        vectors: &[Vec<f32>],
        metric: VectorMetric,
        config: IvfPqConfig,
    ) -> AilakeResult<Self> {
        let n = vectors.len();
        if n == 0 {
            return Err(AilakeError::Catalog(
                "IVF-PQ training requires at least 1 vector".into(),
            ));
        }
        let dim = vectors[0].len();

        let normed_storage: Vec<Vec<f32>>;
        let vecs: &[Vec<f32>] = if metric == VectorMetric::Cosine {
            normed_storage = vectors.iter().map(|v| l2_normalize(v)).collect();
            &normed_storage
        } else {
            vectors
        };

        let nlist = config.nlist.min(n);
        if nlist < config.nlist {
            warn!(
                "ailake: IVF-PQ nlist clamped from {} to {} (n={} vectors); \
                 consider using HNSW for small datasets",
                config.nlist, nlist, n
            );
        }
        let nprobe = config.nprobe.min(nlist);
        let pq_m = find_valid_pq_m(config.pq_m, dim);

        info!(
            "ailake: training IVF-PQ index — n={} dim={} nlist={} nprobe={} pq_m={}",
            n, dim, nlist, nprobe, pq_m
        );

        // Train coarse centroids + PQ codebook, using GPU k-means when available.
        let coarse_centroids = kmeans_dispatch(vecs, nlist, config.max_iter);

        // Assign each vector to its nearest coarse centroid
        let assignments: Vec<usize> = vecs
            .iter()
            .map(|v| nearest_idx(v, &coarse_centroids))
            .collect();

        // Train global PQ on all vectors
        let pq = PQCodebook::train_with_kmeans(
            vecs,
            pq_m,
            config.pq_k.min(256),
            config.max_iter,
            kmeans_dispatch,
        )
        .map_err(|e| AilakeError::Catalog(format!("PQ training failed: {e}")))?;

        // Build inverted lists
        let mut inv_row_ids = vec![Vec::new(); nlist];
        let mut inv_codes = vec![Vec::new(); nlist];

        for (i, (v, &list_idx)) in vecs.iter().zip(assignments.iter()).enumerate() {
            let codes = pq.encode(v);
            inv_row_ids[list_idx].push(row_ids[i].0);
            inv_codes[list_idx].extend_from_slice(&codes);
        }

        Ok(IvfPqIndex {
            config: IvfPqConfig {
                nlist,
                nprobe,
                pq_m,
                ..config
            },
            metric,
            dim,
            coarse_centroids,
            pq,
            inv_row_ids,
            inv_codes,
        })
    }

    /// Search for approximate nearest neighbors.
    ///
    /// `nprobe` overrides `config.nprobe` when `Some`. `ef` is ignored (HNSW compat shim).
    pub fn search(&self, query: &[f32], top_k: usize, nprobe: Option<usize>) -> Vec<(RowId, f32)> {
        let nprobe = nprobe.unwrap_or(self.config.nprobe).min(self.config.nlist);

        let q_normed: Vec<f32>;
        let q: &[f32] = if self.metric == VectorMetric::Cosine {
            q_normed = l2_normalize(query);
            &q_normed
        } else {
            query
        };

        // Select nprobe nearest coarse centroids
        let mut c_dists: Vec<(usize, f32)> = self
            .coarse_centroids
            .iter()
            .enumerate()
            .map(|(i, c)| (i, l2_sq(q, c)))
            .collect();
        c_dists.sort_unstable_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        c_dists.truncate(nprobe);

        // Precompute ADC table once for the full query
        let adc_table = self.pq.compute_adc_table(q);

        // Scan selected inverted lists
        let pq_m = self.config.pq_m;
        let mut candidates: Vec<(RowId, f32)> = Vec::new();

        for (list_idx, _) in &c_dists {
            let row_ids = &self.inv_row_ids[*list_idx];
            let codes_flat = &self.inv_codes[*list_idx];

            for (j, &rid) in row_ids.iter().enumerate() {
                let codes = &codes_flat[j * pq_m..(j + 1) * pq_m];
                let dist = self.pq.adc_distance(codes, &adc_table);
                candidates.push((RowId(rid), dist));
            }
        }

        candidates.sort_unstable_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        candidates.truncate(top_k);
        candidates
    }

    pub fn node_count(&self) -> u64 {
        self.inv_row_ids.iter().map(|l| l.len() as u64).sum()
    }

    pub fn dim(&self) -> usize {
        self.dim
    }
}

// ── Serialization ──────────────────────────────────────────────────────────────

#[derive(Serialize, Deserialize)]
struct IvfPqSnapshot {
    nlist: usize,
    nprobe: usize,
    pq_m: usize,
    pq_k: usize,
    max_iter: usize,
    dim: usize,
    metric: u8,
    coarse_flat: Vec<f32>, // [nlist * dim]
    pq: PQCodebook,
    inv_row_ids: Vec<Vec<u64>>,
    inv_codes: Vec<Vec<u8>>, // flat per list: len == inv_row_ids[i].len() * pq_m
}

pub struct IvfPqSerializer;

impl IvfPqSerializer {
    pub fn to_bytes(index: &IvfPqIndex) -> AilakeResult<Vec<u8>> {
        let coarse_flat: Vec<f32> = index
            .coarse_centroids
            .iter()
            .flat_map(|c| c.iter().copied())
            .collect();
        let snap = IvfPqSnapshot {
            nlist: index.config.nlist,
            nprobe: index.config.nprobe,
            pq_m: index.config.pq_m,
            pq_k: index.config.pq_k,
            max_iter: index.config.max_iter,
            dim: index.dim,
            metric: metric_to_u8(index.metric),
            coarse_flat,
            pq: index.pq.clone(),
            inv_row_ids: index.inv_row_ids.clone(),
            inv_codes: index.inv_codes.clone(),
        };
        bincode::serialize(&snap).map_err(|e| AilakeError::Bincode(e.to_string()))
    }

    pub fn from_bytes(bytes: &[u8]) -> AilakeResult<IvfPqIndex> {
        let snap: IvfPqSnapshot =
            bincode::deserialize(bytes).map_err(|e| AilakeError::Bincode(e.to_string()))?;
        let metric = u8_to_metric(snap.metric)?;
        let coarse_centroids: Vec<Vec<f32>> = snap
            .coarse_flat
            .chunks_exact(snap.dim)
            .map(|c| c.to_vec())
            .collect();
        Ok(IvfPqIndex {
            config: IvfPqConfig {
                nlist: snap.nlist,
                nprobe: snap.nprobe,
                pq_m: snap.pq_m,
                pq_k: snap.pq_k,
                max_iter: snap.max_iter,
            },
            metric,
            dim: snap.dim,
            coarse_centroids,
            pq: snap.pq,
            inv_row_ids: snap.inv_row_ids,
            inv_codes: snap.inv_codes,
        })
    }
}

// ── Helpers ────────────────────────────────────────────────────────────────────

fn l2_normalize(v: &[f32]) -> Vec<f32> {
    let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm < 1e-9 {
        v.to_vec()
    } else {
        v.iter().map(|x| x / norm).collect()
    }
}

fn l2_sq(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| (x - y).powi(2)).sum()
}

fn nearest_idx(v: &[f32], centroids: &[Vec<f32>]) -> usize {
    centroids
        .iter()
        .enumerate()
        .map(|(i, c)| (i, l2_sq(v, c)))
        .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
        .map(|(i, _)| i)
        .unwrap_or(0)
}

/// Find largest M <= requested that divides dim.
pub fn find_valid_pq_m(requested: usize, dim: usize) -> usize {
    for m in (1..=requested).rev() {
        if dim.is_multiple_of(m) {
            return m;
        }
    }
    1
}

fn metric_to_u8(m: VectorMetric) -> u8 {
    match m {
        VectorMetric::Cosine => 0,
        VectorMetric::Euclidean => 1,
        VectorMetric::DotProduct => 2,
    }
}

fn u8_to_metric(v: u8) -> AilakeResult<VectorMetric> {
    match v {
        0 => Ok(VectorMetric::Cosine),
        1 => Ok(VectorMetric::Euclidean),
        2 => Ok(VectorMetric::DotProduct),
        _ => Err(AilakeError::Catalog(format!("unknown metric byte: {v}"))),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_vecs(n: usize, dim: usize) -> (Vec<RowId>, Vec<Vec<f32>>) {
        let row_ids: Vec<RowId> = (0..n).map(|i| RowId(i as u64)).collect();
        let vecs: Vec<Vec<f32>> = (0..n)
            .map(|i| {
                let mut v = vec![0.0f32; dim];
                v[i % dim] = 1.0;
                v
            })
            .collect();
        (row_ids, vecs)
    }

    #[test]
    fn train_and_search_basic() {
        let dim = 8;
        let (ids, vecs) = make_vecs(64, dim);
        let config = IvfPqConfig {
            nlist: 4,
            nprobe: 2,
            pq_m: 2,
            pq_k: 4,
            max_iter: 10,
        };
        let idx = IvfPqIndex::train(&ids, &vecs, VectorMetric::Euclidean, config).unwrap();
        assert_eq!(idx.node_count(), 64);

        let query = vecs[0].clone();
        let results = idx.search(&query, 5, None);
        assert!(!results.is_empty());
        // Top result should be close to query
        assert!(results[0].1 < 0.1, "nearest should be approximate self");
    }

    #[test]
    fn train_cosine_normalizes() {
        let dim = 4;
        let (ids, vecs) = make_vecs(32, dim);
        let config = IvfPqConfig {
            nlist: 4,
            nprobe: 2,
            pq_m: 2,
            pq_k: 4,
            max_iter: 10,
        };
        let idx = IvfPqIndex::train(&ids, &vecs, VectorMetric::Cosine, config).unwrap();
        let results = idx.search(&vecs[0], 1, None);
        assert!(!results.is_empty());
    }

    #[test]
    fn serialize_roundtrip() {
        let dim = 8;
        let (ids, vecs) = make_vecs(32, dim);
        let config = IvfPqConfig {
            nlist: 4,
            nprobe: 2,
            pq_m: 2,
            pq_k: 4,
            max_iter: 10,
        };
        let idx = IvfPqIndex::train(&ids, &vecs, VectorMetric::Euclidean, config).unwrap();
        let bytes = IvfPqSerializer::to_bytes(&idx).unwrap();
        let idx2 = IvfPqSerializer::from_bytes(&bytes).unwrap();

        assert_eq!(idx2.node_count(), idx.node_count());
        assert_eq!(idx2.dim(), idx.dim());

        let q = vecs[0].clone();
        let r1 = idx.search(&q, 5, None);
        let r2 = idx2.search(&q, 5, None);
        assert_eq!(r1.len(), r2.len());
        for (a, b) in r1.iter().zip(r2.iter()) {
            assert_eq!(a.0, b.0, "row_ids should match after roundtrip");
        }
    }

    #[test]
    fn nlist_clamped_to_n() {
        let dim = 4;
        let (ids, vecs) = make_vecs(10, dim); // fewer vectors than default nlist
        let config = IvfPqConfig {
            nlist: 256, // will be clamped to 10
            nprobe: 8,
            pq_m: 2,
            pq_k: 4,
            max_iter: 5,
        };
        let idx = IvfPqIndex::train(&ids, &vecs, VectorMetric::Euclidean, config).unwrap();
        assert!(idx.config.nlist <= 10);
        assert_eq!(idx.node_count(), 10);
    }
}
