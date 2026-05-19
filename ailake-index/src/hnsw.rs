use ailake_core::{RowId, VectorMetric};
use ailake_vec::{cosine_distance, dot_product, euclidean_distance};
use rayon::prelude::*;

#[derive(Debug, Clone)]
pub struct HnswConfig {
    /// Max connections per node (M in HNSW).
    pub m: usize,
    pub ef_construction: usize,
    pub max_elements: usize,
}

impl Default for HnswConfig {
    fn default() -> Self {
        Self {
            m: 16,
            ef_construction: 200,
            max_elements: 1_000_000,
        }
    }
}

pub struct HnswBuilder {
    pub(crate) config: HnswConfig,
    pub(crate) metric: VectorMetric,
    pub(crate) dim: u32,
    pub(crate) vectors: Vec<(RowId, Vec<f32>)>,
}

impl HnswBuilder {
    pub fn new(dim: u32, metric: VectorMetric, config: HnswConfig) -> Self {
        Self {
            config,
            metric,
            dim,
            vectors: Vec::new(),
        }
    }

    pub fn insert(&mut self, row_id: RowId, vector: Vec<f32>) {
        self.vectors.push((row_id, vector));
    }

    pub fn build(self) -> HnswIndex {
        HnswIndex {
            config: self.config,
            metric: self.metric,
            dim: self.dim,
            vectors: self.vectors,
        }
    }
}

pub struct HnswIndex {
    pub(crate) config: HnswConfig,
    pub(crate) metric: VectorMetric,
    pub(crate) dim: u32,
    /// Stored for serialization and brute-force search.
    pub(crate) vectors: Vec<(RowId, Vec<f32>)>,
}

impl HnswIndex {
    /// Top-k search. Uses GPU when available (requires `gpu` feature + CUDA at runtime);
    /// falls back to parallel CPU brute-force otherwise.
    /// Returns (row_id, distance) sorted ascending by distance.
    pub fn search(&self, query: &[f32], top_k: usize, ef: usize) -> Vec<(RowId, f32)> {
        // Try GPU path first (compiled away when `gpu` feature is not enabled).
        #[cfg(feature = "gpu")]
        if let Some(results) = crate::gpu::try_gpu_search(query, &self.vectors, self.metric, top_k)
        {
            return results;
        }

        self.cpu_search(query, top_k, ef)
    }

    /// Parallel CPU brute-force. Used when GPU is unavailable or not compiled in.
    fn cpu_search(&self, query: &[f32], top_k: usize, _ef: usize) -> Vec<(RowId, f32)> {
        let mut results: Vec<(RowId, f32)> = self
            .vectors
            .par_iter()
            .map(|(id, v)| (*id, self.distance(query, v)))
            .collect();
        results.sort_unstable_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        results.truncate(top_k);
        results
    }

    pub fn node_count(&self) -> u64 {
        self.vectors.len() as u64
    }

    pub fn metric(&self) -> VectorMetric {
        self.metric
    }

    pub fn dim(&self) -> u32 {
        self.dim
    }

    fn distance(&self, a: &[f32], b: &[f32]) -> f32 {
        match self.metric {
            VectorMetric::Cosine => cosine_distance(a, b),
            VectorMetric::Euclidean => euclidean_distance(a, b),
            VectorMetric::DotProduct => -dot_product(a, b),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_index(vecs: Vec<Vec<f32>>) -> HnswIndex {
        let mut b = HnswBuilder::new(
            vecs[0].len() as u32,
            VectorMetric::Cosine,
            Default::default(),
        );
        for (i, v) in vecs.into_iter().enumerate() {
            b.insert(RowId::new(i as u64), v);
        }
        b.build()
    }

    #[test]
    fn top1_is_exact_match() {
        let idx = make_index(vec![
            vec![1.0, 0.0, 0.0],
            vec![0.0, 1.0, 0.0],
            vec![0.0, 0.0, 1.0],
        ]);
        let results = idx.search(&[1.0, 0.0, 0.0], 1, 50);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].0, RowId::new(0));
        assert!(results[0].1 < 1e-5);
    }

    #[test]
    fn top_k_returns_k() {
        let idx = make_index(vec![
            vec![1.0, 0.0],
            vec![0.8, 0.2],
            vec![0.0, 1.0],
            vec![-1.0, 0.0],
        ]);
        let results = idx.search(&[1.0, 0.0], 2, 50);
        assert_eq!(results.len(), 2);
    }

    #[test]
    fn node_count() {
        let idx = make_index(vec![vec![1.0, 0.0]; 5]);
        assert_eq!(idx.node_count(), 5);
    }
}
