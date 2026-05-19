//! GPU-accelerated brute-force vector search via candle-core + CUDA.
//!
//! Compiled only when the `gpu` feature is enabled.
//! At runtime, returns `None` if no CUDA-capable device is found; the caller
//! must then fall back to the CPU path.

#[cfg(feature = "gpu")]
pub use gpu_impl::try_gpu_search;

#[cfg(feature = "gpu")]
mod gpu_impl {
    use ailake_core::{RowId, VectorMetric};
    use candle_core::{DType, Device, Tensor};

    /// Run brute-force top-k vector search on the GPU.
    ///
    /// Returns `None` when:
    /// - no CUDA-capable GPU is detected at runtime
    /// - candle fails to allocate GPU tensors (OOM, driver error, etc.)
    ///
    /// Callers must fall back to the CPU path in those cases.
    pub fn try_gpu_search(
        query: &[f32],
        vectors: &[(RowId, Vec<f32>)],
        metric: VectorMetric,
        top_k: usize,
    ) -> Option<Vec<(RowId, f32)>> {
        if vectors.is_empty() {
            return Some(vec![]);
        }

        // Detect GPU at runtime; fall back gracefully if none found.
        let dev = match Device::cuda_if_available(0) {
            Ok(d) if d.is_cuda() => d,
            _ => return None,
        };

        let n = vectors.len();
        let dim = query.len();

        // Build DB tensor [N, dim] on GPU
        let flat_db: Vec<f32> = vectors
            .iter()
            .flat_map(|(_, v)| v.iter().copied())
            .collect();
        let db = Tensor::from_vec(flat_db, (n, dim), &dev).ok()?; // [N, dim]
        let q = Tensor::from_vec(query.to_vec(), (1, dim), &dev).ok()?; // [1, dim]

        let distances: Vec<f32> = match metric {
            VectorMetric::DotProduct => {
                // distance = -dot(q, v)  so lower = more similar
                let dots = q.matmul(&db.t().ok()?).ok()?.squeeze(0).ok()?; // [N]
                dots.neg().ok()?.to_vec1().ok()?
            }
            VectorMetric::Cosine => {
                let q_n = normalize_rows_2d(&q)?; // [1, dim]
                let db_n = normalize_rows_2d(&db)?; // [N, dim]
                let cos_sim = q_n.matmul(&db_n.t().ok()?).ok()?.squeeze(0).ok()?; // [N]
                let ones = Tensor::ones(n, DType::F32, &dev).ok()?;
                ones.sub(&cos_sim).ok()?.to_vec1().ok()?
            }
            VectorMetric::Euclidean => {
                // ||q - v||²  = broadcast_sub then sqr + sum_keepdim + sqrt
                let diff = q.broadcast_sub(&db).ok()?; // [N, dim]
                diff.sqr()
                    .ok()?
                    .sum_keepdim(1)
                    .ok()?
                    .squeeze(1)
                    .ok()?
                    .sqrt()
                    .ok()?
                    .to_vec1()
                    .ok()?
            }
        };

        // Top-k selection (small result set — cheap on CPU)
        let mut indexed: Vec<(usize, f32)> = distances.into_iter().enumerate().collect();
        indexed.sort_unstable_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        indexed.truncate(top_k);

        Some(
            indexed
                .into_iter()
                .map(|(i, d)| (vectors[i].0, d))
                .collect(),
        )
    }

    fn normalize_rows_2d(t: &Tensor) -> Option<Tensor> {
        let norms = t.sqr().ok()?.sum_keepdim(1).ok()?.sqrt().ok()?; // [rows, 1]
        t.broadcast_div(&norms).ok()
    }
}
