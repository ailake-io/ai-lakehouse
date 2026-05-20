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
    /// `flat_vecs` is a contiguous row-major array: flat_vecs[i*dim..(i+1)*dim] = vector i.
    ///
    /// Returns `None` when:
    /// - no CUDA-capable GPU is detected at runtime
    /// - candle fails to allocate GPU tensors (OOM, driver error, etc.)
    pub fn try_gpu_search(
        query: &[f32],
        row_ids: &[u64],
        flat_vecs: &[f32],
        dim: usize,
        metric: VectorMetric,
        top_k: usize,
    ) -> Option<Vec<(RowId, f32)>> {
        let n = row_ids.len();
        if n == 0 {
            return Some(vec![]);
        }

        let dev = match Device::cuda_if_available(0) {
            Ok(d) if d.is_cuda() => d,
            _ => return None,
        };

        let db = Tensor::from_slice(flat_vecs, (n, dim), &dev).ok()?;
        let q = Tensor::from_slice(query, (1, dim), &dev).ok()?;

        let distances: Vec<f32> = match metric {
            VectorMetric::DotProduct => {
                let dots = q.matmul(&db.t().ok()?).ok()?.squeeze(0).ok()?;
                dots.neg().ok()?.to_vec1().ok()?
            }
            VectorMetric::Cosine => {
                let q_n = normalize_rows_2d(&q)?;
                let db_n = normalize_rows_2d(&db)?;
                let cos_sim = q_n.matmul(&db_n.t().ok()?).ok()?.squeeze(0).ok()?;
                let ones = Tensor::ones(n, DType::F32, &dev).ok()?;
                ones.sub(&cos_sim).ok()?.to_vec1().ok()?
            }
            VectorMetric::Euclidean => {
                let diff = q.broadcast_sub(&db).ok()?;
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

        let mut indexed: Vec<(usize, f32)> = distances.into_iter().enumerate().collect();
        indexed.sort_unstable_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        indexed.truncate(top_k);

        Some(
            indexed
                .into_iter()
                .map(|(i, d)| (RowId::new(row_ids[i]), d))
                .collect(),
        )
    }

    fn normalize_rows_2d(t: &Tensor) -> Option<Tensor> {
        let norms = t.sqr().ok()?.sum_keepdim(1).ok()?.sqrt().ok()?;
        t.broadcast_div(&norms).ok()
    }
}
