//! GPU-accelerated brute-force vector search via candle-core + CUDA.
//!
//! Compiled only when the `gpu` feature is enabled.
//! At runtime, returns `None` if no CUDA-capable device is found; the caller
//! must then fall back to the CPU path.

#[cfg(feature = "gpu")]
pub use gpu_impl::{try_gpu_kmeans, try_gpu_search, try_gpu_search_batch};

/// No-op stub: returns `None` so callers degrade to CPU without feature-gating at call site.
#[cfg(not(feature = "gpu"))]
pub fn try_gpu_search_batch(
    _queries: &[&[f32]],
    _row_ids: &[u64],
    _flat_vecs: &[f32],
    _dim: usize,
    _metric: ailake_core::VectorMetric,
    _top_k: usize,
) -> Option<Vec<Vec<(ailake_core::RowId, f32)>>> {
    None
}

#[cfg(feature = "gpu")]
mod gpu_impl {
    use ailake_core::{RowId, VectorMetric};
    use candle_core::{DType, Device, Tensor};

    /// Acquire the cached CUDA device.
    ///
    /// Returns `None` when:
    /// - `detect_cuda()` found no CUDA driver / devices (fast OnceLock path)
    /// - candle fails to initialise the device (OOM, driver mismatch, etc.)
    fn cuda_device() -> Option<Device> {
        // Fast early-exit via libloading probe — avoids candle init overhead on CPU-only hosts.
        if !crate::hardware::detect_cuda() {
            return None;
        }
        match Device::cuda_if_available(0) {
            Ok(d) if d.is_cuda() => Some(d),
            _ => None,
        }
    }

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

        let dev = cuda_device()?;

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

    /// GPU batch top-k search: computes a [Q × N] distance matrix in one matmul.
    ///
    /// `queries` is a slice of Q query vectors (each of length `dim`).
    /// `flat_vecs` is row-major [N × dim].
    ///
    /// Returns `None` when no CUDA device is available or on any GPU error;
    /// the caller must fall back to CPU flat scan.
    pub fn try_gpu_search_batch(
        queries: &[&[f32]],
        row_ids: &[u64],
        flat_vecs: &[f32],
        dim: usize,
        metric: VectorMetric,
        top_k: usize,
    ) -> Option<Vec<Vec<(RowId, f32)>>> {
        let n = row_ids.len();
        let q_count = queries.len();
        if n == 0 || q_count == 0 {
            return Some(vec![vec![]; q_count]);
        }

        let dev = cuda_device()?;

        // Upload DB vectors [N, dim] once.
        let db = Tensor::from_slice(flat_vecs, (n, dim), &dev).ok()?;

        // Flatten queries into [Q, dim].
        let q_flat: Vec<f32> = queries.iter().flat_map(|q| q.iter().copied()).collect();
        let q_tensor = Tensor::from_slice(&q_flat, (q_count, dim), &dev).ok()?;

        // Compute [Q, N] distance matrix.
        let dist_matrix: Vec<Vec<f32>> = match metric {
            VectorMetric::DotProduct => {
                let dots = q_tensor.matmul(&db.t().ok()?).ok()?; // [Q, N]
                dots.neg().ok()?.to_vec2::<f32>().ok()?
            }
            VectorMetric::Cosine => {
                let q_n = normalize_rows_2d(&q_tensor)?;
                let db_n = normalize_rows_2d(&db)?;
                let cos_sim = q_n.matmul(&db_n.t().ok()?).ok()?; // [Q, N]
                let ones = Tensor::ones((q_count, n), DType::F32, &dev).ok()?;
                ones.sub(&cos_sim).ok()?.to_vec2::<f32>().ok()?
            }
            VectorMetric::Euclidean => {
                // ||q - d||^2 = ||q||^2 + ||d||^2 - 2·q·dᵀ
                let q_sq = q_tensor.sqr().ok()?.sum_keepdim(1).ok()?; // [Q, 1]
                let db_sq = db.sqr().ok()?.sum_keepdim(1).ok()?.t().ok()?; // [1, N]
                let cross = q_tensor.matmul(&db.t().ok()?).ok()?; // [Q, N]
                let cross2 = cross.affine(2.0, 0.0).ok()?;
                let sq = q_sq.broadcast_add(&db_sq).ok()?.broadcast_sub(&cross2).ok()?;
                sq.sqrt().ok()?.to_vec2::<f32>().ok()?
            }
        };

        // Top-k per query — CPU sort (k << N).
        let results = dist_matrix
            .into_iter()
            .map(|dists| {
                let mut indexed: Vec<(usize, f32)> = dists.into_iter().enumerate().collect();
                indexed.sort_unstable_by(|a, b| {
                    a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal)
                });
                indexed.truncate(top_k);
                indexed
                    .into_iter()
                    .map(|(i, d)| (RowId::new(row_ids[i]), d))
                    .collect()
            })
            .collect();

        Some(results)
    }

    /// GPU k-means via candle matmul. Returns `None` when no CUDA device is found.
    ///
    /// Distance computation stays on GPU (the bottleneck); centroid update runs on
    /// CPU to avoid complex scatter ops. Transfers per iteration: n×4 bytes (assignments)
    /// + k×dim×4 bytes (new centroids) — negligible vs the n×k×dim matmul cost.
    pub fn try_gpu_kmeans(
        vectors: &[Vec<f32>],
        k: usize,
        max_iter: usize,
    ) -> Option<Vec<Vec<f32>>> {
        let n = vectors.len();
        if n == 0 {
            return Some(vec![]);
        }
        let dim = vectors[0].len();
        let k = k.min(n);

        let dev = cuda_device()?;

        // Flatten + upload all vectors to GPU once
        let flat: Vec<f32> = vectors.iter().flat_map(|v| v.iter().copied()).collect();
        let x = Tensor::from_slice(&flat, (n, dim), &dev).ok()?;

        // Init centroids via evenly-spaced sampling (deterministic)
        let step = n / k;
        let mut centroids_flat: Vec<f32> = (0..k)
            .flat_map(|i| vectors[(i * step) % n].iter().copied())
            .collect();
        let mut centroids = Tensor::from_slice(&centroids_flat, (k, dim), &dev).ok()?;

        // ||x||^2 — constant across iterations
        let x_sq = x.sqr().ok()?.sum_keepdim(1).ok()?; // [n, 1]

        let mut prev_asgn: Vec<u32> = vec![];

        for _ in 0..max_iter {
            // ||x - c||^2 = ||x||^2 + ||c||^2 - 2·x·cᵀ  (all on GPU)
            let c_sq = centroids.sqr().ok()?.sum_keepdim(1).ok()?; // [k, 1]
            let cross = x.matmul(&centroids.t().ok()?).ok()?; // [n, k]
            let cross2 = cross.affine(2.0, 0.0).ok()?;
            let base = x_sq.broadcast_add(&c_sq.t().ok()?).ok()?; // [n, k]
            let dists = base.sub(&cross2).ok()?; // [n, k]

            // Argmin per row → [n] u32 (small D→H transfer)
            let asgn: Vec<u32> = dists.argmin(candle_core::D::Minus1).ok()?.to_vec1().ok()?;

            if asgn == prev_asgn {
                break;
            }

            // Centroid update on CPU
            let mut new_flat = vec![0.0f32; k * dim];
            let mut counts = vec![0usize; k];
            for (i, &ci) in asgn.iter().enumerate() {
                let ci = ci as usize;
                for (d, &v) in vectors[i].iter().enumerate() {
                    new_flat[ci * dim + d] += v;
                }
                counts[ci] += 1;
            }
            for j in 0..k {
                if counts[j] > 0 {
                    let inv = 1.0 / counts[j] as f32;
                    for d in 0..dim {
                        new_flat[j * dim + d] *= inv;
                    }
                } else {
                    // Empty cluster: keep previous centroid
                    new_flat[j * dim..(j + 1) * dim]
                        .copy_from_slice(&centroids_flat[j * dim..(j + 1) * dim]);
                }
            }

            // Upload new centroids to GPU (small H→D transfer)
            centroids = Tensor::from_slice(&new_flat, (k, dim), &dev).ok()?;
            centroids_flat = new_flat;
            prev_asgn = asgn;
        }

        Some(centroids_flat.chunks(dim).map(|c| c.to_vec()).collect())
    }
}
