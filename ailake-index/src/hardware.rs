//! Hardware capability detection for adaptive index selection.
//!
//! Chooses between HNSW (low overhead, any hardware) and IVF-PQ (GPU k-means,
//! parallel CPU) based on available compute resources and dataset size.

/// Minimum vectors to justify IVF-PQ training (k-means + PQ codebook overhead).
const MIN_VECTORS_FOR_IVF_PQ: usize = 5_000;

/// Minimum logical CPU cores to consider "powerful" when no GPU is available.
const MIN_CORES_FOR_IVF_PQ: usize = 8;

/// Detected hardware capabilities.
pub struct HardwareProfile {
    /// CUDA-capable GPU found at runtime (requires `gpu` feature at compile time).
    pub has_cuda: bool,
    /// Logical CPU cores available to rayon's global thread pool.
    pub cpu_logical_cores: usize,
    /// x86_64: AVX2 support detected via CPUID.
    pub has_avx2: bool,
    /// x86_64: AVX-512F support detected via CPUID.
    pub has_avx512: bool,
}

impl HardwareProfile {
    /// Probe the current machine's capabilities.
    pub fn detect() -> Self {
        Self {
            has_cuda: detect_cuda(),
            cpu_logical_cores: rayon::current_num_threads(),
            has_avx2: detect_avx2(),
            has_avx512: detect_avx512(),
        }
    }

    /// True when IVF-PQ training is justified for a dataset of `n_vectors` vectors.
    ///
    /// Returns false when:
    /// - `n_vectors < MIN_VECTORS_FOR_IVF_PQ` (k-means clusters would be meaningless)
    /// - Neither GPU nor a sufficiently parallel CPU is available
    pub fn recommend_ivf_pq(&self, n_vectors: usize) -> bool {
        if n_vectors < MIN_VECTORS_FOR_IVF_PQ {
            return false;
        }
        self.has_cuda || self.cpu_logical_cores >= MIN_CORES_FOR_IVF_PQ
    }
}

fn detect_cuda() -> bool {
    #[cfg(feature = "gpu")]
    {
        use candle_core::Device;
        matches!(Device::cuda_if_available(0), Ok(d) if d.is_cuda())
    }
    #[cfg(not(feature = "gpu"))]
    false
}

fn detect_avx2() -> bool {
    #[cfg(target_arch = "x86_64")]
    {
        std::is_x86_feature_detected!("avx2")
    }
    #[cfg(not(target_arch = "x86_64"))]
    false
}

fn detect_avx512() -> bool {
    #[cfg(target_arch = "x86_64")]
    {
        std::is_x86_feature_detected!("avx512f")
    }
    #[cfg(not(target_arch = "x86_64"))]
    false
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn detect_runs_without_panic() {
        let p = HardwareProfile::detect();
        assert!(p.cpu_logical_cores >= 1);
    }

    #[test]
    fn small_dataset_always_hnsw() {
        let p = HardwareProfile {
            has_cuda: true,
            cpu_logical_cores: 64,
            has_avx2: true,
            has_avx512: true,
        };
        assert!(!p.recommend_ivf_pq(MIN_VECTORS_FOR_IVF_PQ - 1));
    }

    #[test]
    fn large_dataset_gpu_picks_ivf_pq() {
        let p = HardwareProfile {
            has_cuda: true,
            cpu_logical_cores: 1,
            has_avx2: false,
            has_avx512: false,
        };
        assert!(p.recommend_ivf_pq(MIN_VECTORS_FOR_IVF_PQ));
    }

    #[test]
    fn large_dataset_many_cores_picks_ivf_pq() {
        let p = HardwareProfile {
            has_cuda: false,
            cpu_logical_cores: MIN_CORES_FOR_IVF_PQ,
            has_avx2: false,
            has_avx512: false,
        };
        assert!(p.recommend_ivf_pq(MIN_VECTORS_FOR_IVF_PQ));
    }

    #[test]
    fn large_dataset_weak_hardware_picks_hnsw() {
        let p = HardwareProfile {
            has_cuda: false,
            cpu_logical_cores: MIN_CORES_FOR_IVF_PQ - 1,
            has_avx2: false,
            has_avx512: false,
        };
        assert!(!p.recommend_ivf_pq(MIN_VECTORS_FOR_IVF_PQ));
    }
}
