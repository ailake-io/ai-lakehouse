//! Hardware capability detection for adaptive index selection.
//!
//! Chooses between HNSW (low overhead, any hardware) and IVF-PQ (GPU k-means,
//! parallel CPU) based on available compute resources and dataset size.

/// Minimum vectors to justify IVF-PQ training (k-means + PQ codebook overhead).
const MIN_VECTORS_FOR_IVF_PQ: usize = 5_000;

/// Minimum logical CPU cores (exclusive) to consider "powerful" when no GPU is available.
/// IVF-PQ requires strictly more than this value (i.e. > 8).
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
        self.has_cuda || self.cpu_logical_cores > MIN_CORES_FOR_IVF_PQ
    }
}

/// Platform-specific CUDA driver library name.
#[cfg(target_os = "linux")]
const CUDA_DRIVER_LIB: &str = "libcuda.so.1";
#[cfg(windows)]
const CUDA_DRIVER_LIB: &str = "nvcuda.dll";
/// macOS has no CUDA support.
#[cfg(not(any(target_os = "linux", windows)))]
const CUDA_DRIVER_LIB: &str = "";

/// CUDA driver API result code — 0 means success.
type CUresult = i32;

pub fn detect_cuda() -> bool {
    use std::sync::OnceLock;
    static HAS_CUDA: OnceLock<bool> = OnceLock::new();
    *HAS_CUDA.get_or_init(probe_cuda_driver)
}

/// Dynamically load the CUDA driver library and count devices.
///
/// Uses dlopen (Linux) / LoadLibrary (Windows) so the binary starts cleanly on
/// machines without a CUDA driver — the library simply won't be found and we
/// return false instead of crashing at startup.
fn probe_cuda_driver() -> bool {
    if CUDA_DRIVER_LIB.is_empty() {
        return false;
    }

    // Safety: libloading loads and immediately queries then drops the library.
    // All function pointers are used only within this scope.
    let lib = match unsafe { libloading::Library::new(CUDA_DRIVER_LIB) } {
        Ok(l) => l,
        Err(_) => return false, // driver not installed
    };

    // cuInit must succeed before any other driver API call.
    let cu_init: libloading::Symbol<unsafe extern "C" fn(u32) -> CUresult> =
        match unsafe { lib.get(b"cuInit\0") } {
            Ok(f) => f,
            Err(_) => return false,
        };
    if unsafe { cu_init(0) } != 0 {
        return false;
    }

    // cuDeviceGetCount returns the number of CUDA-capable devices.
    let cu_count: libloading::Symbol<unsafe extern "C" fn(*mut i32) -> CUresult> =
        match unsafe { lib.get(b"cuDeviceGetCount\0") } {
            Ok(f) => f,
            Err(_) => return false,
        };
    let mut count = 0i32;
    let rc = unsafe { cu_count(&mut count) };
    rc == 0 && count > 0
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
            cpu_logical_cores: MIN_CORES_FOR_IVF_PQ + 1,
            has_avx2: false,
            has_avx512: false,
        };
        assert!(p.recommend_ivf_pq(MIN_VECTORS_FOR_IVF_PQ));
    }

    #[test]
    fn large_dataset_exactly_threshold_picks_hnsw() {
        // strictly > 8, so exactly 8 cores → HNSW
        let p = HardwareProfile {
            has_cuda: false,
            cpu_logical_cores: MIN_CORES_FOR_IVF_PQ,
            has_avx2: false,
            has_avx512: false,
        };
        assert!(!p.recommend_ivf_pq(MIN_VECTORS_FOR_IVF_PQ));
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
