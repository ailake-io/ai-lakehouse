// SPDX-License-Identifier: MIT OR Apache-2.0
//! Hardware capability detection for adaptive index selection.
//!
//! Detection priority: AMD ROCm → NVIDIA CUDA → CPU SIMD.
//! AMD is checked first because ROCm installations often provide a CUDA compatibility
//! layer (`libcuda.so.1`), which would incorrectly report as NVIDIA without the priority check.

use tracing::{debug, info, warn};

/// Minimum vectors to justify IVF-PQ training (k-means + PQ codebook overhead).
const MIN_VECTORS_FOR_IVF_PQ: usize = 5_000;

/// Minimum logical CPU cores (exclusive) to consider "powerful" when no GPU is available.
/// IVF-PQ requires strictly more than this value (i.e. > 8).
const MIN_CORES_FOR_IVF_PQ: usize = 8;

/// Active GPU/CPU compute backend.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HardwareBackend {
    /// No GPU detected — use SIMD-accelerated CPU kernels (rayon).
    CpuSimd,
    /// NVIDIA GPU with CUDA driver (cuBLAS SGEMM via libloading — no compile-time SDK required).
    NvidiaCuda,
    /// AMD GPU with ROCm/HIP driver (hipBLAS SGEMM via libloading).
    AmdRocm,
}

/// Detected hardware capabilities.
pub struct HardwareProfile {
    /// Active GPU backend (highest priority GPU, or CpuSimd when none found).
    pub backend: HardwareBackend,
    /// CUDA-capable GPU found at runtime via libloading (`libcuda.so.1` + `libcublas.so`).
    pub has_cuda: bool,
    /// AMD ROCm/HIP GPU found at runtime.
    pub has_rocm: bool,
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
        let backend = detect_backend();
        Self {
            backend,
            has_cuda: backend == HardwareBackend::NvidiaCuda,
            has_rocm: backend == HardwareBackend::AmdRocm,
            cpu_logical_cores: rayon::current_num_threads(),
            has_avx2: detect_avx2(),
            has_avx512: detect_avx512(),
        }
    }

    /// True when IVF-PQ training is justified for a dataset of `n_vectors` vectors.
    ///
    /// Returns false when:
    /// - `n_vectors < MIN_VECTORS_FOR_IVF_PQ` (k-means clusters would be meaningless)
    /// - Neither GPU (CUDA or ROCm) nor a sufficiently parallel CPU is available
    pub fn recommend_ivf_pq(&self, n_vectors: usize) -> bool {
        if n_vectors < MIN_VECTORS_FOR_IVF_PQ {
            return false;
        }
        self.has_cuda || self.has_rocm || self.cpu_logical_cores > MIN_CORES_FOR_IVF_PQ
    }
}

// ── Backend detection ─────────────────────────────────────────────────────────

use std::sync::OnceLock;

static BACKEND: OnceLock<HardwareBackend> = OnceLock::new();

/// Returns the best available compute backend, probing once per process.
///
/// Priority: AMD ROCm > NVIDIA CUDA > CPU SIMD.
/// AMD is checked first to correctly identify ROCm machines that also expose
/// a CUDA compatibility layer.
pub fn detect_backend() -> HardwareBackend {
    *BACKEND.get_or_init(|| {
        let backend = if probe_rocm_driver() {
            HardwareBackend::AmdRocm
        } else if probe_cuda_driver() {
            HardwareBackend::NvidiaCuda
        } else {
            HardwareBackend::CpuSimd
        };
        match backend {
            HardwareBackend::AmdRocm => {
                info!("ailake: GPU backend selected — AMD ROCm (hipBLAS SGEMM via libloading)");
            }
            HardwareBackend::NvidiaCuda => {
                info!("ailake: GPU backend selected — NVIDIA CUDA (cuBLAS SGEMM via libloading)");
            }
            HardwareBackend::CpuSimd => {
                info!(
                    "ailake: no GPU detected — using CPU SIMD backend (rayon + AVX2/NEON); \
                     to enable GPU acceleration install the NVIDIA CUDA runtime \
                     (libcudart + libcublas) or AMD ROCm (libamdhip64 + libhipblas)"
                );
            }
        }
        backend
    })
}

/// True only when an NVIDIA CUDA GPU is the active backend.
/// Returns false on AMD ROCm machines (even those with a CUDA compat layer).
pub fn detect_cuda() -> bool {
    detect_backend() == HardwareBackend::NvidiaCuda
}

/// True only when an AMD ROCm/HIP GPU is the active backend.
pub fn detect_rocm() -> bool {
    detect_backend() == HardwareBackend::AmdRocm
}

// ── Library names ─────────────────────────────────────────────────────────────

#[cfg(target_os = "linux")]
const CUDA_DRIVER_LIB: &str = "libcuda.so.1";
#[cfg(windows)]
const CUDA_DRIVER_LIB: &str = "nvcuda.dll";
#[cfg(not(any(target_os = "linux", windows)))]
const CUDA_DRIVER_LIB: &str = "";

#[cfg(target_os = "linux")]
const ROCM_DRIVER_LIB: &str = "libamdhip64.so";
#[cfg(windows)]
const ROCM_DRIVER_LIB: &str = "amdhip64.dll";
#[cfg(not(any(target_os = "linux", windows)))]
const ROCM_DRIVER_LIB: &str = "";

/// CUDA/HIP driver API result code — 0 means success.
type GpuResult = i32;

// ── CUDA probe ────────────────────────────────────────────────────────────────

fn probe_cuda_driver() -> bool {
    if CUDA_DRIVER_LIB.is_empty() {
        return false;
    }
    let lib = match unsafe { libloading::Library::new(CUDA_DRIVER_LIB) } {
        Ok(l) => l,
        Err(e) => {
            debug!(
                "ailake: CUDA driver library `{}` not found ({}); \
                 GPU acceleration unavailable — install the NVIDIA CUDA driver to enable it",
                CUDA_DRIVER_LIB, e
            );
            return false;
        }
    };

    let cu_init: libloading::Symbol<unsafe extern "C" fn(u32) -> GpuResult> =
        match unsafe { lib.get(b"cuInit\0") } {
            Ok(f) => f,
            Err(e) => {
                warn!(
                    "ailake: `{}` loaded but `cuInit` symbol missing ({}); \
                     CUDA installation may be incomplete — falling back to CPU",
                    CUDA_DRIVER_LIB, e
                );
                return false;
            }
        };
    let rc = unsafe { cu_init(0) };
    if rc != 0 {
        warn!(
            "ailake: cuInit(0) returned error code {} — CUDA driver present but no usable GPU \
             or driver not initialised; falling back to CPU SIMD",
            rc
        );
        return false;
    }

    let cu_count: libloading::Symbol<unsafe extern "C" fn(*mut i32) -> GpuResult> =
        match unsafe { lib.get(b"cuDeviceGetCount\0") } {
            Ok(f) => f,
            Err(e) => {
                warn!(
                    "ailake: `cuDeviceGetCount` symbol missing in `{}` ({}); \
                     falling back to CPU",
                    CUDA_DRIVER_LIB, e
                );
                return false;
            }
        };
    let mut count = 0i32;
    let rc = unsafe { cu_count(&mut count) };
    if rc == 0 && count == 0 {
        warn!(
            "ailake: CUDA driver initialised but no CUDA-capable devices found (count=0); \
             falling back to CPU SIMD"
        );
        return false;
    }
    rc == 0 && count > 0
}

// ── ROCm probe ────────────────────────────────────────────────────────────────

/// Dynamically probe the AMD HIP runtime library.
///
/// Uses `hipInit(0)` + `hipGetDeviceCount` via libloading — no ROCm toolkit
/// required at compile time. Returns false when:
/// - `libamdhip64.so` is not installed
/// - `hipInit` fails (no GPU / driver not loaded)
/// - no ROCm-capable devices found
fn probe_rocm_driver() -> bool {
    if ROCM_DRIVER_LIB.is_empty() {
        return false;
    }
    let lib = match unsafe { libloading::Library::new(ROCM_DRIVER_LIB) } {
        Ok(l) => l,
        Err(e) => {
            debug!(
                "ailake: ROCm library `{}` not found ({}); \
                 AMD GPU acceleration unavailable — install the ROCm runtime to enable it",
                ROCM_DRIVER_LIB, e
            );
            return false;
        }
    };

    // hipInit(0) must succeed before any other HIP driver call.
    let hip_init: libloading::Symbol<unsafe extern "C" fn(u32) -> GpuResult> =
        match unsafe { lib.get(b"hipInit\0") } {
            Ok(f) => f,
            Err(e) => {
                warn!(
                    "ailake: `{}` loaded but `hipInit` symbol missing ({}); \
                     ROCm installation may be incomplete — falling back to CPU",
                    ROCM_DRIVER_LIB, e
                );
                return false;
            }
        };
    let rc = unsafe { hip_init(0) };
    if rc != 0 {
        warn!(
            "ailake: hipInit(0) returned error code {} — ROCm driver present but no usable GPU \
             or driver not initialised; falling back to CPU SIMD",
            rc
        );
        return false;
    }

    let hip_count: libloading::Symbol<unsafe extern "C" fn(*mut i32) -> GpuResult> =
        match unsafe { lib.get(b"hipGetDeviceCount\0") } {
            Ok(f) => f,
            Err(e) => {
                warn!(
                    "ailake: `hipGetDeviceCount` symbol missing in `{}` ({}); \
                     falling back to CPU",
                    ROCM_DRIVER_LIB, e
                );
                return false;
            }
        };
    let mut count = 0i32;
    let rc = unsafe { hip_count(&mut count) };
    if rc == 0 && count == 0 {
        warn!(
            "ailake: ROCm driver initialised but no ROCm-capable devices found (count=0); \
             falling back to CPU SIMD"
        );
        return false;
    }
    rc == 0 && count > 0
}

// ── SIMD detection ────────────────────────────────────────────────────────────

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

// ── Tests ─────────────────────────────────────────────────────────────────────

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
            backend: HardwareBackend::NvidiaCuda,
            has_cuda: true,
            has_rocm: false,
            cpu_logical_cores: 64,
            has_avx2: true,
            has_avx512: true,
        };
        assert!(!p.recommend_ivf_pq(MIN_VECTORS_FOR_IVF_PQ - 1));
    }

    #[test]
    fn large_dataset_cuda_picks_ivf_pq() {
        let p = HardwareProfile {
            backend: HardwareBackend::NvidiaCuda,
            has_cuda: true,
            has_rocm: false,
            cpu_logical_cores: 1,
            has_avx2: false,
            has_avx512: false,
        };
        assert!(p.recommend_ivf_pq(MIN_VECTORS_FOR_IVF_PQ));
    }

    #[test]
    fn large_dataset_rocm_picks_ivf_pq() {
        let p = HardwareProfile {
            backend: HardwareBackend::AmdRocm,
            has_cuda: false,
            has_rocm: true,
            cpu_logical_cores: 1,
            has_avx2: false,
            has_avx512: false,
        };
        assert!(p.recommend_ivf_pq(MIN_VECTORS_FOR_IVF_PQ));
    }

    #[test]
    fn large_dataset_many_cores_picks_ivf_pq() {
        let p = HardwareProfile {
            backend: HardwareBackend::CpuSimd,
            has_cuda: false,
            has_rocm: false,
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
            backend: HardwareBackend::CpuSimd,
            has_cuda: false,
            has_rocm: false,
            cpu_logical_cores: MIN_CORES_FOR_IVF_PQ,
            has_avx2: false,
            has_avx512: false,
        };
        assert!(!p.recommend_ivf_pq(MIN_VECTORS_FOR_IVF_PQ));
    }

    #[test]
    fn large_dataset_weak_hardware_picks_hnsw() {
        let p = HardwareProfile {
            backend: HardwareBackend::CpuSimd,
            has_cuda: false,
            has_rocm: false,
            cpu_logical_cores: MIN_CORES_FOR_IVF_PQ - 1,
            has_avx2: false,
            has_avx512: false,
        };
        assert!(!p.recommend_ivf_pq(MIN_VECTORS_FOR_IVF_PQ));
    }

    #[test]
    fn backend_consistency_cuda() {
        let p = HardwareProfile {
            backend: HardwareBackend::NvidiaCuda,
            has_cuda: true,
            has_rocm: false,
            cpu_logical_cores: 4,
            has_avx2: false,
            has_avx512: false,
        };
        assert!(p.has_cuda);
        assert!(!p.has_rocm);
        assert_eq!(p.backend, HardwareBackend::NvidiaCuda);
    }

    #[test]
    fn backend_consistency_rocm() {
        let p = HardwareProfile {
            backend: HardwareBackend::AmdRocm,
            has_cuda: false,
            has_rocm: true,
            cpu_logical_cores: 4,
            has_avx2: false,
            has_avx512: false,
        };
        assert!(!p.has_cuda);
        assert!(p.has_rocm);
        assert_eq!(p.backend, HardwareBackend::AmdRocm);
    }
}
