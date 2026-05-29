// SPDX-License-Identifier: MIT OR Apache-2.0
//! GPU-accelerated vector search and k-means.
//!
//! Two independent GPU backends, both via runtime `libloading` — no compile-time
//! GPU SDK required. Either backend returns `None` when its hardware/libraries
//! are absent; callers fall back to CPU automatically.
//!
//!   - NVIDIA CUDA: cuBLAS SGEMM via dlopen of `libcudart` + `libcublas`.
//!   - AMD ROCm:    hipBLAS SGEMM via dlopen of `libamdhip64` + `libhipblas`.

pub use nvidia_impl::{try_nvidia_kmeans, try_nvidia_search_batch};
pub use rocm_impl::{try_rocm_kmeans, try_rocm_search_batch};

// ── NVIDIA CUDA backend ───────────────────────────────────────────────────────
//
// Always compiled. Returns `None` at runtime when:
//   - No NVIDIA CUDA driver found (`detect_cuda()` is false)
//   - cuBLAS / CUDA runtime libraries not installed
//   - Any GPU allocation or compute error
//
// SGEMM formulation identical to ROCm backend; only library names and
// operation constants differ (CUBLAS_OP_N=0/CUBLAS_OP_T=1 vs HIP 111/112).

mod nvidia_impl {
    use std::ffi::c_void;

    use ailake_core::{RowId, VectorMetric};
    use libloading::{Library, Symbol};
    use tracing::warn;

    // cudaMemcpyKind constants
    const H2D: i32 = 1; // cudaMemcpyHostToDevice
    const D2H: i32 = 2; // cudaMemcpyDeviceToHost

    // cublasOperation_t constants
    const OP_T: i32 = 1; // CUBLAS_OP_T — transpose
    const OP_N: i32 = 0; // CUBLAS_OP_N — no-transpose

    // Type aliases for CUDA runtime function pointers.
    type CudaMallocFn = unsafe extern "C" fn(*mut *mut c_void, usize) -> i32;
    type CudaFreeFn = unsafe extern "C" fn(*mut c_void) -> i32;
    type CudaMemcpyFn = unsafe extern "C" fn(*mut c_void, *const c_void, usize, i32) -> i32;
    type CudaSyncFn = unsafe extern "C" fn() -> i32;

    #[cfg(target_os = "linux")]
    const RT_LIBS: &[&str] = &["libcudart.so", "libcudart.so.12", "libcudart.so.11"];
    #[cfg(windows)]
    const RT_LIBS: &[&str] = &["cudart64_12.dll", "cudart64_11.dll"];
    #[cfg(not(any(target_os = "linux", windows)))]
    const RT_LIBS: &[&str] = &[];

    #[cfg(target_os = "linux")]
    const BLAS_LIBS: &[&str] = &["libcublas.so", "libcublas.so.12", "libcublas.so.11"];
    #[cfg(windows)]
    const BLAS_LIBS: &[&str] = &["cublas64_12.dll", "cublas64_11.dll"];
    #[cfg(not(any(target_os = "linux", windows)))]
    const BLAS_LIBS: &[&str] = &[];

    // cuBLAS SGEMM function pointer type (v2 API, stable since CUDA 4.1).
    type SgemmFn = unsafe extern "C" fn(
        *mut c_void, // handle
        i32,         // transa
        i32,         // transb
        i32,         // m
        i32,         // n
        i32,         // k
        *const f32,  // alpha
        *const c_void,
        i32, // A, lda
        *const c_void,
        i32,        // B, ldb
        *const f32, // beta
        *mut c_void,
        i32, // C, ldc
    ) -> i32;

    /// RAII guard that frees a CUDA device buffer on drop.
    struct DevBuf {
        ptr: *mut c_void,
        free_fn: CudaFreeFn,
    }

    impl Drop for DevBuf {
        fn drop(&mut self) {
            if !self.ptr.is_null() {
                unsafe { (self.free_fn)(self.ptr) };
            }
        }
    }

    /// RAII guard that destroys a cuBLAS handle on drop.
    struct BlasHandle {
        handle: *mut c_void,
        destroy_fn: unsafe extern "C" fn(*mut c_void) -> i32,
    }

    impl Drop for BlasHandle {
        fn drop(&mut self) {
            if !self.handle.is_null() {
                unsafe { (self.destroy_fn)(self.handle) };
            }
        }
    }

    fn try_open(names: &[&str]) -> Option<Library> {
        names
            .iter()
            .find_map(|name| unsafe { Library::new(name) }.ok())
    }

    // ── Public API ────────────────────────────────────────────────────────────

    /// Batch top-k vector search on an NVIDIA GPU via cuBLAS SGEMM.
    ///
    /// Computes the [Q×N] distance matrix in a single SGEMM call, then sorts
    /// top-k on CPU. Returns `None` when no CUDA device is found or on any
    /// GPU error — the caller must fall back to CPU.
    pub fn try_nvidia_search_batch(
        queries: &[&[f32]],
        row_ids: &[u64],
        flat_vecs: &[f32],
        dim: usize,
        metric: VectorMetric,
        top_k: usize,
    ) -> Option<Vec<Vec<(RowId, f32)>>> {
        if !crate::hardware::detect_cuda() {
            return None;
        }
        if RT_LIBS.is_empty() || BLAS_LIBS.is_empty() {
            return None;
        }
        let q = queries.len();
        if row_ids.is_empty() || q == 0 {
            return Some(vec![vec![]; q]);
        }
        let result = batch_inner(queries, row_ids, flat_vecs, dim, metric, top_k);
        if result.is_none() {
            warn!(
                "ailake: NVIDIA GPU search failed at runtime (cuBLAS error or allocation failure); \
                 falling back to CPU SIMD — check CUDA runtime libraries and available GPU memory"
            );
        }
        result
    }

    /// k-means on an NVIDIA GPU via cuBLAS SGEMM.
    ///
    /// Distance matrix (assignment step) computed on GPU.
    /// Centroid update runs on CPU. Returns `None` on any GPU error.
    pub fn try_nvidia_kmeans(
        vectors: &[Vec<f32>],
        k: usize,
        max_iter: usize,
    ) -> Option<Vec<Vec<f32>>> {
        if !crate::hardware::detect_cuda() {
            return None;
        }
        if RT_LIBS.is_empty() || BLAS_LIBS.is_empty() {
            return None;
        }
        if vectors.is_empty() {
            return Some(vec![]);
        }
        let n = vectors.len();
        let dim = vectors[0].len();
        let k = k.min(n);
        let result = kmeans_inner(vectors, k, max_iter, n, dim);
        if result.is_none() {
            warn!(
                "ailake: NVIDIA GPU k-means failed at runtime (cuBLAS error or allocation failure); \
                 falling back to CPU k-means — check CUDA runtime libraries and available GPU memory"
            );
        }
        result
    }

    // ── Internal helpers ──────────────────────────────────────────────────────

    unsafe fn load_cuda_fns(
        rt: &Library,
    ) -> Option<(CudaMallocFn, CudaFreeFn, CudaMemcpyFn, CudaSyncFn)> {
        let malloc_sym: Symbol<CudaMallocFn> = rt.get(b"cudaMalloc\0").ok()?;
        let free_sym: Symbol<CudaFreeFn> = rt.get(b"cudaFree\0").ok()?;
        let memcpy_sym: Symbol<CudaMemcpyFn> = rt.get(b"cudaMemcpy\0").ok()?;
        let sync_sym: Symbol<CudaSyncFn> = rt.get(b"cudaDeviceSynchronize\0").ok()?;
        Some((*malloc_sym, *free_sym, *memcpy_sym, *sync_sym))
    }

    unsafe fn upload(
        data: &[f32],
        malloc_fn: CudaMallocFn,
        free_fn: CudaFreeFn,
        memcpy_fn: CudaMemcpyFn,
    ) -> Option<DevBuf> {
        let bytes = std::mem::size_of_val(data);
        let mut ptr: *mut c_void = std::ptr::null_mut();
        if malloc_fn(&mut ptr, bytes) != 0 {
            return None;
        }
        let buf = DevBuf { ptr, free_fn };
        if memcpy_fn(ptr, data.as_ptr() as *const c_void, bytes, H2D) != 0 {
            return None;
        }
        Some(buf)
    }

    unsafe fn alloc_dev(
        len: usize,
        malloc_fn: CudaMallocFn,
        free_fn: CudaFreeFn,
    ) -> Option<DevBuf> {
        let bytes = len * std::mem::size_of::<f32>();
        let mut ptr: *mut c_void = std::ptr::null_mut();
        if malloc_fn(&mut ptr, bytes) != 0 {
            return None;
        }
        Some(DevBuf { ptr, free_fn })
    }

    fn normalize_rows(mut data: Vec<f32>, dim: usize) -> Vec<f32> {
        for row in data.chunks_mut(dim) {
            let norm: f32 = row.iter().map(|x| x * x).sum::<f32>().sqrt();
            if norm > 1e-8 {
                row.iter_mut().for_each(|x| *x /= norm);
            }
        }
        data
    }

    fn batch_inner(
        queries: &[&[f32]],
        row_ids: &[u64],
        flat_vecs: &[f32],
        dim: usize,
        metric: VectorMetric,
        top_k: usize,
    ) -> Option<Vec<Vec<(RowId, f32)>>> {
        let n = row_ids.len();
        let q = queries.len();

        let rt = try_open(RT_LIBS)?;
        let blas_lib = try_open(BLAS_LIBS)?;

        let (cuda_malloc, cuda_free, cuda_memcpy, cuda_sync) = unsafe { load_cuda_fns(&rt) }?;

        let blas_create: Symbol<unsafe extern "C" fn(*mut *mut c_void) -> i32> =
            unsafe { blas_lib.get(b"cublasCreate_v2\0") }.ok()?;
        let blas_destroy: unsafe extern "C" fn(*mut c_void) -> i32 = *unsafe {
            blas_lib.get::<unsafe extern "C" fn(*mut c_void) -> i32>(b"cublasDestroy_v2\0")
        }
        .ok()?;
        let sgemm: Symbol<SgemmFn> = unsafe { blas_lib.get(b"cublasSgemm_v2\0") }.ok()?;

        let mut raw_handle: *mut c_void = std::ptr::null_mut();
        if unsafe { blas_create(&mut raw_handle) } != 0 {
            return None;
        }
        let _blas = BlasHandle {
            handle: raw_handle,
            destroy_fn: blas_destroy,
        };

        let q_flat: Vec<f32>;
        let db_data: &[f32];
        let q_data: &[f32];
        let q_owned;
        let db_owned;

        match metric {
            VectorMetric::Cosine => {
                q_owned = normalize_rows(
                    queries.iter().flat_map(|q| q.iter().copied()).collect(),
                    dim,
                );
                db_owned = normalize_rows(flat_vecs.to_vec(), dim);
                q_data = &q_owned;
                db_data = &db_owned;
            }
            _ => {
                q_flat = queries.iter().flat_map(|q| q.iter().copied()).collect();
                q_data = &q_flat;
                db_data = flat_vecs;
            }
        }

        let db_dev = unsafe { upload(db_data, cuda_malloc, cuda_free, cuda_memcpy) }?;
        let q_dev = unsafe { upload(q_data, cuda_malloc, cuda_free, cuda_memcpy) }?;
        let c_dev = unsafe { alloc_dev(n * q, cuda_malloc, cuda_free) }?;

        // SGEMM: C[N×Q col-major] = alpha * db[N×dim]^T * queries[Q×dim]
        // C[n + q*N] = dot(db[n], query[q])
        let (alpha, beta) = match metric {
            VectorMetric::DotProduct => (-1.0f32, 0.0f32),
            VectorMetric::Cosine => (-1.0f32, 0.0f32),
            VectorMetric::Euclidean => (-2.0f32, 0.0f32),
        };

        let rc = unsafe {
            sgemm(
                raw_handle,
                OP_T,
                OP_N,
                n as i32,
                q as i32,
                dim as i32,
                &alpha,
                db_dev.ptr as *const c_void,
                dim as i32,
                q_dev.ptr as *const c_void,
                dim as i32,
                &beta,
                c_dev.ptr,
                n as i32,
            )
        };
        if rc != 0 {
            return None;
        }
        if unsafe { cuda_sync() } != 0 {
            return None;
        }

        let mut c_host = vec![0.0f32; n * q];
        if unsafe {
            cuda_memcpy(
                c_host.as_mut_ptr() as *mut c_void,
                c_dev.ptr as *const c_void,
                n * q * std::mem::size_of::<f32>(),
                D2H,
            )
        } != 0
        {
            return None;
        }

        let db_sq: Option<Vec<f32>> = if matches!(metric, VectorMetric::Euclidean) {
            Some(
                (0..n)
                    .map(|ni| {
                        flat_vecs[ni * dim..(ni + 1) * dim]
                            .iter()
                            .map(|x| x * x)
                            .sum()
                    })
                    .collect(),
            )
        } else {
            None
        };

        let results = (0..q)
            .map(|qi| {
                let dists: Vec<f32> = (0..n)
                    .map(|ni| {
                        let raw = c_host[ni + qi * n];
                        match metric {
                            VectorMetric::DotProduct => raw,
                            VectorMetric::Cosine => 1.0 + raw,
                            VectorMetric::Euclidean => {
                                let q_sq: f32 = queries[qi].iter().map(|x| x * x).sum();
                                (q_sq + db_sq.as_ref().unwrap()[ni] + raw).max(0.0).sqrt()
                            }
                        }
                    })
                    .collect();

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

    fn kmeans_inner(
        vectors: &[Vec<f32>],
        k: usize,
        max_iter: usize,
        n: usize,
        dim: usize,
    ) -> Option<Vec<Vec<f32>>> {
        let rt = try_open(RT_LIBS)?;
        let blas_lib = try_open(BLAS_LIBS)?;

        let (cuda_malloc, cuda_free, cuda_memcpy, cuda_sync) = unsafe { load_cuda_fns(&rt) }?;

        let blas_create: Symbol<unsafe extern "C" fn(*mut *mut c_void) -> i32> =
            unsafe { blas_lib.get(b"cublasCreate_v2\0") }.ok()?;
        let blas_destroy: unsafe extern "C" fn(*mut c_void) -> i32 = *unsafe {
            blas_lib.get::<unsafe extern "C" fn(*mut c_void) -> i32>(b"cublasDestroy_v2\0")
        }
        .ok()?;
        let sgemm: Symbol<SgemmFn> = unsafe { blas_lib.get(b"cublasSgemm_v2\0") }.ok()?;

        let mut raw_handle: *mut c_void = std::ptr::null_mut();
        if unsafe { blas_create(&mut raw_handle) } != 0 {
            return None;
        }
        let _blas = BlasHandle {
            handle: raw_handle,
            destroy_fn: blas_destroy,
        };

        let flat: Vec<f32> = vectors.iter().flat_map(|v| v.iter().copied()).collect();
        let x_dev = unsafe { upload(&flat, cuda_malloc, cuda_free, cuda_memcpy) }?;

        let x_sq: Vec<f32> = vectors
            .iter()
            .map(|v| v.iter().map(|x| x * x).sum())
            .collect();

        let step = n / k;
        let mut centroids_flat: Vec<f32> = (0..k)
            .flat_map(|i| vectors[(i * step) % n].iter().copied())
            .collect();

        let mut prev_asgn: Vec<u32> = vec![];

        for _ in 0..max_iter {
            let c_dev = unsafe { upload(&centroids_flat, cuda_malloc, cuda_free, cuda_memcpy) }?;
            let cross_dev = unsafe { alloc_dev(k * n, cuda_malloc, cuda_free) }?;

            // SGEMM: cross[K×N col-major] = -2 * centroids[K×dim] * vectors[N×dim]^T
            let alpha = -2.0f32;
            let beta = 0.0f32;
            let rc = unsafe {
                sgemm(
                    raw_handle,
                    OP_T,
                    OP_N,
                    k as i32,
                    n as i32,
                    dim as i32,
                    &alpha,
                    c_dev.ptr as *const c_void,
                    dim as i32,
                    x_dev.ptr as *const c_void,
                    dim as i32,
                    &beta,
                    cross_dev.ptr,
                    k as i32,
                )
            };
            if rc != 0 {
                return None;
            }
            if unsafe { cuda_sync() } != 0 {
                return None;
            }

            let mut cross_host = vec![0.0f32; k * n];
            if unsafe {
                cuda_memcpy(
                    cross_host.as_mut_ptr() as *mut c_void,
                    cross_dev.ptr as *const c_void,
                    k * n * std::mem::size_of::<f32>(),
                    D2H,
                )
            } != 0
            {
                return None;
            }

            let c_sq: Vec<f32> = centroids_flat
                .chunks(dim)
                .map(|c| c.iter().map(|x| x * x).sum())
                .collect();

            let asgn: Vec<u32> = (0..n)
                .map(|ni| {
                    let base = &cross_host[ni * k..(ni + 1) * k];
                    let best = (0..k)
                        .min_by(|&a, &b| {
                            let da = x_sq[ni] + c_sq[a] + base[a];
                            let db = x_sq[ni] + c_sq[b] + base[b];
                            da.partial_cmp(&db).unwrap_or(std::cmp::Ordering::Equal)
                        })
                        .unwrap_or(0);
                    best as u32
                })
                .collect();

            if asgn == prev_asgn {
                break;
            }

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
                    new_flat[j * dim..(j + 1) * dim]
                        .iter_mut()
                        .for_each(|x| *x *= inv);
                } else {
                    new_flat[j * dim..(j + 1) * dim]
                        .copy_from_slice(&centroids_flat[j * dim..(j + 1) * dim]);
                }
            }

            centroids_flat = new_flat;
            prev_asgn = asgn;
        }

        Some(centroids_flat.chunks(dim).map(|c| c.to_vec()).collect())
    }
}

// ── AMD ROCm backend ─────────────────────────────────────────────────────────
//
// Always compiled (no feature gate). Returns `None` at runtime when:
//   - No AMD HIP driver found (`detect_rocm()` is false)
//   - hipBLAS library not installed
//   - Any GPU allocation or compute error
//
// Distance matrix computed via hipBLAS SGEMM. Norm computation and argmin
// run on CPU (O((n+k)·dim) vs O(n·k·dim) for SGEMM — negligible overhead).

mod rocm_impl {
    use std::ffi::c_void;

    use ailake_core::{RowId, VectorMetric};
    use libloading::{Library, Symbol};
    use tracing::warn;

    // hipMemcpyKind constants
    const H2D: i32 = 1; // hipMemcpyHostToDevice
    const D2H: i32 = 2; // hipMemcpyDeviceToHost

    // hipblasOperation_t constants (same values as cuBLAS)
    const OP_T: i32 = 112; // HIPBLAS_OP_T — transpose
    const OP_N: i32 = 111; // HIPBLAS_OP_N — no-transpose

    // Type aliases for HIP runtime function pointers (avoids clippy::type_complexity).
    type HipMallocFn = unsafe extern "C" fn(*mut *mut c_void, usize) -> i32;
    type HipFreeFn = unsafe extern "C" fn(*mut c_void) -> i32;
    type HipMemcpyFn = unsafe extern "C" fn(*mut c_void, *const c_void, usize, i32) -> i32;
    type HipSyncFn = unsafe extern "C" fn() -> i32;

    #[cfg(target_os = "linux")]
    const HIP_LIB: &str = "libamdhip64.so";
    #[cfg(windows)]
    const HIP_LIB: &str = "amdhip64.dll";
    #[cfg(not(any(target_os = "linux", windows)))]
    const HIP_LIB: &str = "";

    #[cfg(target_os = "linux")]
    const BLAS_LIB: &str = "libhipblas.so";
    #[cfg(windows)]
    const BLAS_LIB: &str = "hipblas.dll";
    #[cfg(not(any(target_os = "linux", windows)))]
    const BLAS_LIB: &str = "";

    // hipBLAS SGEMM function pointer type.
    type SgemmFn = unsafe extern "C" fn(
        *mut c_void, // handle
        i32,         // transa
        i32,         // transb
        i32,         // m
        i32,         // n
        i32,         // k
        *const f32,  // alpha
        *const c_void,
        i32, // A, lda
        *const c_void,
        i32,        // B, ldb
        *const f32, // beta
        *mut c_void,
        i32, // C, ldc
    ) -> i32;

    /// RAII guard that frees a HIP device buffer on drop.
    struct DevBuf {
        ptr: *mut c_void,
        free_fn: unsafe extern "C" fn(*mut c_void) -> i32,
    }

    impl Drop for DevBuf {
        fn drop(&mut self) {
            if !self.ptr.is_null() {
                unsafe { (self.free_fn)(self.ptr) };
            }
        }
    }

    /// RAII guard that destroys a hipBLAS handle on drop.
    struct BlasHandle {
        handle: *mut c_void,
        destroy_fn: unsafe extern "C" fn(*mut c_void) -> i32,
    }

    impl Drop for BlasHandle {
        fn drop(&mut self) {
            if !self.handle.is_null() {
                unsafe { (self.destroy_fn)(self.handle) };
            }
        }
    }

    // ── Public API ────────────────────────────────────────────────────────────

    /// Batch top-k vector search on an AMD ROCm GPU via hipBLAS SGEMM.
    ///
    /// Computes the [Q×N] distance matrix in a single SGEMM call, then sorts
    /// top-k on the CPU. Returns `None` when no ROCm device is found or on any
    /// GPU error — the caller must fall back to CPU.
    pub fn try_rocm_search_batch(
        queries: &[&[f32]],
        row_ids: &[u64],
        flat_vecs: &[f32],
        dim: usize,
        metric: VectorMetric,
        top_k: usize,
    ) -> Option<Vec<Vec<(RowId, f32)>>> {
        if !crate::hardware::detect_rocm() {
            return None;
        }
        if HIP_LIB.is_empty() || BLAS_LIB.is_empty() {
            return None;
        }
        let n = row_ids.len();
        let q = queries.len();
        if n == 0 || q == 0 {
            return Some(vec![vec![]; q]);
        }
        let result = batch_inner(queries, row_ids, flat_vecs, dim, metric, top_k);
        if result.is_none() {
            warn!(
                "ailake: AMD ROCm GPU search failed at runtime (hipBLAS error or allocation failure); \
                 falling back to CPU SIMD — check ROCm runtime libraries and available GPU memory"
            );
        }
        result
    }

    /// k-means on an AMD ROCm GPU.
    ///
    /// Distance matrix (assignment step) computed via hipBLAS SGEMM.
    /// Centroid update runs on CPU. Returns `None` on any GPU error.
    pub fn try_rocm_kmeans(
        vectors: &[Vec<f32>],
        k: usize,
        max_iter: usize,
    ) -> Option<Vec<Vec<f32>>> {
        if !crate::hardware::detect_rocm() {
            return None;
        }
        if HIP_LIB.is_empty() || BLAS_LIB.is_empty() {
            return None;
        }
        let n = vectors.len();
        if n == 0 {
            return Some(vec![]);
        }
        let dim = vectors[0].len();
        let k = k.min(n);
        let result = kmeans_inner(vectors, k, max_iter, n, dim);
        if result.is_none() {
            warn!(
                "ailake: AMD ROCm GPU k-means failed at runtime (hipBLAS error or allocation failure); \
                 falling back to CPU k-means — check ROCm runtime libraries and available GPU memory"
            );
        }
        result
    }

    // ── Internal helpers ──────────────────────────────────────────────────────

    /// Load HIP memory functions from `libamdhip64`.
    ///
    /// Returns: (malloc_fn, free_fn, memcpy_fn, sync_fn)
    unsafe fn load_hip_fns(
        lib: &Library,
    ) -> Option<(HipMallocFn, HipFreeFn, HipMemcpyFn, HipSyncFn)> {
        let malloc_sym: Symbol<HipMallocFn> = lib.get(b"hipMalloc\0").ok()?;
        let free_sym: Symbol<HipFreeFn> = lib.get(b"hipFree\0").ok()?;
        let memcpy_sym: Symbol<HipMemcpyFn> = lib.get(b"hipMemcpy\0").ok()?;
        let sync_sym: Symbol<HipSyncFn> = lib.get(b"hipDeviceSynchronize\0").ok()?;
        Some((*malloc_sym, *free_sym, *memcpy_sym, *sync_sym))
    }

    /// Allocate a device buffer and upload host data.
    unsafe fn upload(
        data: &[f32],
        malloc_fn: HipMallocFn,
        free_fn: HipFreeFn,
        memcpy_fn: HipMemcpyFn,
    ) -> Option<DevBuf> {
        let bytes = std::mem::size_of_val(data);
        let mut ptr: *mut c_void = std::ptr::null_mut();
        if malloc_fn(&mut ptr, bytes) != 0 {
            return None;
        }
        let buf = DevBuf { ptr, free_fn };
        if memcpy_fn(ptr, data.as_ptr() as *const c_void, bytes, H2D) != 0 {
            return None;
        }
        Some(buf)
    }

    /// Allocate an uninitialised device buffer.
    unsafe fn alloc_dev(len: usize, malloc_fn: HipMallocFn, free_fn: HipFreeFn) -> Option<DevBuf> {
        let bytes = len * std::mem::size_of::<f32>();
        let mut ptr: *mut c_void = std::ptr::null_mut();
        if malloc_fn(&mut ptr, bytes) != 0 {
            return None;
        }
        Some(DevBuf { ptr, free_fn })
    }

    /// Per-row L2 normalisation (CPU, in-place on owned Vec).
    fn normalize_rows(mut data: Vec<f32>, dim: usize) -> Vec<f32> {
        for row in data.chunks_mut(dim) {
            let norm: f32 = row.iter().map(|x| x * x).sum::<f32>().sqrt();
            if norm > 1e-8 {
                row.iter_mut().for_each(|x| *x /= norm);
            }
        }
        data
    }

    fn batch_inner(
        queries: &[&[f32]],
        row_ids: &[u64],
        flat_vecs: &[f32],
        dim: usize,
        metric: VectorMetric,
        top_k: usize,
    ) -> Option<Vec<Vec<(RowId, f32)>>> {
        let n = row_ids.len();
        let q = queries.len();
        // Load libraries
        let hip = unsafe { Library::new(HIP_LIB) }.ok()?;
        let blas_lib = unsafe { Library::new(BLAS_LIB) }.ok()?;

        let (hip_malloc, hip_free, hip_memcpy, hip_sync) = unsafe { load_hip_fns(&hip) }?;

        let blas_create: Symbol<unsafe extern "C" fn(*mut *mut c_void) -> i32> =
            unsafe { blas_lib.get(b"hipblasCreate\0") }.ok()?;
        let blas_destroy: unsafe extern "C" fn(*mut c_void) -> i32 = *unsafe {
            blas_lib.get::<unsafe extern "C" fn(*mut c_void) -> i32>(b"hipblasDestroy\0")
        }
        .ok()?;
        let sgemm: Symbol<SgemmFn> = unsafe { blas_lib.get(b"hipblasSgemm\0") }.ok()?;

        // Create hipBLAS handle
        let mut raw_handle: *mut c_void = std::ptr::null_mut();
        if unsafe { blas_create(&mut raw_handle) } != 0 {
            return None;
        }
        let _blas = BlasHandle {
            handle: raw_handle,
            destroy_fn: blas_destroy,
        };

        // Optionally normalise before computing dot products (Cosine metric)
        let q_flat: Vec<f32>;
        let db_data: &[f32];
        let q_data: &[f32];
        let q_owned;
        let db_owned;

        match metric {
            VectorMetric::Cosine => {
                q_owned = normalize_rows(
                    queries.iter().flat_map(|q| q.iter().copied()).collect(),
                    dim,
                );
                db_owned = normalize_rows(flat_vecs.to_vec(), dim);
                q_data = &q_owned;
                db_data = &db_owned;
            }
            _ => {
                q_flat = queries.iter().flat_map(|q| q.iter().copied()).collect();
                q_data = &q_flat;
                db_data = flat_vecs;
            }
        }

        // Upload matrices
        let db_dev = unsafe { upload(db_data, hip_malloc, hip_free, hip_memcpy) }?;
        let q_dev = unsafe { upload(q_data, hip_malloc, hip_free, hip_memcpy) }?;
        let c_dev = unsafe { alloc_dev(n * q, hip_malloc, hip_free) }?;

        // SGEMM: C[N×Q col-major] = db[N×dim row-major] * queries[Q×dim row-major]ᵀ
        //
        // BLAS col-major call: C = op(A) * op(B)
        //   op(A) = db^T  (OP_T, col-major dim×N → transpose to N×dim), lda=dim
        //   op(B) = queries (OP_N, col-major dim×Q), ldb=dim
        //   m=N, n=Q, k=dim → C is N×Q col-major, ldc=N
        //
        // Result: C[n + q*N] = dot(db[n], query[q])
        let (alpha, beta) = match metric {
            VectorMetric::DotProduct => (-1.0f32, 0.0f32), // negate → min-distance semantics
            VectorMetric::Cosine => (-1.0f32, 0.0f32),     // 1 − cos added below
            VectorMetric::Euclidean => (-2.0f32, 0.0f32),  // −2·q·dᵀ; norms added below
        };

        let rc = unsafe {
            sgemm(
                raw_handle,
                OP_T,
                OP_N,
                n as i32,
                q as i32,
                dim as i32,
                &alpha,
                db_dev.ptr as *const c_void,
                dim as i32,
                q_dev.ptr as *const c_void,
                dim as i32,
                &beta,
                c_dev.ptr,
                n as i32,
            )
        };
        if rc != 0 {
            return None;
        }
        if unsafe { hip_sync() } != 0 {
            return None;
        }

        // Copy result back: c_host[n + q*N] = dot(db[n], query[q])
        let mut c_host = vec![0.0f32; n * q];
        if unsafe {
            hip_memcpy(
                c_host.as_mut_ptr() as *mut c_void,
                c_dev.ptr as *const c_void,
                n * q * std::mem::size_of::<f32>(),
                D2H,
            )
        } != 0
        {
            return None;
        }

        // Pre-compute per-vector norms for Euclidean distance
        let db_sq: Option<Vec<f32>> = if matches!(metric, VectorMetric::Euclidean) {
            Some(
                (0..n)
                    .map(|ni| {
                        flat_vecs[ni * dim..(ni + 1) * dim]
                            .iter()
                            .map(|x| x * x)
                            .sum()
                    })
                    .collect(),
            )
        } else {
            None
        };

        let results = (0..q)
            .map(|qi| {
                let dists: Vec<f32> = (0..n)
                    .map(|ni| {
                        let raw = c_host[ni + qi * n];
                        match metric {
                            // raw = −dot → already min-distance order
                            VectorMetric::DotProduct => raw,
                            // raw = −cos_sim → 1 − cos_sim = 1 + raw
                            VectorMetric::Cosine => 1.0 + raw,
                            // raw = −2·q·d → add ||q||² + ||d||², clamp, sqrt
                            VectorMetric::Euclidean => {
                                let q_sq: f32 = queries[qi].iter().map(|x| x * x).sum();
                                (q_sq + db_sq.as_ref().unwrap()[ni] + raw).max(0.0).sqrt()
                            }
                        }
                    })
                    .collect();

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

    fn kmeans_inner(
        vectors: &[Vec<f32>],
        k: usize,
        max_iter: usize,
        n: usize,
        dim: usize,
    ) -> Option<Vec<Vec<f32>>> {
        let hip = unsafe { Library::new(HIP_LIB) }.ok()?;
        let blas_lib = unsafe { Library::new(BLAS_LIB) }.ok()?;

        let (hip_malloc, hip_free, hip_memcpy, hip_sync) = unsafe { load_hip_fns(&hip) }?;

        let blas_create: Symbol<unsafe extern "C" fn(*mut *mut c_void) -> i32> =
            unsafe { blas_lib.get(b"hipblasCreate\0") }.ok()?;
        let blas_destroy: unsafe extern "C" fn(*mut c_void) -> i32 = *unsafe {
            blas_lib.get::<unsafe extern "C" fn(*mut c_void) -> i32>(b"hipblasDestroy\0")
        }
        .ok()?;
        let sgemm: Symbol<SgemmFn> = unsafe { blas_lib.get(b"hipblasSgemm\0") }.ok()?;

        let mut raw_handle: *mut c_void = std::ptr::null_mut();
        if unsafe { blas_create(&mut raw_handle) } != 0 {
            return None;
        }
        let _blas = BlasHandle {
            handle: raw_handle,
            destroy_fn: blas_destroy,
        };

        // Upload all vectors once (never changes across iterations)
        let flat: Vec<f32> = vectors.iter().flat_map(|v| v.iter().copied()).collect();
        let x_dev = unsafe { upload(&flat, hip_malloc, hip_free, hip_memcpy) }?;

        // Per-vector squared norms (constant — computed once on CPU)
        let x_sq: Vec<f32> = vectors
            .iter()
            .map(|v| v.iter().map(|x| x * x).sum())
            .collect();

        // Initialise centroids via evenly-spaced sampling (deterministic)
        let step = n / k;
        let mut centroids_flat: Vec<f32> = (0..k)
            .flat_map(|i| vectors[(i * step) % n].iter().copied())
            .collect();

        let mut prev_asgn: Vec<u32> = vec![];

        for _ in 0..max_iter {
            // Upload current centroids
            let c_dev = unsafe { upload(&centroids_flat, hip_malloc, hip_free, hip_memcpy) }?;

            // Result buffer for cross = −2 * centroids * vectors^T, shape [K×N col-major]
            // c_cross[n*K .. (n+1)*K] gives the partial distances for vector n
            let cross_dev = unsafe { alloc_dev(k * n, hip_malloc, hip_free) }?;

            // SGEMM: cross[K×N col-major] = −2 * centroids[K×dim] * vectors[N×dim]^T
            //   op(A) = centroids^T (OP_T, col-major dim×K → K×dim), lda=dim
            //   op(B) = vectors (OP_N, col-major dim×N), ldb=dim
            //   m=K, n=N, k=dim → result K×N col-major, ldc=K
            let alpha = -2.0f32;
            let beta = 0.0f32;
            let rc = unsafe {
                sgemm(
                    raw_handle,
                    OP_T,
                    OP_N,
                    k as i32,
                    n as i32,
                    dim as i32,
                    &alpha,
                    c_dev.ptr as *const c_void,
                    dim as i32,
                    x_dev.ptr as *const c_void,
                    dim as i32,
                    &beta,
                    cross_dev.ptr,
                    k as i32,
                )
            };
            if rc != 0 {
                return None;
            }
            if unsafe { hip_sync() } != 0 {
                return None;
            }

            let mut cross_host = vec![0.0f32; k * n];
            if unsafe {
                hip_memcpy(
                    cross_host.as_mut_ptr() as *mut c_void,
                    cross_dev.ptr as *const c_void,
                    k * n * std::mem::size_of::<f32>(),
                    D2H,
                )
            } != 0
            {
                return None;
            }

            // Per-centroid squared norms (CPU, k*dim work)
            let c_sq: Vec<f32> = centroids_flat
                .chunks(dim)
                .map(|c| c.iter().map(|x| x * x).sum())
                .collect();

            // Argmin assignment: dists[n][ci] = x_sq[n] + c_sq[ci] + cross[n*K + ci]
            let asgn: Vec<u32> = (0..n)
                .map(|ni| {
                    let base = &cross_host[ni * k..(ni + 1) * k];
                    let best = (0..k)
                        .min_by(|&a, &b| {
                            let da = x_sq[ni] + c_sq[a] + base[a];
                            let db = x_sq[ni] + c_sq[b] + base[b];
                            da.partial_cmp(&db).unwrap_or(std::cmp::Ordering::Equal)
                        })
                        .unwrap_or(0);
                    best as u32
                })
                .collect();

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
                    new_flat[j * dim..(j + 1) * dim]
                        .iter_mut()
                        .for_each(|x| *x *= inv);
                } else {
                    // Empty cluster: keep previous centroid
                    new_flat[j * dim..(j + 1) * dim]
                        .copy_from_slice(&centroids_flat[j * dim..(j + 1) * dim]);
                }
            }

            centroids_flat = new_flat;
            prev_asgn = asgn;
        }

        Some(centroids_flat.chunks(dim).map(|c| c.to_vec()).collect())
    }
}
