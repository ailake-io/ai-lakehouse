// SPDX-License-Identifier: MIT OR Apache-2.0
// Hardware capability detection — mirrors ailake_index::hardware (Rust).
//
// Detection priority: AMD ROCm → NVIDIA CUDA → CPU SIMD.
// AMD checked first: ROCm installations often expose libcuda.so.1,
// so checking ROCm first avoids misidentifying AMD GPUs as NVIDIA.
//
// Runtime probes via dlopen (Linux/macOS) or LoadLibraryA (Windows).
// No GPU toolkit required at compile time — driver libraries only.
#pragma once

#include <cstddef>
#include <cstdint>

#if defined(__linux__) || defined(__APPLE__)
#  include <dlfcn.h>
#elif defined(_WIN32)
#  define WIN32_LEAN_AND_MEAN
#  include <windows.h>
#endif

#if defined(__x86_64__) || defined(_M_X64)
#  include <cpuid.h>
#endif

#include <mutex>
#include <optional>
#include <thread>

namespace ailake {

// ---------------------------------------------------------------------------
// Constants — match Rust implementation
// ---------------------------------------------------------------------------

static constexpr size_t kMinVectorsForIvfPq = 5'000;
static constexpr size_t kMinCoresForIvfPq   = 8;

// ---------------------------------------------------------------------------
// Backend enum
// ---------------------------------------------------------------------------

enum class Backend {
    CpuSimd,    ///< No GPU — use AVX2/AVX-512 CPU kernels
    NvidiaCuda, ///< NVIDIA CUDA driver present and initialized
    AmdRocm,    ///< AMD ROCm/HIP driver present and initialized
};

inline const char* backend_name(Backend b) {
    switch (b) {
        case Backend::NvidiaCuda: return "nvidia-cuda";
        case Backend::AmdRocm:   return "amd-rocm";
        default:                 return "cpu-simd";
    }
}

// ---------------------------------------------------------------------------
// HardwareProfile
// ---------------------------------------------------------------------------

struct HardwareProfile {
    Backend backend         = Backend::CpuSimd;
    bool    has_cuda        = false;
    bool    has_rocm        = false;
    size_t  cpu_cores       = 1;
    bool    has_avx2        = false;
    bool    has_avx512      = false;

    bool has_gpu() const noexcept { return has_cuda || has_rocm; }

    /// True when IVF-PQ training is justified for n_vectors vectors.
    bool recommend_ivf_pq(size_t n_vectors) const noexcept {
        if (n_vectors < kMinVectorsForIvfPq) return false;
        return has_cuda || has_rocm || cpu_cores > kMinCoresForIvfPq;
    }
};

// ---------------------------------------------------------------------------
// Runtime CPUID detection (x86_64 only)
// ---------------------------------------------------------------------------

namespace detail {

inline void detect_simd(bool& avx2, bool& avx512) {
    avx2 = false; avx512 = false;
#if defined(__x86_64__) || defined(_M_X64)
    uint32_t eax = 0, ebx = 0, ecx = 0, edx = 0;
    // Leaf 0 — max leaf
    __get_cpuid(0, &eax, &ebx, &ecx, &edx);
    if (eax < 7) return;
    // Leaf 7, sub-leaf 0
    __get_cpuid_count(7, 0, &eax, &ebx, &ecx, &edx);
    avx2   = (ebx >> 5)  & 1;
    avx512 = (ebx >> 16) & 1; // AVX-512F
#endif
}

// ---------------------------------------------------------------------------
// Dynamic library probe helpers
// ---------------------------------------------------------------------------

#if defined(__linux__)
    static constexpr const char* kCudaLib  = "libcuda.so.1";
    static constexpr const char* kRocmLib  = "libamdhip64.so";
#elif defined(_WIN32)
    static constexpr const char* kCudaLib  = "nvcuda.dll";
    static constexpr const char* kRocmLib  = "amdhip64.dll";
#elif defined(__APPLE__)
    static constexpr const char* kCudaLib  = ""; // No CUDA on modern macOS
    static constexpr const char* kRocmLib  = ""; // No ROCm on macOS
#else
    static constexpr const char* kCudaLib  = "";
    static constexpr const char* kRocmLib  = "";
#endif

inline void* open_lib(const char* name) {
    if (!name || name[0] == '\0') return nullptr;
#if defined(__linux__) || defined(__APPLE__)
    return dlopen(name, RTLD_LAZY | RTLD_LOCAL);
#elif defined(_WIN32)
    return (void*)LoadLibraryA(name);
#else
    return nullptr;
#endif
}

inline void* get_sym(void* lib, const char* sym) {
#if defined(__linux__) || defined(__APPLE__)
    return dlsym(lib, sym);
#elif defined(_WIN32)
    return (void*)GetProcAddress((HMODULE)lib, sym);
#else
    return nullptr;
#endif
}

inline void close_lib(void* lib) {
    if (!lib) return;
#if defined(__linux__) || defined(__APPLE__)
    dlclose(lib);
#elif defined(_WIN32)
    FreeLibrary((HMODULE)lib);
#endif
}

// CUDA driver API types
using CuResult = int;
using FnCuInit          = CuResult(*)(unsigned int);
using FnCuDeviceGetCount= CuResult(*)(int*);

// HIP driver API types
using HipError = int;
using FnHipInit          = HipError(*)(unsigned int);
using FnHipDeviceGetCount= HipError(*)(int*);

inline bool probe_cuda() {
    void* lib = open_lib(kCudaLib);
    if (!lib) return false;
    bool ok = false;
    auto cuInit = (FnCuInit)get_sym(lib, "cuInit");
    if (cuInit && cuInit(0) == 0) {
        auto cuCount = (FnCuDeviceGetCount)get_sym(lib, "cuDeviceGetCount");
        if (cuCount) {
            int count = 0;
            ok = cuCount(&count) == 0 && count > 0;
        }
    }
    close_lib(lib);
    return ok;
}

inline bool probe_rocm() {
    void* lib = open_lib(kRocmLib);
    if (!lib) return false;
    bool ok = false;
    auto hipInit = (FnHipInit)get_sym(lib, "hipInit");
    if (hipInit && hipInit(0) == 0) {
        auto hipCount = (FnHipDeviceGetCount)get_sym(lib, "hipGetDeviceCount");
        if (hipCount) {
            int count = 0;
            ok = hipCount(&count) == 0 && count > 0;
        }
    }
    close_lib(lib);
    return ok;
}

} // namespace detail

// ---------------------------------------------------------------------------
// detect_hardware() — probed once per process via std::once_flag
// ---------------------------------------------------------------------------

inline const HardwareProfile& detect_hardware() {
    static HardwareProfile profile;
    static std::once_flag  flag;

    std::call_once(flag, []() {
        // Priority: ROCm > CUDA > CPU (AMD may expose libcuda.so.1)
        if (detail::probe_rocm()) {
            profile.backend  = Backend::AmdRocm;
            profile.has_rocm = true;
        } else if (detail::probe_cuda()) {
            profile.backend  = Backend::NvidiaCuda;
            profile.has_cuda = true;
        }
        profile.cpu_cores = std::thread::hardware_concurrency();
        detail::detect_simd(profile.has_avx2, profile.has_avx512);
    });

    return profile;
}

} // namespace ailake
