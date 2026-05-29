// SPDX-License-Identifier: MIT OR Apache-2.0
// Hardware capability detection — mirrors ailake_index::hardware (Rust).
//
// Detection priority: AMD ROCm → NVIDIA CUDA → CPU.
// AMD checked first: ROCm installations often expose libcuda.so.1,
// which would be misidentified as NVIDIA without the priority check.
//
// No cgo, no external deps — uses filesystem probes and syscall.
package ailake

import (
	"os"
	"runtime"
	"sync"
)

// Backend is the active compute backend.
type Backend int

const (
	BackendCPU        Backend = iota
	BackendNvidiaCUDA         // NVIDIA CUDA driver present and initialized
	BackendAMDROCm            // AMD ROCm/HIP driver present and initialized
)

func (b Backend) String() string {
	switch b {
	case BackendNvidiaCUDA:
		return "nvidia-cuda"
	case BackendAMDROCm:
		return "amd-rocm"
	default:
		return "cpu"
	}
}

// HardwareProfile mirrors ailake_index::HardwareProfile.
type HardwareProfile struct {
	Backend     Backend
	HasCUDA     bool
	HasROCm     bool
	CPUCores    int
	HasAVX2     bool // x86_64 only
	HasAVX512   bool // x86_64 only
}

// MinVectorsForIvfPq is the minimum dataset size that justifies IVF-PQ training.
const MinVectorsForIvfPq = 5_000

// MinCoresForIvfPq is the minimum CPU core count (exclusive) for IVF-PQ without GPU.
const MinCoresForIvfPq = 8

// RecommendIvfPq returns true when IVF-PQ is preferable to HNSW for n vectors.
func (h *HardwareProfile) RecommendIvfPq(nVectors int) bool {
	if nVectors < MinVectorsForIvfPq {
		return false
	}
	return h.HasCUDA || h.HasROCm || h.CPUCores > MinCoresForIvfPq
}

// HasGPU reports whether any GPU backend is available.
func (h *HardwareProfile) HasGPU() bool {
	return h.HasCUDA || h.HasROCm
}

// ---------------------------------------------------------------------------
// Detection — probed once per process via sync.Once
// ---------------------------------------------------------------------------

var (
	profileOnce sync.Once
	globalProfile HardwareProfile
)

// DetectHardware probes hardware capabilities once and caches the result.
// Thread-safe; subsequent calls return the cached profile.
func DetectHardware() *HardwareProfile {
	profileOnce.Do(func() {
		globalProfile = probeHardware()
	})
	return &globalProfile
}

func probeHardware() HardwareProfile {
	p := HardwareProfile{
		CPUCores: runtime.NumCPU(),
	}

	// Priority: ROCm > CUDA > CPU
	// Check ROCm first (ROCm may expose libcuda.so.1 on some installs).
	if probeROCm() {
		p.Backend = BackendAMDROCm
		p.HasROCm = true
	} else if probeCUDA() {
		p.Backend = BackendNvidiaCUDA
		p.HasCUDA = true
	} else {
		p.Backend = BackendCPU
	}

	// SIMD detection (x86_64 only via cpuid; arm/other → false)
	p.HasAVX2, p.HasAVX512 = detectSIMD()
	return p
}

// ---------------------------------------------------------------------------
// CUDA probe — filesystem heuristics (no cgo, no dlopen)
//
// Probes: /dev/nvidia0 (device node), /proc/driver/nvidia/version (Linux),
//         /dev/dxg (Windows WSL2/native CUDA).
// Same signals used by nvidia-smi and CUDA runtime.
// ---------------------------------------------------------------------------

func probeCUDA() bool {
	switch runtime.GOOS {
	case "linux":
		// /dev/nvidia0 exists when NVIDIA kernel module is loaded and at least
		// one GPU is present. Does NOT require CUDA toolkit — only the driver.
		if _, err := os.Stat("/dev/nvidia0"); err == nil {
			return true
		}
		// Secondary: /proc/driver/nvidia/version (older kernels)
		if _, err := os.Stat("/proc/driver/nvidia/version"); err == nil {
			return true
		}
	case "windows":
		// NVIDIA Management Library (NVML) installs nvcuda.dll; check via
		// environment path. WSL2 exposes /dev/dxg.
		if _, err := os.Stat("/dev/dxg"); err == nil {
			return true
		}
	case "darwin":
		// Apple Silicon GPUs use Metal, not CUDA. Intel Macs had eGPU CUDA
		// support via NVIDIA web drivers (discontinued). No CUDA on modern macOS.
		return false
	}
	return false
}

// ---------------------------------------------------------------------------
// ROCm probe — filesystem heuristics
//
// /dev/kfd    — Kernel Fusion Driver, required for ROCm HIP runtime
// /dev/dri/renderD128 — DRM render node (present for all GPU including NVIDIA)
//                        so we check kfd first
// ---------------------------------------------------------------------------

func probeROCm() bool {
	if runtime.GOOS != "linux" {
		return false // ROCm is Linux-only
	}
	// /dev/kfd is the ROCm kernel driver device node. Not present on NVIDIA-only.
	if _, err := os.Stat("/dev/kfd"); err == nil {
		// Verify at least one AMD GPU render node exists
		for _, path := range []string{
			"/sys/module/amdgpu",
			"/sys/class/kfd/kfd",
		} {
			if _, err := os.Stat(path); err == nil {
				return true
			}
		}
		// /dev/kfd alone is strong enough signal
		return true
	}
	return false
}

// ---------------------------------------------------------------------------
// SIMD detection — x86_64 CPUID
// Other architectures: always false (NEON/SVE not auto-detected yet)
// ---------------------------------------------------------------------------

func detectSIMD() (hasAVX2, hasAVX512 bool) {
	if runtime.GOARCH != "amd64" {
		return false, false
	}
	// CPUID leaf 7, sub-leaf 0: EBX bit 5 = AVX2, bit 16 = AVX-512F
	// We call the assembly stub defined in simd_amd64.s.
	// If the stub is not compiled (non-amd64), the linker simply omits it.
	ebx := cpuidLeaf7EBX()
	hasAVX2   = (ebx>>5)&1 == 1
	hasAVX512 = (ebx>>16)&1 == 1
	return
}

// cpuidLeaf7EBX is implemented in simd_amd64.s for amd64 builds.
// For all other architectures it is a stub returning 0.
func cpuidLeaf7EBX() uint32
