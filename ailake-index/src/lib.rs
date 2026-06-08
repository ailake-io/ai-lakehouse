// SPDX-License-Identifier: MIT OR Apache-2.0
//! ailake-index — HNSW and IVF-PQ index lifecycle
//!
//! Search backend priority (all detected at runtime, no build-time GPU SDK required):
//!   1. NVIDIA CUDA — cuBLAS SGEMM via libloading dlopen of libcudart + libcublas
//!   2. AMD ROCm    — hipBLAS SGEMM via libloading dlopen of libamdhip64 + libhipblas
//!   3. CPU rayon   — parallel brute-force, always available

pub mod binary;
pub mod gpu;
pub mod hardware;
pub mod hnsw;
pub mod ivf_pq;
pub mod mmap_loader;
pub mod rabitq;
pub mod serialize;

pub use binary::{BinaryConfig, BinaryIndex, BinarySerializer};
pub use hardware::{detect_backend, detect_cuda, detect_rocm, HardwareBackend, HardwareProfile};
pub use hnsw::{HnswBuilder, HnswConfig, HnswIndex};
pub use ivf_pq::{find_valid_pq_m, IvfPqCodebook, IvfPqConfig, IvfPqIndex, IvfPqSerializer};
pub use mmap_loader::MmapLoader;
pub use rabitq::{RaBitQConfig, RaBitQIndex, RaBitQSerializer};
pub use serialize::HnswSerializer;

use ailake_core::RowId;

/// Unified index type: dispatches search to HNSW, IVF-PQ, RaBitQ, or Binary.
pub enum AnyIndex {
    Hnsw(HnswIndex),
    IvfPq(IvfPqIndex),
    RaBitQ(RaBitQIndex),
    Binary(BinaryIndex),
}

impl AnyIndex {
    /// Search. `ef` is used for HNSW; ignored for flat indices.
    pub fn search(&self, query: &[f32], top_k: usize, ef: usize) -> Vec<(RowId, f32)> {
        match self {
            AnyIndex::Hnsw(idx) => idx.search(query, top_k, ef),
            AnyIndex::IvfPq(idx) => idx.search(query, top_k, None),
            AnyIndex::RaBitQ(idx) => idx.search(query, top_k, Some(3)),
            AnyIndex::Binary(idx) => idx.search(query, top_k, Some(3)),
        }
    }

    pub fn node_count(&self) -> u64 {
        match self {
            AnyIndex::Hnsw(idx) => idx.node_count(),
            AnyIndex::IvfPq(idx) => idx.node_count(),
            AnyIndex::RaBitQ(idx) => idx.node_count(),
            AnyIndex::Binary(idx) => idx.node_count(),
        }
    }

    /// Quantize stored vectors to F16 for HNSW search (no-op for flat indices).
    pub fn quantize_to_f16(&mut self) {
        if let AnyIndex::Hnsw(idx) = self {
            idx.quantize_to_f16();
        }
    }

    pub fn is_hnsw(&self) -> bool {
        matches!(self, AnyIndex::Hnsw(_))
    }

    pub fn is_ivf_pq(&self) -> bool {
        matches!(self, AnyIndex::IvfPq(_))
    }

    pub fn is_rabitq(&self) -> bool {
        matches!(self, AnyIndex::RaBitQ(_))
    }

    pub fn is_binary(&self) -> bool {
        matches!(self, AnyIndex::Binary(_))
    }
}
