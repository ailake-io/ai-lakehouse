//! ailake-index — HNSW and IVF-PQ index lifecycle
//!
//! Search backend priority:
//!   1. GPU (candle-core + CUDA) — compiled when `--features gpu`, used when CUDA available at runtime
//!   2. Parallel CPU brute-force (rayon) — always available, no special hardware needed
//!
//! Build with `cargo build --features ailake-index/gpu` to enable GPU support.

pub mod gpu;
pub mod hardware;
pub mod hnsw;
pub mod ivf_pq;
pub mod mmap_loader;
pub mod serialize;

pub use hardware::{detect_backend, detect_cuda, detect_rocm, HardwareBackend, HardwareProfile};
pub use hnsw::{HnswBuilder, HnswConfig, HnswIndex};
pub use ivf_pq::{find_valid_pq_m, IvfPqConfig, IvfPqIndex, IvfPqSerializer};
pub use mmap_loader::MmapLoader;
pub use serialize::HnswSerializer;

use ailake_core::RowId;

/// Unified index type: dispatches search to either HNSW or IVF-PQ.
pub enum AnyIndex {
    Hnsw(HnswIndex),
    IvfPq(IvfPqIndex),
}

impl AnyIndex {
    /// Search with HNSW `ef` parameter (ignored for IVF-PQ, which uses `config.nprobe`).
    pub fn search(&self, query: &[f32], top_k: usize, ef: usize) -> Vec<(RowId, f32)> {
        match self {
            AnyIndex::Hnsw(idx) => idx.search(query, top_k, ef),
            AnyIndex::IvfPq(idx) => idx.search(query, top_k, None),
        }
    }

    pub fn node_count(&self) -> u64 {
        match self {
            AnyIndex::Hnsw(idx) => idx.node_count(),
            AnyIndex::IvfPq(idx) => idx.node_count(),
        }
    }

    /// Quantize stored vectors to F16 for HNSW search (no-op for IVF-PQ).
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
}
