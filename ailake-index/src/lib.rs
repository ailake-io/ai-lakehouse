//! ailake-index — HNSW index lifecycle
//!
//! Search backend priority:
//!   1. GPU (candle-core + CUDA) — compiled when `--features gpu`, used when CUDA available at runtime
//!   2. Parallel CPU brute-force (rayon) — always available, no special hardware needed
//!
//! Build with `cargo build --features ailake-index/gpu` to enable GPU support.

pub mod gpu;
pub mod hnsw;
pub mod mmap_loader;
pub mod serialize;

pub use hnsw::{HnswBuilder, HnswConfig, HnswIndex};
pub use mmap_loader::MmapLoader;
pub use serialize::HnswSerializer;
