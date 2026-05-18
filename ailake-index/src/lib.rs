//! ailake-index — HNSW index lifecycle
//!
//! Wraps hnsw_rs. Handles: build, search, bincode serialization, mmap loading.

pub mod hnsw;
pub mod serialize;
pub mod mmap_loader;

pub use hnsw::{HnswBuilder, HnswIndex, HnswConfig};
pub use serialize::HnswSerializer;
pub use mmap_loader::MmapLoader;
