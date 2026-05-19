//! ailake-index — HNSW index lifecycle
//!
//! Wraps hnsw_rs. Handles: build, search, bincode serialization, mmap loading.

pub mod hnsw;
pub mod mmap_loader;
pub mod serialize;

pub use hnsw::{HnswBuilder, HnswConfig, HnswIndex};
pub use mmap_loader::MmapLoader;
pub use serialize::HnswSerializer;
