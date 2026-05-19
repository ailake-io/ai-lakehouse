//! ailake-store — object storage abstraction
//!
//! Thin wrapper over object_store crate.
//! The get_range method is critical for partial S3 reads of the HNSW footer.

pub mod local;
pub mod store;

pub use local::LocalStore;
pub use store::Store;
