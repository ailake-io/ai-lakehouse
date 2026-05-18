//! ailake-store — object storage abstraction
//!
//! Thin wrapper over object_store crate.
//! The get_range method is critical for partial S3 reads of the HNSW footer.

pub mod store;
pub mod local;

pub use store::Store;
pub use local::LocalStore;
