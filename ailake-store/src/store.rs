use std::ops::Range;

use ailake_core::AilakeResult;
use async_trait::async_trait;
use bytes::Bytes;

/// Unified object storage abstraction.
/// All methods are async; implementations cover local filesystem and cloud (Phase 2).
#[async_trait]
pub trait Store: Send + Sync {
    async fn get(&self, path: &str) -> AilakeResult<Bytes>;

    /// Partial read — critical for S3 HNSW footer reads.
    async fn get_range(&self, path: &str, range: Range<u64>) -> AilakeResult<Bytes>;

    async fn put(&self, path: &str, data: Bytes) -> AilakeResult<()>;

    async fn list(&self, prefix: &str) -> AilakeResult<Vec<String>>;

    async fn file_size(&self, path: &str) -> AilakeResult<u64>;

    async fn exists(&self, path: &str) -> AilakeResult<bool>;

    async fn delete(&self, path: &str) -> AilakeResult<()>;
}
