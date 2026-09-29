// SPDX-License-Identifier: MIT OR Apache-2.0
use std::ops::Range;

use ailake_core::AilakeResult;
use async_trait::async_trait;
use bytes::Bytes;

/// Unified object storage abstraction.
/// All methods are async; implementations cover local filesystem and cloud (Phase 2).
#[async_trait]
pub trait Store: Send + Sync {
    /// Validate that a catalog table location can be addressed by this store.
    ///
    /// Object stores generally resolve locations by URI/prefix and need no
    /// local-path validation. Local implementations override this to reject
    /// an absolute catalog location outside their configured root.
    fn validate_location(&self, _location: &str) -> AilakeResult<()> {
        Ok(())
    }

    /// Acquire a best-effort per-table commit lock. Backends without an
    /// atomic create primitive keep the historical no-op behavior; LocalStore
    /// overrides this with an exclusive lock file.
    async fn try_acquire_lock(&self, _path: &str) -> AilakeResult<bool> {
        Ok(true)
    }

    /// Acquire a lease and return its fencing token. Object-store backends
    /// implement this with a conditional create/update; local backends retain
    /// the historical process lock and use token `0`.
    async fn try_acquire_lock_fenced(&self, path: &str) -> AilakeResult<Option<u64>> {
        Ok(self.try_acquire_lock(path).await?.then_some(0))
    }

    /// Release a lock acquired with [`Store::try_acquire_lock`].
    async fn release_lock(&self, _path: &str) -> AilakeResult<()> {
        Ok(())
    }

    /// Release a lease only when the caller still owns `fencing_token`.
    async fn release_lock_fenced(&self, path: &str, fencing_token: u64) -> AilakeResult<()> {
        let _ = fencing_token;
        self.release_lock(path).await
    }

    /// Renew a lock lease. Backends with non-expiring process locks can keep
    /// the default successful result.
    async fn renew_lock(&self, _path: &str) -> AilakeResult<bool> {
        Ok(true)
    }

    /// Renew a lease only when the caller still owns `fencing_token`.
    async fn renew_lock_fenced(&self, path: &str, fencing_token: u64) -> AilakeResult<bool> {
        let _ = fencing_token;
        self.renew_lock(path).await
    }

    /// Validate the current fencing token immediately before a shared-state or
    /// data-plane commit. A stale worker must stop instead of publishing after
    /// its lease was acquired by another process.
    async fn check_lock_fence(&self, _path: &str, _fencing_token: u64) -> AilakeResult<bool> {
        Ok(true)
    }

    async fn get(&self, path: &str) -> AilakeResult<Bytes>;

    /// Partial read — critical for S3 HNSW footer reads.
    async fn get_range(&self, path: &str, range: Range<u64>) -> AilakeResult<Bytes>;

    async fn put(&self, path: &str, data: Bytes) -> AilakeResult<()>;

    async fn list(&self, prefix: &str) -> AilakeResult<Vec<String>>;

    async fn file_size(&self, path: &str) -> AilakeResult<u64>;

    async fn exists(&self, path: &str) -> AilakeResult<bool>;

    async fn delete(&self, path: &str) -> AilakeResult<()>;
}
