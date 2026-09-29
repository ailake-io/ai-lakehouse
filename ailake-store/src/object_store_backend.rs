// SPDX-License-Identifier: MIT OR Apache-2.0
use std::collections::HashMap;
use std::ops::Range;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

use ailake_core::{AilakeError, AilakeResult};
use async_trait::async_trait;
use bytes::Bytes;
use futures::StreamExt;
use object_store::{path::Path, ObjectStore, PutMode, UpdateVersion};

use crate::store::Store;

/// Wraps any `object_store::ObjectStore` (S3, GCS, Azure, in-memory) behind the
/// unified `Store` trait. All paths are resolved relative to `prefix`.
pub struct ObjectStoreBackend {
    inner: Arc<dyn ObjectStore>,
    /// Base prefix prepended to every path (e.g. "my-table/"). May be empty.
    prefix: String,
    lock_tokens: std::sync::Mutex<HashMap<String, HeldLock>>,
    lock_sequence: AtomicU64,
}

#[derive(Clone)]
struct HeldLock {
    owner: String,
    fencing_token: u64,
}

const LOCK_LEASE_MS: u64 = 120_000;

impl ObjectStoreBackend {
    pub fn new(store: Arc<dyn ObjectStore>, prefix: impl Into<String>) -> Self {
        let mut prefix = prefix.into();
        if !prefix.is_empty() && !prefix.ends_with('/') {
            prefix.push('/');
        }
        Self {
            inner: store,
            prefix,
            lock_tokens: std::sync::Mutex::new(HashMap::new()),
            lock_sequence: AtomicU64::new(0),
        }
    }

    fn resolve(&self, path: &str) -> Path {
        // Absolute URI (e.g. "s3://bucket/warehouse/ns/table/data/part.parquet", the shape
        // HadoopCatalog stores in DataFileEntry.path when the warehouse root itself is
        // absolute — see hadoop.rs's `warehouse_prefix` logic) — the object key is
        // everything after "scheme://bucket/". `self.prefix` must NOT be prepended again,
        // it's already encoded in the URI. This mirrors the override behavior
        // `LocalStore::full_path` gets for free from `PathBuf::join` with an absolute path.
        if let Some((_, after_scheme)) = path.split_once("://") {
            let key = after_scheme.split_once('/').map_or("", |(_, key)| key);
            return Path::from(key);
        }
        let full = format!("{}{}", self.prefix, path.trim_start_matches('/'));
        Path::from(full.as_str())
    }

    fn lock_token(&self, _path: &str) -> String {
        let sequence = self.lock_sequence.fetch_add(1, Ordering::Relaxed);
        format!("pid={}-{}-{}", std::process::id(), now_ms(), sequence)
    }

    fn remember_lock(&self, path: &str, owner: String, fencing_token: u64) {
        if let Ok(mut tokens) = self.lock_tokens.lock() {
            tokens.insert(
                path.to_string(),
                HeldLock {
                    owner,
                    fencing_token,
                },
            );
        }
    }

    fn remembered_lock(&self, path: &str) -> Option<HeldLock> {
        self.lock_tokens.lock().ok()?.get(path).cloned()
    }

    fn forget_lock(&self, path: &str) {
        if let Ok(mut tokens) = self.lock_tokens.lock() {
            tokens.remove(path);
        }
    }
}

fn now_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as u64
}

fn lease_payload(owner: &str, fencing_token: u64) -> bytes::Bytes {
    bytes::Bytes::from(format!(
        "owner={owner}\nfence={fencing_token}\nexpires_ms={}\n",
        now_ms() + LOCK_LEASE_MS
    ))
}

fn lease_owner_fence_and_expiry(data: &[u8]) -> (Option<&str>, Option<u64>, Option<u64>) {
    let text = std::str::from_utf8(data).unwrap_or_default();
    let owner = text.lines().find_map(|line| line.strip_prefix("owner="));
    let fence = text
        .lines()
        .find_map(|line| line.strip_prefix("fence=")?.parse::<u64>().ok());
    let expires = text
        .lines()
        .find_map(|line| line.strip_prefix("expires_ms=")?.parse::<u64>().ok());
    (owner, fence, expires)
}

fn lease_is_expired(meta: &object_store::ObjectMeta, data: &[u8]) -> bool {
    let (_, _, expires) = lease_owner_fence_and_expiry(data);
    expires
        .map(|expires| expires <= now_ms())
        .unwrap_or_else(|| {
            let modified = meta.last_modified.timestamp_millis().max(0) as u64;
            now_ms().saturating_sub(modified) > LOCK_LEASE_MS
        })
}

#[async_trait]
impl Store for ObjectStoreBackend {
    async fn try_acquire_lock(&self, path: &str) -> AilakeResult<bool> {
        Ok(self.try_acquire_lock_fenced(path).await?.is_some())
    }

    async fn try_acquire_lock_fenced(&self, path: &str) -> AilakeResult<Option<u64>> {
        let p = self.resolve(path);
        let owner = self.lock_token(path);
        let (meta, current) = match self.inner.head(&p).await {
            Ok(meta) => {
                let current = self
                    .inner
                    .get(&p)
                    .await
                    .map_err(|e| AilakeError::Store(e.to_string()))?
                    .bytes()
                    .await
                    .map_err(|e| AilakeError::Store(e.to_string()))?;
                (Some(meta), Some(current))
            }
            Err(object_store::Error::NotFound { .. }) => (None, None),
            Err(error) => return Err(AilakeError::Store(error.to_string())),
        };

        let (next_fence, mode) = match (&meta, &current) {
            (None, None) => (1, PutMode::Create),
            (Some(meta), Some(current)) => {
                if !lease_is_expired(meta, current) {
                    return Ok(None);
                }
                let (_, current_fence, _) = lease_owner_fence_and_expiry(current);
                (
                    current_fence.unwrap_or(0).saturating_add(1),
                    PutMode::Update(UpdateVersion {
                        e_tag: meta.e_tag.clone(),
                        version: meta.version.clone(),
                    }),
                )
            }
            _ => unreachable!("lock metadata and content are read together"),
        };
        match self
            .inner
            .put_opts(&p, lease_payload(&owner, next_fence).into(), mode.into())
            .await
        {
            Ok(_) => {
                self.remember_lock(path, owner, next_fence);
                Ok(Some(next_fence))
            }
            Err(object_store::Error::Precondition { .. })
            | Err(object_store::Error::AlreadyExists { .. }) => Ok(None),
            Err(object_store::Error::NotFound { .. }) if meta.is_some() => {
                self.try_acquire_lock_fenced(path).await
            }
            Err(error) => Err(AilakeError::Store(error.to_string())),
        }
    }

    async fn renew_lock(&self, path: &str) -> AilakeResult<bool> {
        let Some(held) = self.remembered_lock(path) else {
            return Ok(false);
        };
        self.renew_lock_fenced(path, held.fencing_token).await
    }

    async fn renew_lock_fenced(&self, path: &str, fencing_token: u64) -> AilakeResult<bool> {
        let Some(held) = self.remembered_lock(path) else {
            return Ok(false);
        };
        if held.fencing_token != fencing_token {
            return Ok(false);
        }
        let p = self.resolve(path);
        let meta = match self.inner.head(&p).await {
            Ok(meta) => meta,
            Err(object_store::Error::NotFound { .. }) => {
                self.forget_lock(path);
                return Ok(false);
            }
            Err(error) => return Err(AilakeError::Store(error.to_string())),
        };
        let current = self
            .inner
            .get(&p)
            .await
            .map_err(|e| AilakeError::Store(e.to_string()))?
            .bytes()
            .await
            .map_err(|e| AilakeError::Store(e.to_string()))?;
        let (owner, fence, _) = lease_owner_fence_and_expiry(&current);
        if owner != Some(held.owner.as_str())
            || fence != Some(held.fencing_token)
            || lease_is_expired(&meta, &current)
        {
            self.forget_lock(path);
            return Ok(false);
        }
        let version = UpdateVersion {
            e_tag: meta.e_tag,
            version: meta.version,
        };
        match self
            .inner
            .put_opts(
                &p,
                lease_payload(&held.owner, held.fencing_token).into(),
                PutMode::Update(version).into(),
            )
            .await
        {
            Ok(_) => Ok(true),
            Err(object_store::Error::Precondition { .. }) => Ok(false),
            Err(error) => Err(AilakeError::Store(error.to_string())),
        }
    }

    async fn release_lock(&self, path: &str) -> AilakeResult<()> {
        let Some(held) = self.remembered_lock(path) else {
            return Ok(());
        };
        self.release_lock_fenced(path, held.fencing_token).await
    }

    async fn release_lock_fenced(&self, path: &str, fencing_token: u64) -> AilakeResult<()> {
        let Some(held) = self.remembered_lock(path) else {
            return Ok(());
        };
        if held.fencing_token != fencing_token {
            return Ok(());
        }
        let p = self.resolve(path);
        let meta = match self.inner.head(&p).await {
            Ok(meta) => meta,
            Err(object_store::Error::NotFound { .. }) => {
                self.forget_lock(path);
                return Ok(());
            }
            Err(error) => return Err(AilakeError::Store(error.to_string())),
        };
        let current = match self.inner.get(&p).await {
            Ok(result) => result
                .bytes()
                .await
                .map_err(|e| AilakeError::Store(e.to_string()))?,
            Err(object_store::Error::NotFound { .. }) => {
                self.forget_lock(path);
                return Ok(());
            }
            Err(error) => return Err(AilakeError::Store(error.to_string())),
        };
        let (owner, fence, _) = lease_owner_fence_and_expiry(&current);
        if owner == Some(held.owner.as_str()) && fence == Some(held.fencing_token) {
            let version = UpdateVersion {
                e_tag: meta.e_tag,
                version: meta.version,
            };
            match self
                .inner
                .put_opts(
                    &p,
                    bytes::Bytes::from(format!(
                        "owner={}\nfence={}\nexpires_ms={}\n",
                        held.owner,
                        held.fencing_token,
                        now_ms().saturating_sub(1),
                    ))
                    .into(),
                    PutMode::Update(version).into(),
                )
                .await
            {
                Ok(_) | Err(object_store::Error::Precondition { .. }) => {}
                Err(error) => return Err(AilakeError::Store(error.to_string())),
            }
        }
        self.forget_lock(path);
        Ok(())
    }

    async fn check_lock_fence(&self, path: &str, fencing_token: u64) -> AilakeResult<bool> {
        let Some(held) = self.remembered_lock(path) else {
            return Ok(false);
        };
        if held.fencing_token != fencing_token {
            return Ok(false);
        }
        let p = self.resolve(path);
        let meta = match self.inner.head(&p).await {
            Ok(meta) => meta,
            Err(object_store::Error::NotFound { .. }) => return Ok(false),
            Err(error) => return Err(AilakeError::Store(error.to_string())),
        };
        let current = self
            .inner
            .get(&p)
            .await
            .map_err(|e| AilakeError::Store(e.to_string()))?
            .bytes()
            .await
            .map_err(|e| AilakeError::Store(e.to_string()))?;
        let (owner, fence, _) = lease_owner_fence_and_expiry(&current);
        Ok(owner == Some(held.owner.as_str())
            && fence == Some(fencing_token)
            && !lease_is_expired(&meta, &current))
    }

    async fn get(&self, path: &str) -> AilakeResult<Bytes> {
        let p = self.resolve(path);
        self.inner
            .get(&p)
            .await
            .map_err(|e| AilakeError::Store(e.to_string()))?
            .bytes()
            .await
            .map_err(|e| AilakeError::Store(e.to_string()))
    }

    async fn get_range(&self, path: &str, range: Range<u64>) -> AilakeResult<Bytes> {
        let p = self.resolve(path);
        let byte_range = range.start as usize..range.end as usize;
        self.inner
            .get_range(&p, byte_range)
            .await
            .map_err(|e| AilakeError::Store(e.to_string()))
    }

    async fn put(&self, path: &str, data: Bytes) -> AilakeResult<()> {
        let p = self.resolve(path);
        self.inner
            .put(&p, data.into())
            .await
            .map_err(|e| AilakeError::Store(e.to_string()))?;
        Ok(())
    }

    async fn list(&self, prefix: &str) -> AilakeResult<Vec<String>> {
        let p = self.resolve(prefix);
        let base_prefix = self.prefix.clone();
        let mut stream = self.inner.list(Some(&p));
        let mut paths = Vec::new();
        while let Some(item) = stream.next().await {
            let meta = item.map_err(|e| AilakeError::Store(e.to_string()))?;
            let full = meta.location.to_string();
            // Strip the store prefix to return a relative path
            let rel = if full.starts_with(&base_prefix) {
                full[base_prefix.len()..].to_string()
            } else {
                full
            };
            paths.push(rel);
        }
        paths.sort();
        Ok(paths)
    }

    async fn file_size(&self, path: &str) -> AilakeResult<u64> {
        let p = self.resolve(path);
        let meta = self
            .inner
            .head(&p)
            .await
            .map_err(|e| AilakeError::Store(e.to_string()))?;
        Ok(meta.size as u64)
    }

    async fn exists(&self, path: &str) -> AilakeResult<bool> {
        let p = self.resolve(path);
        match self.inner.head(&p).await {
            Ok(_) => Ok(true),
            Err(object_store::Error::NotFound { .. }) => Ok(false),
            Err(e) => Err(AilakeError::Store(e.to_string())),
        }
    }

    async fn delete(&self, path: &str) -> AilakeResult<()> {
        let p = self.resolve(path);
        self.inner
            .delete(&p)
            .await
            .map_err(|e| AilakeError::Store(e.to_string()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use object_store::memory::InMemory;

    fn backend(prefix: &str) -> ObjectStoreBackend {
        ObjectStoreBackend::new(Arc::new(InMemory::new()), prefix)
    }

    #[test]
    fn relative_path_gets_configured_prefix() {
        let b = backend("warehouse");
        assert_eq!(
            b.resolve("data/part.parquet").as_ref(),
            "warehouse/data/part.parquet"
        );
    }

    /// Regression: an absolute URI (the shape `HadoopCatalog` stores in `DataFileEntry.path`
    /// when the warehouse root itself is absolute — see `hadoop.rs`'s `warehouse_prefix`
    /// logic) used to be concatenated onto `prefix` verbatim, producing a garbage
    /// double-prefixed key like `"warehouse/s3://my-bucket/ns/table/data/part.parquet"`
    /// instead of resolving to the real object key.
    #[test]
    fn absolute_uri_ignores_configured_prefix() {
        let b = backend("warehouse");
        let abs = "s3://my-bucket/ns/table/data/part.parquet";
        assert_eq!(b.resolve(abs).as_ref(), "ns/table/data/part.parquet");
    }

    #[test]
    fn absolute_uri_variants() {
        let b = backend("warehouse");
        assert_eq!(b.resolve("gs://bucket/a/b.parquet").as_ref(), "a/b.parquet");
        assert_eq!(
            b.resolve("az://container/x/y.parquet").as_ref(),
            "x/y.parquet"
        );
    }

    #[tokio::test]
    async fn absolute_uri_round_trips_through_get_put() {
        let b = backend("warehouse");
        let abs = "s3://my-bucket/ns/table/data/part.parquet";
        b.put(abs, Bytes::from_static(b"hello")).await.unwrap();
        assert_eq!(b.get(abs).await.unwrap(), Bytes::from_static(b"hello"));
    }

    #[tokio::test]
    async fn conditional_lock_is_exclusive_and_releasable() {
        let inner: Arc<dyn ObjectStore> = Arc::new(InMemory::new());
        let first = ObjectStoreBackend::new(Arc::clone(&inner), "warehouse");
        let second = ObjectStoreBackend::new(inner, "warehouse");
        let first_fence = first
            .try_acquire_lock_fenced("metadata/job.lock")
            .await
            .unwrap()
            .unwrap();
        assert_eq!(first_fence, 1);
        assert!(first
            .check_lock_fence("metadata/job.lock", first_fence)
            .await
            .unwrap());
        assert!(!second.try_acquire_lock("metadata/job.lock").await.unwrap());
        assert!(first.renew_lock("metadata/job.lock").await.unwrap());
        first
            .release_lock_fenced("metadata/job.lock", first_fence)
            .await
            .unwrap();
        let second_fence = second
            .try_acquire_lock_fenced("metadata/job.lock")
            .await
            .unwrap()
            .unwrap();
        assert_eq!(second_fence, 2);
        assert!(!first
            .check_lock_fence("metadata/job.lock", first_fence)
            .await
            .unwrap());
        second
            .release_lock_fenced("metadata/job.lock", second_fence)
            .await
            .unwrap();
    }
}
