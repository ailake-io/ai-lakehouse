// SPDX-License-Identifier: MIT OR Apache-2.0
use std::ops::Range;
use std::sync::Arc;

use ailake_core::{AilakeError, AilakeResult};
use async_trait::async_trait;
use bytes::Bytes;
use futures::StreamExt;
use object_store::{path::Path, ObjectStore};

use crate::store::Store;

/// Wraps any `object_store::ObjectStore` (S3, GCS, Azure, in-memory) behind the
/// unified `Store` trait. All paths are resolved relative to `prefix`.
pub struct ObjectStoreBackend {
    inner: Arc<dyn ObjectStore>,
    /// Base prefix prepended to every path (e.g. "my-table/"). May be empty.
    prefix: String,
}

impl ObjectStoreBackend {
    pub fn new(store: Arc<dyn ObjectStore>, prefix: impl Into<String>) -> Self {
        let mut prefix = prefix.into();
        if !prefix.is_empty() && !prefix.ends_with('/') {
            prefix.push('/');
        }
        Self {
            inner: store,
            prefix,
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
}

#[async_trait]
impl Store for ObjectStoreBackend {
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
}
