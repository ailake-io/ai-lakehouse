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
        Self { inner: store, prefix }
    }

    fn resolve(&self, path: &str) -> Path {
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
