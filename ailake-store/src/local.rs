// SPDX-License-Identifier: MIT OR Apache-2.0
use std::ops::Range;
use std::path::{Path, PathBuf};

use ailake_core::{AilakeError, AilakeResult};
use async_trait::async_trait;
use bytes::Bytes;
use tokio::io::{AsyncReadExt, AsyncSeekExt};

use crate::store::Store;

pub struct LocalStore {
    root: PathBuf,
}

impl LocalStore {
    pub fn new(root: impl AsRef<Path>) -> Self {
        let path = root.as_ref();
        // Strip file:// scheme if the caller passes a file:// URI.
        // Without this, PathBuf::from("file:///abs/path") is a RELATIVE path
        // starting with the literal segment "file:" — not an absolute path.
        let clean = path
            .to_str()
            .and_then(|s| s.strip_prefix("file://"))
            .map(PathBuf::from)
            .unwrap_or_else(|| path.to_path_buf());
        Self { root: clean }
    }

    fn full_path(&self, path: &str) -> PathBuf {
        // Strip file:// scheme so callers can pass absolute file:// URIs.
        // PathBuf::join with an absolute path ignores self.root, so
        // "file:///abs/path" → "/abs/path" resolves correctly.
        let clean = path.strip_prefix("file://").unwrap_or(path);
        self.root.join(clean)
    }
}

#[async_trait]
impl Store for LocalStore {
    async fn get(&self, path: &str) -> AilakeResult<Bytes> {
        let data = tokio::fs::read(self.full_path(path)).await?;
        Ok(Bytes::from(data))
    }

    async fn get_range(&self, path: &str, range: Range<u64>) -> AilakeResult<Bytes> {
        let mut file = tokio::fs::File::open(self.full_path(path)).await?;
        file.seek(std::io::SeekFrom::Start(range.start)).await?;
        let len = (range.end - range.start) as usize;
        let mut buf = vec![0u8; len];
        file.read_exact(&mut buf).await?;
        Ok(Bytes::from(buf))
    }

    async fn put(&self, path: &str, data: Bytes) -> AilakeResult<()> {
        let full = self.full_path(path);
        if let Some(parent) = full.parent() {
            tokio::fs::create_dir_all(parent).await?;
        }
        tokio::fs::write(full, data).await?;
        Ok(())
    }

    async fn list(&self, prefix: &str) -> AilakeResult<Vec<String>> {
        let dir = self.full_path(prefix);
        if !dir.exists() {
            return Ok(vec![]);
        }
        let mut entries = Vec::new();
        let mut read_dir = tokio::fs::read_dir(&dir).await?;
        while let Some(entry) = read_dir.next_entry().await? {
            let path = entry.path();
            if path.is_file() {
                let rel = path
                    .strip_prefix(&self.root)
                    .map_err(|e| AilakeError::Store(e.to_string()))?
                    .to_string_lossy()
                    .to_string();
                entries.push(rel);
            }
        }
        entries.sort();
        Ok(entries)
    }

    async fn file_size(&self, path: &str) -> AilakeResult<u64> {
        let meta = tokio::fs::metadata(self.full_path(path)).await?;
        Ok(meta.len())
    }

    async fn exists(&self, path: &str) -> AilakeResult<bool> {
        Ok(self.full_path(path).exists())
    }

    async fn delete(&self, path: &str) -> AilakeResult<()> {
        tokio::fs::remove_file(self.full_path(path)).await?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[tokio::test]
    async fn put_get_roundtrip() {
        let dir = TempDir::new().unwrap();
        let store = LocalStore::new(dir.path());
        let data = Bytes::from("hello ailake");
        store.put("test.bin", data.clone()).await.unwrap();
        let got = store.get("test.bin").await.unwrap();
        assert_eq!(got, data);
    }

    #[tokio::test]
    async fn get_range_reads_partial() {
        let dir = TempDir::new().unwrap();
        let store = LocalStore::new(dir.path());
        let data = Bytes::from(b"abcdefghijklmnop".as_ref());
        store.put("test.bin", data).await.unwrap();
        let partial = store.get_range("test.bin", 4..8).await.unwrap();
        assert_eq!(partial.as_ref(), b"efgh");
    }

    #[tokio::test]
    async fn list_returns_files() {
        let dir = TempDir::new().unwrap();
        let store = LocalStore::new(dir.path());
        store.put("data/a.parquet", Bytes::from("a")).await.unwrap();
        store.put("data/b.parquet", Bytes::from("b")).await.unwrap();
        let files = store.list("data").await.unwrap();
        assert_eq!(files.len(), 2);
    }

    #[tokio::test]
    async fn file_size_correct() {
        let dir = TempDir::new().unwrap();
        let store = LocalStore::new(dir.path());
        store
            .put("x.bin", Bytes::from(vec![0u8; 42]))
            .await
            .unwrap();
        assert_eq!(store.file_size("x.bin").await.unwrap(), 42);
    }
}
