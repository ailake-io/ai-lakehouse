// SPDX-License-Identifier: MIT OR Apache-2.0
use std::ops::Range;
use std::path::{Component, Path, PathBuf};

use ailake_core::{AilakeError, AilakeResult};
use async_trait::async_trait;
use bytes::Bytes;
use tokio::io::AsyncWriteExt;
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

    fn full_path(&self, path: &str) -> AilakeResult<PathBuf> {
        // Strip file:// scheme so callers can pass absolute file:// URIs.
        // Absolute file:// paths are accepted only when they remain inside the
        // configured root. Relative paths must never contain `..` components.
        let clean = path.strip_prefix("file://").unwrap_or(path);
        let requested = Path::new(clean);
        if requested
            .components()
            .any(|component| matches!(component, Component::ParentDir))
        {
            return Err(AilakeError::InvalidArgument(format!(
                "path traversal is not allowed: {path}"
            )));
        }

        let candidate = if requested.is_absolute() {
            requested.to_path_buf()
        } else {
            self.root.join(requested)
        };
        let root = normalize_path(&self.root);
        let candidate = normalize_path(&candidate);
        if !candidate.starts_with(&root) {
            return Err(AilakeError::InvalidArgument(format!(
                "path escapes LocalStore root: {path}"
            )));
        }
        Ok(candidate)
    }

    /// Validate an Iceberg table location before a write starts. Relative
    /// locations are intentionally accepted: Hadoop-style catalogs resolve
    /// those relative to this store's root. Absolute locations must remain
    /// inside the configured root.
    pub fn validate_location(&self, location: &str) -> AilakeResult<()> {
        self.full_path(location).map(|_| ())
    }
}

fn normalize_path(path: &Path) -> PathBuf {
    let mut normalized = PathBuf::new();
    for component in path.components() {
        match component {
            Component::CurDir => {}
            Component::ParentDir => {
                normalized.pop();
            }
            other => normalized.push(other.as_os_str()),
        }
    }
    normalized
}

#[async_trait]
impl Store for LocalStore {
    fn validate_location(&self, location: &str) -> AilakeResult<()> {
        LocalStore::validate_location(self, location)
    }

    async fn try_acquire_lock(&self, path: &str) -> AilakeResult<bool> {
        let full = self.full_path(path)?;
        if let Some(parent) = full.parent() {
            tokio::fs::create_dir_all(parent).await?;
        }
        match tokio::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&full)
            .await
        {
            Ok(mut file) => {
                let owner = format!("pid={}\n", std::process::id());
                file.write_all(owner.as_bytes()).await?;
                Ok(true)
            }
            Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => {
                // A crashed process must not permanently wedge a table. Locks
                // owned by a dead local process are immediately stale; locks
                // without a readable PID retain the five-minute safety valve.
                let owner_dead = tokio::fs::read_to_string(&full)
                    .await
                    .ok()
                    .and_then(|contents| {
                        contents
                            .lines()
                            .find_map(|line| line.strip_prefix("pid=")?.parse::<u32>().ok())
                    })
                    .map(|pid| {
                        pid != std::process::id()
                            && !std::path::Path::new(&format!("/proc/{pid}")).exists()
                    })
                    .unwrap_or(false);
                let stale = tokio::fs::metadata(&full)
                    .await
                    .and_then(|meta| meta.modified())
                    .ok()
                    .and_then(|modified| modified.elapsed().ok())
                    .is_some_and(|age| age > std::time::Duration::from_secs(300));
                if owner_dead || stale {
                    let _ = tokio::fs::remove_file(&full).await;
                    return self.try_acquire_lock(path).await;
                }
                Ok(false)
            }
            Err(error) => Err(error.into()),
        }
    }

    async fn release_lock(&self, path: &str) -> AilakeResult<()> {
        let full = self.full_path(path)?;
        let owned = tokio::fs::read_to_string(&full)
            .await
            .ok()
            .and_then(|contents| {
                contents
                    .lines()
                    .find_map(|line| line.strip_prefix("pid=")?.parse::<u32>().ok())
            })
            .is_some_and(|pid| pid == std::process::id());
        if !owned {
            return Ok(());
        }
        match tokio::fs::remove_file(full).await {
            Ok(()) => Ok(()),
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(()),
            Err(error) => Err(error.into()),
        }
    }

    async fn renew_lock(&self, path: &str) -> AilakeResult<bool> {
        let full = self.full_path(path)?;
        let contents = match tokio::fs::read_to_string(&full).await {
            Ok(contents) => contents,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(false),
            Err(error) => return Err(error.into()),
        };
        let owned = contents.lines().any(|line| {
            line.strip_prefix("pid=")
                .and_then(|value| value.parse::<u32>().ok())
                .is_some_and(|pid| pid == std::process::id())
        });
        if !owned {
            return Ok(false);
        }
        tokio::fs::write(full, format!("pid={}\n", std::process::id())).await?;
        Ok(true)
    }

    async fn get(&self, path: &str) -> AilakeResult<Bytes> {
        let data = tokio::fs::read(self.full_path(path)?).await?;
        Ok(Bytes::from(data))
    }

    async fn get_range(&self, path: &str, range: Range<u64>) -> AilakeResult<Bytes> {
        if range.end < range.start {
            return Err(AilakeError::InvalidArgument(format!(
                "invalid byte range: {}..{}",
                range.start, range.end
            )));
        }
        let mut file = tokio::fs::File::open(self.full_path(path)?).await?;
        file.seek(std::io::SeekFrom::Start(range.start)).await?;
        let len = (range.end - range.start) as usize;
        let mut buf = vec![0u8; len];
        file.read_exact(&mut buf).await?;
        Ok(Bytes::from(buf))
    }

    async fn put(&self, path: &str, data: Bytes) -> AilakeResult<()> {
        let full = self.full_path(path)?;
        if let Some(parent) = full.parent() {
            tokio::fs::create_dir_all(parent).await?;
        }
        tokio::fs::write(full, data).await?;
        Ok(())
    }

    async fn list(&self, prefix: &str) -> AilakeResult<Vec<String>> {
        let dir = self.full_path(prefix)?;
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
        let meta = tokio::fs::metadata(self.full_path(path)?).await?;
        Ok(meta.len())
    }

    async fn exists(&self, path: &str) -> AilakeResult<bool> {
        Ok(self.full_path(path)?.exists())
    }

    async fn delete(&self, path: &str) -> AilakeResult<()> {
        tokio::fs::remove_file(self.full_path(path)?).await?;
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

    #[tokio::test]
    async fn commit_lock_is_exclusive_and_releasable() {
        let dir = TempDir::new().unwrap();
        let store = LocalStore::new(dir.path());
        assert!(store.try_acquire_lock("metadata/table.lock").await.unwrap());
        assert!(!store.try_acquire_lock("metadata/table.lock").await.unwrap());
        store.release_lock("metadata/table.lock").await.unwrap();
        assert!(store.try_acquire_lock("metadata/table.lock").await.unwrap());
        store.release_lock("metadata/table.lock").await.unwrap();
    }

    #[tokio::test]
    async fn dead_process_lock_is_recovered_immediately() {
        let dir = TempDir::new().unwrap();
        let store = LocalStore::new(dir.path());
        let lock = dir.path().join("metadata/job.lock");
        std::fs::create_dir_all(lock.parent().unwrap()).unwrap();
        std::fs::write(&lock, b"pid=4294967294\n").unwrap();
        assert!(store.try_acquire_lock("metadata/job.lock").await.unwrap());
        store.release_lock("metadata/job.lock").await.unwrap();
    }

    #[tokio::test]
    async fn rejects_parent_directory_traversal() {
        let dir = TempDir::new().unwrap();
        let store = LocalStore::new(dir.path());
        let error = store
            .put("../outside.bin", Bytes::from("nope"))
            .await
            .unwrap_err();
        assert!(error.to_string().contains("path traversal"));
    }

    #[tokio::test]
    async fn rejects_absolute_path_outside_root() {
        let dir = TempDir::new().unwrap();
        let store = LocalStore::new(dir.path());
        let error = store
            .put("file:///tmp/ailake-outside.bin", Bytes::from("nope"))
            .await
            .unwrap_err();
        assert!(error.to_string().contains("escapes LocalStore root"));
    }
}
