// SPDX-License-Identifier: MIT OR Apache-2.0
use std::collections::HashMap;
use std::ops::Range;
use std::path::{Component, Path, PathBuf};

use ailake_core::{AilakeError, AilakeResult};
use async_trait::async_trait;
use bytes::Bytes;
use tokio::io::{AsyncReadExt, AsyncSeekExt};
use tokio::sync::Mutex;

use crate::store::Store;

pub struct LocalStore {
    root: PathBuf,
    // Keep the OS lock handles alive until release_lock or process exit.
    locks: Mutex<HashMap<PathBuf, std::fs::File>>,
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
        Self {
            root: clean,
            locks: Mutex::new(HashMap::new()),
        }
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
        // Lexical prefix checks alone can be bypassed by a symlink inside the
        // store root. Resolve the nearest existing ancestor so new files are
        // checked too, then append only the still-missing suffix.
        let canonical_root = canonicalize_with_missing_suffix(&root).unwrap_or(root.clone());
        let mut ancestor = candidate.as_path();
        let mut suffix = Vec::new();
        loop {
            if let Ok(metadata) = std::fs::symlink_metadata(ancestor) {
                if metadata.file_type().is_symlink() {
                    return Err(AilakeError::InvalidArgument(format!(
                        "symlinks are not allowed in LocalStore paths: {path}"
                    )));
                }
                break;
            }
            let Some(name) = ancestor.file_name() else {
                break;
            };
            suffix.push(name.to_os_string());
            let Some(parent) = ancestor.parent() else {
                break;
            };
            ancestor = parent;
        }
        let resolved = canonicalize_with_missing_suffix(ancestor)
            .map(|mut path| {
                for part in suffix.iter().rev() {
                    path.push(part);
                }
                path
            })
            .unwrap_or(candidate.clone());
        if !candidate.starts_with(&root) || !resolved.starts_with(&canonical_root) {
            return Err(AilakeError::InvalidArgument(format!(
                "path escapes LocalStore root: {path}"
            )));
        }
        Ok(resolved)
    }

    /// Validate an Iceberg table location before a write starts. Relative
    /// locations are intentionally accepted: Hadoop-style catalogs resolve
    /// those relative to this store's root. Absolute locations must remain
    /// inside the configured root.
    pub fn validate_location(&self, location: &str) -> AilakeResult<()> {
        self.full_path(location).map(|_| ())
    }
}

fn canonicalize_with_missing_suffix(path: &Path) -> Option<PathBuf> {
    let mut ancestor = path;
    let mut suffix = Vec::new();
    while std::fs::symlink_metadata(ancestor).is_err() {
        suffix.push(ancestor.file_name()?.to_os_string());
        ancestor = ancestor.parent()?;
    }
    let mut resolved = std::fs::canonicalize(ancestor).ok()?;
    for part in suffix.iter().rev() {
        resolved.push(part);
    }
    Some(resolved)
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
        let mut locks = self.locks.lock().await;
        if locks.contains_key(&full) {
            return Ok(false);
        }
        let file = tokio::fs::OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(false)
            .open(&full)
            .await?;
        let mut file = file.into_std().await;
        match file.try_lock() {
            Ok(()) => {
                use std::io::Write;
                file.set_len(0)?;
                writeln!(file, "pid={}", std::process::id())?;
                locks.insert(full, file);
                Ok(true)
            }
            Err(std::fs::TryLockError::WouldBlock) => Ok(false),
            Err(std::fs::TryLockError::Error(error)) => Err(error.into()),
        }
    }

    async fn release_lock(&self, path: &str) -> AilakeResult<()> {
        let full = self.full_path(path)?;
        // Keep the lock file itself in place: unlinking it while held lets a
        // second process create and lock a different inode at the same path.
        let mut locks = self.locks.lock().await;
        if let Some(file) = locks.get(&full) {
            file.unlock()?;
        }
        locks.remove(&full);
        Ok(())
    }

    async fn renew_lock(&self, path: &str) -> AilakeResult<bool> {
        let full = self.full_path(path)?;
        Ok(self.locks.lock().await.contains_key(&full))
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
        let root = canonicalize_with_missing_suffix(&normalize_path(&self.root))
            .unwrap_or_else(|| normalize_path(&self.root));
        let mut entries = Vec::new();
        let mut read_dir = tokio::fs::read_dir(&dir).await?;
        while let Some(entry) = read_dir.next_entry().await? {
            let path = entry.path();
            if path.is_file() {
                let rel = path
                    .strip_prefix(&root)
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
    async fn commit_lock_is_exclusive_across_store_instances() {
        let dir = TempDir::new().unwrap();
        let first = LocalStore::new(dir.path());
        let second = LocalStore::new(dir.path());
        assert!(first.try_acquire_lock("metadata/table.lock").await.unwrap());
        assert!(!second
            .try_acquire_lock("metadata/table.lock")
            .await
            .unwrap());
        first.release_lock("metadata/table.lock").await.unwrap();
        assert!(second
            .try_acquire_lock("metadata/table.lock")
            .await
            .unwrap());
        second.release_lock("metadata/table.lock").await.unwrap();
    }

    #[tokio::test]
    async fn commit_lock_is_exclusive_across_processes() {
        let dir = TempDir::new().unwrap();
        let store = LocalStore::new(dir.path());
        assert!(store
            .try_acquire_lock("metadata/process.lock")
            .await
            .unwrap());
        run_lock_probe(dir.path(), false);
        store.release_lock("metadata/process.lock").await.unwrap();
        run_lock_probe(dir.path(), true);
    }

    #[tokio::test]
    async fn commit_lock_process_probe() {
        let Ok(root) = std::env::var("AILAKE_LOCK_PROBE_ROOT") else {
            return;
        };
        let expected = std::env::var("AILAKE_LOCK_PROBE_EXPECTED")
            .map(|value| value == "true")
            .unwrap_or(false);
        let store = LocalStore::new(root);
        let acquired = store
            .try_acquire_lock("metadata/process.lock")
            .await
            .unwrap();
        assert_eq!(acquired, expected);
        if acquired {
            store.release_lock("metadata/process.lock").await.unwrap();
        }
    }

    fn run_lock_probe(root: &Path, expected: bool) {
        let status = std::process::Command::new(std::env::current_exe().unwrap())
            .args(["--exact", "local::tests::commit_lock_process_probe"])
            .env("AILAKE_LOCK_PROBE_ROOT", root)
            .env("AILAKE_LOCK_PROBE_EXPECTED", expected.to_string())
            .status()
            .unwrap();
        assert!(status.success(), "lock probe process failed: {status}");
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
        let outside = TempDir::new().unwrap();
        let store = LocalStore::new(dir.path());
        let outside_file = outside.path().join("ailake-outside.bin");
        let error = store
            .put(outside_file.to_str().unwrap(), Bytes::from("nope"))
            .await
            .unwrap_err();
        assert!(error.to_string().contains("escapes LocalStore root"));
        assert!(!outside_file.exists());
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn rejects_symlink_escape() {
        use std::os::unix::fs::symlink;

        let root = TempDir::new().unwrap();
        let outside = TempDir::new().unwrap();
        symlink(outside.path(), root.path().join("outside-link")).unwrap();
        let store = LocalStore::new(root.path());
        let error = store
            .put("outside-link/escaped.bin", Bytes::from("nope"))
            .await
            .unwrap_err();
        assert!(error.to_string().contains("symlinks are not allowed"));
        assert!(!outside.path().join("escaped.bin").exists());
    }

    #[tokio::test]
    async fn creates_files_under_a_not_yet_existing_root() {
        let dir = TempDir::new().unwrap();
        let root = dir.path().join("new-root");
        let store = LocalStore::new(&root);
        store
            .put("nested/file.bin", Bytes::from("ok"))
            .await
            .unwrap();
        assert_eq!(
            store.get("nested/file.bin").await.unwrap(),
            Bytes::from("ok")
        );
    }

    #[tokio::test]
    async fn list_supports_a_relative_store_root() {
        let dir = TempDir::new_in(".").unwrap();
        let store = LocalStore::new(dir.path());
        store.put("data/a.bin", Bytes::from("ok")).await.unwrap();
        assert_eq!(store.list("data").await.unwrap(), vec!["data/a.bin"]);
    }
}
