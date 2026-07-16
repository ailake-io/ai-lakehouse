use std::ops::Range;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Mutex;

use ailake_core::{AilakeError, AilakeResult};
use async_trait::async_trait;
use bytes::Bytes;

use crate::store::Store;

/// A `Store` wrapper that injects faults for testing error-handling paths.
///
/// Each operation can be independently configured to fail (return an error)
/// or succeed normally by delegating to the inner `Store`.
///
/// # Example
///
/// ```ignore
/// let inner = LocalStore::new("/tmp/test");
/// let store = FailStore::new(inner)
///     .with_fail_get(true)
///     .with_fail_put(true);
/// let result = store.get("foo.parquet").await;  // returns Err
/// ```
pub struct FailStore {
    inner: Box<dyn Store>,
    fail_get: AtomicBool,
    fail_get_nth: Mutex<Option<u64>>,
    fail_get_range: AtomicBool,
    fail_put: AtomicBool,
    fail_list: AtomicBool,
    fail_file_size: AtomicBool,
    fail_exists: AtomicBool,
    fail_delete: AtomicBool,
    custom_error: String,
}

impl FailStore {
    pub fn new(inner: impl Store + 'static) -> Self {
        Self {
            inner: Box::new(inner),
            fail_get: AtomicBool::new(false),
            fail_get_nth: Mutex::new(None),
            fail_get_range: AtomicBool::new(false),
            fail_put: AtomicBool::new(false),
            fail_list: AtomicBool::new(false),
            fail_file_size: AtomicBool::new(false),
            fail_exists: AtomicBool::new(false),
            fail_delete: AtomicBool::new(false),
            custom_error: "FailStore: injected fault".into(),
        }
    }

    pub fn with_custom_error(mut self, msg: impl Into<String>) -> Self {
        self.custom_error = msg.into();
        self
    }

    pub fn with_fail_get(self, fail: bool) -> Self {
        self.fail_get.store(fail, Ordering::Release);
        if !fail {
            *self.fail_get_nth.lock().unwrap() = None;
        }
        self
    }

    pub fn with_fail_get_range(self, fail: bool) -> Self {
        self.fail_get_range.store(fail, Ordering::Release);
        self
    }

    pub fn with_fail_put(self, fail: bool) -> Self {
        self.fail_put.store(fail, Ordering::Release);
        self
    }

    pub fn with_fail_list(self, fail: bool) -> Self {
        self.fail_list.store(fail, Ordering::Release);
        self
    }

    pub fn with_fail_file_size(self, fail: bool) -> Self {
        self.fail_file_size.store(fail, Ordering::Release);
        self
    }

    pub fn with_fail_exists(self, fail: bool) -> Self {
        self.fail_exists.store(fail, Ordering::Release);
        self
    }

    pub fn with_fail_delete(self, fail: bool) -> Self {
        self.fail_delete.store(fail, Ordering::Release);
        self
    }

    pub fn with_fail_all(self, fail: bool) -> Self {
        self.with_fail_get(fail)
            .with_fail_get_range(fail)
            .with_fail_put(fail)
            .with_fail_list(fail)
            .with_fail_file_size(fail)
            .with_fail_exists(fail)
            .with_fail_delete(fail)
    }

    /// Set fail_get so only the Nth get() call fails (1-indexed).
    /// All other calls succeed. Resets after the fail (call N+1 succeeds).
    /// Panics if `n` is 0.
    pub fn with_fail_get_nth(self, n: u64) -> Self {
        assert!(n > 0, "with_fail_get_nth: n must be >= 1, got {n}");
        self.fail_get.store(true, Ordering::Release);
        *self.fail_get_nth.lock().unwrap() = Some(n);
        self
    }

    pub fn set_fail_get(&self, fail: bool) {
        self.fail_get.store(fail, Ordering::Release);
        if !fail {
            *self.fail_get_nth.lock().unwrap() = None;
        }
    }

    pub fn set_fail_put(&self, fail: bool) {
        self.fail_put.store(fail, Ordering::Release);
    }

    fn err(&self) -> AilakeError {
        AilakeError::Store(self.custom_error.clone())
    }

    fn should_fail(flag: &AtomicBool) -> bool {
        flag.load(Ordering::Acquire)
    }

    fn should_fail_get(&self) -> bool {
        if !self.fail_get.load(Ordering::Acquire) {
            return false;
        }
        let mut nth = self.fail_get_nth.lock().unwrap();
        match *nth {
            Some(1) => {
                // This call is the one that should fail — consume the counter and
                // clear fail_get so the very next call succeeds, matching the
                // "resets after the fail" contract documented on `with_fail_get_nth`.
                *nth = None;
                self.fail_get.store(false, Ordering::Release);
                true
            }
            Some(n) => {
                // Decrement the counter, don't fail yet
                *nth = Some(n - 1);
                false
            }
            None => {
                // No nth counter — fail every call
                true
            }
        }
    }
}

#[async_trait]
impl Store for FailStore {
    async fn get(&self, path: &str) -> AilakeResult<Bytes> {
        if self.should_fail_get() {
            return Err(self.err());
        }
        self.inner.get(path).await
    }

    async fn get_range(&self, path: &str, range: Range<u64>) -> AilakeResult<Bytes> {
        if Self::should_fail(&self.fail_get_range) {
            return Err(self.err());
        }
        self.inner.get_range(path, range).await
    }

    async fn put(&self, path: &str, data: Bytes) -> AilakeResult<()> {
        if Self::should_fail(&self.fail_put) {
            return Err(self.err());
        }
        self.inner.put(path, data).await
    }

    async fn list(&self, prefix: &str) -> AilakeResult<Vec<String>> {
        if Self::should_fail(&self.fail_list) {
            return Err(self.err());
        }
        self.inner.list(prefix).await
    }

    async fn file_size(&self, path: &str) -> AilakeResult<u64> {
        if Self::should_fail(&self.fail_file_size) {
            return Err(self.err());
        }
        self.inner.file_size(path).await
    }

    async fn exists(&self, path: &str) -> AilakeResult<bool> {
        if Self::should_fail(&self.fail_exists) {
            return Err(self.err());
        }
        self.inner.exists(path).await
    }

    async fn delete(&self, path: &str) -> AilakeResult<()> {
        if Self::should_fail(&self.fail_delete) {
            return Err(self.err());
        }
        self.inner.delete(path).await
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::local::LocalStore;
    use tempfile::TempDir;

    #[tokio::test]
    async fn passthrough_when_no_faults() {
        let dir = TempDir::new().unwrap();
        let inner = LocalStore::new(dir.path());
        let store = FailStore::new(inner);
        store.put("ok.bin", Bytes::from("data")).await.unwrap();
        let got = store.get("ok.bin").await.unwrap();
        assert_eq!(got.as_ref(), b"data");
    }

    #[tokio::test]
    async fn fail_get_returns_error() {
        let dir = TempDir::new().unwrap();
        let inner = LocalStore::new(dir.path());
        let store = FailStore::new(inner).with_fail_get(true);
        let err = store.get("any.bin").await.unwrap_err();
        assert!(format!("{:?}", err).contains("injected fault"));
    }

    #[tokio::test]
    async fn fail_put_returns_error() {
        let dir = TempDir::new().unwrap();
        let inner = LocalStore::new(dir.path());
        let store = FailStore::new(inner).with_fail_put(true);
        let err = store.put("x.bin", Bytes::from("x")).await.unwrap_err();
        assert!(format!("{:?}", err).contains("injected fault"));
    }

    #[tokio::test]
    async fn fail_get_nth_only_fails_on_nth_call() {
        let dir = TempDir::new().unwrap();
        let inner = LocalStore::new(dir.path());
        let store = FailStore::new(inner).with_fail_get_nth(2);

        // First call succeeds
        store.put("nth.bin", Bytes::from("data")).await.unwrap();
        let first = store.get("nth.bin").await;
        assert!(first.is_ok(), "first get should succeed");

        // Second call fails
        let second = store.get("nth.bin").await;
        assert!(second.is_err(), "second get should fail");
        assert!(format!("{:?}", second).contains("injected fault"));

        // Third call succeeds — the nth failure resets fail_get automatically,
        // matching the "resets after the fail" contract on `with_fail_get_nth`.
        let third = store.get("nth.bin").await;
        assert!(
            third.is_ok(),
            "third get should succeed (auto-reset after nth fail)"
        );
    }

    #[tokio::test]
    async fn fail_get_nth_many_succeed_after() {
        let dir = TempDir::new().unwrap();
        let inner = LocalStore::new(dir.path());
        let store = FailStore::new(inner).with_fail_get_nth(3);

        // Succeed, succeed, fail, then auto-reset — no manual toggle needed.
        store.put("nth2.bin", Bytes::from("x")).await.unwrap();
        assert!(store.get("nth2.bin").await.is_ok(), "call 1");
        assert!(store.get("nth2.bin").await.is_ok(), "call 2");
        assert!(store.get("nth2.bin").await.is_err(), "call 3 (fail)");
        assert!(store.get("nth2.bin").await.is_ok(), "call 4 (auto-reset)");
    }

    #[tokio::test]
    async fn fail_get_range_returns_error() {
        let dir = TempDir::new().unwrap();
        let inner = LocalStore::new(dir.path());
        let store = FailStore::new(inner).with_fail_get_range(true);
        let err = store.get_range("x.bin", 0..1).await.unwrap_err();
        assert!(format!("{:?}", err).contains("injected fault"));
    }

    #[tokio::test]
    async fn fail_list_returns_error() {
        let dir = TempDir::new().unwrap();
        let inner = LocalStore::new(dir.path());
        let store = FailStore::new(inner).with_fail_list(true);
        let err = store.list("prefix").await.unwrap_err();
        assert!(format!("{:?}", err).contains("injected fault"));
    }

    #[tokio::test]
    async fn fail_all_fails_everything() {
        let dir = TempDir::new().unwrap();
        let inner = LocalStore::new(dir.path());
        let store = FailStore::new(inner).with_fail_all(true);
        assert!(store.get("x.bin").await.is_err());
        assert!(store.get_range("x.bin", 0..1).await.is_err());
        assert!(store.put("x.bin", Bytes::new()).await.is_err());
        assert!(store.list("x").await.is_err());
        assert!(store.file_size("x.bin").await.is_err());
        assert!(store.exists("x.bin").await.is_err());
        assert!(store.delete("x.bin").await.is_err());
    }

    #[tokio::test]
    async fn dynamic_toggle() {
        let dir = TempDir::new().unwrap();
        let inner = LocalStore::new(dir.path());
        let store = FailStore::new(inner);
        store.put("tog.bin", Bytes::from("x")).await.unwrap();

        // Enable failure
        store.set_fail_get(true);
        assert!(store.get("tog.bin").await.is_err());

        // Disable
        store.set_fail_get(false);
        assert!(store.get("tog.bin").await.is_ok());
    }

    #[tokio::test]
    async fn custom_error_message() {
        let dir = TempDir::new().unwrap();
        let inner = LocalStore::new(dir.path());
        let store = FailStore::new(inner)
            .with_custom_error("custom failure")
            .with_fail_get(true);
        let err = store.get("x.bin").await.unwrap_err();
        let msg = format!("{:?}", err);
        assert!(msg.contains("custom failure"), "got: {msg}");
    }
}
