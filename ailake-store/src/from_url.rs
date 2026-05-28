// SPDX-License-Identifier: MIT OR Apache-2.0
use std::sync::Arc;

use ailake_core::{AilakeError, AilakeResult};

use crate::store::Store;

/// Build a [`Store`] from a URL, using environment-based credentials for each cloud.
///
/// Supported schemes:
///
/// | URL | Backend | Credentials |
/// |-----|---------|-------------|
/// | `s3://bucket/prefix` | S3 | `Default` (env → `~/.aws` → IMDSv2 → WebIdentity) |
/// | `s3a://bucket/prefix` | S3 | same |
/// | `gs://bucket/prefix` | GCS | `ApplicationDefault` (`GOOGLE_APPLICATION_CREDENTIALS` → metadata) |
/// | `az://container/prefix` | Azure Blob | `ManagedIdentity` (IMDS); account from `AZURE_STORAGE_ACCOUNT_NAME` env |
/// | `/path` or `file:///path` | LocalStore | — |
///
/// Each cloud scheme requires the corresponding feature flag:
/// `store-s3`, `store-gcs`, `store-azure`.
///
/// For explicit credential config (static keys, IRSA, service-account JSON, client secrets)
/// use the typed builders: [`s3_store`](crate::s3::s3_store),
/// [`gcs_store`](crate::gcs::gcs_store), [`azure_store`](crate::azure::azure_store).
pub fn store_from_url(url: &str) -> AilakeResult<Arc<dyn Store>> {
    if let Some(rest) = url
        .strip_prefix("s3://")
        .or_else(|| url.strip_prefix("s3a://"))
    {
        return s3_from_rest(rest);
    }
    if let Some(rest) = url.strip_prefix("gs://") {
        return gcs_from_rest(rest);
    }
    if let Some(rest) = url.strip_prefix("az://") {
        return azure_from_rest(rest);
    }
    if url.starts_with('/') || url.starts_with("file://") {
        let path = url.strip_prefix("file://").unwrap_or(url);
        return Ok(Arc::new(crate::LocalStore::new(path)));
    }
    Err(AilakeError::Store(format!(
        "store_from_url: unsupported scheme in \"{url}\" \
         (supported: s3://, s3a://, gs://, az://, file://, /path)"
    )))
}

// ── s3:// ──────────────────────────────────────────────────────────────────

#[cfg(feature = "store-s3")]
fn s3_from_rest(rest: &str) -> AilakeResult<Arc<dyn Store>> {
    use crate::s3::{s3_store, S3Config, S3Credentials};

    let (bucket, prefix) = split_bucket_prefix(rest);
    let region = std::env::var("AWS_DEFAULT_REGION")
        .or_else(|_| std::env::var("AWS_REGION"))
        .unwrap_or_else(|_| "us-east-1".to_string());

    let backend = s3_store(
        S3Config {
            bucket,
            region,
            endpoint: None,
            allow_http: false,
            credentials: S3Credentials::Default,
        },
        prefix,
    )?;
    Ok(Arc::new(backend))
}

#[cfg(not(feature = "store-s3"))]
fn s3_from_rest(_rest: &str) -> AilakeResult<Arc<dyn Store>> {
    Err(AilakeError::Store(
        "store_from_url: s3:// requires feature \"store-s3\"".into(),
    ))
}

// ── gs:// ──────────────────────────────────────────────────────────────────

#[cfg(feature = "store-gcs")]
fn gcs_from_rest(rest: &str) -> AilakeResult<Arc<dyn Store>> {
    use crate::gcs::{gcs_store, GcsConfig, GcsCredentials};

    let (bucket, prefix) = split_bucket_prefix(rest);
    let backend = gcs_store(
        GcsConfig {
            bucket,
            credentials: GcsCredentials::ApplicationDefault,
        },
        prefix,
    )?;
    Ok(Arc::new(backend))
}

#[cfg(not(feature = "store-gcs"))]
fn gcs_from_rest(_rest: &str) -> AilakeResult<Arc<dyn Store>> {
    Err(AilakeError::Store(
        "store_from_url: gs:// requires feature \"store-gcs\"".into(),
    ))
}

// ── az:// ──────────────────────────────────────────────────────────────────

#[cfg(feature = "store-azure")]
fn azure_from_rest(rest: &str) -> AilakeResult<Arc<dyn Store>> {
    use crate::azure::{azure_store, AzureConfig, AzureCredentials};

    // az://container/prefix
    let (container, prefix) = split_bucket_prefix(rest);
    let account_name = std::env::var("AZURE_STORAGE_ACCOUNT_NAME").map_err(|_| {
        AilakeError::Store(
            "store_from_url: az:// requires AZURE_STORAGE_ACCOUNT_NAME env var".into(),
        )
    })?;
    let backend = azure_store(
        AzureConfig {
            account_name,
            container,
            credentials: AzureCredentials::ManagedIdentity { client_id: None },
        },
        prefix,
    )?;
    Ok(Arc::new(backend))
}

#[cfg(not(feature = "store-azure"))]
fn azure_from_rest(_rest: &str) -> AilakeResult<Arc<dyn Store>> {
    Err(AilakeError::Store(
        "store_from_url: az:// requires feature \"store-azure\"".into(),
    ))
}

// ── helpers ────────────────────────────────────────────────────────────────

/// `"bucket/some/prefix"` → `("bucket", "some/prefix")`
/// `"bucket"` → `("bucket", "")`
#[cfg(any(feature = "store-s3", feature = "store-gcs", feature = "store-azure"))]
fn split_bucket_prefix(s: &str) -> (String, String) {
    match s.find('/') {
        Some(i) => (s[..i].to_string(), s[i + 1..].to_string()),
        None => (s.to_string(), String::new()),
    }
}

#[cfg(all(
    test,
    any(feature = "store-s3", feature = "store-gcs", feature = "store-azure")
))]
mod tests {
    use super::split_bucket_prefix;

    #[test]
    fn split_with_prefix() {
        let (b, p) = split_bucket_prefix("my-bucket/warehouse/tbl");
        assert_eq!(b, "my-bucket");
        assert_eq!(p, "warehouse/tbl");
    }

    #[test]
    fn split_no_prefix() {
        let (b, p) = split_bucket_prefix("my-bucket");
        assert_eq!(b, "my-bucket");
        assert_eq!(p, "");
    }
}
