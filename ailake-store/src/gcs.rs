use std::sync::Arc;

use ailake_core::{AilakeError, AilakeResult};
use object_store::gcp::GoogleCloudStorageBuilder;

use crate::ObjectStoreBackend;

pub enum GcsCredentials {
    /// Path to a JSON service account key file.
    ServiceAccountFile(String),
    /// Inline JSON service account key (from secrets manager or env var).
    ServiceAccountJson(String),
    /// Application Default Credentials: reads `GOOGLE_APPLICATION_CREDENTIALS` env var
    /// (path to a key file or Workload Identity Federation config), then falls back to
    /// the GCE metadata server — covers GKE Workload Identity and Cloud Run automatically.
    ApplicationDefault,
}

pub struct GcsConfig {
    pub bucket: String,
    pub credentials: GcsCredentials,
}

/// Build an [`ObjectStoreBackend`] pointing at a GCS bucket.
///
/// `prefix` is prepended to every path — set to the table root key
/// (e.g. `"warehouse/my_table/"`) so callers use relative paths.
#[cfg(feature = "store-gcs")]
pub fn gcs_store(config: GcsConfig, prefix: impl Into<String>) -> AilakeResult<ObjectStoreBackend> {
    let mut b = GoogleCloudStorageBuilder::new().with_bucket_name(&config.bucket);

    match config.credentials {
        GcsCredentials::ServiceAccountFile(path) => {
            b = b.with_service_account_path(path);
        }
        GcsCredentials::ServiceAccountJson(json) => {
            b = b.with_service_account_key(json);
        }
        GcsCredentials::ApplicationDefault => {
            // object_store reads GOOGLE_APPLICATION_CREDENTIALS automatically;
            // if absent, falls to GCE metadata server (Workload Identity).
        }
    }

    let store = b.build().map_err(|e| AilakeError::Store(e.to_string()))?;
    Ok(ObjectStoreBackend::new(Arc::new(store), prefix))
}
