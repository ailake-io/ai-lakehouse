// SPDX-License-Identifier: MIT OR Apache-2.0
//! ailake-store — object storage abstraction
//!
//! Thin wrapper over object_store crate.
//! The get_range method is critical for partial S3 reads of the HNSW footer.
//!
//! ## Quick start
//!
//! Use `store_from_url` for env-based auth, or the typed builders for explicit credentials:
//!
//! ```rust,ignore
//! // URL-based (env credentials)
//! let store = ailake_store::store_from_url("s3://my-bucket/warehouse/my_table")?;
//!
//! // Explicit static credentials (dev / CI)
//! use ailake_store::s3::{s3_store, S3Config, S3Credentials};
//! let store = s3_store(S3Config {
//!     bucket: "my-bucket".into(),
//!     region: "us-east-1".into(),
//!     endpoint: None,
//!     allow_http: false,
//!     credentials: S3Credentials::Static {
//!         access_key_id: "AKIA...".into(),
//!         secret_access_key: "secret".into(),
//!         session_token: None,
//!     },
//! }, "warehouse/my_table/")?;
//! ```

pub mod fail_store;
pub mod from_url;
pub mod local;
pub mod object_store_backend;
pub mod store;

#[cfg(feature = "store-azure")]
pub mod azure;
#[cfg(feature = "store-gcs")]
pub mod gcs;
#[cfg(feature = "store-s3")]
pub mod s3;

pub use from_url::store_from_url;
pub use local::LocalStore;
pub use object_store_backend::ObjectStoreBackend;
pub use store::Store;

#[cfg(feature = "store-azure")]
pub use azure::{azure_store, AzureConfig, AzureCredentials};
#[cfg(feature = "store-gcs")]
pub use gcs::{gcs_store, GcsConfig, GcsCredentials};
#[cfg(feature = "store-s3")]
pub use s3::{s3_store, S3Config, S3Credentials};
