//! ailake-catalog — Iceberg catalog operations
//!
//! Implements CatalogProvider for all supported backends.
//! This is the only crate that reads/writes metadata.json and manifests.
//!
//! See docs/architecture/CATALOG_BACKENDS.md for backend details.

pub mod databricks;
pub mod hadoop;
pub mod metadata;
pub mod provider;
pub mod rest;
pub mod snapshot;

#[cfg(feature = "catalog-glue")]
pub mod glue;

#[cfg(feature = "catalog-nessie")]
pub mod nessie;

#[cfg(feature = "catalog-jdbc")]
pub mod jdbc;

pub use databricks::{databricks_aws, databricks_azure, databricks_gcp, DatabricksAuth};
pub use hadoop::HadoopCatalog;
pub use provider::{
    decode_centroid, make_data_file_entry, new_snapshot_id, CatalogProvider, DataFileEntry,
    NewSnapshot, SnapshotId, SnapshotOperation, TableIdent, TableMetadata, TableProperties,
    VectorIndexInfo,
};
pub use rest::{RestCatalog, RestCatalogAuth, RestCatalogConfig};

#[cfg(feature = "catalog-glue")]
pub use glue::{GlueCatalog, GlueCatalogConfig};

#[cfg(feature = "catalog-nessie")]
pub use nessie::{NessieBranch, NessieCatalog, NessieCatalogConfig};

#[cfg(feature = "catalog-jdbc")]
pub use jdbc::JdbcCatalog;
