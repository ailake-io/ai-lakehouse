// SPDX-License-Identifier: MIT OR Apache-2.0
//! ailake-catalog — Iceberg catalog operations
//!
//! Implements CatalogProvider for all supported backends.
//! This is the only crate that reads/writes metadata.json and manifests.
//!
//! See docs/architecture/CATALOG_BACKENDS.md for backend details.

pub mod avro_manifest;
pub mod avro_raw;
#[cfg(feature = "rest-catalog")]
pub mod databricks;
pub mod hadoop;
pub mod metadata;
pub mod provider;
#[cfg(feature = "rest-catalog")]
pub mod rest;
pub mod snapshot;

#[cfg(feature = "catalog-glue")]
pub mod glue;

#[cfg(feature = "catalog-nessie")]
pub mod nessie;

#[cfg(feature = "catalog-jdbc")]
pub mod jdbc;

#[cfg(feature = "rest-catalog")]
pub use databricks::{databricks_aws, databricks_azure, databricks_gcp, DatabricksAuth};
pub use hadoop::HadoopCatalog;
pub use provider::{
    decode_centroid, encode_centroid_b64, make_data_file_entry, make_data_file_entry_indexing,
    make_multi_column_data_file_entry, new_snapshot_id, CatalogProvider, DataFileEntry,
    ExtraVectorIndex, IcebergSchemaUpdate, IndexStatus, NewSnapshot, SnapshotId, SnapshotOperation,
    TableIdent, TableMetadata, TableProperties, VectorIndexInfo,
};
#[cfg(feature = "rest-catalog")]
pub use rest::{RestCatalog, RestCatalogAuth, RestCatalogConfig};

#[cfg(feature = "catalog-glue")]
pub use glue::{GlueCatalog, GlueCatalogConfig};

#[cfg(feature = "catalog-nessie")]
pub use nessie::{NessieBranch, NessieCatalog, NessieCatalogConfig};

#[cfg(feature = "catalog-jdbc")]
pub use jdbc::JdbcCatalog;
