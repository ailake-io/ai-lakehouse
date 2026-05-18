//! ailake-catalog — Iceberg catalog operations
//!
//! Implements CatalogProvider for all supported backends.
//! This is the only crate that reads/writes metadata.json and Avro manifests.
//!
//! See docs/architecture/CATALOG_BACKENDS.md for backend details.

pub mod provider;
pub mod metadata;
pub mod snapshot;
pub mod hadoop;
pub mod rest;

#[cfg(feature = "catalog-glue")]
pub mod glue;

#[cfg(feature = "catalog-nessie")]
pub mod nessie;

#[cfg(feature = "catalog-jdbc")]
pub mod jdbc;

pub use provider::{
    CatalogProvider, DataFileEntry, NewSnapshot, SnapshotId, SnapshotOperation,
    TableIdent, TableMetadata, TableProperties, decode_centroid, make_data_file_entry,
    new_snapshot_id,
};
pub use hadoop::HadoopCatalog;
pub use rest::RestCatalog;
