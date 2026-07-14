// SPDX-License-Identifier: MIT OR Apache-2.0
//! ailake-catalog — Iceberg catalog operations
//!
//! Implements CatalogProvider for all supported backends.
//! This is the only crate that reads/writes metadata.json and manifests.
//!
//! See docs/architecture/CATALOG_BACKENDS.md for backend details.

pub mod avro_manifest;
pub mod avro_raw;
pub mod column_stats;
#[cfg(feature = "rest-catalog")]
pub mod databricks;
pub mod hadoop;
mod manifest_commit;
pub mod metadata;
pub mod provider;
pub mod puffin;
#[cfg(feature = "rest-catalog")]
pub mod rest;
pub mod schema_evolution;
pub mod snapshot;

#[cfg(feature = "catalog-glue")]
pub mod glue;

#[cfg(feature = "catalog-nessie")]
pub mod nessie;

#[cfg(feature = "catalog-jdbc")]
pub mod jdbc;

#[cfg(feature = "catalog-ducklake")]
pub mod ducklake;

pub use avro_manifest::{
    build_manifest_entry_schema, read_equality_delete_values, write_equality_delete_avro,
    write_equality_delete_manifest, write_manifest_list_multi_typed, write_partition_stats_parquet,
};
pub use column_stats::{extract_column_stats, FieldStats};
#[cfg(feature = "rest-catalog")]
pub use databricks::{databricks_aws, databricks_azure, databricks_gcp, DatabricksAuth};
pub use hadoop::HadoopCatalog;
pub use metadata::{BlobRef, IcebergPartitionStatsRef, IcebergStatisticsRef};
pub use provider::{
    decode_centroid, encode_centroid_b64, make_data_file_entry, make_data_file_entry_indexing,
    make_multi_column_data_file_entry, new_snapshot_id, CatalogProvider, DataFileEntry,
    EqualityDeleteFile, ExtraVectorIndex, IcebergSchemaUpdate, IndexStatus, NewSnapshot,
    PartitionField, PartitionSpec, SchemaField, SnapshotId, SnapshotOperation, TableIdent,
    TableMetadata, TableProperties, VectorIndexInfo,
};
pub use puffin::{
    AilakePuffinReader, AilakePuffinWriter, BM25BloomEntry, VectorStatEntry, BLOB_TYPE_BM25_BLOOM,
    BLOB_TYPE_VECTOR_STATS,
};
#[cfg(feature = "rest-catalog")]
pub use rest::{RestCatalog, RestCatalogAuth, RestCatalogConfig};
pub use schema_evolution::{AddColumnRequest, RenameColumnRequest, SchemaEvolution};

#[cfg(feature = "catalog-glue")]
pub use glue::{GlueCatalog, GlueCatalogConfig};

#[cfg(feature = "catalog-nessie")]
pub use nessie::{NessieBranch, NessieCatalog, NessieCatalogConfig};

#[cfg(feature = "catalog-jdbc")]
pub use jdbc::JdbcCatalog;

#[cfg(feature = "catalog-ducklake")]
pub use ducklake::DuckLakeCatalog;
