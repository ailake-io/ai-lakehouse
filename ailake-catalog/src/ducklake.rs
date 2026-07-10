// SPDX-License-Identifier: MIT OR Apache-2.0
// DuckLakeCatalog: stores Iceberg-equivalent metadata in a real DuckLake catalog,
// driven entirely through the real DuckDB `ducklake` extension (no hand-rolled
// catalog DDL — see docs/guides/DUCKLAKE_CATALOG.md for the source-level research
// this is built on).
//
// Design:
//   - The `ducklake` extension owns everything DuckLake-native: schemas, tables,
//     columns, and snapshots. We only ever touch it through sanctioned public SQL:
//     CREATE/ALTER/DROP TABLE, `CALL ducklake_add_data_files(...)`,
//     `ducklake_list_files(...)`, and plain `DELETE FROM lake.tbl WHERE filename = ?`
//     (DuckLake exposes `filename` as a real, filterable virtual column — confirmed
//     by reading `DuckLakeTableEntry::GetVirtualColumns`/`GetRowIdColumns` in
//     duckdb/ducklake). That row-predicate DELETE is the correct way to make a
//     file's rows logically vanish for *any* DuckLake reader — but (also confirmed
//     against a live `ducklake` extension, not just docs) it only attaches a
//     deletion vector: the file keeps showing up in `ducklake_list_files` until a
//     maintenance pass (`ducklake_expire_snapshots` + `ducklake_cleanup_files`)
//     reclaims it. There is no sanctioned "drop this one file, right now" primitive.
//   - AI-Lake's own per-file vector metadata (centroid, radius, HNSW offset/len,
//     index status, embedding model, etc.) never fits DuckLake's fixed
//     `ducklake_data_file` schema, so it lives in a sidecar table
//     (`main.ailake_vector_index`) in the *same* DuckDB connection but outside the
//     `ducklake:` attachment — a plain table, no DuckLake versioning overhead.
//   - Because DuckLake's own file list can't be used to tell "still active" apart
//     from "logically deleted but not yet reclaimed", the sidecar's own `active`
//     flag is authoritative for `list_files`. `ducklake_list_files()` is only
//     consulted to catch "foreign" files: paths DuckLake knows about that have no
//     sidecar row *at all* (written by a generic DuckDB/DuckLake client, never
//     through AI-Lake) — those come back with `centroid_b64: None`, i.e.
//     `DataFileEntry::is_foreign() == true`, the same "foreign write" contract
//     already established for the Iceberg backends (ADR-018, CLAUDE.md §5A).
//
// Known v1 limitations (see docs/guides/DUCKLAKE_CATALOG.md):
//   - Not atomic across the `lake` attachment and the `main` sidecar table: DuckDB
//     refuses to write to two attached databases in one transaction ("a single
//     transaction can only write to a single attached database"). `commit_snapshot`
//     and `evolve_schema` therefore commit in two phases (`lake` first, `main`
//     second). A crash between the two only ever degrades gracefully — a
//     just-written file looks "foreign" (see below) until the second phase catches
//     up, and an orphaned sidecar row for an already-retired file is never
//     surfaced — never wrong or corrupt data. See the comment on `commit_snapshot`.
//   - Retired files are not physically reclaimed by this module — their bytes and
//     DuckLake catalog rows stick around (with a deletion vector attached) until an
//     operator runs DuckLake's own `ducklake_expire_snapshots` /
//     `ducklake_cleanup_files` maintenance calls. Same class of follow-up cleanup
//     Iceberg's `expire_snapshots` requires; not wired into this module to avoid
//     guessing at retention-window semantics we haven't verified against a live
//     multi-snapshot scenario.
//   - Single-writer: the metadata catalog is a local DuckDB/SQLite file, so only one
//     process should write to a given table at a time (same class of constraint as
//     SQLite itself). Multi-writer production use needs a Postgres-backed DuckLake
//     catalog — out of scope for this phase.
//   - `list_files(_, Some(snapshot_id))` only supports the table's *current*
//     snapshot id (returned by `load_table`) — arbitrary point-in-time time-travel
//     isn't wired up. No caller in this codebase requests anything else.
//   - The `ducklake` extension is not bundled; `connect()` runs `INSTALL ducklake;
//     LOAD ducklake;` which fetches it from DuckDB's extension repository on first
//     use. Needs network access (or a pre-populated extension directory) once per
//     machine.
//   - Table columns beyond the primary vector column are declared lazily via
//     `evolve_schema`/`add_vector_column` (`ALTER TABLE ... ADD COLUMN`), mirroring
//     the other backends. Physical Parquet files may carry columns not yet declared
//     in DuckLake — `ducklake_add_data_files(..., ignore_extra_columns => true)`
//     accepts this, but those columns aren't selectable through plain DuckDB SQL
//     until declared.

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use ailake_core::{AilakeError, AilakeResult};
use async_trait::async_trait;
use duckdb::Connection;
use tokio::sync::Mutex as AsyncMutex;
use uuid::Uuid;

use crate::provider::{
    CatalogProvider, DataFileEntry, DeletionVector, EqualityDeleteFile, ExtraVectorIndex,
    IndexStatus, NewSnapshot, SnapshotId, SnapshotOperation, TableIdent, TableMetadata,
    TableProperties,
};
use crate::schema_evolution::SchemaEvolution;

pub struct DuckLakeCatalog {
    conn: Arc<AsyncMutex<Connection>>,
    catalog_alias: String,
    warehouse: String,
}

fn quote_ident(name: &str) -> String {
    format!("\"{}\"", name.replace('"', "\"\""))
}

fn table_key(table: &TableIdent) -> String {
    format!("{}.{}", table.namespace, table.name)
}

fn cat_err(context: &str, e: impl std::fmt::Display) -> AilakeError {
    AilakeError::Catalog(format!("DuckLake {context}: {e}"))
}

fn index_status_to_str(s: &IndexStatus) -> &'static str {
    match s {
        IndexStatus::Ready => "ready",
        IndexStatus::Indexing => "indexing",
        IndexStatus::Failed => "failed",
    }
}

fn index_status_from_str(s: &str) -> IndexStatus {
    match s {
        "indexing" => IndexStatus::Indexing,
        "failed" => IndexStatus::Failed,
        _ => IndexStatus::Ready,
    }
}

/// Map an Iceberg type string (as used by `SchemaEvolution`/`AddColumnRequest`) to
/// the closest DuckDB SQL type. Unknown/complex types fall back to VARCHAR, same
/// safe-default policy `SchemaFiller` uses for the Iceberg backends.
fn iceberg_type_to_duckdb(t: &str) -> String {
    match t {
        "int" => "INTEGER".to_string(),
        "long" => "BIGINT".to_string(),
        "float" => "FLOAT".to_string(),
        "double" => "DOUBLE".to_string(),
        "boolean" => "BOOLEAN".to_string(),
        "string" => "VARCHAR".to_string(),
        "date" => "DATE".to_string(),
        "timestamp" => "TIMESTAMP".to_string(),
        "timestamptz" => "TIMESTAMPTZ".to_string(),
        "binary" => "BLOB".to_string(),
        "uuid" => "UUID".to_string(),
        _ => "VARCHAR".to_string(),
    }
}

impl DuckLakeCatalog {
    /// Connect to a DuckLake-backed catalog and ensure the sidecar tables exist.
    ///
    /// `root_db_path`: local DuckDB file that owns AI-Lake's own sidecar tables
    /// (`main.ailake_*`) — separate from DuckLake's own metadata store.
    /// `ducklake_meta_path`: path passed to `ATTACH 'ducklake:<path>' AS <alias>`
    /// (a DuckDB-backed DuckLake metadata catalog — Postgres-as-metadata is a
    /// stretch goal, out of scope here).
    /// `data_path`: directory DuckLake writes/expects table data files under
    /// (`DATA_PATH` attach option).
    pub async fn connect(
        root_db_path: &str,
        ducklake_meta_path: &str,
        data_path: &str,
        catalog_alias: &str,
        warehouse: &str,
    ) -> AilakeResult<Self> {
        let root_db_path = root_db_path.to_string();
        let ducklake_meta_path = ducklake_meta_path.to_string();
        let data_path = data_path.to_string();
        let alias = catalog_alias.to_string();

        let conn = tokio::task::spawn_blocking(move || -> AilakeResult<Connection> {
            let conn =
                Connection::open(&root_db_path).map_err(|e| cat_err("connect (root db)", e))?;
            conn.execute_batch("INSTALL ducklake; LOAD ducklake;")
                .map_err(|e| cat_err("extension load", e))?;
            let attach_sql = format!(
                "ATTACH 'ducklake:{}' AS {} (DATA_PATH '{}');",
                ducklake_meta_path.replace('\'', "''"),
                quote_ident(&alias),
                data_path.replace('\'', "''"),
            );
            conn.execute_batch(&attach_sql)
                .map_err(|e| cat_err("attach", e))?;
            migrate_sidecar_tables(&conn)?;
            Ok(conn)
        })
        .await
        .map_err(|e| cat_err("connect task", e))??;

        Ok(Self {
            conn: Arc::new(AsyncMutex::new(conn)),
            catalog_alias: catalog_alias.to_string(),
            warehouse: warehouse.trim_end_matches('/').to_string(),
        })
    }

    fn qualified_table(&self, table: &TableIdent) -> String {
        format!(
            "{}.{}.{}",
            quote_ident(&self.catalog_alias),
            quote_ident(&table.namespace),
            quote_ident(&table.name)
        )
    }

    fn table_root(&self, table: &TableIdent) -> String {
        format!("{}/{}/{}", self.warehouse, table.namespace, table.name)
    }
}

fn migrate_sidecar_tables(conn: &Connection) -> AilakeResult<()> {
    conn.execute_batch(
        r#"
        CREATE TABLE IF NOT EXISTS main.ailake_tables (
            table_key         VARCHAR PRIMARY KEY,
            namespace         VARCHAR NOT NULL,
            table_name        VARCHAR NOT NULL,
            table_uuid        VARCHAR NOT NULL,
            format_version    INTEGER NOT NULL,
            location          VARCHAR NOT NULL,
            last_snapshot_id  BIGINT
        );

        CREATE TABLE IF NOT EXISTS main.ailake_table_props (
            table_key VARCHAR NOT NULL,
            key       VARCHAR NOT NULL,
            value     VARCHAR NOT NULL,
            PRIMARY KEY (table_key, key)
        );

        CREATE TABLE IF NOT EXISTS main.ailake_vector_index (
            table_key                 VARCHAR NOT NULL,
            path                      VARCHAR NOT NULL,
            record_count              BIGINT NOT NULL,
            file_size_bytes           BIGINT NOT NULL,
            centroid_b64              VARCHAR,
            radius                    DOUBLE,
            hnsw_offset               BIGINT,
            hnsw_len                  BIGINT,
            vector_column             VARCHAR,
            vector_dim                INTEGER,
            extra_vector_indexes_json VARCHAR,
            index_status              VARCHAR NOT NULL DEFAULT 'ready',
            index_error               VARCHAR,
            batch_id                  VARCHAR,
            embedding_model           VARCHAR,
            partition_value           VARCHAR,
            deletion_vector_json      VARCHAR,
            first_row_id              BIGINT,
            active                    BOOLEAN NOT NULL DEFAULT true,
            PRIMARY KEY (table_key, path)
        );

        CREATE TABLE IF NOT EXISTS main.ailake_equality_deletes (
            table_key          VARCHAR NOT NULL,
            path               VARCHAR NOT NULL,
            equality_ids_json  VARCHAR NOT NULL,
            record_count       BIGINT NOT NULL,
            file_size_bytes    BIGINT NOT NULL,
            PRIMARY KEY (table_key, path)
        );
        "#,
    )
    .map_err(|e| cat_err("sidecar migrate", e))
}

// ── DataFileEntry <-> sidecar row ──────────────────────────────────────────────

fn upsert_file_entry(conn: &Connection, key: &str, e: &DataFileEntry) -> AilakeResult<()> {
    let extra_json = serde_json::to_string(&e.extra_vector_indexes)
        .map_err(|err| cat_err("serialize extra_vector_indexes", err))?;
    let dv_json = match &e.deletion_vector {
        Some(dv) => Some(
            serde_json::to_string(dv).map_err(|err| cat_err("serialize deletion_vector", err))?,
        ),
        None => None,
    };
    conn.execute(
        r#"
        INSERT INTO main.ailake_vector_index (
            table_key, path, record_count, file_size_bytes, centroid_b64, radius,
            hnsw_offset, hnsw_len, vector_column, vector_dim, extra_vector_indexes_json,
            index_status, index_error, batch_id, embedding_model, partition_value,
            deletion_vector_json, first_row_id, active
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, true)
        ON CONFLICT (table_key, path) DO UPDATE SET
            record_count = excluded.record_count,
            file_size_bytes = excluded.file_size_bytes,
            centroid_b64 = excluded.centroid_b64,
            radius = excluded.radius,
            hnsw_offset = excluded.hnsw_offset,
            hnsw_len = excluded.hnsw_len,
            vector_column = excluded.vector_column,
            vector_dim = excluded.vector_dim,
            extra_vector_indexes_json = excluded.extra_vector_indexes_json,
            index_status = excluded.index_status,
            index_error = excluded.index_error,
            batch_id = excluded.batch_id,
            embedding_model = excluded.embedding_model,
            partition_value = excluded.partition_value,
            deletion_vector_json = excluded.deletion_vector_json,
            first_row_id = excluded.first_row_id,
            active = true
        "#,
        duckdb::params![
            key,
            e.path,
            e.record_count as i64,
            e.file_size_bytes as i64,
            e.centroid_b64,
            e.radius.map(|r| r as f64),
            e.hnsw_offset.map(|v| v as i64),
            e.hnsw_len.map(|v| v as i64),
            e.vector_column,
            e.vector_dim.map(|v| v as i32),
            extra_json,
            index_status_to_str(&e.index_status),
            e.index_error,
            e.batch_id,
            e.embedding_model,
            e.partition_value,
            dv_json,
            e.first_row_id,
        ],
    )
    .map_err(|err| cat_err("upsert file entry", err))?;
    Ok(())
}

/// Soft-retire a file: AI-Lake stops considering it active. The underlying
/// DuckLake row-predicate `DELETE` (see `commit_snapshot`) already marks every row
/// in the file as logically deleted for *any* DuckLake reader — but DuckLake has no
/// sanctioned "drop this whole file now" primitive (row DELETE only attaches a
/// deletion vector; the file keeps showing up in `ducklake_list_files` until a
/// maintenance pass like `ducklake_expire_snapshots`/`ducklake_cleanup_files`
/// reclaims it — see module doc comment). So the sidecar row's `active` flag, not
/// deletion, is what `query_active_files` treats as authoritative for exclusion.
/// The row is kept (not deleted) so the foreign-file detection below can still
/// tell "AI-Lake retired this on purpose" apart from "AI-Lake never saw this path".
fn retire_file_entry(conn: &Connection, key: &str, path: &str) -> AilakeResult<()> {
    conn.execute(
        "UPDATE main.ailake_vector_index SET active = false WHERE table_key = ? AND path = ?",
        duckdb::params![key, path],
    )
    .map_err(|e| cat_err("retire file entry", e))?;
    Ok(())
}

/// Active files for a table. The sidecar table (`active = true` rows) is
/// authoritative for which files AI-Lake currently considers part of the table —
/// see `retire_file_entry` for why DuckLake's own file list can't be used for
/// exclusion. `ducklake_list_files` is only consulted to catch "foreign" files:
/// paths DuckLake knows about that AI-Lake has literally no sidecar row for at all
/// (written by a generic DuckDB/DuckLake client, never through AI-Lake) — those
/// come back with `centroid_b64: None`, matching `DataFileEntry::is_foreign()`,
/// the same contract already established for the Iceberg backends (ADR-018).
fn query_active_files(
    conn: &Connection,
    alias: &str,
    table: &TableIdent,
    key: &str,
) -> AilakeResult<Vec<DataFileEntry>> {
    let mut stmt = conn
        .prepare(
            "SELECT path, record_count, file_size_bytes, centroid_b64, radius, hnsw_offset,
                    hnsw_len, vector_column, vector_dim, extra_vector_indexes_json, index_status,
                    index_error, batch_id, embedding_model, partition_value, deletion_vector_json,
                    first_row_id
             FROM main.ailake_vector_index WHERE table_key = ? AND active = true",
        )
        .map_err(|e| cat_err("sidecar prepare", e))?;
    let mut out: Vec<DataFileEntry> = stmt
        .query_map(duckdb::params![key], row_to_entry)
        .map_err(|e| cat_err("sidecar query", e))?
        .collect::<Result<Vec<_>, _>>()
        .map_err(|e| cat_err("sidecar rows", e))?;

    let mut known_stmt = conn
        .prepare("SELECT path FROM main.ailake_vector_index WHERE table_key = ?")
        .map_err(|e| cat_err("known paths prepare", e))?;
    let known_paths: HashSet<String> = known_stmt
        .query_map(duckdb::params![key], |row| row.get::<_, String>(0))
        .map_err(|e| cat_err("known paths query", e))?
        .collect::<Result<HashSet<_>, _>>()
        .map_err(|e| cat_err("known paths rows", e))?;

    let list_files_sql = format!(
        "SELECT data_file, data_file_size_bytes FROM ducklake_list_files('{}', '{}', schema => '{}') WHERE data_file IS NOT NULL",
        alias.replace('\'', "''"),
        table.name.replace('\'', "''"),
        table.namespace.replace('\'', "''"),
    );
    let mut stmt = conn
        .prepare(&list_files_sql)
        .map_err(|e| cat_err("list_files prepare", e))?;
    let ducklake_files: Vec<(String, i64)> = stmt
        .query_map([], |row| {
            let path: String = row.get(0)?;
            let size: i64 = row.get(1)?;
            Ok((path, size))
        })
        .map_err(|e| cat_err("list_files query", e))?
        .collect::<Result<Vec<_>, _>>()
        .map_err(|e| cat_err("list_files rows", e))?;

    for (path, size) in ducklake_files {
        if !known_paths.contains(&path) {
            out.push(DataFileEntry {
                path,
                record_count: 0,
                file_size_bytes: size as u64,
                centroid_b64: None,
                radius: None,
                hnsw_offset: None,
                hnsw_len: None,
                vector_column: None,
                vector_dim: None,
                extra_vector_indexes: vec![],
                index_status: IndexStatus::Ready,
                index_error: None,
                batch_id: None,
                embedding_model: None,
                partition_value: None,
                deletion_vector: None,
                first_row_id: None,
            });
        }
    }
    Ok(out)
}

fn row_to_entry(row: &duckdb::Row) -> duckdb::Result<DataFileEntry> {
    let path: String = row.get(0)?;
    let record_count: i64 = row.get(1)?;
    let file_size_bytes: i64 = row.get(2)?;
    let centroid_b64: Option<String> = row.get(3)?;
    let radius: Option<f64> = row.get(4)?;
    let hnsw_offset: Option<i64> = row.get(5)?;
    let hnsw_len: Option<i64> = row.get(6)?;
    let vector_column: Option<String> = row.get(7)?;
    let vector_dim: Option<i32> = row.get(8)?;
    let extra_json: Option<String> = row.get(9)?;
    let index_status: String = row.get(10)?;
    let index_error: Option<String> = row.get(11)?;
    let batch_id: Option<String> = row.get(12)?;
    let embedding_model: Option<String> = row.get(13)?;
    let partition_value: Option<String> = row.get(14)?;
    let dv_json: Option<String> = row.get(15)?;
    let first_row_id: Option<i64> = row.get(16)?;

    let extra_vector_indexes: Vec<ExtraVectorIndex> = extra_json
        .as_deref()
        .and_then(|s| serde_json::from_str(s).ok())
        .unwrap_or_default();
    let deletion_vector: Option<DeletionVector> = dv_json
        .as_deref()
        .and_then(|s| serde_json::from_str(s).ok());

    Ok(DataFileEntry {
        path,
        record_count: record_count as u64,
        file_size_bytes: file_size_bytes as u64,
        centroid_b64,
        radius: radius.map(|r| r as f32),
        hnsw_offset: hnsw_offset.map(|v| v as u64),
        hnsw_len: hnsw_len.map(|v| v as u64),
        vector_column,
        vector_dim: vector_dim.map(|v| v as u32),
        extra_vector_indexes,
        index_status: index_status_from_str(&index_status),
        index_error,
        batch_id,
        embedding_model,
        partition_value,
        deletion_vector,
        first_row_id,
    })
}

// ── CatalogProvider ───────────────────────────────────────────────────────────

#[async_trait]
impl CatalogProvider for DuckLakeCatalog {
    async fn create_table(&self, name: &TableIdent, props: &TableProperties) -> AilakeResult<()> {
        let key = table_key(name);
        let location = self.table_root(name);
        let table_uuid = Uuid::new_v4().to_string();
        let vector_col = quote_ident(&props.policy.column_name);
        let full_table = self.qualified_table(name);
        let schema_ref = format!(
            "{}.{}",
            quote_ident(&self.catalog_alias),
            quote_ident(&name.namespace)
        );

        let mut extra_col_sql = String::new();
        if let Some(part_col) = &props.policy.partition_by {
            let ty = iceberg_type_to_duckdb(
                props
                    .partition_column_type
                    .as_deref()
                    .or(props.policy.partition_column_type.as_deref())
                    .unwrap_or("string"),
            );
            extra_col_sql = format!(", {} {}", quote_ident(part_col), ty);
        }

        let mut properties: HashMap<String, String> = HashMap::new();
        properties.insert("ailake.format-version".to_string(), "1".to_string());
        properties.insert(
            "ailake.vector-column".to_string(),
            props.policy.column_name.clone(),
        );
        properties.insert(
            "ailake.vector-dim".to_string(),
            props.policy.dim.to_string(),
        );
        properties.insert(
            "ailake.vector-metric".to_string(),
            format!("{:?}", props.policy.metric).to_lowercase(),
        );
        properties.insert(
            "ailake.vector-precision".to_string(),
            format!("{:?}", props.policy.precision).to_lowercase(),
        );
        if let Some(m) = props.policy.hnsw_m {
            properties.insert("ailake.hnsw-m".to_string(), m.to_string());
        }
        if let Some(ef) = props.policy.hnsw_ef_construction {
            properties.insert("ailake.hnsw-ef-construction".to_string(), ef.to_string());
        }
        if props.policy.pre_normalize {
            properties.insert("ailake.pre-normalize".to_string(), "true".to_string());
        }
        if let Some(modality) = props.policy.modality {
            properties.insert(
                format!("ailake.modality-{}", props.policy.column_name),
                modality.as_str().to_string(),
            );
        }
        if let Some(col) = &props.policy.partition_by {
            properties.insert("ailake.partition-by".to_string(), col.clone());
        }
        for (k, v) in &props.extra {
            properties.insert(k.clone(), v.clone());
        }

        let conn = self.conn.lock().await;
        conn.execute_batch(&format!("CREATE SCHEMA IF NOT EXISTS {schema_ref};"))
            .map_err(|e| cat_err("create schema", e))?;
        conn.execute_batch(&format!(
            "CREATE TABLE {full_table} ({vector_col} BLOB{extra_col_sql});"
        ))
        .map_err(|e| cat_err("create table", e))?;

        conn.execute(
            "INSERT INTO main.ailake_tables (table_key, namespace, table_name, table_uuid, format_version, location, last_snapshot_id)
             VALUES (?, ?, ?, ?, ?, ?, NULL)",
            duckdb::params![key, name.namespace, name.name, table_uuid, props.format_version as i32, location],
        )
        .map_err(|e| cat_err("register table", e))?;

        for (k, v) in &properties {
            conn.execute(
                "INSERT INTO main.ailake_table_props (table_key, key, value) VALUES (?, ?, ?)
                 ON CONFLICT (table_key, key) DO UPDATE SET value = excluded.value",
                duckdb::params![key, k, v],
            )
            .map_err(|e| cat_err("write table props", e))?;
        }
        Ok(())
    }

    async fn load_table(&self, name: &TableIdent) -> AilakeResult<TableMetadata> {
        let key = table_key(name);
        let conn = self.conn.lock().await;
        let (table_uuid, format_version, location, last_snapshot_id): (
            String,
            i32,
            String,
            Option<i64>,
        ) = conn
            .query_row(
                "SELECT table_uuid, format_version, location, last_snapshot_id
                 FROM main.ailake_tables WHERE table_key = ?",
                duckdb::params![key],
                |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?, row.get(3)?)),
            )
            .map_err(|_| {
                AilakeError::Catalog(format!("table not found: {}.{}", name.namespace, name.name))
            })?;

        let mut stmt = conn
            .prepare("SELECT key, value FROM main.ailake_table_props WHERE table_key = ?")
            .map_err(|e| cat_err("load props prepare", e))?;
        let properties: HashMap<String, String> = stmt
            .query_map(duckdb::params![key], |row| {
                let k: String = row.get(0)?;
                let v: String = row.get(1)?;
                Ok((k, v))
            })
            .map_err(|e| cat_err("load props query", e))?
            .collect::<Result<HashMap<_, _>, _>>()
            .map_err(|e| cat_err("load props rows", e))?;

        Ok(TableMetadata {
            table_uuid,
            format_version,
            location,
            properties,
            current_snapshot_id: last_snapshot_id,
            current_statistics_path: None,
            schema_fields: vec![],
            equality_delete_files: vec![],
            partition_spec: None,
        })
    }

    async fn commit_snapshot(
        &self,
        table: &TableIdent,
        snapshot: NewSnapshot,
    ) -> AilakeResult<SnapshotId> {
        let key = table_key(table);
        let snap_id = snapshot.snapshot_id;
        let full_table = self.qualified_table(table);
        let alias = self.catalog_alias.clone();

        let conn = self.conn.lock().await;

        // Diff old-vs-new file list for Overwrite/Replace (callers already rebuild
        // `snapshot.files` as the complete new state — same contract as
        // hadoop.rs/jdbc.rs). Append/Delete: `snapshot.files` are entries being
        // added on top of the current active set (Delete's real payload is
        // `equality_delete_files`, not a file removed from `files`).
        let (to_add, to_remove, metadata_only): (
            Vec<DataFileEntry>,
            Vec<String>,
            Vec<DataFileEntry>,
        ) = match snapshot.operation {
            SnapshotOperation::Append | SnapshotOperation::Delete => {
                (snapshot.files.clone(), vec![], vec![])
            }
            SnapshotOperation::Overwrite | SnapshotOperation::Replace => {
                let current = query_active_files(&conn, &alias, table, &key)?;
                let current_paths: HashSet<String> =
                    current.iter().map(|f| f.path.clone()).collect();
                let new_paths: HashSet<String> =
                    snapshot.files.iter().map(|f| f.path.clone()).collect();
                let removed: Vec<String> = current_paths.difference(&new_paths).cloned().collect();
                // Anything in the new state — whether the path already existed
                // (metadata-only patch, e.g. deferred index status / deletion
                // vector) or is brand new — gets upserted into the sidecar table.
                // Only genuinely new paths need `ducklake_add_data_files`.
                let added: Vec<DataFileEntry> = snapshot
                    .files
                    .iter()
                    .filter(|f| !current_paths.contains(&f.path))
                    .cloned()
                    .collect();
                let metadata_only: Vec<DataFileEntry> = snapshot
                    .files
                    .iter()
                    .filter(|f| current_paths.contains(&f.path))
                    .cloned()
                    .collect();
                (added, removed, metadata_only)
            }
        };

        // DuckDB forbids writing to more than one attached database within a single
        // transaction ("a single transaction can only write to a single attached
        // database"). `lake` (real DuckLake) and `main` (our sidecar) are separate
        // attachments, so this commits in two phases instead of one atomic
        // transaction. `lake` goes first — it's the source of truth for which files
        // are active. If phase 2 (sidecar) fails or the process dies between the
        // two, a just-added file is simply missing its vector metadata until repaired
        // (surfaces as `is_foreign() == true`, same degraded-but-safe path already
        // used for files written by generic DuckDB/DuckLake clients — see the module
        // doc comment) and a just-removed file's orphaned sidecar row is never
        // surfaced by `query_active_files` (it only iterates DuckLake's own active
        // list). Neither failure mode returns wrong or corrupt data.
        conn.execute_batch("BEGIN TRANSACTION;")
            .map_err(|e| cat_err("begin (lake)", e))?;
        let lake_result: AilakeResult<()> = (|| {
            for path in &to_remove {
                conn.execute(
                    &format!("DELETE FROM {full_table} WHERE filename = ?"),
                    duckdb::params![path],
                )
                .map_err(|e| cat_err("retire file", e))?;
            }
            for entry in &to_add {
                let call_sql = format!(
                    "CALL ducklake_add_data_files('{}', '{}', '{}', schema => '{}', ignore_extra_columns => true);",
                    alias.replace('\'', "''"),
                    table.name.replace('\'', "''"),
                    entry.path.replace('\'', "''"),
                    table.namespace.replace('\'', "''"),
                );
                conn.execute_batch(&call_sql)
                    .map_err(|e| cat_err("add_data_files", e))?;
            }
            Ok(())
        })();
        match lake_result {
            Ok(()) => conn
                .execute_batch("COMMIT;")
                .map_err(|e| cat_err("commit (lake)", e))?,
            Err(e) => {
                let _ = conn.execute_batch("ROLLBACK;");
                return Err(e);
            }
        }

        conn.execute_batch("BEGIN TRANSACTION;")
            .map_err(|e| cat_err("begin (sidecar)", e))?;
        let sidecar_result: AilakeResult<()> = (|| {
            for path in &to_remove {
                retire_file_entry(&conn, &key, path)?;
            }
            for entry in to_add.iter().chain(metadata_only.iter()) {
                upsert_file_entry(&conn, &key, entry)?;
            }
            for eq in &snapshot.equality_delete_files {
                upsert_equality_delete(&conn, &key, eq)?;
            }
            conn.execute(
                "UPDATE main.ailake_tables SET last_snapshot_id = ? WHERE table_key = ?",
                duckdb::params![snap_id, key],
            )
            .map_err(|e| cat_err("update snapshot ptr", e))?;
            for (k, v) in &snapshot.extra_properties {
                conn.execute(
                    "INSERT INTO main.ailake_table_props (table_key, key, value) VALUES (?, ?, ?)
                     ON CONFLICT (table_key, key) DO UPDATE SET value = excluded.value",
                    duckdb::params![key, k, v],
                )
                .map_err(|e| cat_err("write extra props", e))?;
            }
            Ok(())
        })();
        match sidecar_result {
            Ok(()) => {
                conn.execute_batch("COMMIT;")
                    .map_err(|e| cat_err("commit (sidecar)", e))?;
                Ok(snap_id)
            }
            Err(e) => {
                let _ = conn.execute_batch("ROLLBACK;");
                Err(e)
            }
        }
    }

    async fn list_files(
        &self,
        table: &TableIdent,
        snapshot_id: Option<SnapshotId>,
    ) -> AilakeResult<Vec<DataFileEntry>> {
        let key = table_key(table);
        let conn = self.conn.lock().await;
        if let Some(requested) = snapshot_id {
            let current: Option<i64> = conn
                .query_row(
                    "SELECT last_snapshot_id FROM main.ailake_tables WHERE table_key = ?",
                    duckdb::params![key],
                    |row| row.get(0),
                )
                .map_err(|_| {
                    AilakeError::Catalog(format!(
                        "table not found: {}.{}",
                        table.namespace, table.name
                    ))
                })?;
            if current != Some(requested) {
                return Err(AilakeError::Catalog(
                    "DuckLakeCatalog v1 only supports listing the current snapshot \
                     (arbitrary point-in-time time-travel isn't wired up yet)"
                        .to_string(),
                ));
            }
        }
        query_active_files(&conn, &self.catalog_alias, table, &key)
    }

    async fn drop_table(&self, name: &TableIdent) -> AilakeResult<()> {
        let key = table_key(name);
        let full_table = self.qualified_table(name);
        let conn = self.conn.lock().await;
        conn.execute_batch(&format!("DROP TABLE IF EXISTS {full_table};"))
            .map_err(|e| cat_err("drop table", e))?;
        conn.execute(
            "DELETE FROM main.ailake_vector_index WHERE table_key = ?",
            duckdb::params![key],
        )
        .map_err(|e| cat_err("drop sidecar vector rows", e))?;
        conn.execute(
            "DELETE FROM main.ailake_equality_deletes WHERE table_key = ?",
            duckdb::params![key],
        )
        .map_err(|e| cat_err("drop sidecar delete rows", e))?;
        conn.execute(
            "DELETE FROM main.ailake_table_props WHERE table_key = ?",
            duckdb::params![key],
        )
        .map_err(|e| cat_err("drop sidecar props", e))?;
        conn.execute(
            "DELETE FROM main.ailake_tables WHERE table_key = ?",
            duckdb::params![key],
        )
        .map_err(|e| cat_err("drop table registry row", e))?;
        Ok(())
    }

    async fn evolve_schema(
        &self,
        table: &TableIdent,
        evolution: SchemaEvolution,
    ) -> AilakeResult<i32> {
        let key = table_key(table);
        let full_table = self.qualified_table(table);
        let conn = self.conn.lock().await;

        // Two phases for the same reason as `commit_snapshot`: DuckDB won't let one
        // transaction write to both the `lake` attachment (ALTER TABLE) and `main`
        // (property sidecar).
        conn.execute_batch("BEGIN TRANSACTION;")
            .map_err(|e| cat_err("evolve begin (lake)", e))?;
        let ddl_result: AilakeResult<()> = (|| {
            for rename in &evolution.renames {
                conn.execute_batch(&format!(
                    "ALTER TABLE {full_table} RENAME COLUMN {} TO {};",
                    quote_ident(&rename.old_name),
                    quote_ident(&rename.new_name)
                ))
                .map_err(|e| cat_err("rename column", e))?;
            }
            for add in &evolution.adds {
                let ty = iceberg_type_to_duckdb(&add.iceberg_type);
                conn.execute_batch(&format!(
                    "ALTER TABLE {full_table} ADD COLUMN {} {};",
                    quote_ident(&add.name),
                    ty
                ))
                .map_err(|e| cat_err("add column", e))?;
            }
            Ok(())
        })();
        match ddl_result {
            Ok(()) => conn
                .execute_batch("COMMIT;")
                .map_err(|e| cat_err("evolve commit (lake)", e))?,
            Err(e) => {
                let _ = conn.execute_batch("ROLLBACK;");
                return Err(e);
            }
        }

        conn.execute_batch("BEGIN TRANSACTION;")
            .map_err(|e| cat_err("evolve begin (sidecar)", e))?;
        let result: AilakeResult<()> = (|| {
            for (k, v) in &evolution.extra_properties {
                conn.execute(
                    "INSERT INTO main.ailake_table_props (table_key, key, value) VALUES (?, ?, ?)
                     ON CONFLICT (table_key, key) DO UPDATE SET value = excluded.value",
                    duckdb::params![key, k, v],
                )
                .map_err(|e| cat_err("evolve write props", e))?;
            }
            Ok(())
        })();

        match result {
            Ok(()) => {
                conn.execute_batch("COMMIT;")
                    .map_err(|e| cat_err("evolve commit", e))?;
                // DuckLake tracks its own schema versioning internally; AI-Lake's
                // `schema-id` concept doesn't map onto it 1:1. Return a monotonic
                // stand-in derived from the sidecar table's prop count so callers
                // get a changing value across successive evolutions.
                let count: i64 = conn
                    .query_row(
                        "SELECT count(*) FROM main.ailake_table_props WHERE table_key = ?",
                        duckdb::params![key],
                        |row| row.get(0),
                    )
                    .unwrap_or(0);
                Ok(count as i32)
            }
            Err(e) => {
                let _ = conn.execute_batch("ROLLBACK;");
                Err(e)
            }
        }
    }

    async fn list_equality_deletes(
        &self,
        table: &TableIdent,
        _snapshot_id: Option<SnapshotId>,
    ) -> AilakeResult<Vec<EqualityDeleteFile>> {
        let key = table_key(table);
        let conn = self.conn.lock().await;
        let mut stmt = conn
            .prepare(
                "SELECT path, equality_ids_json, record_count, file_size_bytes
                 FROM main.ailake_equality_deletes WHERE table_key = ?",
            )
            .map_err(|e| cat_err("list eq deletes prepare", e))?;
        let out: Vec<EqualityDeleteFile> = stmt
            .query_map(duckdb::params![key], |row| {
                let path: String = row.get(0)?;
                let ids_json: String = row.get(1)?;
                let record_count: i64 = row.get(2)?;
                let file_size_bytes: i64 = row.get(3)?;
                Ok((path, ids_json, record_count, file_size_bytes))
            })
            .map_err(|e| cat_err("list eq deletes query", e))?
            .collect::<Result<Vec<_>, _>>()
            .map_err(|e| cat_err("list eq deletes rows", e))?
            .into_iter()
            .map(
                |(path, ids_json, record_count, file_size_bytes)| EqualityDeleteFile {
                    path,
                    equality_ids: serde_json::from_str(&ids_json).unwrap_or_default(),
                    record_count: record_count as u64,
                    file_size_bytes: file_size_bytes as u64,
                },
            )
            .collect();
        Ok(out)
    }
}

fn upsert_equality_delete(
    conn: &Connection,
    key: &str,
    eq: &EqualityDeleteFile,
) -> AilakeResult<()> {
    let ids_json = serde_json::to_string(&eq.equality_ids)
        .map_err(|e| cat_err("serialize equality_ids", e))?;
    conn.execute(
        "INSERT INTO main.ailake_equality_deletes (table_key, path, equality_ids_json, record_count, file_size_bytes)
         VALUES (?, ?, ?, ?, ?)
         ON CONFLICT (table_key, path) DO UPDATE SET
            equality_ids_json = excluded.equality_ids_json,
            record_count = excluded.record_count,
            file_size_bytes = excluded.file_size_bytes",
        duckdb::params![key, eq.path, ids_json, eq.record_count as i64, eq.file_size_bytes as i64],
    )
    .map_err(|e| cat_err("upsert equality delete", e))?;
    Ok(())
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn quote_ident_escapes_double_quotes() {
        assert_eq!(quote_ident("simple"), "\"simple\"");
        assert_eq!(quote_ident("weird\"name"), "\"weird\"\"name\"");
    }

    #[test]
    fn table_key_format() {
        let t = TableIdent::new("default", "docs");
        assert_eq!(table_key(&t), "default.docs");
    }

    #[cfg(feature = "catalog-ducklake")]
    mod live {
        use super::super::*;
        use crate::provider::{
            new_snapshot_id, DataFileEntry, IndexStatus, NewSnapshot, SnapshotOperation,
        };
        use ailake_core::{VectorMetric, VectorPrecision, VectorStoragePolicy};
        use tempfile::TempDir;

        fn policy() -> VectorStoragePolicy {
            VectorStoragePolicy {
                column_name: "embedding".into(),
                dim: 4,
                metric: VectorMetric::Cosine,
                precision: VectorPrecision::F16,
                pq: None,
                keep_raw_for_reranking: true,
                pre_normalize: false,
                hnsw_m: None,
                hnsw_ef_construction: None,
                ivf_residual: false,
                embedding_model: None,
                modality: None,
                partition_by: None,
                partition_value: None,
                partition_column_type: None,
                partition_fields: vec![],
            }
        }

        fn entry(path: &str, record_count: u64) -> DataFileEntry {
            DataFileEntry {
                path: path.to_string(),
                record_count,
                file_size_bytes: 1024,
                centroid_b64: Some("AACAPwAAAEAAAEBAAACAQA==".to_string()),
                radius: Some(0.3),
                hnsw_offset: Some(512),
                hnsw_len: Some(256),
                vector_column: Some("embedding".into()),
                vector_dim: Some(4),
                extra_vector_indexes: vec![],
                index_status: IndexStatus::Ready,
                index_error: None,
                batch_id: None,
                embedding_model: None,
                partition_value: None,
                deletion_vector: None,
                first_row_id: None,
            }
        }

        async fn write_source_parquet(conn_path: &str, out_path: &str, n: i64) {
            let conn = duckdb::Connection::open_in_memory().unwrap();
            conn.execute_batch(&format!(
                "COPY (SELECT (i::VARCHAR)::BLOB AS embedding FROM range({n}) t(i)) TO '{out_path}' (FORMAT PARQUET);"
            ))
            .unwrap();
            let _ = conn_path;
        }

        #[tokio::test]
        async fn create_insert_search_drop_roundtrip() {
            let dir = TempDir::new().unwrap();
            let root_db = dir.path().join("ailake_root.db");
            let meta_db = dir.path().join("ducklake_meta.db");
            let data_path = dir.path().join("data");
            std::fs::create_dir_all(&data_path).unwrap();

            let catalog = DuckLakeCatalog::connect(
                root_db.to_str().unwrap(),
                meta_db.to_str().unwrap(),
                data_path.to_str().unwrap(),
                "lake",
                dir.path().to_str().unwrap(),
            )
            .await
            .unwrap();

            let table = TableIdent::new("default", "docs");
            let props = TableProperties {
                policy: policy(),
                extra: HashMap::new(),
                format_version: 2,
                partition_column_type: None,
            };
            catalog.create_table(&table, &props).await.unwrap();

            let meta = catalog.load_table(&table).await.unwrap();
            assert_eq!(meta.format_version, 2);
            assert!(meta.properties.contains_key("ailake.vector-column"));

            let file1 = data_path.join("part-00001.parquet");
            write_source_parquet(meta_db.to_str().unwrap(), file1.to_str().unwrap(), 10).await;

            let snap = NewSnapshot {
                snapshot_id: new_snapshot_id(),
                parent_snapshot_id: None,
                files: vec![entry(file1.to_str().unwrap(), 10)],
                operation: SnapshotOperation::Append,
                iceberg_schema: None,
                extra_properties: HashMap::new(),
                bloom_filters: vec![],
                equality_delete_files: vec![],
            };
            let snap_id = catalog.commit_snapshot(&table, snap).await.unwrap();

            let files = catalog.list_files(&table, Some(snap_id)).await.unwrap();
            assert_eq!(files.len(), 1);
            assert!(!files[0].is_foreign());
            assert_eq!(files[0].record_count, 10);

            // Overwrite: retire file1, add file2 — same pattern compaction/backfill use.
            let file2 = data_path.join("part-00002.parquet");
            write_source_parquet(meta_db.to_str().unwrap(), file2.to_str().unwrap(), 15).await;
            let snap2 = NewSnapshot {
                snapshot_id: new_snapshot_id(),
                parent_snapshot_id: Some(snap_id),
                files: vec![entry(file2.to_str().unwrap(), 15)],
                operation: SnapshotOperation::Overwrite,
                iceberg_schema: None,
                extra_properties: HashMap::new(),
                bloom_filters: vec![],
                equality_delete_files: vec![],
            };
            let snap2_id = catalog.commit_snapshot(&table, snap2).await.unwrap();
            let files_after = catalog.list_files(&table, Some(snap2_id)).await.unwrap();
            assert_eq!(files_after.len(), 1);
            assert_eq!(files_after[0].path, file2.to_str().unwrap());
            assert_eq!(files_after[0].record_count, 15);

            // evolve_schema: add a plain column, confirm ALTER TABLE really landed.
            let evolution =
                SchemaEvolution::new().add_column(crate::schema_evolution::AddColumnRequest {
                    name: "chunk_text".to_string(),
                    iceberg_type: "string".to_string(),
                    required: false,
                    initial_default: None,
                    write_default: None,
                    doc: None,
                });
            catalog.evolve_schema(&table, evolution).await.unwrap();

            catalog.drop_table(&table).await.unwrap();
            assert!(catalog.load_table(&table).await.is_err());
        }
    }
}
