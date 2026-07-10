# DuckLake Catalog Backend

`DuckLakeCatalog` (`ailake-catalog`, feature `catalog-ducklake`) is an
alternative `CatalogProvider` backend that stores AI-Lake table metadata in a
real [DuckLake](https://ducklake.select) catalog, driven entirely through the
real DuckDB `ducklake` extension — no hand-rolled catalog DDL. It sits
alongside `HadoopCatalog` (file-based, default), `JdbcCatalog`
(Postgres/MySQL/SQLite pointer), `GlueCatalog`, and `NessieCatalog`.

Like `JdbcCatalog`/`GlueCatalog`/`NessieCatalog`, this is a catalog *backend*
— it implements `CatalogProvider` and is fully tested on its own, but (as of
this writing) none of the alternative backends are wired into
`ailake-cli`/`ailake-py`'s catalog selection surface yet. That wiring is a
separate, not-yet-started piece of work for every alternative backend, not a
DuckLake-specific gap.

## Why not just hand-roll the DuckLake catalog schema?

DuckLake's own catalog tables (`ducklake_data_file`, `ducklake_snapshot`,
`ducklake_table`, …) are public and documented, and an earlier draft of this
backend considered writing to them directly (the same style
`JdbcCatalog`/`HadoopCatalog` use for their own metadata formats). That draft
was rejected: DuckLake's *bootstrap* invariants (initial `ducklake_metadata`
rows, how `next_catalog_id`/`next_file_id` counters get allocated across
concurrent transactions, etc.) are not fully documented publicly, and getting
them wrong would produce a catalog file that looks fine until a real DuckDB
`ducklake` extension tries to read it. Instead, this backend never writes to
DuckLake's own tables directly — it only calls sanctioned public SQL (`CREATE
TABLE`, `ALTER TABLE`, `CALL ducklake_add_data_files(...)`, `DELETE FROM
lake.tbl WHERE filename = ?`, `ducklake_list_files(...)`) and lets the real
extension own its own bootstrap and invariants.

## Design

```
DuckDB connection
├── main.*              — AI-Lake's own sidecar tables (plain DuckDB, no
│                          DuckLake versioning): ailake_tables, ailake_table_props,
│                          ailake_vector_index, ailake_equality_deletes
└── ATTACH 'ducklake:<meta.db>' AS lake (DATA_PATH '<warehouse>/data')
    └── lake.<namespace>.<table>  — real DuckLake-managed table
```

- **DuckLake owns**: schemas, tables, columns, snapshots, and the row-level
  content of each table (via its own `ducklake_data_file`/deletion-vector
  machinery).
- **The sidecar owns**: everything DuckLake's fixed schema has no room for —
  centroid, radius, HNSW offset/length, index build status, embedding model,
  partition value, deletion vector pointer, batch id. One row per
  `(table_key, path)`.
- **`list_files` authority**: the sidecar's `active` boolean, not DuckLake's
  own file list. See "The retirement problem" below for why.
- **Foreign-file detection**: a path `ducklake_list_files()` returns that has
  *no* sidecar row at all (active or retired) was written by something other
  than AI-Lake — it comes back with `centroid_b64: None`
  (`DataFileEntry::is_foreign() == true`), the same contract already
  established for the Iceberg backends (ADR-018, `CLAUDE.md` §5A).

## The retirement problem (why this took several iterations to get right)

The natural-looking design — call `ducklake_add_data_files` for new files and
`DELETE FROM lake.tbl WHERE filename = ?` for files being replaced (used by
compaction, backfill, memory decay, and embedding-model migration, which all
follow the same "write new file(s), retire old file(s)" pattern) — turns out
to be subtly wrong, confirmed against a live `ducklake` extension (not just
documentation):

```sql
DELETE FROM lake.default.docs WHERE filename = '/path/to/old_file.parquet';
SELECT count(*) FROM lake.default.docs;                          -- 0, rows really gone
SELECT data_file FROM ducklake_list_files('lake', 'docs', schema => 'default');
-- still returns /path/to/old_file.parquet
```

Row-predicate `DELETE` is real and correct — any DuckLake reader (bare
`duckdb` CLI, Spark/Trino once a DuckLake connector exists, etc.) sees the
rows as gone — but it only attaches a deletion vector. The file itself stays
registered until a maintenance pass
(`ducklake_expire_snapshots`/`ducklake_cleanup_files`) reclaims it. There is
no sanctioned "drop this one file, right now" primitive.

So `list_files()` cannot use "is this path still in `ducklake_list_files()`"
to decide what AI-Lake considers active — it would keep returning
already-retired files forever. Instead:

1. `commit_snapshot` still issues the real `DELETE FROM lake.tbl WHERE
   filename = ?` for retired files, so any DuckLake-native reader gets
   correct results immediately.
2. The sidecar row for that path is *not* deleted — its `active` flag flips
   to `false`. This is what makes `list_files()` stop returning it, and
   keeps the path distinguishable from "AI-Lake never saw this file"
   (foreign-file detection above).
3. Physical space reclamation (deleting the bytes, dropping the catalog row
   entirely) is left to an operator running DuckLake's own
   `ducklake_expire_snapshots`/`ducklake_cleanup_files` periodically — the
   same class of follow-up maintenance Iceberg's `expire_snapshots` needs.
   Not wired into this module: doing it safely needs a retention-window
   policy verified against a live multi-snapshot scenario, which wasn't
   exercised while building this backend.

## Two-phase commit (not atomic across `lake` and `main`)

DuckDB refuses to write to two attached databases in a single transaction:

```
Conversion Error / TransactionContext Error: Attempting to write to database
"main" in a transaction that has already modified database "lake" - a single
transaction can only write to a single attached database.
```

(Confirmed the hard way — the first version of this backend wrapped both
sides in one `BEGIN`/`COMMIT` and errored on the very first `Overwrite`
commit.) `commit_snapshot` and `evolve_schema` therefore run two sequential
transactions: `lake` first (source of truth for which files/columns exist),
`main` second (sidecar metadata). A crash between the two phases degrades
gracefully, never incorrectly:

- Phase 1 committed, phase 2 didn't: the file is real and queryable through
  DuckLake, but until phase 2 catches up it looks **foreign** to AI-Lake
  (no vector metadata) rather than missing.
- A retirement's phase 1 (`DELETE`) committed, phase 2 (flipping `active =
  false`) didn't: the row is logically gone for any DuckLake reader; the
  stale sidecar row is simply never looked at again once a *subsequent*
  successful commit updates it, and in the meantime `query_active_files`
  still returns it as if active — a narrow, self-resolving inconsistency
  window, not data loss.

## Known v1 limitations

- **No multi-writer support**: the metadata catalog is a local DuckDB file
  (SQLite-class single-writer constraint). Concurrent processes writing to
  the same table are not safe. A Postgres-backed DuckLake metadata catalog
  would lift this but is out of scope here (see `ailake-catalog/Cargo.toml`
  — v1 scopes to DuckDB-as-metadata only).
- **`list_files(_, Some(snapshot_id))`** only accepts the table's current
  snapshot id (as returned by `load_table`) — arbitrary point-in-time
  time-travel isn't wired up. No caller in this codebase requests anything
  else (`grep -rn "list_files(" ailake-query ailake-cli` — every call site
  passes `None` or `table_meta.current_snapshot_id`, never an arbitrary past
  id).
- **Retired files are not physically reclaimed** by this module — see "The
  retirement problem" above.
- **Network dependency on first use**: the `ducklake` extension is not
  bundled with the `duckdb` crate; `connect()` runs `INSTALL ducklake; LOAD
  ducklake;`, which fetches the extension from DuckDB's extension repository
  the first time it runs on a machine.
- **Lazy column declaration**: like the Iceberg backends, only the primary
  vector column (and the partition column, if any) is declared at
  `create_table` time. Other columns arrive via `evolve_schema`/
  `add_vector_column` (`ALTER TABLE ... ADD COLUMN`). Physical Parquet files
  may carry columns not yet declared to DuckLake —
  `ducklake_add_data_files(..., ignore_extra_columns => true)` accepts this,
  but those columns aren't selectable through plain DuckDB SQL until
  declared.

## Sidecar schema

```sql
CREATE TABLE main.ailake_tables (
    table_key VARCHAR PRIMARY KEY, namespace VARCHAR, table_name VARCHAR,
    table_uuid VARCHAR, format_version INTEGER, location VARCHAR,
    last_snapshot_id BIGINT
);
CREATE TABLE main.ailake_table_props (table_key, key, value VARCHAR, PRIMARY KEY (table_key, key));
CREATE TABLE main.ailake_vector_index (
    table_key, path VARCHAR, record_count BIGINT, file_size_bytes BIGINT,
    centroid_b64 VARCHAR, radius DOUBLE, hnsw_offset BIGINT, hnsw_len BIGINT,
    vector_column VARCHAR, vector_dim INTEGER, extra_vector_indexes_json VARCHAR,
    index_status VARCHAR, index_error VARCHAR, batch_id VARCHAR,
    embedding_model VARCHAR, partition_value VARCHAR, deletion_vector_json VARCHAR,
    first_row_id BIGINT, active BOOLEAN, PRIMARY KEY (table_key, path)
);
CREATE TABLE main.ailake_equality_deletes (
    table_key, path VARCHAR, equality_ids_json VARCHAR,
    record_count BIGINT, file_size_bytes BIGINT, PRIMARY KEY (table_key, path)
);
```

## Usage

```rust
use ailake_catalog::DuckLakeCatalog;

let catalog = DuckLakeCatalog::connect(
    "/warehouse/catalog/ailake_root.db",   // AI-Lake's own sidecar tables
    "/warehouse/catalog/ducklake_meta.db", // DuckLake's own metadata store
    "/warehouse/data",                     // DATA_PATH — where DuckLake writes/expects data files
    "lake",                                // catalog alias used in ATTACH
    "/warehouse",                          // warehouse root (for table_root() paths)
).await?;
```

`CatalogProvider` methods implemented: `create_table`, `load_table`,
`commit_snapshot` (Append/Overwrite/Replace/Delete all supported —
Overwrite/Replace diff old-vs-new file lists and retire/add as needed;
Delete's real payload is `equality_delete_files`, stored in the sidecar and
returned by `list_equality_deletes`), `list_files`, `drop_table`,
`evolve_schema` (real `ALTER TABLE ADD/RENAME COLUMN`).
