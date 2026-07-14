# DuckLake Catalog Backend

`DuckLakeCatalog` (`ailake-catalog`, feature `catalog-ducklake`) is an
alternative `CatalogProvider` backend that stores AI-Lake table metadata in a
real [DuckLake](https://ducklake.select) catalog, driven entirely through the
real DuckDB `ducklake` extension — no hand-rolled catalog DDL. It sits
alongside `HadoopCatalog` (file-based, default), `JdbcCatalog`
(Postgres/MySQL/SQLite pointer), `GlueCatalog`, and `NessieCatalog`.

Wired into `ailake-cli` behind the same feature flag (`--catalog ducklake`,
see "CLI usage" below). `JdbcCatalog`/`GlueCatalog`/`NessieCatalog` remain
backend-only (implemented and tested, but not selectable from the CLI or
`ailake-py`) — that wiring is separate, not-yet-started work for each of
those, not something this backend's CLI wiring implies is now easy/expected
for the others (DuckLake's embedded, local-file nature is what made a single
`--catalog ducklake` flag — no extra connection-string flags — a reasonable
CLI surface; JDBC/Glue/Nessie need real connection config a CLI flag alone
can't sensibly default).

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

## In-place rewrites are rejected (and why)

DuckLake records each file's zone-map stats (per-column min/max) **and its exact
footer size** at `ducklake_add_data_files` time, and trusts both afterwards.
Verified against a live extension:

- A **same-length** in-place rewrite silently returns wrong filtered rows — the
  stale zone-map prunes the file (`WHERE w > 5` returned 0 with every row at
  9.9).
- A **changed-length** rewrite breaks every subsequent native read of the file
  outright (`Parquet footer length stored in file is not equal to footer length
  provided`) — including the row-`DELETE` needed to retire the path, so there
  is no sanctioned SQL that can repair the registration afterwards.

Consequences wired into the code:

- `CatalogProvider::supports_in_place_rewrite()` returns `false` for this
  backend. `MemoryDecayJob` (the one values-changing in-place rewriter) now
  writes decayed files to a fresh `data/decayed-<ts>-<idx>.parquet` path and
  retires the old one on every backend — uniform behavior, no DuckLake special
  case at the writer level.
- Deferred writes and deferred compaction (`--deferred`) are **refused up
  front** with a clear error on this backend: their background index build
  patches the data file in place at its committed path by design, and the
  physical rewrite happens *before* any commit — too late for a commit-time
  guard to prevent the file from becoming natively unreadable.
- `commit_snapshot` backs all of this up with a hard error if a same-path
  entry ever arrives with a different `file_size_bytes` than the registered
  one (the entry's size always derives from the bytes the writer actually
  produced, so a mismatch proves an in-place rewrite): failing loudly beats
  registering a file whose native reads are already broken.

## Known v1 limitations

- **Deferred writes/compaction not supported** — `ailake insert --deferred` and
  `ailake compact --deferred` error out immediately (see "In-place rewrites are
  rejected" above). Blocking writes and compaction are fully supported.
- **Row-level deletes are invisible to DuckLake-native readers** —
  `delete_where` (equality deletes) and `delete_rows` (V3 deletion vectors)
  live in AI-Lake's sidecar and are applied by AI-Lake readers only; a plain
  `SELECT ... FROM lake.<ns>.<table>` still sees the deleted rows. This is
  asymmetric with file *retirement* (compaction, decay, backfill), which uses
  a real row-`DELETE` that native readers do observe. Possible future fix:
  translate equality deletes into `DELETE FROM lake.tbl WHERE col IN (...)`
  when the predicate column is DuckLake-declared.
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
  retirement problem" above. `CatalogProvider::retires_files_physically()`
  returns `false` for this backend specifically so generic callers (currently
  `CompactionExecutor` — `ailake-query/src/compaction.rs`) know not to
  `store.delete()` a file's bytes right after `commit_snapshot` retires it;
  doing so would leave the file's still-registered `ducklake_list_files()`
  entry dangling and break any subsequent DuckLake-native SQL read of it.
- **Network dependency on first use**: the `ducklake` extension is not
  bundled with the `duckdb` crate; `connect()` runs `INSTALL ducklake; LOAD
  ducklake;`, which fetches the extension from DuckDB's extension repository
  the first time it runs on a machine.
- **Lazy column declaration**: like the Iceberg backends, only the primary
  vector column (and the partition column, if any) is declared at
  `create_table` time. Other columns arrive via `evolve_schema`/
  `add_vector_column` (`ALTER TABLE ... ADD COLUMN`). `ducklake_add_data_files`
  is always called with `ignore_extra_columns => true` (source file has
  columns DuckLake hasn't been told about yet) and `allow_missing => true`
  (source file predates a column DuckLake now expects — the common case for
  any file written before an `evolve_schema` call; the default
  `allow_missing => false` rejects such a file outright, confirmed against a
  live extension while building this backend). Columns DuckLake doesn't know
  about yet aren't selectable through plain DuckDB SQL until declared.

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
    "/warehouse",                          // warehouse root — see "path resolution" below
).await?;
```

The `warehouse` argument matters beyond `table_root()`: `DataFileEntry::path`
is warehouse-relative by convention (`Store::get`/`put` and every other
`CatalogProvider` backend use this convention too), but DuckDB's
`ducklake_add_data_files`/`filename` predicate need a real filesystem path —
`commit_snapshot`/`list_files`'s foreign-file check resolve relative paths
against `warehouse` internally (`resolve_path` in `ducklake.rs`); an
already-absolute path or a URI-scheme path is left untouched. Get `warehouse`
wrong (e.g. pointing at the table directory instead of the true warehouse
root) and file registration fails with something like `No files found that
match the pattern "data/part-00000.parquet"` — found and fixed the hard way
while wiring this into `ailake-cli` (see "Real bugs found" below).

`CatalogProvider` methods implemented: `create_table`, `load_table`,
`commit_snapshot` (Append/Overwrite/Replace/Delete all supported —
Overwrite/Replace diff old-vs-new file lists and retire/add as needed;
Delete's real payload is `equality_delete_files`, stored in the sidecar and
returned by `list_equality_deletes`), `list_files`, `drop_table`,
`evolve_schema` (real `ALTER TABLE ADD/RENAME COLUMN`).

## CLI usage

```bash
# --catalog ducklake only supports a local filesystem --store (no s3://, gs://,
# az://). Layout under --store: catalog/{ailake_root,ducklake_meta}.db, data/ —
# created on first use. Requires the ailake-cli catalog-ducklake build feature.
ailake --store /warehouse --catalog ducklake create default.docs \
    --dim 4 --metric cosine --precision f16 --column embedding

ailake --store /warehouse --catalog ducklake insert default.docs batch.parquet \
    --embeddings embedding --metric cosine --precision f16

ailake --store /warehouse --catalog ducklake search default.docs \
    --query "0.1,0.2,0.3,0.4" --top-k 10

ailake --store /warehouse --catalog ducklake evolve default.docs --add tag:string

ailake --store /warehouse --catalog ducklake compact default.docs

ailake --store /warehouse --catalog ducklake info default.docs
```

Build with the feature: `cargo build -p ailake-cli --features catalog-ducklake`
(not part of the default build — it pulls in `duckdb`'s bundled C++ build,
several minutes on a cold cache). Without the feature, `--catalog ducklake`
fails fast with a clear "built without catalog-ducklake feature" error rather
than silently falling back to Hadoop.

`drop_table` has no CLI subcommand for *any* backend (not a DuckLake-specific
gap) — use the Rust API directly if you need it.

## Real bugs found wiring this into `ailake-cli`

Verified by actually running the built `ailake` binary against a real DuckLake
catalog end to end (create → insert → evolve → insert-after-evolve → search →
compact → info), not just unit tests. Three real bugs caught this way — two in
this backend's own code, one pre-existing in the shared read path (reproduces
identically with `HadoopCatalog`, found and fixed in the same pass since it
blocked verifying `compact` through DuckLake):

1. **Relative-path resolution was missing entirely (`ducklake.rs`).**
   `TableWriter` writes `DataFileEntry::path` as warehouse-relative
   (`"data/part-00000.parquet"`), the same convention every other backend and
   `Store::get`/`put` use. The first version of `commit_snapshot` passed this
   straight to `ducklake_add_data_files`, which resolves relative paths
   against DuckDB's own process working directory, not the warehouse — failed
   immediately with `No files found that match the pattern
   "data/part-00000.parquet"` on the very first real `ailake insert`. Fixed by
   resolving to an absolute path (`resolve_path`) only at the SQL call sites,
   keeping `DataFileEntry::path` itself warehouse-relative everywhere else so
   `store.get()` downstream keeps working.
2. **`ducklake_add_data_files` needs `allow_missing => true`, not just
   `ignore_extra_columns => true` (`ducklake.rs`).** Once `evolve_schema` adds
   a column, any subsequently-inserted file that predates that column (the
   normal case — AI-Lake never rewrites old files just because the schema
   grew) was rejected outright by the extension's default `allow_missing =>
   false`. Fixed by always passing both flags.
3. **`ParquetVectorReader::read_all()` (`ailake-parquet/src/reader.rs`) always
   decoded the vector column as F16, ignoring the file's actual stored
   precision.** Every AI-Lake writer embeds `ailake.precision` in the Parquet
   file's own KV metadata, but the reader used by `AilakeFileReader::
   read_parquet()` — the path both compaction and the scanner's foreign-file
   flat scan go through — never read it back, unconditionally calling
   `Quantizer::f16_bytes_to_f32` (2 bytes/element). For any table written with
   `--precision f32` (4 bytes/element), this silently produced a `Vec<f32>` of
   the wrong length (byte-width mismatch), surfacing downstream as a
   `debug_assert_eq!` panic in `ailake-vec`'s `cosine_distance`
   (`dimension mismatch 8 vs 4` for `dim=4`) in debug builds — and would have
   silently misbehaved instead of panicking in release builds, where the
   assert compiles out. Affects **any** F32-precision table run through
   `ailake compact`, independent of catalog backend — not a DuckLake-specific
   or synthetic-test-only bug. Fixed by reading `ailake.precision` from the
   file's own KV metadata and decoding accordingly (`f16`/`f32` supported;
   other precisions error clearly instead of silently misreading — I8's
   scaling params aren't currently persisted anywhere to reconstruct exact
   values from raw bytes, a separate, pre-existing gap not addressed here).
   Absent `ailake.precision` (raw external source files fed to `ailake
   insert`, which predate any AI-Lake writer touching them) still defaults to
   F16, preserving that path's existing documented contract.
