# Iceberg REST Catalog Backend

`RestCatalog` (`ailake-catalog`, feature `rest-catalog`) implements `CatalogProvider`
against any [Iceberg REST Catalog spec](https://iceberg.apache.org/spec/#rest-catalog)
server — Apache Polaris, Azure Databricks Unity Catalog, GCP BigLake Metastore, AWS S3
Tables, Project Nessie (REST mode), Gravitino, or any other spec-compliant
implementation. The Rust implementation existed before this guide, but was never
wired into any consumer surface (CLI, `ailake-py`, `ailake-jni`) — nobody could
actually reach it. This guide covers the wiring, config, and what live testing found.

## Design

Same split as `HadoopCatalog`/`DuckLakeCatalog`: the REST server owns table
*metadata* (`metadata.json`, schema, snapshots, table registration) via the REST
protocol; a separate `Store` (local filesystem, S3, GCS, Azure) still handles the
physical *data* — Parquet files and Avro manifests. `RestCatalog::new(config, store)`
takes both independently. `RestCatalogConfig.warehouse` tells the *server* where it
should consider new tables' storage location to be (used to build the `location`
field in `create_table` requests) — it does not have to match the `Store`'s own root,
but for a local-filesystem `Store` it should point at the same physical directory the
`Store` writes to, or reads will fail to find the files the catalog thinks exist
there (see "Known limitations" below).

## Auth strategies

```rust
pub enum RestCatalogAuth {
    None,                                    // open dev catalogs (local Polaris/Nessie)
    Bearer(String),                          // pre-obtained token — CI, Workload Identity
    OAuth2 { token_endpoint, client_id, client_secret, scope },  // client-credentials flow, token cached
}
```

## CLI usage

```
ailake --catalog rest --rest-uri http://localhost:8181 \
       --rest-warehouse /path/matching/--store \
       --store /path/matching/--rest-warehouse \
       create default.mytable --dim 1536
```

Flags: `--rest-uri` (required), `--rest-prefix`, `--rest-warehouse`, `--rest-auth
none|bearer|oauth2`, `--rest-token`, `--rest-oauth-token-endpoint`,
`--rest-oauth-client-id`, `--rest-oauth-client-secret`, `--rest-oauth-scope`. All
have `AILAKE_REST_*` env var fallbacks (avoids putting secrets on the command line /
in shell history). Requires the `catalog-rest` build feature
(`cargo build --features catalog-rest`) — off by default, matching `catalog-ducklake`'s
opt-in pattern (keeps `reqwest` out of the default binary for users who don't need it).

## Python usage

```python
import ailake

catalog_opts = {
    "catalog": "rest",
    "rest_uri": "http://localhost:8181",
    "rest_warehouse": "/path/matching/table_path",
    # "rest_auth": "bearer", "rest_token": "...",  # or oauth2 fields
}
t = ailake.open_table("/path/matching/rest_warehouse", dim=1536, catalog_opts=catalog_opts)
```

`catalog_opts` is a plain `dict[str, str]` accepted by `open_table`, `Table`,
`SearchQuery`, and the module-level `search`/`search_text`/`search_with_data`/
`search_multimodal`/`migrate_embeddings`/`decay_memories`/`compact`/`delete_rows`/
`add_column`/`rename_column`/`delete_where`/`add_vector_column`/
`backfill_vector_column` functions. Omit it (or pass `None`) for the default —
unchanged `HadoopCatalog` behavior. `ailake-py` has no `store_from_url` equivalent
yet, so `path` is always a local filesystem path regardless of catalog backend — a
separate, pre-existing gap, not closed here (S3/GCS/Azure aren't reachable from any
Python/JNI binding today, only from `ailake-cli`).

## JNI usage (Spark / Trino / Flink)

Every `ailake_*_json` C-ABI function accepts the same `catalog`/`rest_*` fields
flattened into its JSON request body, alongside the existing `warehouse` field:

```json
{"warehouse": "...", "table": "...", "catalog": "rest", "rest_uri": "http://localhost:8181", ...}
```

`ailake_vector_search_json`/`do_search`'s raw-pointer legacy entry point (no JSON
body) stays Hadoop-only — there's nowhere to carry the config.

### Spark

All 10 `AilakeNative.scala` methods take a trailing `catalogOpts: Map[String, String]`
(default empty = Hadoop catalog), which `AilakeCatalog`/`AilakeDataSource` also
populate automatically from catalog/writer options:

```scala
// spark.sql.catalog.<name>.catalog / .rest-uri / .rest-auth / .rest-token / ...
spark.conf.set("spark.sql.catalog.ailake.catalog", "rest")
spark.conf.set("spark.sql.catalog.ailake.rest-uri", "https://catalog.example.com")

// or the DataFrame/SQL API directly:
import io.ailake.spark.implicits._
spark.ailakeSearch(tableUri, queryVec, topK = 10,
  catalogOpts = Map("catalog" -> "rest", "rest_uri" -> "https://catalog.example.com"))

df.write.format("io.ailake.spark.AilakeDataSource")
  .option("tableUri", tableUri)
  .option("catalog", "rest")
  .option("rest-uri", "https://catalog.example.com")
  .save()
```

### Trino

Only `CALL ailake.system.compact()` supports REST catalog today, via new
`ailake.catalog`/`ailake.rest-*` catalog properties:

```properties
# etc/catalog/ailake.properties
connector.name=ailake
ailake.table-uri=s3://my-lake/docs/
ailake.catalog=rest
ailake.rest-uri=https://catalog.example.com
ailake.rest-auth=bearer
ailake.rest-token=...
```

`search`/`search_full`/`search_multimodal`/`INSERT` do **not** support it yet —
those go through `VectorScanHandles.kt`'s JSON-serialized Trino table/split
handle classes (`VectorScanTableHandle`, `ScanTableHandle`,
`AilakeIngestTableHandle`, etc), which that file's own doc comments flag as
having a real prior history of subtle Jackson serialization bugs (a handle
field silently not round-tripping coordinator→worker, only caught via a live
Trino server test) — extending them needs the same live verification, out of
reach in an offline sandbox. Tracked as a follow-up.

### Flink

Add `'catalog' = 'rest'` + `'rest-*'` options to either `CREATE TABLE`'s `WITH
(...)` clause (source/sink both — unlike Trino, this reaches search/scan/
insert/DELETE, since Flink's DynamicTableSource/Sink objects use plain Java
serialization, not the JSON-handle round-trip that makes Trino's case risky):

```sql
CREATE TABLE docs_ingest (
  id BIGINT, embedding ARRAY<FLOAT>
) WITH (
  'connector' = 'ailake',
  'warehouse' = 's3://my-lake/',
  'table-name' = 'docs',
  'vector.dim' = '1536',
  'catalog' = 'rest',
  'rest-uri' = 'https://catalog.example.com',
  'rest-auth' = 'bearer',
  'rest-token' = '...'
);
```

## Airflow usage

Set catalog config in the Airflow Connection's `extra` JSON (alongside cloud
credentials) — `AilakeHook.run_cli()` forwards it as `--catalog`/`--rest-*` CLI
flags automatically, single choke point for every hook method and operator:

```json
{
    "catalog": "rest",
    "rest_uri": "https://catalog.example.com",
    "rest_prefix": "my_catalog",
    "rest_warehouse": "s3://my-bucket/warehouse",
    "rest_auth": "bearer",
    "rest_token": "..."
}
```

Omit `catalog` (or set it to `"hadoop"`) for the default local/S3-prefix
metadata-dir catalog — unchanged behavior, no flags added.

## DuckDB usage

Every `ailake_*` SQL function (table and scalar) takes a trailing optional
`catalog_opts_json VARCHAR` parameter — a JSON object with the same
`catalog`/`rest_*` fields, merged into the request sent to the
statically-linked `ailake-jni` binary. Table functions (`ailake_search`,
`ailake_search_multimodal`, `ailake_search_text`, `ailake_scan`) expose it as
a named parameter; scalar functions (`ailake_write_batch`,
`ailake_write_batch_multi`, `ailake_delete_where`, `ailake_evolve_schema`,
`ailake_compact`, `ailake_create_table`) expose it as the last positional
argument in their highest arity overload:

```sql
SELECT * FROM ailake_search(
    'file:///data/my_table', [0.1, 0.2, 0.3]::FLOAT[], 10,
    catalog_opts_json := '{"catalog":"rest","rest_uri":"https://catalog.example.com","rest_auth":"bearer","rest_token":"..."}'
);
```

Malformed JSON (or a non-object value) raises `InvalidInputException` — never
silently falls back to the Hadoop catalog.

## Go usage

`CreateTable`/`WriteBatch`/`Compact`/`DecayMemories`/`Migrate`/`DeleteRows`/
`AddVectorColumn`/`BackfillVectorColumn` accept `CatalogOpts
map[string]string` (either directly, or via their `Options` struct) —
forwarded as `--catalog`/`--rest-*` flags to the `ailake` CLI binary these
functions shell out to. Nil/empty = default Hadoop catalog:

```go
err := ailake.WriteBatch(catalog, "default", "docs", "batch.parquet", ailake.WriteBatchOptions{
    VecCol: "embedding",
    CatalogOpts: map[string]string{
        "catalog":   "rest",
        "rest-uri":  "https://catalog.example.com",
        "rest-auth": "bearer",
        "rest-token": "...",
    },
})
```

`DeleteWhere`/`EvolveSchema` don't accept it yet (see "Known limitations").
`Estimate` needs no catalog config at all — pure math, no warehouse.

## C++ usage

`create_table`/`write_batch`/`write_batch_multi`/`compact`/`decay_memories`/
`migrate`/`delete_rows`/`add_vector_column`/`backfill_vector_column` accept a
`catalog_opts: std::map<std::string, std::string>` field (either directly, or
via their `Options` struct — `CreateTableOptions`/`WriteBatchOptions`/
`CompactOptions`/`MigrateOptions`/`AddVectorColumnOptions`/
`BackfillVectorColumnOptions`) — forwarded as `--catalog`/`--rest-*` flags to
the `ailake` CLI binary these functions shell out to. Empty = default Hadoop
catalog:

```cpp
ailake::WriteBatchOptions opts;
opts.vec_col = "embedding";
opts.catalog_opts["catalog"]    = "rest";
opts.catalog_opts["rest-uri"]   = "https://catalog.example.com";
opts.catalog_opts["rest-auth"]  = "bearer";
opts.catalog_opts["rest-token"] = "...";
ailake::write_batch(warehouse, "default.docs", "batch.parquet", opts);
```

`delete_where`/`evolve_schema` don't accept it yet (see "Known limitations").
`estimate` needs no catalog config at all — pure math, no warehouse.

## Known limitations

- **`ailake-go` and `ailake-cpp`'s write paths are both wired now**
  (`WriteBatchOptions.CatalogOpts`/`CompactOptions.CatalogOpts` in Go,
  `catalog_opts` in C++). Neither has native read-path support (no HTTP
  client in either — would need a new dependency, unlike the mechanical
  CLI-flag-forwarding fix). Both SDKs' `DeleteWhere`/`EvolveSchema`
  (`delete_where`/`evolve_schema` in C++) don't take catalog config yet —
  they predate an `Options`
  struct parameter, and adding one now would be a breaking API change; left
  as a follow-up (new sibling functions, not a signature change).
- **Trino only wires REST catalog into `CALL ailake.system.compact()`** — see
  "JNI usage → Trino" above for why search/scan/multimodal/INSERT are deferred
  (JSON-serialized Trino handle classes with a documented prior serialization
  bug history, needs live-server verification).
- **`Store` root vs. `RestCatalogConfig.warehouse` must be kept in sync manually**
  for local-filesystem storage. The catalog computes each table's `location` from
  `warehouse`; the `Store` resolves `DataFileEntry.path` against its own root
  (`path`/`--store`, independent of `warehouse`). If they don't point at the same
  physical directory, `commit_snapshot` succeeds (the server accepts the metadata)
  but a later `search`/read fails with a plain `No such file or directory` — the
  catalog and the store silently disagree about where the file actually is. No
  validation currently catches this mismatch at write time.
- **`AddPartitionSpec`/`SetDefaultSpec` for tables with real partitioning is
  untested** — the live verification session used only unpartitioned tables (the
  `unchanged` check added to skip a redundant `AddPartitionSpec` — see "Real bugs
  found" below — means partitioned tables *do* still hit that code path, just
  unverified). If it hits the same `-1`-sentinel class of issue `SetCurrentSchema`
  did, the fix is the same shape: use the spec's own `spec-id` explicitly instead of
  `-1`.

## Real bugs found wiring this into the CLI/Python/JNI bindings

Verified live (2026-07) against a real `apache/iceberg-rest-fixture:latest` container
— not mocks, including a full create → insert → commit → search round trip. Four
real, confirmed, fixed bugs — the write-commit path is closed, not just the
read/create side.

1. **`create_table` never created the namespace first.** Spec-compliant REST
   catalogs (unlike `HadoopCatalog`, which just uses a directory implicitly) reject
   `create_table` for a namespace nobody has explicitly registered with
   `NoSuchNamespaceException`. Fixed with `RestCatalog::ensure_namespace` — `POST
   /v1/{prefix}/namespaces`, treating 409 Conflict (already exists) as success, so
   `create_table` stays idempotent with respect to namespace existence, matching
   `HadoopCatalog`'s implicit-namespace behavior from the caller's point of view.

2. **`commit_snapshot`'s `AssertRefSnapshotId` requirement sent Iceberg's on-disk
   "no snapshot yet" sentinel (`-1`) as a literal integer instead of converting it
   to `null`.** `IcebergMetadata`'s plain `#[serde(default)]` deserialization has no
   reason to know `current-snapshot-id: -1` in a freshly created table's
   metadata.json means "no snapshot", not a real snapshot numbered -1 — it reads
   straight into `Some(-1)`. The REST spec's actual semantics for "assert this ref
   does not currently exist" need a real JSON `null`, not `-1` — sending `-1`
   verbatim got every first commit to a brand-new table rejected with
   `CommitFailedException: branch or tag main is missing, expected -1` on all 5 OCC
   retries (same wrong value resent every time, not a real conflict). Fixed by
   treating `meta.current_snapshot_id == Some(-1)` as `None` before building the
   requirement.

3. **`commit_snapshot` unconditionally re-sent the existing partition spec as a
   "new" `AddPartitionSpec` on every schema-patch commit**, even for unpartitioned
   tables where nothing about partitioning changed (the spec being "remapped" is
   just a no-op clone of the empty default spec — see
   `manifest_commit::build_commit`). This is wasteful in general, and this
   fixture's `AddPartitionSpec` response serialization specifically doesn't
   round-trip cleanly (`Cannot convert metadata update action to json:
   add-partition-spec`, HTTP 500). Fixed by comparing the remapped spec against
   what's already registered and only emitting the update when it actually changed.

4. **`commit_snapshot` unconditionally sent `AddSchema` + `SetCurrentSchema` on
   every write commit, even when the schema hadn't actually changed** — the
   schema-patch path (`TableWriter::commit`'s `captured_schema`) fires on *every*
   normal write, not just real schema evolution. Real Iceberg core
   (`TableMetadata.Builder.addSchema`, which every spec-compliant REST server
   delegates to) *reuses an existing schema-id* — silently ignoring whatever id
   the request suggests — whenever the submitted schema is structurally
   identical to one already registered, and is a true no-op when it's identical
   to the table's *current* schema specifically. Both cases made the
   client-predicted `current_schema_id + 1` (or the `-1` "last added" sentinel)
   wrong to reference in the immediately-following `SetCurrentSchema` — this is
   what produced the two *different* errors from the *same* request shape in
   different runs (`IllegalArgumentException: Cannot set current schema to
   unknown schema: N` when dedup reused a *different* existing id;
   `ValidationException: Cannot set last added schema: no schema has been added`
   when it was a true no-op) — a real client-side bug, not a fixture
   inconsistency as first suspected. Fixed by comparing the patch's new fields
   against the table's *current* schema fields (from `meta.schemas`, matched by
   `current_schema_id`) and skipping the whole `AddSchema`/`SetCurrentSchema`/
   name-mapping trio entirely when nothing actually changed — the same
   "skip when unchanged" principle already applied to `AddPartitionSpec` (bug 3).

Verified with 2 real (non-mocked) integration tests in `ailake-catalog/src/rest.rs`
(`live_create_table_auto_creates_namespace`, `live_ensure_namespace_is_idempotent`),
`#[ignore]`d by default (need a running server — see the doc comment above them for
the exact `docker run` command), plus a full `ailake-py` create → insert → commit →
search round trip run 3× against a live server after bug 4's fix, all 3 succeeding.
Full workspace build/test/clippy/fmt clean with and without the
`rest-catalog`/`catalog-rest` features.
