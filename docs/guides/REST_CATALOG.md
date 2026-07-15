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

## Known limitations

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
