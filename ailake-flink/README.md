# ailake-flink

Flink connector exposing AI-Lake vector search, hybrid BM25+vector search,
cross-modal RRF search, search + full-row fetch (no JOIN), `DELETE`/`ALTER
TABLE` pushdown, and Phase 9 agent-memory scalar functions
(decay/migrate/backfill/estimate) as SQL DDL — backed by `libailake_jni.so`
via JNA (same C-ABI Trino and Spark use).

Full reference: [`docs/guides/JVM_INTEGRATION.md`](../docs/guides/JVM_INTEGRATION.md) §6
(build/install/DDL options/Kotlin API/cross-engine tables) and
[`docs/specs/JVM_PLUGINS.md`](../docs/specs/JVM_PLUGINS.md) (architecture, C-ABI reference).

## Build

```bash
cargo build --release -p ailake-jni   # from the repo root — builds libailake_jni.so
cd ailake-flink
./gradlew shadowJar
# → build/libs/ailake-flink-<version>-plugin.jar
```

## Install

```bash
cp build/libs/ailake-flink-*-plugin.jar /opt/flink/lib/
cp ../target/release/libailake_jni.so    /opt/ailake/lib/
echo 'env.java.opts.taskmanager: -Djava.library.path=/opt/ailake/lib' \
    >> /opt/flink/conf/flink-conf.yaml
```

## Quickstart

```sql
CREATE TABLE ailake_docs (
    row_id BIGINT, distance FLOAT, file_path STRING
) WITH (
    'connector'     = 'ailake',
    'warehouse'     = 's3://my-lake/',
    'namespace'     = 'default',
    'table-name'    = 'docs',
    'vector.column' = 'embedding',
    'vector.dim'    = '1536',
    'search.top-k'  = '10',
    'search.ef'     = '50',
    'search.pruning-threshold' = '0.8'   -- unset (default) = no pruning
);

-- Query vector passed via job parameters (Flink SQL has no per-query SET SESSION):
--   flink run -Dailake.query.vector='0.1,0.2,...' -Dailake.top-k=10
SELECT * FROM ailake_docs;
```

Scalar functions (register once per session/job):

```sql
CREATE TEMPORARY FUNCTION ailake_compact AS 'io.ailake.flink.AilakeCompactFunction';
SELECT ailake_compact('s3://my-lake/', 'default', 'docs');

CREATE TEMPORARY FUNCTION ailake_decay_memories AS 'io.ailake.flink.AilakeDecayMemoriesFunction';
SELECT ailake_decay_memories('s3://my-lake/agent-memory/', 'default', 'memories', CAST(0.1 AS FLOAT));
```

Also registers `AilakeCatalog` for DDL-driven `CREATE TABLE`/writes — see
`docs/guides/JVM_INTEGRATION.md` §6 for the full catalog + Kotlin API examples.

## Test

```bash
./gradlew test   # 8 test classes, no running Flink cluster required (integration tests
                 # need AILAKE_NATIVE_LIB pointing at a built libailake_jni.so)
```

## License

MIT OR Apache-2.0 — same as the rest of the AI-Lake SDK.
