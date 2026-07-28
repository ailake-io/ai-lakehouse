# ailake-trino

Trino connector exposing AI-Lake vector search, hybrid BM25+vector search,
cross-modal RRF search, and equality delete / schema evolution / compaction /
table creation as SQL — backed by `libailake_jni.so` via JNA (no cgo, no
Arrow Flight, same C-ABI Spark and Flink use).

Full reference: [`docs/guides/JVM_INTEGRATION.md`](../docs/guides/JVM_INTEGRATION.md) §5
(build/install/session properties/procedures/cross-engine tables) and
[`docs/specs/JVM_PLUGINS.md`](../docs/specs/JVM_PLUGINS.md) (architecture, C-ABI reference).

## Build

```bash
cargo build --release -p ailake-jni   # from the repo root — builds libailake_jni.so
cd trino-plugin
./gradlew shadowJar
# → build/libs/trino-plugin-<version>-plugin.jar
```

## Install

```bash
cp build/libs/trino-plugin-*-plugin.jar $TRINO_HOME/plugin/ailake/
cp ../target/release/libailake_jni.so    /opt/ailake/lib/
```

`etc/catalog/ailake.properties`:

```properties
connector.name       = ailake
ailake.table-uri     = s3://my-lake/docs/
ailake.vector-column = embedding
ailake.vector-dim    = 1536
ailake.metric        = cosine
ailake.precision     = f16
```

`etc/jvm.config` (or `-Djava.library.path`): point the JVM at `/opt/ailake/lib`
so JNA finds `libailake_jni.so`. See `docs/guides/JVM_INTEGRATION.md` §2C for
the full load-order (`ailake.native.lib` system property, `AILAKE_NATIVE_LIB`
env var, standard JNA search path).

## Quickstart

```sql
SET SESSION ailake.query_vector = '0.1,-0.2,0.3,...';
SET SESSION ailake.top_k = 10;
SELECT row_id, distance, file_path FROM ailake.default.search ORDER BY distance;

-- search + full row fetch, no JOIN needed
SELECT * FROM ailake.default.search_full ORDER BY _distance LIMIT 10;

INSERT INTO ailake.default.ingest VALUES (1, ARRAY[0.1, 0.2, 0.3]);
DELETE FROM ailake.default.ingest WHERE id = 5;
ALTER TABLE ailake.default.ingest ADD COLUMN source VARCHAR;

CALL ailake.system.compact();
CALL ailake.system.create_table();
SET SESSION ailake.decay_lambda = 0.1;
CALL ailake.system.decay_memories();
```

## Test

```bash
./gradlew test --info   # 10 test classes, no running Trino server required (some integration
                         # tests skip automatically without AILAKE_BIN/AILAKE_FIXTURE)
```

## License

MIT OR Apache-2.0 — same as the rest of the AI-Lake SDK.
