# duckdb-ailake

DuckDB community extension that exposes AI-Lake vector search and write via SQL table/scalar functions.

Bridges DuckDB to [`ailake-jni`](../ailake-jni) using the same C-ABI as the Spark and Trino plugins — zero additional Rust code required. `ailake-jni` is linked **statically** into `ailake.duckdb_extension` (via [corrosion](https://github.com/corrosion-rs/corrosion), see "Design" below) — no separate `.so` to build or load at runtime.

> **Error handling**: a genuine backend rejection (`ok:false` in the JSON response — e.g. a
> nonexistent table path, `NaN`/`Infinity` embeddings, mismatched `ids`/`embeddings` lengths,
> `top_k` above `ailake_core::MAX_TOP_K` (100,000)) is now raised as a `duckdb::InvalidInputException`
> with the real error message, for every function below except `ailake_delete_where` (which still
> returns `FALSE`, unchanged). This used to be silently folded into an empty result / `-1` / `FALSE`,
> indistinguishable from a genuine zero-match search or no-op.

## Functions

### `ailake_search` — vector similarity search

```sql
SELECT * FROM ailake_search(
    table_path       VARCHAR,    -- path/URI to AI-Lake table root
    query            FLOAT[],    -- query embedding (LIST(FLOAT))
    top_k            INTEGER,    -- number of nearest neighbors
    -- named (optional):
    vec_col          VARCHAR,    -- default 'embedding'
    ef_search        INTEGER,    -- HNSW ef parameter, default 50
    partition_filter VARCHAR,    -- restrict to files with matching partition_value (Phase 9)
    hybrid_text      VARCHAR,    -- BM25 query text for hybrid scoring; NULL = pure vector search
    text_column      VARCHAR,    -- Parquet column for BM25 scoring, default 'chunk_text'
    bm25_weight      FLOAT       -- BM25 weight in RRF fusion [0, 1], default 0.5
) → TABLE(row_id BIGINT, distance FLOAT, file_path VARCHAR)
```

**Examples:**

```sql
-- Load extension
LOAD 'ailake';

-- Basic search
SELECT row_id, distance, file_path
FROM ailake_search('file:///data/my_table', [0.1, 0.2, 0.3]::FLOAT[], 10)
ORDER BY distance;

-- Combine with parquet_scan for full row data (legacy — prefer ailake_scan() below,
-- which does this in one call with no JOIN required)
SELECT p.id, p.text, s.distance
FROM ailake_search('file:///data/docs', my_query_vec, 20) s
JOIN parquet_scan('file:///data/docs/data/*.parquet') p
  ON p.id = s.row_id
ORDER BY s.distance
LIMIT 5;

-- Named parameters
SELECT * FROM ailake_search(
    'file:///data/docs',
    my_vec,
    10,
    vec_col='context_embedding',
    ef_search=100
);

-- Agent isolation (Phase 9) — only files tagged with partition_value='agent-42'
SELECT * FROM ailake_search(
    'file:///data/agents',
    my_vec,
    10,
    partition_filter='agent-42'
) ORDER BY distance;
```

### `ailake_scan` — vector search + full row fetch, no JOIN required

```sql
SELECT * FROM ailake_scan(
    table_path VARCHAR,    -- path/URI to AI-Lake table root
    query      FLOAT[],    -- query embedding (LIST(FLOAT))
    top_k      INTEGER     -- number of nearest neighbors
) → TABLE(<all Parquet columns>, _distance FLOAT)
```

Unlike `ailake_search()`, which returns only `(row_id, distance, file_path)` pointers and needs a manual `JOIN` against `parquet_scan(...)` to get real columns, `ailake_scan()` performs the search and full-row fetch in one native call — every Parquet column comes back alongside `_distance`. Backed by `ailake_scan_json` C-ABI. The full result is fetched at bind time and cached, so `LIMIT` does not reduce Rust-side I/O — use `top_k` to control how many rows are fetched.

**Example:**

```sql
SELECT id, chunk_text, _distance
FROM ailake_scan('file:///data/my_table', [0.1, 0.2, 0.3]::FLOAT[], 10)
ORDER BY _distance;
```

### `ailake_search_multimodal` — cross-modal RRF search (Phase 8)

```sql
SELECT * FROM ailake_search_multimodal(
    table_path       VARCHAR,                -- path/URI to AI-Lake table root
    queries          LIST(STRUCT(           -- one entry per vector column
                         col    VARCHAR,    -- column name
                         query  FLOAT[],   -- query embedding
                         weight FLOAT)),   -- RRF weight (higher = more influential)
    top_k            INTEGER,
    -- named (optional):
    partition_filter VARCHAR               -- restrict to files with matching partition_value (Phase 9)
) → TABLE(row_id BIGINT, rrf_score FLOAT, file_path VARCHAR)
```

Results are **not** automatically sorted — add `ORDER BY rrf_score DESC` to rank them.

**Examples:**

```sql
-- Cross-modal: 70% text + 30% image
SELECT row_id, rrf_score
FROM ailake_search_multimodal(
    'file:///data/media',
    [
        {'col': 'embedding',       'query': [0.1, 0.2, ...]::FLOAT[], 'weight': 0.7},
        {'col': 'image_embedding', 'query': [0.3, 0.4, ...]::FLOAT[], 'weight': 0.3}
    ],
    20
)
ORDER BY rrf_score DESC;

-- Single-column (equivalent to ailake_search but returns rrf_score)
SELECT * FROM ailake_search_multimodal(
    'file:///data/docs',
    [{'col': 'embedding', 'query': my_vec, 'weight': 1.0}],
    10
) ORDER BY rrf_score DESC;

-- Agent isolation (Phase 9)
SELECT * FROM ailake_search_multimodal(
    'file:///data/agents',
    [{'col': 'embedding', 'query': my_vec, 'weight': 1.0}],
    10,
    partition_filter='agent-42'
) ORDER BY rrf_score DESC;
```

A backend rejection (e.g. nonexistent table path) raises `InvalidInputException` — see "Error handling" above.

---

### `ailake_search_text` — full-text search (Phase T — Tantivy FTS)

```sql
SELECT * FROM ailake_search_text(
    table_path   VARCHAR,    -- path/URI to AI-Lake table root
    query_text   VARCHAR,    -- BM25 query string
    top_k        INTEGER,    -- number of results
    -- named (optional):
    text_column      VARCHAR,    -- Parquet column to score, default 'chunk_text'
    partition_filter VARCHAR     -- restrict to files with matching partition_value
) → TABLE(row_id BIGINT, distance FLOAT, file_path VARCHAR)
```

`distance` is the negated BM25 score — lower is more relevant, consistent with vector search convention.

**Examples:**

```sql
-- Pure BM25 search
SELECT row_id, distance, file_path
FROM ailake_search_text('file:///data/docs', 'rust programming async', 10)
ORDER BY distance;

-- Custom text column + agent partition
SELECT * FROM ailake_search_text(
    'file:///data/agents',
    'deployment failure',
    10,
    text_column='chunk_text',
    partition_filter='agent-42'
) ORDER BY distance;
```

A backend rejection raises `InvalidInputException` — see "Error handling" above. Backed by `ailake_search_text_json` C-ABI.

---

### `ailake_write_batch` — ingest embeddings

```sql
-- 3-arg form (defaults: vec_col=embedding, metric=cosine, precision=f16)
SELECT ailake_write_batch(
    table_path      VARCHAR,         -- table root path/URI
    ids             BIGINT[],        -- row identifiers
    embeddings      FLOAT[][]        -- one embedding per id
) → BIGINT  -- snapshot_id; a backend rejection (e.g. NaN/Infinity embeddings) raises
            -- InvalidInputException, not a silent -1 — see "Error handling" above.

-- 6-arg form (explicit options)
SELECT ailake_write_batch(
    table_path,
    ids,
    embeddings,
    vec_col         VARCHAR,         -- embedding column name
    metric          VARCHAR,         -- cosine | euclidean | dot
    precision       VARCHAR          -- f32 | f16 | i8
) → BIGINT

-- Named parameters (single-column partition)
SELECT ailake_write_batch(
    table_path,
    ids,
    embeddings,
    partition_by    VARCHAR,         -- partition column name (e.g. 'agent_id')
    partition_value VARCHAR          -- value for this write (e.g. agent UUID)
) → BIGINT

-- Named parameters (multi-column partition spec + format_version)
SELECT ailake_write_batch(
    table_path,
    ids,
    embeddings,
    partition_fields VARCHAR,        -- JSON array: '[{"column":"topic_id","transform":"identity","column_type":"int"}]'
    format_version   INTEGER         -- 2 (default) or 3 (Iceberg v3)
) → BIGINT

-- Named parameters (Tantivy FTS + pre_normalize)
SELECT ailake_write_batch(
    table_path,
    ids,
    embeddings,
    fts_columns      VARCHAR,        -- JSON array of text column names: '["chunk_text","document_title"]'
    fts_tokenizer    VARCHAR,        -- 'simple' (default) or 'raw'
    pre_normalize    BOOLEAN         -- normalize vectors to unit L2 at write time (~12-20% search speedup for cosine)
) → BIGINT
```

**Examples:**

```sql
SELECT ailake_write_batch(
    'file:///data/my_table',
    [0, 1, 2]::BIGINT[],
    [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]]::FLOAT[][]
);

-- Write to a per-agent shard
SELECT ailake_write_batch(
    'file:///data/agents',
    [0, 1]::BIGINT[],
    [[0.1, 0.2], [0.3, 0.4]]::FLOAT[][],
    partition_by='agent_id',
    partition_value='agent-42'
);

-- Multi-column partition spec with Iceberg v3
SELECT ailake_write_batch(
    'file:///data/topics',
    [0, 1]::BIGINT[],
    [[0.1, 0.2], [0.3, 0.4]]::FLOAT[][],
    partition_fields='[{"column":"topic_id","transform":"identity","column_type":"int"}]',
    format_version=3
);
```

### `ailake_write_batch_multi` — multi-column (multimodal) write (Phase 8)

```sql
SELECT ailake_write_batch_multi(
    table_path      VARCHAR,                 -- table root path/URI
    ids             BIGINT[],                -- row identifiers
    vector_columns  LIST(STRUCT(
                       col        VARCHAR,   -- column name
                       dim        INTEGER,   -- dimensionality
                       embeddings FLOAT[][], -- one embedding per id, same order
                       metric     VARCHAR,   -- cosine | euclidean | dot
                       precision  VARCHAR,   -- f32 | f16 | i8
                       modality   VARCHAR)), -- '' | text | image | audio | video
    -- named (optional):
    namespace       VARCHAR,                 -- default 'default'
    table_name      VARCHAR,                 -- default 'table'
    format_version  INTEGER,                 -- 2 (default) or 3
    deferred        BOOLEAN                  -- default false — persist Parquet
                                              --   immediately, build all HNSW
                                              --   indexes in the background
) → BIGINT  -- snapshot_id; a backend rejection raises InvalidInputException, not a silent
            -- -1 — see "Error handling" above. -1 is still returned if the lib isn't loaded.
```

Writes a batch of rows with **N independent vector columns** (e.g. text + image embeddings on the same row), each getting its own HNSW section in the same AI-Lake file — searchable via `ailake_search_multimodal`'s RRF fusion. The **first entry in `vector_columns` is primary** (used for geometric pruning in the manifest). Backed by `ailake_write_batch_multi_json` C-ABI.

**Example:**

```sql
SELECT ailake_write_batch_multi(
    'file:///data/media',
    [0, 1]::BIGINT[],
    [
        {'col': 'embedding',       'dim': 4, 'embeddings': [[0.1,0.2,0.3,0.4],[0.5,0.6,0.7,0.8]]::FLOAT[][], 'metric': 'cosine', 'precision': 'f16', 'modality': ''},
        {'col': 'image_embedding', 'dim': 2, 'embeddings': [[0.9,1.0],[1.1,1.2]]::FLOAT[][], 'metric': 'cosine', 'precision': 'f16', 'modality': 'image'}
    ]
);
```

### `ailake_create_table` — create an empty table

```sql
SELECT ailake_create_table(
    table_path            VARCHAR,   -- table root path/URI
    dim                   INTEGER,   -- vector dimension
    -- named or positional (optional), in order:
    vector_column          VARCHAR,  -- default 'embedding'
    metric                 VARCHAR,  -- default 'cosine'
    precision              VARCHAR,  -- default 'f16'
    format_version         INTEGER,  -- default 2 (2 or 3)
    hnsw_m                 INTEGER,  -- default -1 (use native default)
    hnsw_ef_construction   INTEGER,  -- default -1 (use native default)
    pre_normalize          BOOLEAN,  -- default false
    modality               VARCHAR,  -- default ''
    partition_by           VARCHAR,  -- default ''
    partition_value        VARCHAR,  -- default ''
    partition_column_type  VARCHAR,  -- default ''
    partition_fields_json  VARCHAR,  -- default ''
    fts_columns            VARCHAR,  -- default ''
    fts_tokenizer          VARCHAR,  -- default ''
    embedding_model        VARCHAR,  -- default ''
    namespace               VARCHAR, -- default 'default'
    table_name               VARCHAR -- default 'table'
) → BOOLEAN  -- TRUE on success, FALSE on any error or if the lib isn't loaded
```

Creates an empty AI-Lake/Iceberg table (schema/policy only, no data files) — useful
when a table needs to exist and be searchable (returning zero rows) before any
embeddings are ready. Backed by `ailake_create_table_json` C-ABI.

**Example:**

```sql
SELECT ailake_create_table('file:///data/my_table', 1536);

SELECT ailake_create_table(
    'file:///data/my_table', 768,
    vector_column := 'image_embedding', metric := 'euclidean'
);
```

### `ailake_delete_where` — logical delete

```sql
SELECT ailake_delete_where(
    table_path VARCHAR,    -- path/URI to AI-Lake table root
    column     VARCHAR,    -- column name to match against
    values     VARCHAR[]   -- values to delete
) → BOOLEAN                -- TRUE on success, FALSE on any error or if the lib isn't loaded
```

Writes an Iceberg equality delete file for all rows where `column` equals any value in `values`. No data files are rewritten. Backed by `ailake_delete_where_json` C-ABI.

**Example:**

```sql
SELECT ailake_delete_where(
    'file:///data/my_table',
    'document_id',
    ['doc-a', 'doc-b', 'doc-c']
);
```

### `ailake_evolve_schema` — metadata-only ADD/RENAME COLUMN

```sql
SELECT ailake_evolve_schema(
    table_path          VARCHAR,  -- path/URI to AI-Lake table root
    add_columns_json    VARCHAR,  -- JSON array: [{"name":"col","type":"string","initial_default":null}]
    rename_columns_json VARCHAR   -- JSON array: [{"from":"old_name","to":"new_name"}]
) → INTEGER  -- new schema_id; a backend rejection raises InvalidInputException, not a
             -- silent -1 — see "Error handling" above. -1 is still returned if the lib
             -- isn't loaded.
```

Either argument may be `'[]'` or `''` to skip. No data files are rewritten. Backed by `ailake_evolve_schema_json` C-ABI.

**Example:**

```sql
SELECT ailake_evolve_schema(
    'file:///data/my_table',
    '[{"name":"score","type":"float","initial_default":0.0}]',
    '[{"from":"old_col","to":"new_col"}]'
);
```

### `ailake_compact` — merge small files

```sql
SELECT ailake_compact(
    table_path          VARCHAR,   -- table root path/URI
    -- named or positional (optional), in order:
    min_files            BIGINT,   -- default 4   — min small files required to trigger
    target_size_bytes    BIGINT,   -- default 128 MiB — target output file size
    max_files_per_pass   BIGINT,   -- default 20  — bounds peak RAM / HNSW rebuild cost
    deferred              BOOLEAN, -- default false — write merged Parquet immediately,
                                   --   build the HNSW index in the background
    namespace             VARCHAR, -- default 'default'
    table_name            VARCHAR  -- default 'table'
) → BIGINT  -- number of files compacted (0 = nothing eligible); a backend rejection
            -- (e.g. missing table) raises InvalidInputException, not a silent -1 —
            -- see "Error handling" above. -1 is still returned if the lib isn't loaded.
```

Compacts small files in an AI-Lake table into a larger merged file. Backed by `ailake_compact_json` C-ABI.

**Example:**

```sql
-- Force a merge even with just 2 small files present
SELECT ailake_compact('file:///data/my_table', min_files := 2);
```

## Build

```bash
cmake -S duckdb-ailake -B duckdb-ailake/build -DCMAKE_BUILD_TYPE=Release
cmake --build duckdb-ailake/build --parallel

# Output: duckdb-ailake/build/ailake.duckdb_extension
```

A single `cmake --build` does everything: builds `ailake-jni` as a static lib (via
[corrosion](https://github.com/corrosion-rs/corrosion), no separate `cargo build` step needed),
builds a real `duckdb_static` from source at the pinned `DUCKDB_VERSION`, and links both plus this
extension's own C++ sources into `ailake.duckdb_extension`. Building `duckdb_static` from source
takes noticeably longer than the old headers-only setup (several minutes) — that's the cost of no
longer depending on the host process to supply DuckDB's symbols at `LOAD` time.

### DuckDB version

The extension must be built against the same DuckDB version as the Python/CLI client:

```bash
cmake -S duckdb-ailake -B duckdb-ailake/build \
  -DCMAKE_BUILD_TYPE=Release \
  -DDUCKDB_VERSION=v1.5.0
```

Match the pip package: `pip install duckdb==1.5.0` (see `.github/workflows/ci-duckdb.yml` for the
version this project's CI actually tests against — keep this section in sync with it). Now that
the extension links a real `duckdb_static` at this exact version/commit (see "Design" below)
instead of resolving symbols from whatever DuckDB happens to be hosting it, a version mismatch
between this setting and the client fails more informatively (ABI/struct-layout mismatch at
`LOAD`) rather than the old silent "works with Python, not CLI" split.

## Load in Python

```python
import duckdb

conn = duckdb.connect(config={"allow_unsigned_extensions": True})
conn.execute("LOAD './duckdb-ailake/build/ailake.duckdb_extension'")

rows = conn.execute("""
    SELECT row_id, distance
    FROM ailake_search('file:///data/docs', [0.1, 0.2, 0.3]::FLOAT[], 5)
    ORDER BY distance
""").fetchall()
```

`ailake-jni` is statically linked into `ailake.duckdb_extension` (see "Design" below) — no
`ctypes.CDLL(...)` pre-load, no `RTLD_GLOBAL`/`sys.setdlopenflags` dance, no `LD_LIBRARY_PATH`.

## Load in DuckDB CLI

```bash
duckdb -unsigned

D LOAD './duckdb-ailake/build/ailake.duckdb_extension';
D SELECT * FROM ailake_search('file:///data/docs', [0.1, 0.2]::FLOAT[], 5);
```

Works against the official `duckdb.org`-distributed CLI binary, not just the Python path — the
extension links against a real `duckdb_static` built from the same DuckDB source/version (see
"Design" below) instead of resolving DuckDB's own symbols from the host process at `LOAD` time,
which is what previously failed here with `undefined symbol:
_ZTIN6duckdb28SimpleNamedParameterFunctionE` against the official CLI binary specifically (the
Python wheel's `_duckdb...so` happened to export that symbol; the CLI binary doesn't).

## Design

- `ailake-jni` is linked **statically** into `ailake.duckdb_extension` via
  [corrosion](https://github.com/corrosion-rs/corrosion) (`ailake-jni`'s `staticlib` crate-type,
  imported and linked at CMake configure/build time — see `CMakeLists.txt`) — no `dlopen`, no
  separate `.so` to ship or find at runtime. `AilakeLib` (`include/ailake_extension.hpp`) declares
  the same 11 C-ABI symbols `extern "C"` and resolves them at **link** time instead of via
  `dlsym`: `ailake_search_json` / `ailake_scan_json` / `ailake_search_multimodal_json` /
  `ailake_search_text_json` / `ailake_write_batch_json` / `ailake_write_batch_multi_json` /
  `ailake_delete_where_json` / `ailake_evolve_schema_json` / `ailake_compact_json` /
  `ailake_create_table_json` / `ailake_free_string`.
- Same JSON-envelope protocol as Spark (`AilakeNative.scala`) and Trino (`AilakeNative.kt`)
- `ailake_search` executes the full search (pruning + HNSW) inside Rust; DuckDB sees a virtual table
- The extension also links against a real `duckdb_static` (built from source at the pinned
  `DUCKDB_VERSION`, not headers-only) instead of resolving `duckdb::*` symbols from the host
  process — this is what makes `LOAD` work against the official CLI binary (see above).
- **Error surfacing**: a real backend rejection (e.g. `top_k` above `ailake_core::MAX_TOP_K`
  (100,000), or a `NaN`/`Infinity` embedding value passed to `ailake_write_batch*`) throws
  `duckdb::InvalidInputException` with the real error message — `ailake_search`/`ailake_scan`/
  `ailake_write_batch`/`ailake_write_batch_multi`/`ailake_evolve_schema`/`ailake_compact` no
  longer fold a genuine error into the same empty-result/`-1`/`FALSE` return used for benign
  "no matches"/no-op outcomes.

## Comparison with Spark and Trino plugins

| Feature | Spark | Trino | DuckDB |
|---|---|---|---|
| Vector search | `VectorScanExec` | `VectorScanRecordSet` | `ailake_search()` table fn |
| Cross-modal search | `searchMultimodal()` | `searchMultimodal()` | `ailake_search_multimodal()` table fn |
| INSERT INTO / write | `AilakeWriteSupport` | `AilakePageSink` | `ailake_write_batch()` scalar fn |
| Multi-column (multimodal) write | `ailakeWriteMulti()` | `ailake.vector-columns` catalog property | `ailake_write_batch_multi()` scalar fn |
| Compact | `spark.ailakeCompact(...)` | `CALL ailake.system.compact()` | `ailake_compact()` scalar fn |
| Catalog integration | `AilakeCatalog` | — | — (use `parquet_scan` for joins) |
| Native lib loading | JNA | JNA | static link (corrosion) |

## Tests

```bash
AILAKE_EXT=./duckdb-ailake/build/ailake.duckdb_extension \
AILAKE_FIXTURE=./compat-fixture \
python duckdb-ailake/test/test_search.py

AILAKE_EXT=./duckdb-ailake/build/ailake.duckdb_extension \
python duckdb-ailake/test/test_write.py
```
