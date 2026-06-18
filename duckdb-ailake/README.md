# duckdb-ailake

DuckDB community extension that exposes AI-Lake vector search and write via SQL table/scalar functions.

Bridges DuckDB to [`libailake_jni.so`](../ailake-jni) using the same C-ABI as the Spark and Trino plugins — zero additional Rust code required.

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

-- Combine with parquet_scan for full row data
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

Returns 0 rows (no error) if `libailake_jni.so` is not loaded or does not export `ailake_search_multimodal_json`.

---

### `ailake_search_text` — pure BM25 full-text search (Phase 9)

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

Returns 0 rows (graceful degradation) when `libailake_jni.so` is not loaded. Backed by `ailake_search_text_json` C-ABI.

---

### `ailake_write_batch` — ingest embeddings

```sql
-- 3-arg form (defaults: vec_col=embedding, metric=cosine, precision=f16)
SELECT ailake_write_batch(
    table_path      VARCHAR,         -- table root path/URI
    ids             BIGINT[],        -- row identifiers
    embeddings      FLOAT[][]        -- one embedding per id
) → BIGINT  -- snapshot_id, or -1 on error

-- 6-arg form (explicit options)
SELECT ailake_write_batch(
    table_path,
    ids,
    embeddings,
    vec_col         VARCHAR,         -- embedding column name
    metric          VARCHAR,         -- cosine | euclidean | dot
    precision       VARCHAR          -- f32 | f16 | i8
) → BIGINT

-- Named parameters (Phase 9 agent partitioning)
SELECT ailake_write_batch(
    table_path,
    ids,
    embeddings,
    partition_by    VARCHAR,         -- partition column name (e.g. 'agent_id')
    partition_value VARCHAR          -- value for this write (e.g. agent UUID)
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
```

## Build

```bash
# 1. Build the native library (Rust)
cargo build --release -p ailake-jni

# 2. Configure and build the extension
cmake -S duckdb-ailake -B duckdb-ailake/build -DCMAKE_BUILD_TYPE=Release
cmake --build duckdb-ailake/build --parallel

# Output: duckdb-ailake/build/ailake.duckdb_extension
```

### DuckDB version

The extension must be built against the same DuckDB version as the Python/CLI client:

```bash
cmake -S duckdb-ailake -B duckdb-ailake/build \
  -DCMAKE_BUILD_TYPE=Release \
  -DDUCKDB_VERSION=v1.1.3
```

Match the pip package: `pip install duckdb==1.1.3`.

## Load in Python

```python
import ctypes, duckdb

# Pre-load native lib so DuckDB extension resolves symbols
ctypes.CDLL("./target/release/libailake_jni.so", ctypes.RTLD_GLOBAL)

conn = duckdb.connect()
conn.execute("LOAD './duckdb-ailake/build/ailake.duckdb_extension'")

rows = conn.execute("""
    SELECT row_id, distance
    FROM ailake_search('file:///data/docs', [0.1, 0.2, 0.3]::FLOAT[], 5)
    ORDER BY distance
""").fetchall()
```

## Load in DuckDB CLI

```bash
# Set LD_LIBRARY_PATH so the extension finds libailake_jni.so
LD_LIBRARY_PATH=./target/release duckdb

D LOAD './duckdb-ailake/build/ailake.duckdb_extension';
D SELECT * FROM ailake_search('file:///data/docs', [0.1, 0.2]::FLOAT[], 5);
```

## Design

- C-ABI bridge: `dlopen("libailake_jni.so")` → `ailake_search_json` / `ailake_search_multimodal_json` / `ailake_write_batch_json`
- Same JSON-envelope protocol as Spark (`AilakeNative.scala`) and Trino (`AilakeNative.kt`)
- `ailake_search` executes the full search (pruning + HNSW) inside Rust; DuckDB sees a virtual table
- Graceful degradation: if `libailake_jni.so` is not found, search returns 0 rows instead of aborting

## Comparison with Spark and Trino plugins

| Feature | Spark | Trino | DuckDB |
|---|---|---|---|
| Vector search | `VectorScanExec` | `VectorScanRecordSet` | `ailake_search()` table fn |
| Cross-modal search | `searchMultimodal()` | `searchMultimodal()` | `ailake_search_multimodal()` table fn |
| INSERT INTO / write | `AilakeWriteSupport` | `AilakePageSink` | `ailake_write_batch()` scalar fn |
| Catalog integration | `AilakeCatalog` | — | — (use `parquet_scan` for joins) |
| Native lib loading | JNA | JNA | `dlopen` |

## Tests

```bash
AILAKE_LIB=./target/release/libailake_jni.so \
AILAKE_EXT=./duckdb-ailake/build/ailake.duckdb_extension \
AILAKE_FIXTURE=./compat-fixture \
python duckdb-ailake/test/test_search.py

AILAKE_LIB=./target/release/libailake_jni.so \
AILAKE_EXT=./duckdb-ailake/build/ailake.duckdb_extension \
python duckdb-ailake/test/test_write.py
```
