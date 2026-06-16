# duckdb-ailake

DuckDB community extension that exposes AI-Lake vector search and write via SQL table/scalar functions.

Bridges DuckDB to [`libailake_jni.so`](../ailake-jni) using the same C-ABI as the Spark and Trino plugins — zero additional Rust code required.

## Functions

### `ailake_search` — vector similarity search

```sql
SELECT * FROM ailake_search(
    table_path   VARCHAR,    -- path/URI to AI-Lake table root
    query        FLOAT[],    -- query embedding (LIST(FLOAT))
    top_k        INTEGER,    -- number of nearest neighbors
    -- named (optional):
    vec_col      VARCHAR     -- default 'embedding'
    ef_search    INTEGER     -- HNSW ef parameter, default 50
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
```

### `ailake_search_multimodal` — cross-modal RRF search (Phase 8)

```sql
SELECT * FROM ailake_search_multimodal(
    table_path  VARCHAR,                -- path/URI to AI-Lake table root
    queries     LIST(STRUCT(           -- one entry per vector column
                    col    VARCHAR,    -- column name
                    query  FLOAT[],   -- query embedding
                    weight FLOAT)),   -- RRF weight (higher = more influential)
    top_k       INTEGER
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
```

Returns 0 rows (no error) if `libailake_jni.so` is not loaded or does not export `ailake_search_multimodal_json`.

---

### `ailake_write_batch` — ingest embeddings

```sql
-- 3-arg form (defaults: vec_col=embedding, metric=cosine, precision=f16)
SELECT ailake_write_batch(
    table_path   VARCHAR,         -- table root path/URI
    ids          BIGINT[],        -- row identifiers
    embeddings   FLOAT[][]        -- one embedding per id
) → BIGINT  -- snapshot_id, or -1 on error

-- 6-arg form (explicit options)
SELECT ailake_write_batch(
    table_path,
    ids,
    embeddings,
    vec_col    VARCHAR,           -- embedding column name
    metric     VARCHAR,           -- cosine | euclidean | dot
    precision  VARCHAR            -- f32 | f16 | i8
) → BIGINT
```

**Example:**

```sql
SELECT ailake_write_batch(
    'file:///data/my_table',
    [0, 1, 2]::BIGINT[],
    [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]]::FLOAT[][]
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
