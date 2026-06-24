# ailake-jni

C-ABI cdylib that exposes the [AI-Lake](https://github.com/ThiagoLange/ai-lakehouse) vector search engine to JVM runtimes (Spark, Trino, Flink) via [JNA](https://github.com/java-native-access/jna).

## Overview

`ailake-jni` compiles to a single shared library (`libailake_jni.so` / `ailake_jni.dll`) that all JVM plugins load at runtime. The API uses a **JSON-envelope pattern** — callers pass a JSON request string and receive a JSON response string — making it callable from any JVM language (Scala, Kotlin, Java) without code generation.

## Exported C-ABI functions

### `ailake_search_json`

```
fn ailake_search_json(request_json: *const c_char) -> *mut c_char
```

Performs a nearest-neighbor vector search on a local AI-Lake table.

**Request JSON:**

```json
{
  "warehouse":        "/path/to/warehouse",
  "namespace":        "default",
  "table":            "my_table",
  "vec_col":          "embedding",
  "dim":              1536,
  "query":            [0.1, -0.2, 0.3, "..."],
  "top_k":            10,
  "ef_search":        50,
  "partition_filter": "agent-42"
}
```

Optional fields:
- `partition_filter` (string, default absent) — restrict search to manifest entries where `partition_value` matches this string. Pruning happens before geometric centroid check and HNSW load (Phase 9).
- `hybrid_text` (string, default absent) — query text for BM25 hybrid scoring. When set, pipeline retrieves `10×top_k` HNSW candidates, scores each by BM25, and fuses via RRF.
- `text_column` (string, default `"chunk_text"`) — Parquet column to score for BM25. Only used when `hybrid_text` is set.
- `bm25_weight` (float, default `0.5`) — relative BM25 weight in RRF fusion. Only used when `hybrid_text` is set.

**Response JSON:**

```json
{
  "ok": true,
  "results": [
    { "row_id": 42, "distance": 0.123, "file_path": "data/part-00001.parquet" }
  ]
}
```

On error: `{"ok": false, "error": "..."}`.

When the query vector dimension does not match the table dimension, the error message names the stored embedding model:
```json
{"ok": false, "error": "query dim=512 does not match table dim=1536 (table model: text-embedding-3-small@v1)"}
```

---

### `ailake_write_batch_json`

```
fn ailake_write_batch_json(request_json: *const c_char) -> *mut c_char
```

Writes a batch of records and their embeddings to an AI-Lake table.

**Request JSON:**

```json
{
  "warehouse":        "/path/to/warehouse",
  "namespace":        "default",
  "table":            "my_table",
  "vec_col":          "embedding",
  "dim":              1536,
  "metric":           "cosine",
  "precision":        "f16",
  "ids":              [1, 2, 3],
  "embeddings":       [[0.1, 0.2, "..."], [0.3, 0.4, "..."], [0.5, 0.6, "..."]],
  "ivf_residual":     false,
  "embedding_model":  "text-embedding-3-small@v1",
  "partition_by":     "agent_id",
  "partition_value":  "agent-42",
  "partition_fields": [{"column":"topic_id","transform":"identity","column_type":"int"}],
  "format_version":   3
}
```

Optional fields:
- `ivf_residual` (bool, default `false`) — enable residual PQ encoding (`vec - cluster_centroid`); improves recall@10 by ~2-4 pp at same storage.
- `embedding_model` (string, default absent) — model identifier stored in Iceberg properties (`ailake.embedding-model`). Format: `"<name>"` or `"<name>@<version>"`.
- `pre_normalize` (bool, default `false`) — normalize vectors to unit L2 at write time; enables `1-dot(a,b)` fast path in HNSW (~12-20% speedup for `cosine` metric). Stored as `ailake.pre-normalize` in Iceberg properties.
- `fts_columns` (array of strings, default `[]`) — text column names to index with Tantivy FTS (e.g. `["chunk_text","document_title"]`). When set, each file receives an `AILK_FTS` section; `ailake_search_text_json` uses O(log N) Tantivy path instead of BM25 brute-force.
- `fts_tokenizer` (string, default `"simple"`) — Tantivy tokenizer: `"simple"` (whitespace + lowercase) or `"raw"` (no tokenization).
- `partition_by` (string, default absent) — single-column Iceberg identity partition column (legacy; prefer `partition_fields` for new tables).
- `partition_value` (string, default absent) — value for `partition_by`. Must be set when `partition_by` is set.
- `partition_fields` (array, default `[]`) — multi-column Iceberg partition spec. Each object: `{column, transform, column_type}`. Supports all Iceberg transforms: `identity`, `year`, `month`, `day`, `hour`, `bucket[N]`, `truncate[N]`. Takes precedence over `partition_by` when non-empty.
- `format_version` (int, default `2`) — Iceberg format version. Set to `3` to enable Iceberg v3.

**Response JSON:** `{"ok": true, "snapshot_id": 7}` or `{"ok": false, "error": "..."}`.

---

### `ailake_delete_where_json`

```
fn ailake_delete_where_json(request_json: *const c_char) -> *mut c_char
```

Writes an Iceberg equality delete file and commits a Delete snapshot. No data files are rewritten.

**Request JSON:**

```json
{
  "warehouse": "/path/to/warehouse",
  "namespace": "default",
  "table":     "my_table",
  "column":    "id",
  "values":    ["doc-1", "doc-2"]
}
```

- `values` empty array → no-op, returns `{"ok": true}` without writing any file.

**Response JSON:** `{"ok": true}` or `{"ok": false, "error": "..."}`.

---

### `ailake_evolve_schema_json`

```
fn ailake_evolve_schema_json(request_json: *const c_char) -> *mut c_char
```

Applies metadata-only schema evolution. No data files are rewritten. Field IDs are stable.

**Request JSON:**

```json
{
  "warehouse":      "/path/to/warehouse",
  "namespace":      "default",
  "table":          "my_table",
  "add_columns":    [{"name": "source_url", "type": "string", "initial_default": ""}],
  "rename_columns": [{"from": "source_url", "to": "url"}]
}
```

- Both arrays empty → no-op, returns current `schema_id`.
- `initial_default` is optional; absent means `null`.

**Response JSON:** `{"ok": true, "new_schema_id": 5}` or `{"ok": false, "error": "..."}`.

---

### `ailake_search_multimodal_json`

```
fn ailake_search_multimodal_json(request_json: *const c_char) -> *mut c_char
```

Cross-modal vector search with Reciprocal Rank Fusion. Accepts N column queries with individual weights; fuses ranked lists via RRF: `score = Σ weight_i / (60 + rank_i)`.

**Request JSON:**

```json
{
  "warehouse":        "/path/to/warehouse",
  "namespace":        "default",
  "table":            "my_table",
  "queries": [
    { "col": "embedding",       "query": [0.1, -0.2, "..."], "weight": 0.7, "dim": 0 },
    { "col": "image_embedding", "query": [0.3,  0.4, "..."], "weight": 0.3, "dim": 0 }
  ],
  "top_k":            10,
  "partition_filter": "agent-42"
}
```

`dim: 0` means auto-detect from table metadata. `partition_filter` is optional — restricts to files with a matching `partition_value` (Phase 9). Each `col` is a vector column name; if the column is the table's primary column, its main HNSW index is used; otherwise the secondary index from `extra_vector_indexes` in the file manifest is used.

**Response JSON:**

```json
{
  "ok": true,
  "results": [
    { "row_id": 42, "rrf_score": 0.0284, "file_path": "data/part-00001.parquet" }
  ]
}
```

`rrf_score` is positive (higher = more relevant). On error: `{"ok": false, "error": "..."}`.

---

### `ailake_search_text_json`

```
fn ailake_search_text_json(request_json: *const c_char) -> *mut c_char
```

Pure BM25 full-text search — no HNSW required. Scans all Parquet files and returns top-k by BM25 score.

**Request JSON:**

```json
{
  "warehouse":        "/path/to/warehouse",
  "namespace":        "default",
  "table":            "my_table",
  "query_text":       "rust programming async",
  "top_k":            10,
  "text_column":      "chunk_text",
  "partition_filter": "agent-42"
}
```

Optional fields:
- `text_columns` (array of strings, default `["chunk_text"]`) — Parquet columns to score. Each column is searched independently; scores are combined. Example: `["chunk_text","document_title"]`.
- `text_column` (string, deprecated alias for `text_columns[0]`) — still accepted for backward compatibility.
- `partition_filter` (string, default absent) — restrict to files tagged with this `partition_value`.

**Response JSON:** `{"ok":true,"results":[{"row_id":N,"distance":F,"file_path":"..."}]}` where `distance` = negated BM25 score (lower = more relevant, consistent with vector search convention).

On error: `{"ok": false, "error": "..."}`.

---

### `ailake_compact_json`

```
fn ailake_compact_json(request_json: *const c_char) -> *mut c_char
```

Triggers a compaction pass on a local AI-Lake table — merges small files into fewer larger files and rebuilds the HNSW index.

**Request JSON:**

```json
{
  "warehouse":         "/path/to/warehouse",
  "namespace":         "default",
  "table":             "my_table",
  "min_files":         4,
  "target_size_bytes": 536870912,
  "max_files_per_pass": 20,
  "deferred":          false
}
```

Optional fields:
- `min_files` (int, default `4`) — only compact when there are at least this many files.
- `target_size_bytes` (int, default 512 MB) — target merged file size.
- `max_files_per_pass` (int, default `20`) — maximum files to merge in one pass.
- `deferred` (bool, default `false`) — when `true`, writes Parquet immediately and builds the HNSW index in a background task.

**Response JSON:** `{"ok":true,"files_compacted":3,"snapshot_id":12}` or `{"ok":false,"error":"..."}`.

---

### `ailake_scan_json`

```
fn ailake_scan_json(request_json: *const c_char) -> *mut c_char
```

Full-read scan: performs nearest-neighbor search and returns complete Parquet row data (all columns) alongside search metadata. Equivalent to `search(…, fetch_data=True)` in Python.

**Request JSON:**

```json
{
  "warehouse":        "/path/to/warehouse",
  "namespace":        "default",
  "table":            "my_table",
  "vec_col":          "embedding",
  "dim":              1536,
  "query":            [0.1, -0.2, "..."],
  "top_k":            10,
  "ef_search":        50,
  "partition_filter": "agent-42"
}
```

**Response JSON:** `{"ok":true,"results":[{"row_id":N,"distance":F,"file_path":"...","columns":{...}}]}` where `columns` contains all Parquet columns for that row.

---

### `ailake_vector_search_json` (legacy binary API)

```
fn ailake_vector_search_json(
    table_uri:  *const c_char,
    query_ptr:  *const f32,
    query_len:  u32,
    top_k:      u32,
) -> *mut c_char
```

Binary-parameter API — accepts a raw `f32` array pointer rather than a JSON-encoded query. Used by the DuckDB extension (which calls into C-ABI directly without JSON marshalling). Hardcodes namespace `"default"`, table `"table"`, `vec_col` `"embedding"`.

- Null `table_uri` or `query_ptr` → returns `[]` (empty JSON array).
- `query_len > 65 536` → returns `{"ok":false,"error":"query_len N exceeds maximum supported dimension (65536)"}`.
- On error → `{"ok":false,"error":"..."}`.
- On success → `{"ok":true,"results":[{"row_id":N,"distance":F,"file_path":"..."}]}`.

**Prefer `ailake_search_json` for new integrations.** This function exists for DuckDB's native binary call path.

### Security constraints

| Limit | Applied in | Value |
|---|---|---|
| Max `query_len` (legacy API) | `ailake_vector_search_json` | 65 536 dimensions |
| Max `ef_search` | `ailake_search_json`, `ailake_scan_json` | 100 000 (clamped via `.min(100_000)`) |

---

### `ailake_free_string`

```
fn ailake_free_string(ptr: *mut c_char)
```

Frees a string returned by any of the above functions. **Always call this** after consuming the response — the JVM garbage collector cannot free Rust-allocated memory.

---

### `ailake_version`

```
fn ailake_version() -> *const c_char
```

Returns the library version as a static null-terminated string. Do **not** free this pointer.

## Usage from JVM plugins

### Kotlin / Trino (JNA)

```kotlin
import com.sun.jna.Library
import com.sun.jna.Native
import com.sun.jna.Pointer

interface AilakeLib : Library {
    fun ailake_search_json(requestJson: String): Pointer
    fun ailake_free_string(ptr: Pointer)
}

val lib: AilakeLib = Native.load("ailake_jni", AilakeLib::class.java)

val request = """{"warehouse":"/data/lake","namespace":"default","table":"docs",
    "vec_col":"embedding","dim":1536,"query":[...],"top_k":10}"""

val ptr = lib.ailake_search_json(request)
val json = ptr.getString(0)
lib.ailake_free_string(ptr)
// parse json...
```

### Scala / Spark (JNA)

```scala
import com.sun.jna.{Library, Native, Pointer}

trait AilakeLib extends Library {
  def ailake_search_json(requestJson: String): Pointer
  def ailake_free_string(ptr: Pointer): Unit
}

val lib = Native.load("ailake_jni", classOf[AilakeLib]).asInstanceOf[AilakeLib]

val ptr = lib.ailake_search_json(requestJson)
val json = ptr.getString(0)
lib.ailake_free_string(ptr)
```

## Library path

The `.so` / `.dll` must be on the native library search path before the JVM starts:

```bash
# Linux
export LD_LIBRARY_PATH=/path/to/lib:$LD_LIBRARY_PATH

# macOS
export DYLD_LIBRARY_PATH=/path/to/lib:$DYLD_LIBRARY_PATH

# JVM flag (all platforms)
-Djava.library.path=/path/to/lib
```

## Building

```bash
cargo build --release -p ailake-jni
# output: target/release/libailake_jni.so  (Linux)
#         target/release/ailake_jni.dll    (Windows)
#         target/release/libailake_jni.dylib (macOS)
```

## License

Licensed under either of [Apache License, Version 2.0](../LICENSE-APACHE) or [MIT License](../LICENSE-MIT) at your option.
