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
  "warehouse":       "/path/to/warehouse",
  "namespace":       "default",
  "table":           "my_table",
  "vec_col":         "embedding",
  "dim":             1536,
  "metric":          "cosine",
  "precision":       "f16",
  "ids":             [1, 2, 3],
  "embeddings":      [[0.1, 0.2, "..."], [0.3, 0.4, "..."], [0.5, 0.6, "..."]],
  "ivf_residual":    false,
  "embedding_model": "text-embedding-3-small@v1",
  "partition_by":    "agent_id",
  "partition_value": "agent-42"
}
```

Optional fields:
- `ivf_residual` (bool, default `false`) — enable residual PQ encoding (`vec - cluster_centroid`); improves recall@10 by ~2-4 pp at same storage.
- `embedding_model` (string, default absent) — model identifier stored in Iceberg properties (`ailake.embedding-model`) and in the per-file Avro `key_metadata`. Format: `"<name>"` or `"<name>@<version>"`. Used for mismatch detection and migration tracking.
- `partition_by` (string, default absent) — Iceberg identity partition column (e.g. `"agent_id"`). Stored in `metadata.json` as an Iceberg partition spec; subsequent readers can prune files by this column (Phase 9).
- `partition_value` (string, default absent) — value for this write, tagged in per-file `key_metadata` JSON in the Avro manifest. Must be set whenever `partition_by` is set.

**Response JSON:** `{"ok": true, "snapshot_id": 7}` or `{"ok": false, "error": "..."}`.

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
- `text_column` (string, default `"chunk_text"`) — Parquet column to score.
- `partition_filter` (string, default absent) — restrict to files tagged with this `partition_value`.

**Response JSON:** `{"ok":true,"results":[{"row_id":N,"distance":F,"file_path":"..."}]}` where `distance` = negated BM25 score (lower = more relevant, consistent with vector search convention).

On error: `{"ok": false, "error": "..."}`.

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
