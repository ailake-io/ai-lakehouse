# ailake — AI-Lake Format Python SDK

**Version**: 0.1.1 — Unified storage for tabular data, embeddings, and HNSW vector index in a single Parquet-compatible file. Apache Iceberg Spec v2/v3 compatible.

## Install

```bash
pip install ailake
```

Requires Python ≥ 3.9. Dependencies: `pyarrow >= 14.0`, `numpy >= 1.24`.

## Quickstart

### Write + search — fluent API (recommended)

```python
import ailake
import numpy as np

# Open or create a table
table = ailake.open_table(
    "./my_table",
    dim=1536,
    metric="cosine",          # cosine | euclidean | dot_product | normalized_cosine
    pre_normalize=True,       # normalize at write time; enables fast 1-dot(a,b) path
    hnsw_m=16,                # HNSW connections per node (default 16)
    hnsw_ef_construction=150,
    embedding_model="text-embedding-3-small",  # tracked in Iceberg metadata
    embedding_model_version="v1",
)

texts = ["Document about AI", "Another document"]
embeddings = np.random.rand(2, 1536).astype(np.float32)

table.insert(texts, embeddings)   # accepts list or numpy array
snapshot_id = table.commit()

# Tabular metadata columns alongside text + embedding
table.insert(
    texts, embeddings,
    extra_columns={"id": [1, 2], "category": ["news", "blog"]},  # type inferred from 1st element
)
table.commit()

# Pattern B — auto-embed without passing embeddings explicitly
def my_embed(texts: list[str]) -> list[list[float]]:
    return np.random.rand(len(texts), 1536).tolist()  # replace with real model

table2 = ailake.open_table("./my_table2", dim=1536, embed_fn=my_embed)
table2.insert(["Document about AI", "Another document"])  # embed_fn called automatically
table2.commit()

# Pointer-only search (default — backward-compatible)
df      = table.search(embeddings[0], top_k=10).to_pandas()   # row_id, distance, file
lf      = table.search(embeddings[0]).limit(5).to_polars()
results = table.search(embeddings[0]).to_list()   # list[dict]

# Full row data — all Parquet columns + _distance
df_full = table.search(embeddings[0], top_k=10, fetch_data=True).to_pandas()
```

### Async API

```python
import ailake, asyncio
import numpy as np

async def main():
    table = ailake.open_table("./my_table", dim=1536)
    await table.insert_async(texts, embeddings)
    await table.commit_async()

    # fluent async chain
    df = await table.search(query_vec).limit(10).to_pandas_async()

    # parallel searches via asyncio.gather
    r1, r2 = await asyncio.gather(
        table.search(q1).to_list_async(),
        table.search(q2).to_list_async(),
    )

asyncio.run(main())
```

### Module-level search

```python
import ailake
import numpy as np

query = np.random.rand(1536).astype(np.float32)

df     = ailake.search("./my_table", query, top_k=10).to_pandas()
lf     = ailake.search("./my_table", query).limit(5).to_polars()
items  = ailake.search("./my_table", query).to_list()
```

### Assemble context for LLMs

```python
import ailake

chunks = [
    {
        "document_id": "doc-1",
        "chunk_index": 0,
        "chunk_text": "AI-Lake stores vectors and tabular data together.",
        "document_title": "AI-Lake Overview",
        "section_path": "Introduction",
        "source_uri": "s3://my-lake/docs/overview.pdf",
        "distance": 0.12,
    },
]

context_xml = ailake.assemble_context(
    chunks=chunks,
    max_tokens=4096,       # token budget (4 chars ≈ 1 token)
    dedup_threshold=0.05,  # drop near-duplicate chunks
)
# Pass context_xml directly to Claude / GPT-4 as a user message
```

## API reference

### `open_table(path, *, ...) → Table`

Opens or creates an AI-Lake table at `path`.

| Parameter | Default | Description |
|---|---|---|
| `path` | required | Table root (local, `s3://`, `gs://`, `az://`) |
| `vector_column` | `"embedding"` | Vector column name |
| `dim` | `1536` | Embedding dimension |
| `metric` | `"cosine"` | `cosine`, `euclidean`, `dot_product`, `normalized_cosine` |
| `pre_normalize` | `False` | Normalize to unit L2 at write; enables `1-dot(a,b)` fast path (~12-20 % speedup) |
| `hnsw_m` | `None` (=16) | HNSW connections per node |
| `hnsw_ef_construction` | `None` (=150) | HNSW build pool size |
| `pq_only` | `False` | Discard raw F16 vectors after index build — only PQ codes stored. ~98 % storage reduction; reranking disabled; recall@10 ~93-95 %. |
| `ivf_residual` | `False` | Encode `vec − cluster_centroid` per IVF cell (residual PQ). Same storage as standard PQ; ~2-4 pp better recall@10. |
| `embedding_model` | `None` | Embedding model name stored in Iceberg properties (`ailake.embedding-model`). Used for mismatch detection and migration tracking. |
| `embedding_model_version` | `None` | Optional model version. Stored as `"<name>@<version>"` in Iceberg properties. |
| `fts_text_columns` | `None` | List of text column names to index with Tantivy FTS (e.g. `["chunk_text", "document_title"]`). When set, each file gets an `AILK_FTS` section; `search_text()` uses O(log N) Tantivy path instead of BM25 brute-force. |
| `embed_fn` | `None` | Auto-embed callable `list[str] → list[list[float]]`. When set, `insert(texts)` and `write_batch(texts)` can be called without passing `embeddings` — the callable is invoked automatically. |
| `partition_by` | `None` | Single-column Iceberg identity partition (e.g. `"agent_id"`). Stored in `metadata.json`. Prefer `partition_fields` for new tables. |
| `partition_value` | `None` | Per-write value for `partition_by`. Tagged in `key_metadata`; used for manifest-level pruning at search time. |
| `partition_fields` | `None` | Multi-column Iceberg partition spec. List of `(column, transform, column_type)` tuples. Supports all Iceberg transforms: `"identity"`, `"year"`, `"month"`, `"day"`, `"hour"`, `"bucket[N]"`, `"truncate[N]"`. Takes precedence over `partition_by`. Example: `[("topic_id","identity","int"),("date","month","date")]`. |
| `format_version` | `2` | Iceberg format version. Set to `3` to write an Iceberg v3 table. |

### `Table`

| Method | Description |
|---|---|
| `insert(texts, embeddings=None, extra_columns=None) → Table` | Buffer a batch. `embeddings`: `list[list[float]]` or numpy array. When `embed_fn` was set on `open_table()`, `embeddings` may be omitted — the callable is invoked automatically. `extra_columns`: `dict[str, list]` of additional tabular columns (e.g. `id`, `category`) written alongside `text`/`embedding`; type per column inferred from its first element (bool/float/int/str). |
| `write_batch_auto_deferred(texts, embeddings=None, extra_columns=None) → Table` | Deferred write — Parquet persisted immediately (~200k vec/s); index (HNSW or IVF-PQ, auto-selected) built in a background thread. Shard served via flat scan until index ready. |
| `write_batch_idempotent(texts, embeddings, batch_id, extra_columns=None) → Table` | No-op if `batch_id` was already committed — safe for retried Airflow tasks / at-least-once pipelines. |
| `write_batch_multi(texts, columns, extra_columns=None) → Table` | N-column multimodal write — see `TableWriter.write_batch_multi` below. |
| `write_batch_multi_deferred(texts, columns, extra_columns=None) → Table` | Deferred variant of `write_batch_multi`. |
| `commit() → int` | Persist as a new Iceberg snapshot; returns snapshot ID. |
| `search(query, top_k=10, fetch_data=False, partition_filter=None, score_fn=None, hybrid_text=None, text_column="chunk_text", bm25_weight=0.5, pruning_threshold=None, ef_search=None, rerank_factor=None) → SearchQuery` | Lazy, chainable search. `query`: `list[float]` or numpy array. `fetch_data=True` returns all Parquet columns + `_distance`. `hybrid_text` enables BM25+vector RRF fusion. `pruning_threshold` skips files whose centroid is farther than this from the query. `ef_search` overrides the HNSW search pool size. `rerank_factor` corrects PQ approximation error on IVF-PQ tables by fetching `top_k * rerank_factor` candidates and reranking with exact distances. Raises `ModelMismatch` if query dim ≠ table dim. |
| `insert_async(...)` | Async variant of `insert`. |
| `write_batch_auto_deferred_async(...)` | Async variant of `write_batch_auto_deferred`. |
| `commit_async() → int` | Async variant of `commit`. |

`Table` is a context manager: `with ailake.open_table(...) as t: ...`

In Jupyter, `table` renders a styled HTML card showing path and vector config.

### `SearchQuery`

Lazy result set — no I/O until materialised.

| Method | Description |
|---|---|
| `limit(n) → SearchQuery` | Cap to *n* nearest neighbours (chainable). |
| `to_list() → list[dict]` | Always pointer-only: `[{"row_id": int, "distance": float, "file": str}, ...]` |
| `to_arrow() → pyarrow.Table` | Full row data (all columns + `_distance`) when `fetch_data=True`; pointer-only `pyarrow.Table` with columns `row_id, distance, file` otherwise. |
| `to_pandas() → pd.DataFrame` | Full row DataFrame when `fetch_data=True`; pointer-only otherwise. |
| `to_polars() → pl.DataFrame` | Full row DataFrame when `fetch_data=True`; pointer-only otherwise. |
| `to_list_async()` | Async variant. |
| `to_arrow_async()` | Async variant. |
| `to_pandas_async()` | Async variant. |
| `to_polars_async()` | Async variant. |

In Jupyter, `results` renders as an HTML table when executed, pending state otherwise.
When `fetch_data=True`, the HTML table shows all Parquet columns.

#### Full-read mode

```python
# Pointer-only (default — backward-compatible)
df = ailake.search("./my_table", query, top_k=10).to_pandas()
# columns: row_id, distance, file

# Full row data — all Parquet columns + _distance
df = ailake.search("./my_table", query, top_k=10, fetch_data=True).to_pandas()
# columns: text, embedding, ..., _distance

# Same via Table handle
df = table.search(query, top_k=10, fetch_data=True).to_pandas()
```

`fetch_data=True` reads each matching Parquet file once and uses `arrow_select::take` to extract only the matched rows — no full table scan.

### `search(path, query, top_k=10, fetch_data=False, partition_filter=None, score_fn=None, hybrid_text=None, text_column="chunk_text", bm25_weight=0.5, pruning_threshold=None, ef_search=None, rerank_factor=None) → SearchQuery`

Module-level search returning the same chainable `SearchQuery`.

- `partition_filter` — restrict to files with matching `partition_value`; pruning at manifest level before HNSW I/O.
- `hybrid_text` — BM25 query string; when set, retrieves `10×top_k` HNSW candidates and fuses via RRF with `bm25_weight`.
- `pruning_threshold` — geometric pruning distance; files whose centroid distance exceeds this are skipped. Default `None` = no pruning.
- `ef_search` — HNSW search pool size. Larger = higher recall, slower. Default `None` = table default (50).
- `rerank_factor` — when set, fetches `top_k * rerank_factor` HNSW candidates and reranks with exact F32 distances — corrects PQ approximation error on IVF-PQ-indexed tables. Default `None` = off. Also honored by `search_with_data`/`scan` and `search_multimodal`.
- `score_fn` — re-ranking callable `(distance: float, row: Any) -> float`. Requires `fetch_data=True`.

### `VectorColSpec(column, dim, metric="cosine", modality=None, precision="f16", pre_normalize=False, hnsw_m=None, hnsw_ef_construction=None)`

Declares one vector column for multi-column writes or searches.

| Arg | Description | Example |
|---|---|---|
| `column` | Parquet column name | `"image_embedding"` |
| `dim` | Embedding dimension | `512` |
| `metric` | Distance metric | `"cosine"` |
| `modality` | Optional tag — stored as `ailake.modality-<column>` | `"text"` / `"image"` / `"audio"` / `"video"` |
| `precision` | Per-column storage precision | `"f16"` (default) / `"f32"` / `"i8"` |
| `pre_normalize` | Normalize this column's vectors to unit L2 at write time | `False` (default) |
| `hnsw_m` / `hnsw_ef_construction` | Per-column HNSW tuning (`None` = table/library default) | `8` / `100` |

### `TableWriter.write_batch_multi(texts, columns, extra_columns=None)`

Write a batch with **N independent vector columns** in one call. Each column gets its own HNSW index in the AILK section of the file footer. `extra_columns` works the same as on `write_batch` (e.g. for `MultimodalContextSchema` fields like `media_uri`/`media_caption`). `write_batch_multi_deferred(...)` is the deferred-index variant.

```python
from ailake import TableWriter, VectorColSpec

text_spec  = VectorColSpec("embedding",       1536, "cosine", "text")
image_spec = VectorColSpec("image_embedding",  512, "cosine", "image")

writer = TableWriter("s3://my-lake/media/", dim=1536, metric="cosine")
writer.write_batch_multi(
    texts,
    [(text_spec, text_embeddings), (image_spec, image_embeddings)],
    extra_columns={"media_uri": media_uris, "media_caption": captions},
)
snapshot_id = writer.commit()
```

### `TableWriter.write_batch_ivf_pq(texts, embeddings, extra_columns=None)` / `write_batch_ivf_pq_deferred(...)`

Forces IVF-PQ indexing regardless of `write_batch_auto_deferred`'s hardware/batch-size heuristic — smaller index, better for S3 sequential-scan workloads. The `_deferred` variant persists Parquet immediately and builds the index in the background.

### `search_multimodal(path, queries, top_k=10, ef_search=None, pruning_threshold=None, rerank_factor=None) → list[dict]`

Cross-modal search: fuse results from N vector columns via **Reciprocal Rank Fusion**.

`rrf_score = Σ weight_i / (60 + rank_i)` — higher is better.

```python
results = ailake.search_multimodal(
    "s3://my-lake/media/",
    queries=[
        ("embedding",       text_vec,  0.7),   # 70% weight on text similarity
        ("image_embedding", image_vec, 0.3),   # 30% weight on image similarity
    ],
    top_k=20,
)
# Returns: [{"row_id": int, "rrf_score": float, "file": str}, ...]
# Ordered by descending rrf_score
```

Each column is searched by its own HNSW. Per-column dimensions are auto-detected
from `ailake.dim-<col>` Iceberg properties written at `commit()` time — no `dim`
argument needed when reading tables written with `write_batch_multi`.

### `Agent(table_path, embed_fn, agent_id=None)` — Phase 9 episodic memory

High-level helper for agent frameworks (LangChain, CrewAI, AutoGen). Wraps `TableWriter` + `search` + `ContextAssembler` with hybrid scoring (distance × recency × importance) and automatic per-agent partition isolation. Metadata (`agent_id`, `session_id`, `mem_type`, `importance`, `created_at`, `last_accessed_at`, `tool_name`, `outcome`, ...) is written as real typed columns — the table stays queryable by any AI-Lake client (Spark/Trino/Flink/DuckDB/CLI), and `ailake.decay_memories(table_path)` works against it directly (it reads the real `last_accessed_at` Timestamp column, not a JSON-packed string).

```python
import ailake

agent = ailake.Agent(
    table_path="s3://my-lake/agents/",
    embed_fn=my_embed_fn,         # list[str] → list[list[float]]
    agent_id="agent-uuid-here",   # isolates reads/writes to this agent's shard
)

# Store a memory with optional importance score
agent.remember("Deployment failed due to OOM on Tuesday", importance=0.9)

# Recall relevant memories — hybrid score = distance × recency × importance
results = agent.recall("deployment issues", top_k=5)

# Log a tool call for later retrieval
agent.log_tool_call(
    name="web_search",
    input={"q": "python asyncio timeout"},
    output={"hits": 5},
    outcome="success",
    latency_ms=120,
)

# Assemble context for LLM prompt (dedup + token budget)
context_xml = agent.assemble_context("why did deployment fail?", max_tokens=4096)
```

| Method | Description |
|---|---|
| `remember(text, importance=1.0)` | Embeds `text` and stores it as an `EpisodicMemorySchema` row tagged with `agent_id`. |
| `recall(query, top_k=5)` | Embeds `query`, searches with `partition_filter=self.agent_id`, applies hybrid score. |
| `log_tool_call(name, input, output, outcome="success", latency_ms=0)` | Stores a `ToolCallSchema` row — searchable by tool name and context. |
| `assemble_context(query, max_tokens=4096)` | `recall()` + `ContextAssembler` — returns prompt-ready XML. |

### `migrate_embeddings(path, old_column, new_column, embed_fn, *, ...)`

Re-embeds all chunks in a table with a new model, committing the result as a new Iceberg snapshot.

```python
ailake.migrate_embeddings(
    path         = "s3://my-lake/docs/",
    old_column   = "embedding",        # existing vector column
    new_column   = "embedding_v2",     # destination column (may be same name)
    embed_fn     = my_embed_fn,        # callable: list[str] → list[list[float]]
    text_column  = "chunk_text",       # source text column
    strategy     = "dual_write_then_cutover",  # or "atomic_replace"
    batch_size   = 512,
    new_model    = "text-embedding-3-large",
    new_model_version = "v1",
    on_progress  = lambda *, files_done, files_total, rows_migrated: print(
        f"{files_done}/{files_total} files, {rows_migrated} rows"
    ),
)
```

| Parameter | Default | Description |
|---|---|---|
| `path` | required | Table root URI |
| `old_column` | required | Existing vector column to migrate from |
| `new_column` | required | Destination vector column |
| `embed_fn` | required | `list[str] → list[list[float]]` callable |
| `text_column` | `"chunk_text"` | Parquet column containing the source text |
| `strategy` | `"dual_write_then_cutover"` | `"dual_write_then_cutover"` (zero downtime, 2× peak storage) or `"atomic_replace"` (lower storage, brief mixed-model window) |
| `batch_size` | `512` | Rows passed to `embed_fn` per call |
| `new_model` | `None` | Model name written to `ailake.embedding-model` after migration |
| `new_model_version` | `None` | Optional version suffix |
| `on_progress` | `None` | Callable invoked after each file with keyword args `files_done`, `files_total`, `rows_migrated` |

### `TableWriter` (low-level — use `open_table()` for most cases)

```python
# Standard HNSW write with model tracking
writer = ailake.TableWriter(
    path, dim=1536, metric="cosine",
    embedding_model="text-embedding-3-small",
    embedding_model_version="v1",
)
writer.write_batch(texts, embeddings)
snapshot_id = writer.commit()

# Pattern B — auto-embed: omit embeddings, SDK calls embed_fn
writer = ailake.TableWriter(
    path, dim=1536,
    embed_fn=lambda texts: my_model.encode(texts).tolist(),
)
writer.write_batch(texts)  # no embeddings arg needed
writer.commit()

# PQ-only — raw vectors discarded after index build (~98 % storage reduction)
writer = ailake.TableWriter(path, dim=1536, metric="cosine", pq_only=True)
writer.write_batch(texts, embeddings)
writer.commit()

# Residual PQ — per-cluster encoding for better recall
writer = ailake.TableWriter(path, dim=1536, metric="cosine", ivf_residual=True)
writer.write_batch(texts, embeddings)
writer.commit()

# Deferred write — Parquet immediate, index background (~200k vec/s)
writer = ailake.TableWriter(path, dim=1536, metric="cosine")
writer.write_batch_auto_deferred(texts, embeddings)
writer.commit()
```

`TableWriter` parameters: same as `open_table()` (includes `pq_only`, `ivf_residual`, `pre_normalize`, `hnsw_m`, `hnsw_ef_construction`, `embedding_model`, `embedding_model_version`, `embed_fn`, `partition_by`, `partition_value`, `partition_fields`, `format_version`).

### `delete_where(path, column, values) → None`

Commits an Iceberg equality delete. No data files are rewritten.

```python
ailake.delete_where("./my_table", "id", ["doc-obsolete-1", "doc-obsolete-2"])
```

### `add_column(path, name, col_type, *, required=False, initial_default=None) → int`

Adds column to live table schema. Returns new `schema_id`. No data files rewritten.

```python
ailake.add_column("./my_table", "source_url", "string", required=False, initial_default="")
```

### `rename_column(path, old_name, new_name) → int`

Renames column. Returns new `schema_id`.

### `hardware_info() → dict[str, str]`

Returns hardware profile of current machine.

```python
info = ailake.hardware_info()
# {
#   "backend":           "cpu-simd",   # or "nvidia-cuda" / "amd-rocm"
#   "has_cuda":          "false",
#   "has_rocm":          "false",
#   "cpu_logical_cores": "16",
#   "has_avx2":          "true",
#   "has_avx512":        "false",
#   "recommend_ivf_pq":  "true",       # true when has GPU OR (cores > 8 AND n >= 5000)
# }
```

Call before `write_batch_auto_deferred` to understand what index type will be selected.

### `compact(path, *, min_files=4, target_size_bytes=536870912, max_files_per_pass=20, deferred=False) → dict`

Native binding — calls `ailake_query::compaction` directly (no external `ailake` CLI binary required). Merges small files into a larger file and rebuilds the HNSW/IVF-PQ index. Returns `{"ok": True, "files_compacted": N, "output_path": str | None}`. No-op when fewer than `min_files` qualify.

```python
result = ailake.compact("s3://my-lake/docs/", min_files=5)
# {"ok": True, "files_compacted": 1, "output_path": "data/compacted-..."}
```

### `estimate(rows, dim, hnsw_m=16, pq_m=None) → list[dict]`

Pure-math storage estimate (no I/O) across 6 precision modes (F32/F16/I8, with/without IVF-PQ, PQ-only). Mirrors `ailake estimate`'s CLI output exactly.

```python
for row in ailake.estimate(rows=1_000_000, dim=1536):
    print(row["mode"], row["total_bytes"], row["recall"])
```

### `add_vector_column(table_path, column, dim, metric="cosine", precision="f16", pre_normalize=False, hnsw_m=None, hnsw_ef_construction=None) → int`

Adds a new vector column to an existing table's schema without rewriting data files. Old files return `null` for this column until `backfill_vector_column` runs. Returns the new schema-id.

### `backfill_vector_column(table_path, column, embed_fn, text_column="chunk_text", batch_size=512) → None`

Backfills a column added via `add_vector_column` across all existing files by embedding `text_column`. Idempotent — files that already have the column are skipped.

### `evolve_schema(path, *, add_columns=None, rename_columns=None) → int`

Applies schema evolution in a single metadata-only call (no data files rewritten). Combines `add_column` + `rename_column` in order. Returns final `schema_id`.

```python
ailake.evolve_schema(
    "s3://my-lake/docs/",
    add_columns=[{"name": "score", "type": "float", "initial_default": 0.0}],
    rename_columns=[{"from": "old_text", "to": "chunk_text"}],
)
```

### `now_ns() → int`

Returns current Unix epoch time in nanoseconds. Use to populate `created_at` / `last_accessed_at` columns (Arrow `Timestamp(ns, UTC)`).

```python
ts = ailake.now_ns()   # e.g. 1750000000000000000
```

### `delete_rows(path, file_path, row_ids) → None`

Low-level Rust binding: physically removes rows from a specific Parquet file within the table, rebuilding the HNSW index. For logical Iceberg deletes (no file rewrite), use `delete_where` instead.

### `search_text(path, query_text, top_k=10, text_columns=None) → list[dict]`

Pure BM25 full-text search — no HNSW required. O(N) brute-force scan; uses Tantivy O(log N) fast path for files that have a Tantivy FTS index embedded (written with `fts_text_columns=`).

```python
# BM25 search (no embedding needed)
hits = ailake.search_text("s3://my-lake/docs/", "rust async programming", top_k=10)
# Returns: [{"row_id": int, "score": float, "file": str}]  (score: higher = more relevant)

# Restrict to specific text columns (default: ["chunk_text"])
hits = ailake.search_text(path, "query", text_columns=["chunk_text", "document_title"])
```

### `scan(path, query, top_k=10, ...) → bytes`

Alias for `search_with_data` — same capability as ailake-go's `Scan()` and ailake-jni's `ailake_scan_json` (search + full-row fetch in one call, no JOIN needed against a separately-registered table). See `search_with_data` below.

### `assemble_context(chunks, max_tokens=4096, dedup_threshold=0.05, group_by_document=True, max_chunks_per_document=10) → dict`

Assembles chunk dicts into structured XML for LLM input. Returns `{"text": str, "chunk_count": int, "token_estimate": int}` (previously returned a bare string). Deduplicates near-identical chunks within the token budget — pass an `"embedding": list[float]` key per chunk to enable cosine-distance dedup; chunks without it are never deduplicated.

```python
ctx = ailake.assemble_context(chunks, max_tokens=2048)
print(ctx["text"])         # XML ready for LLM input
print(ctx["chunk_count"])  # how many chunks made it in
```

## Storage modes and index types

| Mode | `pq_only` | `ivf_residual` | Storage (dim=1536, 1M rows) | Reranking | Recall@10 |
|---|---|---|---|---|---|
| HNSW + F16 raw (default) | `False` | `False` | ~300 GB vectors + ~30 GB HNSW | Yes (exact) | ~97 % |
| IVF-PQ + F16 raw | `False` | `False` | ~300 GB + ~5 GB PQ codes | Yes (exact) | ~93 % inline |
| IVF-PQ residual + raw | `False` | `True` | ~300 GB + ~5 GB | Yes (exact) | ~96 % |
| PQ-only | `True` | `False` | **~5 GB total** | No | ~93-95 % |
| PQ-only residual | `True` | `True` | **~5 GB total** | No | ~94-96 % |

```python
# Deferred write — all modes, instant Parquet commit, index in background
writer = ailake.TableWriter(path, dim=1536, pq_only=True, ivf_residual=True)
writer.write_batch_auto_deferred(texts, embeddings)
writer.commit()
```

## HNSW tuning guide

| Goal | `hnsw_m` | `hnsw_ef_construction` |
|---|---|---|
| Low latency / high QPS | 8 | 100 |
| General purpose (default) | 16 | 150 |
| High recall (RAG) | 24 | 200 |
| Max recall (medical, legal) | 32 | 400 |

## Type checking

Ships `py.typed` (PEP 561) and `ailake/_ailake.pyi` stubs. `mypy` and `pyright` work out of the box with no configuration.

## Iceberg compatibility

Tables are valid Apache Iceberg Spec v2. Spark, Trino, DuckDB, and PyIceberg read tabular columns normally; the HNSW index lives in an extension section that standard Parquet readers silently ignore.

## License

MIT OR Apache-2.0
