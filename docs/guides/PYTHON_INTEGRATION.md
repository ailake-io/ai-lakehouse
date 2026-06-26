# Python Integration Guide

`ailake` is a PyO3-compiled Rust extension exposing the full AI-Lake SDK to
Python. All heavy operations (HNSW search, IVF-PQ, AILK I/O, Iceberg catalog
writes) run in Rust — Python is only glue. Compatible with Python 3.9+.

---

## 1. Installation

```bash
pip install ailake
```

**From source (development):**

```bash
cd ailake-py
pip install maturin
maturin develop --release
```

**Optional extras** (only needed for the corresponding output format):

```bash
pip install pyarrow   # .to_arrow() / .to_pandas() / .to_polars()
pip install pandas
pip install polars
```

---

## 2. Writing data — `TableWriter`

`TableWriter` is the low-level write interface. `Table` / `open_table` wrap it
with a fluent API. Both go to the same Rust `TableWriter` underneath.

```python
import ailake
import numpy as np

writer = ailake.TableWriter(
    "s3://my-lake/docs/",
    vector_column="embedding",
    dim=1536,
    metric="cosine",
)

texts = ["chunk one", "chunk two", "chunk three"]
embs  = np.random.rand(3, 1536).astype(np.float32)

writer.write_batch(texts, embs.tolist())
snapshot_id = writer.commit()
print(f"committed snapshot {snapshot_id}")
```

**All `TableWriter` parameters:**

```python
writer = ailake.TableWriter(
    "s3://my-lake/docs/",

    # Vector column
    vector_column           = "embedding",        # default
    dim                     = 1536,
    metric                  = "cosine",           # cosine | euclidean | dot_product | normalized_cosine
    pre_normalize           = False,              # normalize to unit-L2 at write (~12-20% search speedup)

    # HNSW tuning (None → use table default stored in Iceberg metadata)
    hnsw_m                  = None,
    hnsw_ef_construction    = None,

    # IVF-PQ options
    pq_only                 = False,              # discard raw F16 after index build (saves ~95% vector storage)
    ivf_residual            = False,              # encode PQ residuals from cluster centroid (+2-4pp recall)

    # Model tracking
    embedding_model         = "text-embedding-3-small",
    embedding_model_version = "2024-01",

    # Auto-embed (omit embeddings arg in write_batch when set)
    embed_fn                = None,               # Callable[[list[str]], list[list[float]]]

    # Partitioning
    partition_by            = "agent_id",         # single partition column (simple)
    partition_value         = "agent-001",
    partition_column_type   = "string",
    partition_fields        = None,               # list of (name, transform, type) for compound keys
    partition_values        = None,               # dict for compound partition values

    # Iceberg
    format_version          = 2,                  # 2 or 3

    # BM25 / FTS
    bm25_text_column        = "chunk_text",       # column for BM25 hybrid search stats
    fts_text_columns        = ["chunk_text"],      # columns to index with Tantivy FTS
    fts_tokenizer           = "default",
)
```

**Write methods on `TableWriter`:**

| Method | Description |
|---|---|
| `write_batch(texts, embeddings, extra_columns)` | Buffer batch; call `commit()` to persist |
| `write_batch_auto_deferred(texts, embeddings)` | Parquet persisted immediately (~200k vec/s); index built async |
| `write_batch_idempotent(texts, embeddings, batch_id)` | No-op if `batch_id` already committed (Airflow restart-safe) |
| `write_batch_multi(texts, [(spec, embs), ...])` | N independent vector columns, each with own HNSW |
| `write_batch_multi_deferred(texts, [...])` | Deferred variant of `write_batch_multi` |
| `commit()` | Persist all buffered batches as new Iceberg snapshot; returns snapshot id |

**Extra columns:**

```python
writer.write_batch(
    texts,
    embs.tolist(),
    extra_columns={
        "language":   ["en", "en", "pt"],
        "score":      [0.9, 0.8, 0.95],
        "is_premium": [True, False, True],
    },
)
```

Types inferred: `bool → Boolean`, `float → Float32`, `int → Int64`, `str → Utf8`.

---

## 3. Fluent API — `open_table` / `Table`

`open_table` returns a `Table` handle combining writer + search in one object.

```python
import ailake

table = ailake.open_table(
    "s3://my-lake/docs/",
    dim=1536,
    metric="cosine",
)

# Write
table.insert(["hello world", "rust embeddings"], embeddings=embs.tolist())
table.commit()

# Search
results = table.search(query_vec, top_k=10)
df = results.to_pandas()
```

**`open_table` / `Table.__init__` parameters:**

Same as `TableWriter` minus partition/format_version (use `TableWriter` directly for those).
Key extras on `Table`:

```python
table = ailake.open_table(
    "s3://my-lake/docs/",
    embed_fn=my_embed_fn,   # auto-embed: table.insert(texts) without embeddings arg
    bm25_text_column="chunk_text",
    fts_text_columns=["chunk_text"],
)
```

**Context manager (auto-commit not applied — commit explicitly):**

```python
with ailake.open_table("s3://my-lake/docs/") as table:
    table.insert(texts, embeddings=embs.tolist())
    table.commit()
```

---

## 4. Vector search — `SearchQuery`

`search()` (module-level or `Table.search()`) returns a lazy `SearchQuery` —
executed only when you call a materialisation method.

```python
import ailake

query_vec = [...]   # list[float] or numpy array

# Module-level
results = ailake.search("s3://my-lake/docs/", query_vec, top_k=10)

# Via Table handle
results = table.search(query_vec, top_k=10)
```

**Materialisation methods:**

```python
results.to_list()    # list[dict] — always [{row_id, distance, file}]
results.to_arrow()   # pyarrow.Table
results.to_pandas()  # pandas.DataFrame
results.to_polars()  # polars.DataFrame
len(results)         # executes if not yet executed
for r in results:    # iterate dicts
    print(r["row_id"], r["distance"])
```

**Full row data (`fetch_data=True`):**

```python
results = ailake.search(
    "s3://my-lake/docs/", query_vec, top_k=10,
    fetch_data=True,   # returns all Parquet columns + _distance
)
df = results.to_pandas()
# columns: chunk_id, chunk_text, embedding, ..., _distance
```

**All search parameters:**

```python
results = ailake.search(
    "s3://my-lake/docs/",
    query_vec,
    top_k             = 10,
    fetch_data        = False,
    partition_filter  = "agent-001",  # restrict to one partition (manifest-level)
    hybrid_text       = "rust async", # BM25+vector RRF (requires bm25_text_column at write)
    text_column       = "chunk_text",
    bm25_weight       = 0.5,          # 0.0=pure vector, 1.0=pure BM25
    pruning_threshold = 0.5,          # geometric pruning; None=no pruning
    ef_search         = 50,           # HNSW ef_search; None=top_k*5
    score_fn          = None,         # Callable[(distance, row) → float]; requires fetch_data=True
)
```

**Chaining:**

```python
results = ailake.search(path, query, top_k=100).limit(5)
```

**Async variants:**

```python
async def search_docs():
    results = ailake.search(path, query_vec, top_k=10)
    df = await results.to_pandas_async()
    return df
```

All materialisation methods have `_async` counterparts:
`to_list_async()`, `to_arrow_async()`, `to_pandas_async()`, `to_polars_async()`.

**`Table.search()` has the same signature** as module-level `search()` minus `path`.

**Custom re-ranking with `score_fn`:**

```python
import math

def recency_score(distance: float, row: dict) -> float:
    days_old = row.get("days_since_update", 0)
    recency  = math.exp(-0.1 * days_old)
    return distance / (recency + 1e-6)

results = ailake.search(
    path, query_vec, top_k=50,
    fetch_data=True,
    score_fn=recency_score,
)
```

**Jupyter / JupyterLab:** `SearchQuery` has `_repr_html_()` — renders as a
styled table in notebooks without calling any materialisation method explicitly.

---

## 5. Full-text search

```python
# BM25 — O(N) brute-force (no HNSW involved)
hits = ailake.search_text(
    "s3://my-lake/docs/",
    "machine learning embeddings",
    top_k=10,
    text_column="chunk_text",
)
# hits: list[dict] with row_id, distance (negated BM25 score), file

# Tantivy FTS (O(log N)) — available when table written with fts_text_columns
# search_text() uses Tantivy when present, BM25 fallback for legacy files.
```

---

## 6. Hybrid search (BM25 + vector)

Pass `hybrid_text` to `search()` to fuse BM25 and vector via RRF:

```python
results = ailake.search(
    "s3://my-lake/docs/",
    query_vec,
    top_k=10,
    hybrid_text="geometric pruning vector index",
    bm25_weight=0.4,        # relative BM25 contribution
    text_column="chunk_text",
)
df = results.to_pandas()
```

Requires `bm25_text_column="chunk_text"` at write time so BM25 IDF stats are
accumulated.

---

## 7. Multimodal search (cross-modal RRF)

```python
import ailake

text_vec  = [...]   # dim=1536
image_vec = [...]   # dim=512

results = ailake.search_multimodal(
    "s3://my-lake/media/",
    queries=[
        ("embedding",       text_vec,  0.7),
        ("image_embedding", image_vec, 0.3),
    ],
    top_k=20,
)
# results: list[dict] with row_id, rrf_score, file — sorted descending by rrf_score
```

**Writing multimodal tables:**

```python
text_spec  = ailake.VectorColSpec("embedding",       1536, "cosine", "text")
image_spec = ailake.VectorColSpec("image_embedding",  512, "cosine", "image")

writer = ailake.TableWriter("s3://my-lake/media/", dim=1536)
writer.write_batch_multi(
    texts,
    [(text_spec, text_embs), (image_spec, image_embs)],
)
writer.commit()
```

---

## 8. Full row data — `search_with_data`

Returns Arrow IPC bytes — useful when you need all Parquet columns with zero
copy to numpy/pandas:

```python
import io, pyarrow as pa

ipc_bytes = ailake.search_with_data(
    "s3://my-lake/docs/",
    query_vec,
    top_k=10,
    partition_filter="agent-001",  # optional
)
table = pa.ipc.open_file(io.BytesIO(ipc_bytes)).read_all()
df = table.to_pandas()  # all Parquet columns + _distance
```

---

## 9. LLM context assembly

`assemble_context` turns search results into structured XML for Claude / GPT-4:

```python
results = ailake.search(path, query_vec, top_k=20, fetch_data=True)
df = results.to_pandas()

chunks = [
    {
        "document_id":   row["document_id"],
        "chunk_index":   int(row["chunk_index"]),
        "chunk_text":    row["chunk_text"],
        "document_title": row.get("title", ""),
        "source_uri":    row.get("source_url", ""),
        "distance":      float(row["_distance"]),
    }
    for _, row in df.iterrows()
]

context_xml = ailake.assemble_context(chunks, max_tokens=4096)
# Feed context_xml into your LLM prompt
```

---

## 10. Compaction

Merges many small Parquet files into a larger file, rebuilding the index:

```python
result = ailake.compact(
    "s3://my-lake/docs/",
    min_files=4,                   # only compact when ≥4 small files exist
    target_size_bytes=128*1024*1024, # 128 MiB target
    max_files_per_pass=20,
    deferred=False,                # True = Parquet now, index async
)
print(result)
# {"ok": True, "files_compacted": 1, "output_path": "data/compacted-..."}
# {"ok": True, "files_compacted": 0}  ← nothing to compact
# {"ok": True, "files_compacted": 0, "warning": "ailake CLI not found; skipping"}
```

`compact()` requires the `ailake` CLI. When the binary is not found, it returns
gracefully with `warning` instead of raising.

---

## 11. Schema evolution

```python
# Add columns (metadata-only — no data files rewritten)
ailake.evolve_schema(
    "s3://my-lake/docs/",
    add_columns=[
        {"name": "language",    "type": "string", "initial_default": "en"},
        {"name": "page_number", "type": "int",    "initial_default": None},
    ],
    rename_columns=[
        {"from": "old_text", "to": "chunk_text"},
    ],
)

# Individual operations
ailake.add_column(
    "s3://my-lake/docs/",
    name="score", col_type="float",
    required=False, initial_default=0.0,
    doc="Quality score from reranker",
)
ailake.rename_column("s3://my-lake/docs/", "old_name", "new_name")
```

---

## 12. Deletes

```python
# Equality delete — Iceberg equality delete file; no data rewrite
ailake.delete_where(
    "s3://my-lake/docs/",
    column="chunk_id",
    values=["uuid-aaa", "uuid-bbb"],
)

# Row-level delete — positional delete for specific row IDs in a file
ailake.delete_rows(
    table_path="s3://my-lake/docs/",
    file_path="data/part-00001.parquet",
    row_ids=[0, 5, 12],
)
```

---

## 13. Embedding model migration

Re-embed an entire table with a new model without downtime:

```python
import openai

def embed(texts: list[str]) -> list[list[float]]:
    resp = openai.embeddings.create(
        model="text-embedding-3-large", input=texts
    )
    return [d.embedding for d in resp.data]

ailake.migrate_embeddings(
    "s3://my-lake/docs/",
    old_column="embedding",         # existing column
    new_column="embedding_v2",      # migrated column (may equal old for in-place upgrade)
    embed_fn=embed,
    text_column="chunk_text",
    strategy="dual_write_then_cutover",  # or "atomic_replace"
    batch_size=512,
    new_model="text-embedding-3-large",
    on_progress=lambda files_done, files_total, rows_migrated:
        print(f"{files_done}/{files_total} files, {rows_migrated} rows"),
)
```

---

## 14. Agent memory — `Agent`

High-level helper for agent frameworks (LangChain, CrewAI, AutoGen).
Hybrid scoring: `score = distance / (recency_weight × importance)`.

```python
import ailake, openai

def embed(texts):
    resp = openai.embeddings.create(model="text-embedding-3-small", input=texts)
    return [d.embedding for d in resp.data]

agent = ailake.Agent(
    "s3://my-lake/agent-memory/",
    embed_fn=embed,
    agent_id="agent-42",     # stable across sessions
    session_id="sess-001",   # current session
    metric="cosine",
    lambda_=0.099,           # half-life ≈ 7 days; 0.693 = daily, 0.023 = monthly
)

# Store memories
mem_id  = agent.remember("User prefers concise responses", importance=0.9)
call_id = agent.log_tool_call(
    "web_search",
    input={"q": "Rust tokio docs"},
    output={"hits": 5},
    outcome="success",
    latency_ms=120,
)
agent.commit()

# Recall with hybrid scoring
query_vec = embed(["async programming"])[0]
memories = agent.recall(query_vec, top_k=5, oversample=3)
for m in memories:
    print(f"[score={m['score']:.3f}  recency={m['recency']:.3f}] {m['text'][:80]}")

# Context for LLM
xml = agent.assemble_context(query_vec, max_tokens=4096)

# Context manager — auto-commits on exit
with ailake.Agent(path, embed_fn=embed, agent_id="agent-42") as agent:
    agent.remember("Meeting notes: discussed Rust performance", importance=0.7)
    # commit() called automatically on __exit__
```

**Async:**

```python
async def run():
    mem_id = await agent.remember_async("async memory", importance=0.8)
    memories = await agent.recall_async(query_vec, top_k=5)
    await agent.commit_async()
```

---

## 15. Working memory buffer

Bounded in-memory FIFO — short-term context before draining to disk:

```python
wm = ailake.WorkingMemoryBuffer(max_rows=200)

wm.push("step 1 result", embedding=step1_emb, importance=0.9)
wm.push("step 2 result", embedding=step2_emb, importance=0.6)

# Brute-force search in buffer
hits = wm.search(query_vec, top_k=3)

# Drain to long-term storage
if wm.is_full():
    wm.drain_to_table(writer)   # calls writer.write_batch internally
    writer.commit()
```

---

## 16. Memory decay

Recompute `recency_weight` for all rows in an episodic memory table (call
nightly to naturally down-rank stale memories):

```python
updated = ailake.decay_memories(
    "s3://my-lake/agent-memory/",
    decay_lambda=0.099,   # half-life ≈ 7 days
)
print(f"{updated} files updated")
```

---

## 17. Hardware detection

```python
info = ailake.hardware_info()
print(info)
# {
#   "backend": "nvidia_cuda",     # cpu | nvidia_cuda | amd_rocm
#   "has_avx2": "true",
#   "has_avx512": "false",
#   "cuda_device": "NVIDIA A100",
# }
```

---

## 18. Deferred indexing (high-throughput ingest)

```python
writer = ailake.TableWriter("s3://my-lake/docs/", dim=1536)

# Parquet committed immediately; HNSW/IVF-PQ built in background thread
writer.write_batch_auto_deferred(texts, embs.tolist())
writer.commit()
# While index builds, search falls back to exact flat scan (GPU-accelerated when available)
```

Or via `Table`:

```python
table = ailake.open_table("s3://my-lake/docs/", dim=1536)
await table.write_batch_auto_deferred_async(texts, embs)
await table.commit_async()
```

---

## 19. Full example — RAG pipeline

```python
"""
RAG pipeline:
  1. Embed + write documents
  2. Hybrid search (BM25+vector)
  3. Assemble LLM context
  4. Call Claude
"""
import ailake
import openai

client = openai.OpenAI()

def embed(texts: list[str]) -> list[list[float]]:
    resp = client.embeddings.create(model="text-embedding-3-small", input=texts)
    return [d.embedding for d in resp.data]

TABLE = "s3://my-lake/docs/"

# ── Ingest ──────────────────────────────────────────────────────────────────
docs = [
    "AI-Lake stores HNSW indexes inside Parquet files.",
    "Geometric pruning skips files whose centroid is far from the query.",
    "BM25 and vector search are fused via Reciprocal Rank Fusion.",
]
embs = embed(docs)

writer = ailake.TableWriter(
    TABLE, dim=1536, metric="cosine",
    bm25_text_column="chunk_text",
    fts_text_columns=["chunk_text"],
    embedding_model="text-embedding-3-small",
)
writer.write_batch(docs, embs, extra_columns={"chunk_text": docs})
writer.commit()

# ── Search ───────────────────────────────────────────────────────────────────
query = "How does file pruning work in AI-Lake?"
q_vec = embed([query])[0]

results = ailake.search(
    TABLE, q_vec, top_k=5,
    fetch_data=True,
    hybrid_text=query,
    bm25_weight=0.4,
)
df = results.to_pandas()

# ── Context ──────────────────────────────────────────────────────────────────
chunks = [
    {
        "document_id": f"doc-{i}",
        "chunk_index": 0,
        "chunk_text":  row["chunk_text"],
        "distance":    float(row["_distance"]),
    }
    for i, (_, row) in enumerate(df.iterrows())
]
context_xml = ailake.assemble_context(chunks, max_tokens=2048)

# ── LLM ──────────────────────────────────────────────────────────────────────
resp = client.chat.completions.create(
    model="gpt-4o",
    messages=[
        {"role": "system",  "content": f"<context>{context_xml}</context>"},
        {"role": "user",    "content": query},
    ],
)
print(resp.choices[0].message.content)
```

---

## 20. API reference

### Module-level functions

| Function | Description |
|---|---|
| `open_table(path, **kwargs)` | Open/create table; returns `Table` |
| `search(path, query, top_k, ...)` | Returns lazy `SearchQuery` |
| `search_text(path, query_text, top_k, ...)` | BM25 FTS (O(N)) |
| `search_multimodal(path, queries, top_k, ...)` | Cross-modal RRF |
| `search_with_data(path, query, top_k, ...)` | Arrow IPC bytes (full row data) |
| `assemble_context(chunks, max_tokens, ...)` | Format chunks as LLM XML context |
| `compact(path, *, min_files, target_size_bytes, ...)` | Merge small files |
| `evolve_schema(path, *, add_columns, rename_columns)` | Add/rename columns |
| `add_column(path, name, col_type, ...)` | Single column add |
| `rename_column(path, old_name, new_name)` | Single column rename |
| `delete_where(path, column, values)` | Equality delete |
| `delete_rows(table_path, file_path, row_ids)` | Positional delete |
| `migrate_embeddings(path, old_column, new_column, embed_fn, ...)` | Re-embed with new model |
| `decay_memories(path, decay_lambda)` | Recompute recency weights |
| `hardware_info()` | Returns `dict` with backend, SIMD, GPU info |
| `now_ns()` | Nanosecond timestamp from Rust monotonic clock |

### Classes

| Class | Description |
|---|---|
| `TableWriter` | Low-level write interface; all write methods + `commit()` |
| `Table` | Fluent handle: `insert()`, `commit()`, `search()`, async variants |
| `SearchQuery` | Lazy search result; `.to_list/arrow/pandas/polars()` + async |
| `VectorColSpec` | Column spec for multimodal write (`column`, `dim`, `metric`, `modality`) |
| `Agent` | Phase 9: `remember()`, `log_tool_call()`, `recall()`, `assemble_context()`, async |
| `WorkingMemoryBuffer` | Bounded FIFO: `push()`, `search()`, `drain_to_table()` |

### Key `SearchQuery` methods

| Method | Description |
|---|---|
| `.to_list()` | `list[dict]` — `{row_id, distance, file}` always |
| `.to_arrow()` | `pyarrow.Table` (full data when `fetch_data=True`) |
| `.to_pandas()` | `pandas.DataFrame` |
| `.to_polars()` | `polars.DataFrame` |
| `.limit(n)` | Re-cap top_k; resets cached results |
| `len(sq)` | Execute and return count |
| `for r in sq` | Iterate dicts |
| All `*_async()` variants | Thread-executor async versions |

---

## Related docs

- [File Format Spec](../specs/FILE_FORMAT.md) — AILK section layout
- [LLM Context](../specs/LLM_CONTEXT.md) — `LlmContextSchema` fields
- [Go Integration](GO_INTEGRATION.md) — pure-Go client
- [C++ Integration](CPP_INTEGRATION.md) — C++17 header-only client
- [DBT Integration](DBT_INTEGRATION.md) — dbt pipelines with Spark / Trino / DuckDB
- [Demo Notebooks](DEMO_NOTEBOOKS.md) — live Jupyter examples
- [ailake-py source](../../ailake-py/) — PyO3 bindings and tests
