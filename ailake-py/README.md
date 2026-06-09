# ailake — AI-Lake Format Python SDK

Unified storage for tabular data, embeddings, and HNSW vector index in a single Parquet-compatible file. 100% Apache Iceberg Spec v2 compatible.

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
)

texts = ["Document about AI", "Another document"]
embeddings = np.random.rand(2, 1536).astype(np.float32)

table.insert(texts, embeddings)   # accepts list or numpy array
snapshot_id = table.commit()

# Fluent search chain — no I/O until materialised
df      = table.search(embeddings[0], top_k=10).to_pandas()
lf      = table.search(embeddings[0]).limit(5).to_polars()
results = table.search(embeddings[0]).to_list()   # list[dict]
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

### `Table`

| Method | Description |
|---|---|
| `insert(texts, embeddings) → Table` | Buffer a batch. `embeddings`: `list[list[float]]` or numpy array. |
| `commit() → int` | Persist as a new Iceberg snapshot; returns snapshot ID. |
| `search(query, top_k=10) → SearchQuery` | Lazy, chainable search. `query`: `list[float]` or numpy array. |
| `insert_async(...)` | Async variant of `insert`. |
| `commit_async() → int` | Async variant of `commit`. |

`Table` is a context manager: `with ailake.open_table(...) as t: ...`

In Jupyter, `table` renders a styled HTML card showing path and vector config.

### `SearchQuery`

Lazy result set — no I/O until materialised.

| Method | Description |
|---|---|
| `limit(n) → SearchQuery` | Cap to *n* nearest neighbours (chainable). |
| `to_list() → list[dict]` | `[{"row_id": int, "distance": float, "file": str}, ...]` |
| `to_pandas() → pd.DataFrame` | pandas DataFrame. |
| `to_polars() → pl.DataFrame` | polars DataFrame. |
| `to_list_async()` | Async variant. |
| `to_pandas_async()` | Async variant. |
| `to_polars_async()` | Async variant. |

In Jupyter, `results` renders as an HTML table when executed, pending state otherwise.

### `search(path, query, top_k=10) → SearchQuery`

Module-level search returning the same chainable `SearchQuery`.

### `TableWriter` (legacy — still supported)

```python
writer = ailake.TableWriter(path, vector_column="embedding", dim=1536, metric="cosine")
writer.write_batch(texts, embeddings)
snapshot_id = writer.commit()
```

### `assemble_context(chunks, max_tokens=4096, dedup_threshold=0.05) → str`

Assembles chunk dicts into structured XML for LLM input. Deduplicates near-identical chunks within the token budget.

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
