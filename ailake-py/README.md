# ailake — AI-Lake Format Python SDK

Unified storage for tabular data, embeddings, and HNSW vector index in a single Parquet-compatible file. 100% Apache Iceberg Spec v2 compatible.

## Install

```bash
pip install ailake
```

Requires Python ≥ 3.9. Dependencies: `pyarrow >= 14.0`, `numpy >= 1.24`.

## Quickstart

### Write

```python
import ailake
import numpy as np

writer = ailake.TableWriter(
    path="./my_table",
    vector_column="embedding",  # default
    dim=1536,                   # default
    metric="cosine",            # cosine | euclidean | dot_product
    pre_normalize=True,         # normalize to unit L2 at write time (recommended for cosine)
                                # enables NormalizedCosine fast path: 1-dot(a,b), no sqrt
    hnsw_m=16,                  # HNSW connections per node (default 16; 32 = higher recall)
    hnsw_ef_construction=150,   # HNSW build quality (default 150; 400 = max quality)
)

texts = ["Document about AI", "Another document"]
embeddings = np.random.rand(2, 1536).astype(np.float32).tolist()

writer.write_batch(texts=texts, embeddings=embeddings)
snapshot_id = writer.commit()
```

### Search

```python
import ailake
import numpy as np

query = np.random.rand(1536).astype(np.float32).tolist()

results = ailake.search(
    path="./my_table",
    query=query,
    top_k=10,
)

for r in results:
    print(r["row_id"], r["distance"], r["file"])
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

## API

### `TableWriter(path, vector_column="embedding", dim=1536, metric="cosine")`

Opens or creates an AI-Lake table at `path`. Local filesystem only in this release.

| Method | Description |
|---|---|
| `write_batch(texts, embeddings)` | Stage a batch of rows. `texts: list[str]`, `embeddings: list[list[float]]` |
| `commit() -> int` | Commit staged batches as a new Iceberg snapshot. Returns snapshot ID. |

### `search(path, query, top_k=10) -> list[dict]`

Returns up to `top_k` nearest neighbours. Each result: `{"row_id": int, "distance": float, "file": str}`.

### `assemble_context(chunks, max_tokens=4096, dedup_threshold=0.05) -> str`

Assembles a list of chunk dicts into structured XML ready for LLM input. Deduplicates near-identical chunks and respects the token budget.

## Iceberg compatibility

Tables written by `ailake` are valid Apache Iceberg Spec v2 tables. Any Iceberg-compatible engine (Spark, Trino, DuckDB, PyIceberg) reads the tabular columns normally. The HNSW index lives in an AI-Lake extension section that standard Parquet readers silently ignore.

## License

MIT OR Apache-2.0
