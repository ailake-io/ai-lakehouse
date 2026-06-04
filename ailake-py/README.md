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

### TableWriter parameters

| Parameter | Default | Description |
|---|---|---|
| `path` | required | Table root path (local or `s3://`, `gs://`, `az://`) |
| `vector_column` | `"embedding"` | Vector column name |
| `dim` | `1536` | Vector dimension |
| `metric` | `"cosine"` | `cosine`, `euclidean`, `dot_product` |
| `pre_normalize` | `False` | Normalize to unit L2 at write time (recommended for cosine). Enables `1-dot(a,b)` fast path. |
| `hnsw_m` | `None` (=16) | HNSW connections per node. Higher → better recall, more memory. |
| `hnsw_ef_construction` | `None` (=150) | HNSW build pool size. Higher → better quality, slower build. |
| `rabitq` | `False` | Use RaBitQ flat index instead of HNSW: 1 bit/dim = 16× smaller than F16. Better recall than naive binary quantization. Use with `rerank_factor ≥ 3` at search. |
| `rabitq_seed` | `0` | Seed for RaBitQ random rotation matrix. |
| `rabitq_keep_raw` | `True` | Keep raw F16 vectors for exact reranking (recommended). |

HNSW tuning guide:

| Goal | `hnsw_m` | `hnsw_ef_construction` |
|---|---|---|
| Low latency / high QPS | 8 | 100 |
| General purpose (default) | 16 | 150 |
| High recall (RAG) | 24 | 200 |
| Max recall (medical, legal) | 32 | 400 |

### RaBitQ — extreme compression (1 bit/dim)

RaBitQ is a flat index with no graph construction: 1 bit/dim after a random rotation, yielding better recall than naive binary quantization via an unbiased XOR/popcount IP estimator. Write throughput ~300k vec/s (no k-means, no graph). Storage: 200 bytes/vector at dim=1536 (15× smaller than F16).

Use when storage is the primary constraint or write throughput matters more than recall. Pair with `rerank_factor ≥ 3` to recover precision using the stored raw F16 vectors.

```python
import ailake
import numpy as np

# Write with RaBitQ (keep_raw=True stores F16 vectors for reranking)
writer = ailake.TableWriter(
    path="./rabitq_table",
    dim=1536,
    metric="cosine",
    rabitq=True,
    rabitq_seed=42,       # same seed across all shards → comparable distances
    rabitq_keep_raw=True, # recommended: enables reranking
)
writer.write_batch(texts=texts, embeddings=embeddings)
writer.commit()

# Search with reranking for best recall
results = ailake.search(
    path="./rabitq_table",
    query=query,
    top_k=10,
    rerank_factor=3,  # fetch top_30, rerank with raw F16, return top_10
)
```

| Index | Bytes/vector (dim=1536) | Recall@10 (with rerank) | Write (vec/s) |
|---|---|---|---|
| HNSW (F16) | ~3 200 | ≥ 0.95 | ~50k |
| IVF-PQ (M=48) | ~50 | 0.90–0.95 | ~200k |
| RaBitQ (no raw) | **192** | 0.70–0.85 | **~300k** |
| RaBitQ + raw F16 | ~3 264 | **0.85–0.95** | **~300k** |

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
