# airbyte-destination-ailake

Airbyte destination connector for [AI-Lake Format](https://github.com/ThiagoLange/ai-lakehouse) — writes Airbyte records to AI-Lake vector tables (Apache Iceberg + HNSW/IVF-PQ).

## What it does

Each Airbyte stream maps to one AI-Lake table at `{table_base_path}/{stream_name}/`. For every record batch the connector:

1. Extracts the text field from each record (`text_field`, dot-notation supported for nested fields).
2. Embeds texts via the configured backend (`cmd`, `openai`, or `cohere`).
3. Writes the batch to the AI-Lake table (Parquet + HNSW index in the footer).
4. Commits an Iceberg snapshot on each Airbyte state message and at the end of the sync.

Tables created this way are fully compatible with Apache Iceberg — Spark, Trino, DuckDB, and PyIceberg can read tabular data without the AI-Lake SDK. The AI-Lake SDK activates vector-search.

## Installation

```bash
pip install airbyte-destination-ailake          # core (cmd/http embed modes only)
pip install "airbyte-destination-ailake[openai]"  # + OpenAI Embeddings API
pip install "airbyte-destination-ailake[cohere]"  # + Cohere Embed API
pip install "airbyte-destination-ailake[fastembed]"              # + local ONNX embeddings, no API key
pip install "airbyte-destination-ailake[sentence-transformers]"  # + local PyTorch embeddings, no API key
```

## Configuration

| Field | Type | Default | Description |
|---|---|---|---|
| `table_base_path` | string | **required** | S3/GCS/Azure/local base path. Each stream lands at `{base}/{stream}/`. |
| `embed_mode` | `cmd` / `openai` / `cohere` / `http` / `fastembed` / `sentence_transformers` | **required** | Embedding backend. |
| `text_field` | string | `content` | Record field to embed. Dot-notation for nested: `meta.body`. |
| `embedding_dim` | int | `1536` | Vector dimension — must match model output. |
| `embedding_metric` | string | `cosine` | Distance metric: `cosine`, `euclidean`, `dot_product`, `normalized_cosine`. |
| `embedding_model` | string | `` | Stored as `ailake.embedding-model` Iceberg property for model tracking. |
| `embedding_model_version` | string | `` | Optional version suffix. |
| `embed_cmd` | string | `` | Shell command (cmd mode). Stdin: JSON string array. Stdout: JSON float[][]. |
| `openai_api_key` | string | `` | OpenAI API key (secret). |
| `openai_model` | string | `text-embedding-3-small` | OpenAI model name. |
| `openai_base_url` | string | `` | Override for Azure OpenAI or compatible endpoints. |
| `cohere_api_key` | string | `` | Cohere API key (secret). |
| `cohere_model` | string | `embed-english-v3.0` | Cohere model name. |
| `cohere_input_type` | string | `search_document` | Cohere input type. |
| `fastembed_model` | string | `BAAI/bge-small-en-v1.5` | fastembed model name (local ONNX, no API key, no PyTorch). |
| `sentence_transformers_model` | string | `BAAI/bge-small-en-v1.5` | Any Hugging Face sentence-embedding model name (local, no API key). |
| `sentence_transformers_device` | string | `` | `cpu`/`cuda`/etc. Empty = auto-detect. |
| `batch_size` | int | `512` | Records per embed call and per write_batch. |
| `pre_normalize` | bool | `false` | Normalize vectors to unit L2 at write time (recommended for cosine). |
| `pq_only` | bool | `false` | Discard raw F16 vectors after index build — maximum compression, no reranking. |
| `partition_by` | string | `` | Iceberg identity partition column (e.g. `"agent_id"`). When set, every write is tagged with the value of this field read from the Airbyte record. Enables per-agent/per-tenant manifest-level pruning at search time. |
| `partition_fields` | array | `[]` | Multi-column Iceberg partition spec. JSON array of `{column, transform, column_type}` objects. Supports any Iceberg transform (identity, bucket[N], truncate[N], year, month, day, hour). Takes precedence over `partition_by` when both are set. |
| `format_version` | int | `2` | Iceberg format version. Set to `3` to enable Iceberg v3 features (row lineage, Deletion Vectors). Equality deletes are encoded identically on V2 and V3 — there is no V3-native equality-delete field encoding. |
| `hnsw_m` | int | `null` | HNSW connections per node (M parameter). `null` = use table default (16). Higher M → better recall, more memory. |
| `hnsw_ef_construction` | int | `null` | HNSW build pool size. `null` = use table default (150). Higher → better graph quality, slower build. |
| `deferred` | bool | `false` | When `true`, write Parquet immediately (~200k vec/s) and build HNSW index in background; shard served via flat scan until index is ready. |
| `fts_columns` | array | `[]` | Text columns to embed as a Tantivy FTS index alongside the HNSW index. Example: `["chunk_text", "document_title"]`. Enables `search_text()` fast path. |

A record whose embedding comes back with a `NaN`/`Infinity` value (a malformed
`embed_cmd` response, or an API edge case) is rejected at write time with a clear
error instead of being silently written — the sync fails visibly rather than
producing a table with a corrupted vector.

## Embedding modes

### `cmd` — custom command

```json
{
  "embed_mode": "cmd",
  "embed_cmd": "python my_embedder.py"
}
```

The command must read a JSON string array from stdin and write a JSON float-array-of-arrays to stdout:

```bash
# stdin:  ["text one", "text two"]
# stdout: [[0.01, -0.02, ...], [0.03, 0.04, ...]]
```

### `openai` — OpenAI Embeddings API

```json
{
  "embed_mode": "openai",
  "openai_api_key": "sk-...",
  "openai_model": "text-embedding-3-small",
  "embedding_dim": 1536
}
```

For Azure OpenAI set `openai_base_url` to your deployment endpoint.

### `cohere` — Cohere Embed API

```json
{
  "embed_mode": "cohere",
  "cohere_api_key": "...",
  "cohere_model": "embed-english-v3.0",
  "embedding_dim": 1024
}
```

### `fastembed` — local ONNX embeddings (no API key)

```json
{
  "embed_mode": "fastembed",
  "fastembed_model": "BAAI/bge-small-en-v1.5",
  "embedding_dim": 384
}
```

Runs entirely in-process via ONNX Runtime — no PyTorch dependency, smallest install
footprint of the local options. Good default for CI/demos and low-memory environments.
Requires `pip install "airbyte-destination-ailake[fastembed]"`.

### `sentence-transformers` — local embeddings (no API key)

```json
{
  "embed_mode": "sentence_transformers",
  "sentence_transformers_model": "BAAI/bge-small-en-v1.5",
  "sentence_transformers_device": "",
  "embedding_dim": 384
}
```

PyTorch-based — widest model selection (any Hugging Face sentence-embedding model),
supports GPU via `sentence_transformers_device: "cuda"`. Heavier install than `fastembed`.
Requires `pip install "airbyte-destination-ailake[sentence-transformers]"`.

Both local modes default to `BAAI/bge-small-en-v1.5` (dim=384) — a small, fast,
well-regarded open model. Neither requires an API key, an internet connection after the
first model download, or any external service.

## Running locally (CLI)

```bash
# Check connection
airbyte-destination-ailake check --config config.json

# Write records from catalog + stdin messages
airbyte-destination-ailake write --config config.json --catalog catalog.json
```

## Docker

```bash
docker build -t airbyte-destination-ailake:latest .
docker run --rm -v $(pwd):/data airbyte-destination-ailake:latest check --config /data/config.json
```

## Reading the tables after sync

```python
import ailake

results = ailake.search(
    table="s3://my-lake/airbyte/my_stream/",
    query=my_query_embedding,
    top_k=10,
)
```

Or with any Iceberg-compatible engine (no AI-Lake SDK needed for tabular access):

```python
import duckdb
duckdb.sql("INSTALL iceberg; LOAD iceberg;")
duckdb.sql("SELECT content FROM iceberg_scan('s3://my-lake/airbyte/my_stream/')")
```

## Development

```bash
pip install -e ".[dev,openai,cohere]"
pytest tests/
```

## License

MIT OR Apache-2.0
