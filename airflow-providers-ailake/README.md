# apache-airflow-providers-ailake

Apache Airflow provider for [AI-Lake Format](https://github.com/ThiagoLange/ai-lakehouse) — an Iceberg-compatible file format that unifies tabular data, embeddings, and HNSW vector indexes in a single Parquet file.

## Installation

```bash
pip install apache-airflow-providers-ailake
```

## Components

### Hook

`AilakeHook` — connects to an AI-Lake table on object storage (S3/GCS/Azure).

`AilakeHook` wraps the `ailake` CLI — it has no `write_batch` method; writes go
through `AilakeWriteOperator` (below), which points the CLI at a Parquet file
already on disk (e.g. produced by an upstream task).

```python
from airflow_providers_ailake.hooks.ailake import AilakeHook

hook = AilakeHook(ailake_conn_id="ailake_default")

# Vector search
results = hook.search(
    table="default.docs",
    query=my_embedding,
    top_k=10,
    partition_filter="agent-42",   # optional — restrict to this agent's files (Phase 9)
)
```

### Operators

`AilakeWriteOperator` — inserts a Parquet file (already on disk, e.g. from an upstream task) into an AI-Lake table. Wraps `ailake insert <table> <file> --embeddings <col>`.

```python
from airflow_providers_ailake.operators.ailake import AilakeWriteOperator

write_op = AilakeWriteOperator(
    task_id="write_agent_memory",
    ailake_conn_id="ailake_default",
    table="default.agents",
    source_file="{{ ti.xcom_pull(task_ids='generate') }}",  # local path to the Parquet file
    embeddings_column="embedding",
    partition_by="agent_id",      # single-column identity partition
    partition_value="agent-42",   # value tagged on this write
    # multi-column partition spec (takes precedence over partition_by when set):
    partition_fields=[{"column": "topic_id", "transform": "identity", "column_type": "int"}],
    format_version=3,             # Iceberg v3 (default: 2)
    fts_columns=["chunk_text"],   # Tantivy FTS index on these text columns (opt-in)
    fts_tokenizer="default",      # "default" or "raw"
    pre_normalize=False,          # normalize vectors to unit L2 at write time (~12-20% speedup)
    deferred=False,               # build the index asynchronously; Parquet is committed immediately.
                                   # Not combinable with batch_id — when True, batch_id is dropped
                                   # (with a warning) instead of passed to the CLI.
    batch_id="{{ run_id }}_{{ task.task_id }}",  # idempotency key (this is the default)
)

# Multi-column (Phase 8 multimodal) mode — one dict per vector column,
# embeddings_column is ignored when vector_cols is set
multimodal_write_op = AilakeWriteOperator(
    task_id="write_media",
    ailake_conn_id="ailake_default",
    table="default.media",
    source_file="{{ ti.xcom_pull(task_ids='generate') }}",
    vector_cols=[
        {"column": "embedding", "dim": 1536},
        {"column": "image_embedding", "dim": 512, "metric": "euclidean", "modality": "image"},
    ],
)
```

A row with a `NaN`/`Infinity` embedding value is rejected by the CLI at write time; the
underlying `ailake insert` call fails with a clear error (`embedding contains
non-finite value (...); NaN/Infinity embeddings are rejected at write time`), which
surfaces as a task failure rather than a silently accepted bad row.

`AilakeCompactOperator` — compacts small files in an AI-Lake table. Wraps `ailake compact <table>`.

```python
from airflow_providers_ailake.operators.ailake import AilakeCompactOperator

compact_op = AilakeCompactOperator(
    task_id="compact_agent_memory",
    ailake_conn_id="ailake_default",
    table="default.agents",
    target_size=536_870_912,   # bytes, default 512 MiB
    min_files=4,
    max_files_per_pass=20,     # bounds peak RAM / HNSW rebuild cost
    deferred=False,            # build the merged HNSW index in the background
)
```

Returns the number of files compacted, which Airflow's default `do_xcom_push` behavior lands on XCom under the standard `"return_value"` key.

`AilakeFtsSearchOperator` — runs a full-text search (Tantivy O(log N) when FTS index present; BM25 brute-force fallback) and pushes results to XCom.

```python
from airflow_providers_ailake.operators.ailake import AilakeFtsSearchOperator

fts_op = AilakeFtsSearchOperator(
    task_id="fts_search",
    ailake_conn_id="ailake_default",
    table="default.docs",
    query_text="rust async programming",
    text_columns=["chunk_text", "document_title"],  # default: ["chunk_text"]
    top_k=10,
)
```

Pushes results to XCom under key `"fts_results"` (not `do_xcom_push` — that flag doesn't affect this operator; results are pushed explicitly in `execute()`).

`AilakeDeleteWhereOperator` — writes an Iceberg equality delete file and commits a Delete snapshot. No data files are rewritten.

```python
from airflow_providers_ailake.operators.ailake import AilakeDeleteWhereOperator

delete_op = AilakeDeleteWhereOperator(
    task_id="expire_old_records",
    ailake_conn_id="ailake_default",
    table="default.docs",
    column="id",
    values=["doc-obsolete-1", "doc-obsolete-2"],
    # values_xcom_task_id="upstream_task",   # pull values list from XCom instead
    # values_xcom_key="ids_to_delete",
)
```

`AilakeEvolveSchemaOperator` — applies metadata-only schema evolution (add columns, rename columns). Pushes `schema_id` to XCom.

```python
from airflow_providers_ailake.operators.ailake import AilakeEvolveSchemaOperator

evolve_op = AilakeEvolveSchemaOperator(
    task_id="add_source_url_column",
    ailake_conn_id="ailake_default",
    table="default.docs",
    add_columns=[{"name": "source_url", "type": "string"}],
    rename_columns=[],  # e.g. [{"from": "source_url", "to": "url"}]
)
```

`AilakeAddVectorColumnOperator` — adds a new vector column to an existing table schema (no data files rewritten). Old files return null for the new column until `AilakeBackfillVectorColumnOperator` runs. Pushes `schema_id` to XCom.

```python
from airflow_providers_ailake.operators.ailake import AilakeAddVectorColumnOperator

add_col_op = AilakeAddVectorColumnOperator(
    task_id="add_image_embedding_column",
    ailake_conn_id="ailake_default",
    table="default.docs",
    column="image_embedding",
    dim=512,
    metric="cosine",       # default
    precision="f16",       # default
)
```

`AilakeBackfillVectorColumnOperator` — backfills a new vector column in all existing files by re-reading text and calling an external embed command. Idempotent: files already containing the column are skipped. Requires `AilakeAddVectorColumnOperator` to have run first for `column`.

```python
from airflow_providers_ailake.operators.ailake import AilakeBackfillVectorColumnOperator

backfill_op = AilakeBackfillVectorColumnOperator(
    task_id="backfill_image_embedding",
    ailake_conn_id="ailake_default",
    table="default.docs",
    column="image_embedding",
    text_column="image_caption",
    embed_cmd="python3 embed_images.py",  # reads JSON array of strings from stdin,
                                           # writes JSON array of float arrays to stdout
)
```

`AilakeMigrateOperator` — re-embeds a table's vector column via an external embed command (e.g. upgrading to a new embedding model).

```python
from airflow_providers_ailake.operators.ailake import AilakeMigrateOperator

migrate_op = AilakeMigrateOperator(
    task_id="migrate_to_new_model",
    ailake_conn_id="ailake_default",
    table="default.docs",
    embed_cmd="python3 embed_v2.py",
    old_column="embedding",
    new_column="embedding_v2",       # may equal old_column for an in-place upgrade
    text_column="chunk_text",
    strategy="dual-write-then-cutover",  # or "atomic_replace" (lower storage)
    model_name="text-embedding-3-large",
    model_version="v1",
)
```

`AilakeDeleteRowsOperator` — marks specific row positions as deleted within one data file, using Iceberg Deletion Vectors. Distinct from `AilakeDeleteWhereOperator` (equality predicate across the whole table). **Requires the table to have been created with `format_version=3`** — Deletion Vectors are a V3-only Iceberg feature; the CLI raises a clear error on a V2 table.

```python
from airflow_providers_ailake.operators.ailake import AilakeDeleteRowsOperator

delete_rows_op = AilakeDeleteRowsOperator(
    task_id="delete_stale_rows",
    ailake_conn_id="ailake_default",
    table="default.docs",
    file="data/part-00001.parquet",  # as reported by ailake info / get_table_info()
    row_positions=[0, 5, 42],
)
```

`AilakeSearchOperator` — runs a vector similarity search and pushes results to XCom.

```python
from airflow_providers_ailake.operators.ailake import AilakeSearchOperator

search_op = AilakeSearchOperator(
    task_id="recall_memories",
    ailake_conn_id="ailake_default",
    table="default.agents",
    query_vector=[0.1, 0.2, ...],           # or query_xcom_task_id="upstream_task"
    top_k=10,
    partition_filter="agent-42",  # Phase 9 — restrict to this agent's files
)
```

Returns the results list, which Airflow's default `do_xcom_push` behavior lands on XCom under the standard `"return_value"` key — pull with `{{ ti.xcom_pull(task_ids='recall_memories') }}`.

`top_k` (here and on `AilakeHook.search`/`search_text`/`AilakeFtsSearchOperator`) is
capped at `ailake_core::MAX_TOP_K` (100,000) by the underlying CLI — a value above
that fails the task with a clear error instead of risking an unbounded-allocation
crash.

### Sensor

`AilakeSnapshotSensor` — waits until a new Iceberg snapshot appears on the table (useful for triggering downstream DAGs after a write).

`AilakeIndexStatusSensor` — polls `ailake info <table> --format json` until `index_status == "ready"`. Use after a deferred write to gate downstream tasks on the index being fully built.

```python
from airflow_providers_ailake.sensors.ailake import AilakeIndexStatusSensor

wait_for_index = AilakeIndexStatusSensor(
    task_id="wait_for_hnsw_index",
    ailake_conn_id="ailake_default",
    table="default.docs",
    poke_interval=30,
    timeout=600,
)
```

### Additional hook methods

`AilakeHook.get_table_info(table) → dict` — returns table metadata as a dict (`ailake info <table> --format json`), or `{}` if the table doesn't exist yet.

`AilakeHook.get_current_snapshot_id(table) → int | None` — returns the table's current `snapshot_id`, or `None` if no snapshot exists.

`AilakeHook.compact(table, *, target_size=536_870_912, min_files=4, max_files_per_pass=20, deferred=False) → int` — runs compaction on the table via CLI (`--format json`); returns number of files compacted (`0` if nothing qualified).

`AilakeHook.decay_memories(table, *, decay_lambda=0.1) → int` — applies exponential recency decay (`exp(-λ × days_since_access)`) to the `recency_weight` column; returns number of files updated.

`AilakeHook.search_text(table, query_text, text_columns=None, top_k=10, partition_filter=None) → list[dict]` — full-text search via Tantivy (O(log N) fast path) or BM25 brute-force fallback. Wraps `ailake search --text`.

`AilakeHook.delete_where(table, column, values) → None` — logically deletes rows where `column` equals any value in `values` via an Iceberg equality delete file. No-op when `values` is empty.

`AilakeHook.evolve_schema(table, add_columns=None, rename_columns=None) → int` — applies a metadata-only schema evolution; each `add_columns` entry needs `name`/`type` keys (optionally `initial_default`), each `rename_columns` entry needs `from`/`to` keys. Returns the new `schema_id`.

`AilakeHook.migrate(table, *, embed_cmd, old_column="embedding", new_column="embedding_v2", text_column="chunk_text", strategy="dual-write-then-cutover", batch_size=512, model_name=None, model_version=None) → None` — re-embeds a table's vector column via an external embed command. Raises on failure.

`AilakeHook.delete_rows(table, file, row_positions) → None` — marks specific row positions as deleted within one data file using Iceberg Deletion Vectors. Requires `format_version=3`. No-op when `row_positions` is empty.

`AilakeHook.add_vector_column(table, column, dim, *, metric="cosine", precision="f16", pre_normalize=False, hnsw_m=None, hnsw_ef=None) → int` — adds a new vector column to an existing table schema (no data files rewritten). Returns the new `schema_id`, or `-1` when not parseable from CLI output.

`AilakeHook.backfill_vector_column(table, column, *, embed_cmd, text_column="chunk_text", batch_size=512) → None` — backfills a new vector column in all existing files. Requires `add_vector_column` to have been run first for `column`.

`AilakeHook.estimate(rows, dim, *, hnsw_m=16, pq_m=None) → dict` — estimates storage usage before writing (no I/O — pure math). `rows` supports K/M/B suffixes (e.g. `"1M"`). Returns `{"rows", "dim", "hnsw_m", "pq_m", "estimates": [...]}`, or `{}` on parse failure.

## Requirements

- Apache Airflow >= 2.6
- Python >= 3.9
- AI-Lake SDK (`ailake` Python package) installed in the Airflow worker environment

## Links

- [Source](https://github.com/ThiagoLange/ai-lakehouse)
- [Issue tracker](https://github.com/ThiagoLange/ai-lakehouse/issues)
