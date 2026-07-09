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
    deferred=False,               # build the index asynchronously; Parquet is committed immediately
    batch_id="{{ run_id }}_{{ task.task_id }}",  # idempotency key (this is the default)
)
```

`AilakeCompactOperator` — compacts small files in an AI-Lake table. Wraps `ailake compact <table>`.

```python
from airflow_providers_ailake.operators.ailake import AilakeCompactOperator

compact_op = AilakeCompactOperator(
    task_id="compact_agent_memory",
    ailake_conn_id="ailake_default",
    table="default.agents",
    target_size=536_870_912,  # bytes, default 512 MiB
    min_files=4,
)
```

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

`AilakeHook.compact(table, *, target_size=536_870_912, min_files=4, deferred=False) → int` — runs compaction on the table via CLI; returns number of files compacted (`0` if nothing qualified).

`AilakeHook.decay_memories(table, *, decay_lambda=0.1) → int` — applies exponential recency decay (`exp(-λ × days_since_access)`) to the `recency_weight` column; returns number of files updated.

`AilakeHook.search_text(table, query_text, text_columns=None, top_k=10, partition_filter=None) → list[dict]` — full-text search via Tantivy (O(log N) fast path) or BM25 brute-force fallback. Wraps `ailake search --text`.

`AilakeHook.delete_where(table, column, values) → None` — logically deletes rows where `column` equals any value in `values` via an Iceberg equality delete file. No-op when `values` is empty.

`AilakeHook.evolve_schema(table, add_columns=None, rename_columns=None) → int` — applies a metadata-only schema evolution; each `add_columns` entry needs `name`/`type` keys (optionally `initial_default`), each `rename_columns` entry needs `from`/`to` keys. Returns the new `schema_id`.

## Requirements

- Apache Airflow >= 2.6
- Python >= 3.9
- AI-Lake SDK (`ailake` Python package) installed in the Airflow worker environment

## Links

- [Source](https://github.com/ThiagoLange/ai-lakehouse)
- [Issue tracker](https://github.com/ThiagoLange/ai-lakehouse/issues)
