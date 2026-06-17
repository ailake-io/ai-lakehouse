# apache-airflow-providers-ailake

Apache Airflow provider for [AI-Lake Format](https://github.com/ThiagoLange/ai-lakehouse) — an Iceberg-compatible file format that unifies tabular data, embeddings, and HNSW vector indexes in a single Parquet file.

## Installation

```bash
pip install apache-airflow-providers-ailake
```

## Components

### Hook

`AilakeHook` — connects to an AI-Lake table on object storage (S3/GCS/Azure).

```python
from airflow_providers_ailake.hooks.ailake import AilakeHook

hook = AilakeHook(conn_id="ailake_default")

# Vector search
results = hook.search(
    table_path="s3://my-lake/docs/",
    query=my_embedding,
    top_k=10,
    partition_filter="agent-42",   # optional — restrict to this agent's files (Phase 9)
)

# Write a batch
snapshot_id = hook.write_batch(
    table_path="s3://my-lake/docs/",
    texts=["doc 1", "doc 2"],
    embeddings=my_embeddings,
    partition_by="agent_id",       # optional — Iceberg identity partition column (Phase 9)
    partition_value="agent-42",    # optional — value tagged on this write
)
```

### Operators

`AilakeWriteOperator` — writes a batch of rows (with embeddings) to an AI-Lake table.

```python
from airflow_providers_ailake.operators.ailake import AilakeWriteOperator

write_op = AilakeWriteOperator(
    task_id="write_agent_memory",
    conn_id="ailake_default",
    table_path="s3://my-lake/agents/",
    texts_key="texts",            # XCom key for texts
    embeddings_key="embeddings",  # XCom key for embeddings
    partition_by="agent_id",      # Phase 9 — Iceberg identity partition column
    partition_value="agent-42",   # Phase 9 — value for this write
)
```

`AilakeSearchOperator` — runs a vector similarity search and pushes results to XCom.

```python
from airflow_providers_ailake.operators.ailake import AilakeSearchOperator

search_op = AilakeSearchOperator(
    task_id="recall_memories",
    conn_id="ailake_default",
    table_path="s3://my-lake/agents/",
    query_key="query_embedding",  # XCom key for query vector
    top_k=10,
    partition_filter="agent-42",  # Phase 9 — restrict to this agent's files
    do_xcom_push=True,
)
```

### Sensor

`AilakeSnapshotSensor` — waits until a new Iceberg snapshot appears on the table (useful for triggering downstream DAGs after a write).

## Requirements

- Apache Airflow >= 2.6
- Python >= 3.9
- AI-Lake SDK (`ailake` Python package) installed in the Airflow worker environment

## Links

- [Source](https://github.com/ThiagoLange/ai-lakehouse)
- [Issue tracker](https://github.com/ThiagoLange/ai-lakehouse/issues)
