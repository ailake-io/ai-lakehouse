# apache-airflow-providers-ailake

Apache Airflow provider for [AI-Lake Format](https://github.com/ThiagoLange/iceberg-ai-deltalakehouse) — an Iceberg-compatible file format that unifies tabular data, embeddings, and HNSW vector indexes in a single Parquet file.

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
```

### Operators

`AilakeWriteOperator` — writes a batch of rows (with embeddings) to an AI-Lake table.

`AilakeSearchOperator` — runs a vector similarity search and pushes results to XCom.

### Sensor

`AilakeSnapshotSensor` — waits until a new Iceberg snapshot appears on the table (useful for triggering downstream DAGs after a write).

## Requirements

- Apache Airflow >= 2.6
- Python >= 3.9
- AI-Lake SDK (`ailake` Python package) installed in the Airflow worker environment

## Links

- [Source](https://github.com/ThiagoLange/iceberg-ai-deltalakehouse)
- [Issue tracker](https://github.com/ThiagoLange/iceberg-ai-deltalakehouse/issues)
