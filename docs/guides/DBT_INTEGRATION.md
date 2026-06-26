# dbt Integration Guide

AI-Lake tables fit naturally into dbt pipelines: dbt handles SQL transformations
(staging, cleaning, chunking), and the AI-Lake SDK handles the vector-specific step
(embedding ingestion + HNSW index build). The two phases are connected via
**dbt post-hooks** that invoke `ailake_write_batch` on Spark or Trino after each
model materializes.

---

## Conceptual flow

```
raw sources  →  dbt staging/intermediate models  →  dbt final model (Iceberg table)
                                                              │
                                              post-hook: ailake_write_batch()
                                                              │
                                                    AI-Lake file (.parquet + HNSW)
                                                    committed to Iceberg catalog
```

The dbt layer owns **what data to transform**. The AI-Lake hook owns **how to
embed and index it**. The resulting table is a standard Iceberg table readable by
any engine; the HNSW index is embedded in each Parquet file's AILK footer.

---

## 1. Project layout

```
my_project/
├── dbt_project.yml
├── macros/
│   ├── ailake_write_batch.sql        # adapter macro (Spark / Trino)
│   └── ailake_compact.sql            # optional pre-hook for compaction
├── models/
│   ├── staging/
│   │   └── stg_documents.sql
│   ├── intermediate/
│   │   └── int_chunks.sql
│   └── marts/
│       └── ailake_embeddings.sql     # final model with post-hook
└── tests/
    └── ailake_recall.sql             # optional recall assertion
```

---

## 2. dbt_project.yml — global variables

```yaml
# dbt_project.yml
name: my_project
version: "1.0.0"

vars:
  ailake_vec_col:    "embedding"
  ailake_dim:        1536
  ailake_metric:     "cosine"
  ailake_precision:  "f16"
  # path to libailake_jni.so on cluster nodes (Spark / Trino)
  ailake_lib_path:   "/opt/ailake/libailake_jni.so"
  # Iceberg warehouse root
  ailake_warehouse:  "s3://my-lake/warehouse"
```

Override per-model in `config()` when different columns need different settings
(e.g. `dim=512` for a smaller model, `metric=euclidean` for image embeddings).

---

## 3. Macros

### `macros/ailake_write_batch.sql`

```sql
{% macro ailake_write_batch(
    table_path,
    id_col      = "id",
    vec_col     = var("ailake_vec_col"),
    dim         = var("ailake_dim"),
    metric      = var("ailake_metric"),
    precision   = var("ailake_precision")
) %}

  {%- if target.type == "spark" -%}

    -- Spark: call via AilakeNative Scala object loaded by the plugin.
    -- Requires spark-plugin jar and libailake_jni.so on LD_LIBRARY_PATH.
    SELECT ailake_write_batch(
        '{{ table_path }}',
        collect_list(CAST({{ id_col }} AS BIGINT)),
        collect_list({{ vec_col }})
    )

  {%- elif target.type == "trino" -%}

    -- Trino: call via AilakeNative Kotlin function loaded by the plugin.
    SELECT ailake_write_batch(
        '{{ table_path }}',
        array_agg(CAST({{ id_col }} AS BIGINT)),
        array_agg({{ vec_col }})
    )

  {%- elif target.type == "duckdb" -%}

    -- DuckDB: use the duckdb-ailake extension directly.
    -- Requires LOAD 'ailake' executed in the session.
    SELECT ailake_write_batch(
        '{{ table_path }}',
        array_agg(CAST({{ id_col }} AS BIGINT)),
        array_agg({{ vec_col }})
    )

  {%- else -%}

    {{ exceptions.raise_compiler_error(
        "ailake_write_batch: unsupported adapter '" ~ target.type ~ "'. "
        ~ "Supported: spark, trino, duckdb."
    ) }}

  {%- endif -%}

{% endmacro %}
```

### `macros/ailake_compact.sql`

```sql
{% macro ailake_compact(table_path) %}

  {%- if target.type == "spark" -%}
    SELECT ailake_compact('{{ table_path }}')
  {%- elif target.type == "trino" -%}
    SELECT ailake_compact('{{ table_path }}')
  {%- elif target.type == "duckdb" -%}
    -- DuckDB compaction via Python SDK (call from a dbt operation, not a model)
    {{ log("ailake_compact: run `ailake.compact('" ~ table_path ~ "')` via Python SDK", info=True) }}
  {%- endif -%}

{% endmacro %}
```

---

## 4. Models

### `models/staging/stg_documents.sql`

Standard dbt staging — no AI-Lake concern here.

```sql
{{ config(materialized="view") }}

SELECT
    id,
    title,
    body,
    source_url,
    created_at
FROM {{ source("raw", "documents") }}
WHERE body IS NOT NULL
  AND LENGTH(TRIM(body)) > 50
```

### `models/intermediate/int_chunks.sql`

Split documents into chunks. This model is also plain SQL — chunking can be done
with window functions or a custom UDF registered in Spark/Trino.

```sql
{{ config(materialized="table", file_format="parquet") }}

WITH numbered AS (
    SELECT
        id                                          AS document_id,
        title                                       AS document_title,
        -- sliding window chunking: 512-token chunks with 64-token overlap
        explode(ailake_chunk(body, 512, 64))        AS chunk_struct,
        created_at
    FROM {{ ref("stg_documents") }}
)
SELECT
    {{ dbt_utils.generate_surrogate_key(["document_id", "chunk_struct.index"]) }} AS chunk_id,
    document_id,
    document_title,
    chunk_struct.index          AS chunk_index,
    chunk_struct.total          AS total_chunks,
    chunk_struct.text           AS chunk_text,
    created_at
FROM numbered
```

> `ailake_chunk` is a UDF registered by the Spark plugin. For Trino, use the
> equivalent Trino function or a Python dbt macro that calls the chunking logic
> via a subprocess.

### `models/marts/ailake_embeddings.sql`

The final model materializes the Iceberg table and triggers the AI-Lake ingest
via post-hook. The embedding column (`embedding`) must already be present — either
computed by a UDF, fetched from an external embedding API via a prior dbt model,
or populated by a Python script that updates the staging table before this model runs.

```sql
{{
  config(
    materialized     = "incremental",
    incremental_strategy = "append",
    file_format      = "parquet",
    -- Table-level AI-Lake options (override global vars if needed)
    meta = {
      "ailake_vec_col":   "embedding",
      "ailake_dim":       1536,
      "ailake_metric":    "cosine",
      "ailake_precision": "f16"
    },
    post_hook = [
      -- 1. Ingest new rows into AI-Lake (HNSW build happens inline or deferred)
      ailake_write_batch(
        this.render(),
        id_col    = "chunk_id",
        vec_col   = "embedding",
        dim       = 1536,
        metric    = "cosine",
        precision = "f16"
      ),
      -- 2. Compact small files created by incremental runs (optional, run periodically)
      -- ailake_compact(this.render())
    ]
  )
}}

SELECT
    chunk_id,
    document_id,
    document_title,
    chunk_index,
    total_chunks,
    chunk_text,
    embedding,       -- FLOAT[] (dim=1536) — computed upstream
    created_at
FROM {{ ref("int_chunks") }}

{% if is_incremental() %}
WHERE created_at > (SELECT MAX(created_at) FROM {{ this }})
{% endif %}
```

---

## 5. Incremental strategy and compaction

Each incremental run appends new Parquet files with their own HNSW index. Over many
runs, many small files accumulate. Schedule a periodic compaction to merge them:

```yaml
# dbt_project.yml — scheduled operation
operations:
  - name: compact_embeddings
    description: "Merge small AI-Lake files and rebuild HNSW"
```

```sql
-- operations/compact_embeddings.sql
{{ ailake_compact(var("ailake_warehouse") ~ "/marts/ailake_embeddings") }}
```

Run via: `dbt run-operation compact_embeddings --target prod`

Alternatively, trigger compaction from the Python SDK in a Prefect/Airflow task
after the dbt job completes:

```python
import ailake

# Triggered by Airflow after `dbt run --select ailake_embeddings`
ailake.compact(
    "s3://my-lake/warehouse/marts/ailake_embeddings",
    min_files=4,
    target_size_bytes=128 * 1024 * 1024,  # 128 MiB
)
```

---

## 6. Embedding generation strategies

The dbt model above assumes `embedding` is already a column in `int_chunks`.
Three common patterns:

### 6A — UDF registered in Spark (online, blocking)

```python
# spark_session_init.py (run before dbt via --pre-hook or cluster init script)
from pyspark.sql.functions import udf
from pyspark.sql.types import ArrayType, FloatType
import openai

@udf(returnType=ArrayType(FloatType()))
def embed_text(text: str):
    resp = openai.embeddings.create(input=text, model="text-embedding-3-small")
    return resp.data[0].embedding

spark.udf.register("embed_text", embed_text)
```

```sql
-- int_chunks.sql (adds embedding column via UDF)
SELECT *, embed_text(chunk_text) AS embedding
FROM ...
```

### 6B — Pre-computed embeddings table (recommended for large datasets)

Run embedding in a separate Python job (batched, async, with retry logic), write
results to a staging table, then join in dbt:

```sql
-- int_chunks_with_embeddings.sql
SELECT
    c.*,
    e.embedding
FROM {{ ref("int_chunks") }} c
JOIN {{ ref("stg_embeddings") }} e USING (chunk_id)
```

This separates the expensive embedding API call from the dbt transformation DAG,
avoiding timeout issues on large tables.

### 6C — Python dbt model (dbt 1.3+)

```python
# models/intermediate/int_chunks_embedded.py
import pandas as pd
import ailake

def model(dbt, session):
    chunks = dbt.ref("int_chunks").toPandas()

    # Batch embed (replace with your provider)
    import openai
    texts = chunks["chunk_text"].tolist()
    resp = openai.embeddings.create(input=texts, model="text-embedding-3-small")
    chunks["embedding"] = [r.embedding for r in resp.data]

    return session.createDataFrame(chunks)
```

---

## 7. dbt tests — recall assertion

Add a data test that verifies the HNSW index is searchable after each run.
Works with dbt-duckdb or via the DuckDB extension loaded in CI.

```sql
-- tests/ailake_recall.sql
-- Fails if nearest-neighbour search returns 0 results for a known query.
-- Use a fixed test embedding stored in a seed file.

WITH test_query AS (
    SELECT embedding FROM {{ ref("seed_test_queries") }} LIMIT 1
),
results AS (
    SELECT COUNT(*) AS hit_count
    FROM ailake_search(
        '{{ var("ailake_warehouse") }}/marts/ailake_embeddings',
        (SELECT embedding FROM test_query),
        5
    )
)
SELECT * FROM results WHERE hit_count = 0  -- test fails if search returns nothing
```

Run: `dbt test --select ailake_recall`

> Requires `LOAD 'ailake'` in the DuckDB session. Configure via `dbt-duckdb`'s
> `connection_init_sql` setting in `profiles.yml`:

```yaml
# profiles.yml
my_project:
  target: dev
  outputs:
    dev:
      type: duckdb
      path: ":memory:"
      connection_init_sql: |
        LOAD '/opt/ailake/ailake.duckdb_extension';
        SET ailake_lib_path = '/opt/ailake/libailake_jni.so';
```

---

## 8. Spark-specific setup

### Spark session configuration

```python
# spark-submit flags or SparkConf
spark = SparkSession.builder \
    .config("spark.jars", "/opt/ailake/spark-plugin.jar") \
    .config("spark.driver.extraLibraryPath", "/opt/ailake") \
    .config("spark.executor.extraLibraryPath", "/opt/ailake") \
    .config("spark.sql.extensions",
            "org.apache.iceberg.spark.extensions.IcebergSparkSessionExtensions,"
            "ai.ailake.spark.AilakeSparkExtensions") \
    .config("spark.sql.catalog.spark_catalog",
            "org.apache.iceberg.spark.SparkSessionCatalog") \
    .getOrCreate()
```

### dbt-spark profiles.yml

```yaml
my_project:
  target: prod
  outputs:
    prod:
      type: spark
      method: thrift        # or livy, session
      host: spark-thrift-server.internal
      port: 10001
      schema: marts
      connect_retries: 5
      connect_timeout: 60
```

---

## 9. Trino-specific setup

### Trino plugin deployment

```bash
# Copy plugin jar to all coordinators and workers
scp trino-plugin/target/ailake-trino-plugin.jar \
    trino-host:/etc/trino/plugin/ailake/

# Ensure libailake_jni.so is on LD_LIBRARY_PATH for the Trino process
echo 'JAVA_OPTS="${JAVA_OPTS} -Djava.library.path=/opt/ailake"' \
    >> /etc/trino/jvm.config
```

### dbt-trino profiles.yml

```yaml
my_project:
  target: prod
  outputs:
    prod:
      type: trino
      host: trino.internal
      port: 443
      database: iceberg
      schema: marts
      auth: ldap
      user: dbt_service
      http_scheme: https
      session_properties:
        iceberg.target_max_file_size: "134217728"   # 128 MB
```

---

## 10. Full end-to-end example

```bash
# 1. Stage raw documents
dbt run --select stg_documents

# 2. Chunk documents into ~512-token segments
dbt run --select int_chunks

# 3. (Outside dbt) Generate embeddings and write to stg_embeddings
python scripts/embed_chunks.py \
    --input-table int_chunks \
    --output-table stg_embeddings \
    --model text-embedding-3-small

# 4. Join embeddings, materialize Iceberg table, trigger ailake_write_batch
dbt run --select ailake_embeddings

# 5. Verify HNSW is searchable
dbt test --select ailake_recall

# 6. Compact small files (weekly cron or dbt Cloud job)
dbt run-operation compact_embeddings
```

After step 4, `s3://my-lake/warehouse/marts/ailake_embeddings/` contains standard
Iceberg metadata + Parquet files with embedded HNSW indexes. Any Spark/Trino/DuckDB
query reads tabular columns normally; AI-Lake vector search is available via the
plugin or Python SDK.

---

## 11. Known limitations

| Limitation | Workaround |
|---|---|
| `ailake_write_batch` is a SQL function — dbt `post_hook` runs in SQL context only | Works on Spark/Trino where the function is registered by the plugin; for DuckDB targets run a Python post-step |
| Embedding UDFs block the Spark executor thread pool for large tables | Use pattern 6B (pre-computed embeddings table) to decouple API calls from transformation |
| dbt incremental `merge` strategy rewrites rows — HNSW RowIds become stale | Use `incremental_strategy = "append"` only; for updates, trigger a full compaction after the run |
| Compaction is not a native dbt operation | Run via `dbt run-operation` or as a separate Airflow/Prefect task after dbt job |
| `ailake_chunk` UDF requires spark-plugin on the cluster | Implement chunking in a Python dbt model (pattern 6C) if the plugin cannot be deployed |

---

## Related docs

- [JVM Plugins](../specs/JVM_PLUGINS.md) — Spark, Trino, Flink C-ABI functions
- [Compaction](../specs/COMPACTION.md) — file merge strategies and scheduling
- [LLM Context](../specs/LLM_CONTEXT.md) — `LlmContextSchema` fields for RAG
- [SETUP.md §8L](../../SETUP.md#8l-bm25-hybrid-search) — BM25 hybrid search
