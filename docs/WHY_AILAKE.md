# Why AI-Lake

> A technical case for the community — honest about tradeoffs, concrete on benchmarks.

---

## Quick start

```bash
pip install ailake
```

```python
import ailake
import numpy as np

# Open or create a table (local, s3://, gs://, az://)
table = ailake.open_table("./my_docs", dim=1536, metric="cosine")

texts      = ["Introduction to AI-Lake", "Vector search at scale", "Iceberg ACID"]
embeddings = np.random.rand(3, 1536).astype(np.float32)

table.insert(texts, embeddings)
table.commit()

# Search
query = np.random.rand(1536).astype(np.float32)
df = table.search(query, top_k=10).to_pandas()
print(df)  # row_id, distance, file

# Same table readable in Spark — zero configuration, no plugin
# spark.read.format("iceberg").load("./my_docs")
```

Production ingest at ~200k vec/s with deferred index builds:

```python
table.write_batch_auto_deferred(texts, embeddings)
table.commit()
# HNSW or IVF-PQ built in background; shard served via flat scan until ready
```

---

## The problem no one wants to talk about

Most teams building RAG or recommendation systems end up with two separate systems for the same data:

```
Your data lake (Parquet / Iceberg)
  → ETL job (sync, transform, re-embed)
    → Vector DB (Pinecone / Milvus / Weaviate)
      → results joined back to data lake at query time
```

This architecture has a well-known failure mode: **sync drift**. A document deleted from the data lake survives in the vector DB. A batch job fails halfway. The vector index reflects state from six hours ago. Your RAG pipeline returns results that reference documents that no longer exist.

The standard answer is "run more frequent syncs." The real answer is: **you should not have two systems for the same data**.

---

## The "just use Iceberg" objection

Iceberg alone stores embeddings fine:

```sql
CREATE TABLE docs (
  id       BIGINT,
  text     STRING,
  embedding BINARY  -- store any bytes here
) USING iceberg;
```

But similarity search on this table is a full scan:

```sql
-- This reads every embedding in every Parquet file
SELECT id, text, cosine_distance(embedding, ?) AS dist
FROM docs
ORDER BY dist
LIMIT 10;
```

At 10M rows: ~30 GB of embedding data read per query. At 100M rows: ~300 GB. Even with Spark parallelism, this is multiple seconds of I/O and CPU per query. Iceberg has `lower_bound`/`upper_bound` statistics for scalar columns — these mean nothing for vectors.

**Iceberg stores vectors. Iceberg cannot search vectors efficiently.**

---

## The "just use LanceDB" objection

LanceDB is excellent. Fast ingestion, clean Python API, good DX. Use it if it fits.

The honest comparison on SIFT-1M (1M vectors, dim=128, Euclidean):

| Metric | LanceDB | AI-Lake HNSW |
|---|---|---|
| Write throughput | **530k vec/s** | 199k vec/s (deferred) |
| Recall@10 | 88.05% | **99.63%** |
| QPS | 745 | **1,365** |
| p99 latency | 63.34ms | **1.96ms** |

LanceDB uses IVF-PQ by default for fast ingest — it trades recall for write speed. AI-Lake builds HNSW in a background Tokio task (deferred mode), so ingest is still fast and recall is not sacrificed.

**But the deeper issue is not benchmarks.** It is ecosystem fit.

LanceDB uses the Lance format. Lance tables are not readable by Spark, Trino, Athena, DuckDB, Snowflake, or any other Iceberg-compatible engine without conversion. If your organisation already runs BI on Iceberg, LanceDB is a new silo that requires a sync pipeline — the same problem you were trying to solve.

AI-Lake tables **are** Iceberg tables. Spark, Trino, DuckDB, and Athena read them without any plugin:

```python
# Standard PySpark — no AI-Lake plugin installed
df = spark.read.format("iceberg").load("glue_catalog.db.my_ailake_table")
df.filter("category = 'finance'").count()  # works, no modification needed
```

The HNSW index in the file footer is past the final `PAR1` marker — standard Parquet readers stop there per spec. The extension is invisible and inert to anything that does not understand it.

---

## What AI-Lake actually is

A Parquet file with one addition: after the final `PAR1` marker, the writer appends:

```
┌─────────────────────────────────────────┐
│  PARQUET HEADER + ROW GROUPS            │
│  (standard — any reader works)          │
├─────────────────────────────────────────┤
│  AILK HEADER  (64 bytes)                │
│  CENTROID + RADIUS  (dim×4 + 4 bytes)   │
│  HNSW GRAPH  (bincode-serialised)       │
│  AILK TRAILER  (24 bytes)               │
├─────────────────────────────────────────┤
│  PARQUET FOOTER  (schema + KV metadata) │
│  footer_len  (4 bytes LE)               │
│  PAR1  (4 bytes) ← last bytes readers   │
│                    follow the spec to   │
└─────────────────────────────────────────┘
```

Each file is self-contained: data + index + centroid in one object. No separate index files to keep in sync. No external service to keep running.

The Iceberg manifest carries per-file `centroid` and `radius` in its `custom_properties`. Before opening any Parquet file, the SDK computes `distance(query, centroid[i]) - radius[i]` for every file. Files whose nearest possible point is farther than the search threshold are skipped — **without any I/O on the data files themselves**. On tables with thousands of files, 95–99% of S3 objects are never fetched.

---

## When to use AI-Lake

**AI-Lake is the right choice when:**

- You already run Iceberg (Glue, Nessie, Unity Catalog, Polaris) for BI workloads and want vector search on the same tables without a new service.
- Your data governance team requires a single catalog, single access control, and single audit log. Two systems means two places to manage permissions and two audit trails.
- You need ACID consistency between tabular and vector data. A document deleted via `DELETE FROM` must disappear from vector search results in the same snapshot — not "eventually."
- You need time-travel on vectors. `SELECT … AS OF SNAPSHOT 12345` returns results consistent with that historical state of the index.
- You run Spark, Trino, or Flink for transforms and want vector search in the same pipeline without a network hop to an external service.
- You want sub-2ms p99 at production scale with no recall compromise.

**AI-Lake is also the right choice when:**

- You need hybrid BM25 + vector search without running an external FTS cluster. AI-Lake includes a pure-Rust BM25 scorer (`BM25Scorer`) with IDF stats accumulated at write time, fused with HNSW via Reciprocal Rank Fusion in a single search call. No Elasticsearch, no Tantivy dependency.
- You need atomic deletes. `ailake.delete_where(path, "id", [ids])` commits an Iceberg equality delete in one snapshot — the rows vanish from both SQL queries and vector search simultaneously. No dual-system "delete in DB, delete in vector service" race condition.
- You need per-agent/per-tenant isolation at massive scale. `partition_by="agent_id"` tags each file at write time; `partition_filter="agent-42"` at search time prunes to that agent's files at the manifest level — before any HNSW I/O. Isolation cost is O(manifest entries), not O(total vectors).

**AI-Lake is not the right choice when:**

- You are starting from scratch with no Iceberg infrastructure. LanceDB is simpler to get running in a day.
- Your dataset is below ~1M vectors and your latency requirement is above 1 second. Iceberg + DuckDB `array_cosine_similarity` is sufficient.
- You need full-text search as a primary product feature (web-scale FTS, faceting, autocomplete). AI-Lake's BM25 path is a complement to vector search for RAG workloads — not a replacement for Elasticsearch or Solr at search-engine scale.
- You need a managed cloud service with a control plane. AI-Lake is a format and SDK, not a SaaS product.

---

## The ecosystem argument

The deepest reason to use AI-Lake is not performance. It is **avoiding a second system**.

Consider what happens when a document is updated in a dual-system architecture:

```
1. Write new version to data lake (Iceberg commit → snapshot S2)
2. Delete old embedding from vector DB (async, may fail)
3. Embed new version (latency: 50-200ms per doc)
4. Insert new embedding to vector DB (async, may fail)
5. Steps 2-4 are not atomic — window where old and new both exist
```

With AI-Lake:

```
1. Write new version (Iceberg commit → snapshot S2)
   ↳ new HNSW in footer of new file
   ↳ old file enters retention period (time-travel still works)
   ↳ vector search on S2 returns new version — atomically
```

No async pipeline. No failure modes 2-4. One transaction.

The same applies to access control. If you revoke a user's access to a table in your catalog, they lose access to both tabular queries and vector search in one operation. With a dual-system stack, you must revoke in two places — and synchronisation bugs are a real security concern.

---

## Storage and cost

### Default (HNSW + F16 raw vectors)

For 100M records, `text-embedding-3-small` (dim=1536):

| Component | Size |
|---|---|
| Tabular columns (text, metadata) | ~50 GB |
| Vector column (F16 in Parquet) | ~300 GB |
| HNSW footer (10-20% of vectors) | ~30-60 GB |
| **Total** | **~380-410 GB** |

### PQ-only mode (extreme compression)

When raw vectors are not needed for reranking:

```python
table = ailake.open_table("s3://lake/docs/", dim=1536, pq_only=True)
```

| Component | Size |
|---|---|
| Tabular columns | ~50 GB |
| IVF-PQ codes only | ~5 GB |
| **Total** | **~55 GB** — 7× smaller |

Recall@10 ≈ 93–95% (vs 99.6% HNSW). Reranking with raw vectors is disabled.

### Geometric pruning reduces query I/O

On a 10k-file table with good vector clustering, a query touches 50–200 files. At ~15 MB HNSW per file, that is 750 MB–3 GB of footer reads per query instead of 150 GB (all files). S3 range GETs are billed per request and per byte — pruning directly cuts S3 cost.

---

## Compatibility matrix

| Engine | Tabular read | Tabular write | Vector search |
|---|---|---|---|
| **Apache Spark 3.5 / 4.0** | ✅ Native Iceberg | ✅ | ✅ `spark-plugin/` |
| **Trino 430+** | ✅ Native Iceberg | ✅ | ✅ `trino-plugin/` |
| **Apache Flink 1.18+** | ✅ Iceberg connector | ✅ `ailake-flink` | ✅ `ailake-flink` |
| **DuckDB 0.10+** | ✅ Iceberg ext. | ✅ `duckdb-ailake` | ✅ `ailake_search()` |
| **PyIceberg 0.6+** | ✅ | ✅ | via SDK direct |
| **Apache Beam 2.56+** | ✅ `Managed.ICEBERG` | ✅ | via SDK direct |
| **AWS Athena** | ✅ Glue catalog | limited | — |
| **Snowflake** | ✅ Iceberg tables | limited | — |
| **Python (`ailake-py`)** | ✅ PyArrow | ✅ | ✅ fluent `SearchQuery` |
| **Go (`ailake-go`)** | ✅ | ✅ | ✅ |
| **C++17 (`ailake-cpp`)** | ✅ | ✅ | ✅ |

"via SDK direct" means: use the AI-Lake Python/Go/C++ SDK inside a DoFn or UDF to run vector search outside the engine's SQL planner.

---

## Contributing

See [`CONTRIBUTING.md`](../CONTRIBUTING.md) and [`docs/contributing/DECISIONS.md`](contributing/DECISIONS.md) for architecture decisions and the contribution workflow.

File format spec: [`docs/specs/FILE_FORMAT.md`](specs/FILE_FORMAT.md).

Engine integrations: [`docs/specs/INTEGRATIONS.md`](specs/INTEGRATIONS.md).
