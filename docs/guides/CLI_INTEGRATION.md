# CLI Integration Guide

`ailake` is the administrative CLI for AI-Lake — a single Rust binary (`ailake-cli`)
covering table lifecycle (create/insert/compact), search (vector/FTS/hybrid), schema
evolution, deletes, embedding-model migration, storage estimation, and an HTTP server.
It's the natural entry point for shell scripts, cron jobs, CI pipelines, and Airflow
`BashOperator`/`AilakeHook` shell-outs — every SDK guide in this directory
(`PYTHON_INTEGRATION.md`, `GO_INTEGRATION.md`, `CPP_INTEGRATION.md`, `JVM_INTEGRATION.md`)
documents a binding that either wraps this CLI directly or reimplements a subset of it
natively; this guide documents the CLI itself as the baseline capability surface.

All commands, flags, and example output in this guide were run against a real local
table with the actual `ailake` binary — not aspirational.

---

## 1. Installation

```bash
git clone https://github.com/ailake-io/ai-lakehouse.git
cd ai-lakehouse
cargo build --release -p ailake-cli
# binary at target/release/ailake
```

For the optional DuckLake catalog backend (`--catalog ducklake`), build with the
`catalog-ducklake` feature — see `docs/guides/DUCKLAKE_CATALOG.md` (it pulls in a
bundled DuckDB C++ build, so it's opt-in, not part of the default build):

```bash
cargo build --release -p ailake-cli --features catalog-ducklake
```

```bash
ailake --version   # ailake 0.1.3
ailake --help      # full command list
```

---

## 2. Global options

Every subcommand accepts these two:

| Flag | Default | Description |
|---|---|---|
| `--store <STORE>` | `.` (env: `AILAKE_STORE_URL`) | `s3://bucket/prefix`, `gs://bucket/prefix`, `az://container/prefix`, or a local filesystem path |
| `--catalog <CATALOG>` | `hadoop` | `hadoop` or `ducklake`. `ducklake` requires a local filesystem `--store` (no `s3://`/`gs://`/`az://`) and the `catalog-ducklake` build feature |

Table names are `namespace.table` or just `table` (defaults to namespace `default`).
Most subcommands also accept `--format text|json` — `text` is human-readable,
`json` is machine-parseable (one JSON object per invocation, not NDJSON).

**Note:** command errors go to stderr as plain text (`error: <message>`, exit code 1)
regardless of `--format` — they are never JSON-wrapped, even in `--format json` mode.
Scripts checking for command success should check the exit code, not attempt to parse
stderr as JSON.

---

## 3. `create` — new table

```bash
ailake create docs.chunks --dim 32 --metric cosine --store ./lake --fts-columns text
```
```
created table docs.chunks
```

Key flags (run `ailake create --help` for the full list — it's long, covering every
storage/index knob):

| Flag | Default | Notes |
|---|---|---|
| `--dim <N>` | required | vector dimensionality |
| `--metric` | `cosine` | `cosine` \| `euclidean` \| `dot` |
| `--precision` | `f16` | `f32` \| `f16` \| `i8` |
| `--column` | `embedding` | primary vector column name |
| `--pre-normalize` | off | L2-normalize at write time; enables the `NormalizedCosine` fast path (no sqrt in the HNSW hot loop) |
| `--hnsw-m` / `--hnsw-ef` | `16` / `150` | HNSW graph tuning |
| `--pq-only` | off | omit raw vectors from Parquet — ~98% storage cut, no reranking |
| `--modality` | none | `text` \| `image` \| `audio` \| `video` |
| `--format-version` | `2` | `2` or `3` (V3 enables row lineage; equality deletes are Iceberg-V2-encoded on both) |
| `--fts-columns` | none | comma-separated text columns to index with Tantivy FTS at write time |

---

## 4. `insert` — write a Parquet file

Source data must already contain the vector column encoded as **F16 `FixedSizeBinary(dim*2)`**
— this mirrors the physical on-disk column type (see `CLAUDE.md` §5C, §2), not a plain
`list<float32>` column. This is the one sharp edge of the raw CLI compared to the SDKs
(`ailake-py`'s `TableWriter.write_batch()` takes `float32` numpy arrays and quantizes to
F16 internally before calling the same underlying write path `insert` uses) — if you're
hand-building a source Parquet file for `insert` outside of an SDK, encode each float32
to IEEE-754 binary16 and pack `dim*2` bytes per row yourself.

```bash
ailake insert docs.chunks source.parquet --embeddings embedding --store ./lake
```
```
inserted 20 rows into docs.chunks
```

Notable flags:

| Flag | Notes |
|---|---|
| `--vector-cols col:dim:metric[:modality],...` | multi-column (Phase 8 multimodal) mode — ignores `--embeddings`/`--metric`/`--precision` |
| `--batch-id <ID>` | idempotency key — safe re-run for Airflow/orchestrator retries; no-op if already committed |
| `--deferred` | write Parquet immediately, build HNSW in the background (~200k vec/s vs. blocking build); not combinable with `--batch-id` |
| `--fts-columns` | per-insert FTS indexing (independent of `create`'s `--fts-columns` — each file carries its own index) |
| `--partition-by` / `--partition-value` / `--partition-fields` | Iceberg hidden partitioning; `--partition-fields` (JSON array) takes precedence over the single-column `--partition-by` |
| `--hnsw-m` / `--hnsw-ef` | only take effect when this `insert` is the one creating the table — ignored on writes to an already-created table |

---

## 5. `search` — vector, full-text, or hybrid

Vector and full-text (`--text`) are mutually exclusive on their own, but `--hybrid-text`
combines both (BM25 + vector fused via Reciprocal Rank Fusion).

```bash
# vector search
ailake search docs.chunks --store ./lake \
  --query "0.1,0.1,0.1,...,0.1" --top-k 3
```
```
rank   distance     file
1      0.781340     data/part-1783973180891-00000.parquet
2      0.797473     data/part-1783973180891-00000.parquet
3      0.838059     data/part-1783973180891-00000.parquet
```

```bash
# full-text search (Tantivy if the table has an FTS index, else O(N) BM25 fallback)
ailake search docs.chunks --store ./lake \
  --text "chunk 5" --text-columns text --top-k 3
```
```
1: row_id=0 score=0.4919 file=data/part-1783973180891-00000.parquet
2: row_id=1 score=0.4919 file=data/part-1783973180891-00000.parquet
3: row_id=2 score=0.4919 file=data/part-1783973180891-00000.parquet
```

```bash
# JSON output — note row_id is only present here, not in the default text table
ailake search docs.chunks --store ./lake --query "0.1,..." --top-k 2 --format json
```
```json
{"results":[{"distance":0.7813403606414795,"file_path":"data/part-....parquet","rank":1,"row_id":5},{"distance":0.7974725961685181,"file_path":"data/part-....parquet","rank":2,"row_id":4}]}
```

| Flag | Notes |
|---|---|
| `--query` / `--query-file` | comma-separated floats, or a path to a little-endian f32 binary file |
| `--hybrid-text` + `--query`/`--query-file` | enables BM25+vector fusion; `--bm25-weight` (default `0.5`) controls the RRF balance |
| `--pruning-threshold` | geometric pruning aggressiveness (0.0–1.0, lower = more files pruned; default `0.8`) |

The CLI's `search` is pointer-only (row_id/distance/file_path) — it does not fetch full
row data (no-JOIN full-row fetch, `ailake_scan_json`, is exposed via the SDKs and the
JVM plugins' `search_full`/`scan()`/`search.mode='full'`, not the CLI; see
`docs/guides/JVM_INTEGRATION.md` and `docs/guides/PYTHON_INTEGRATION.md` §8).

---

## 6. `compact` — merge small files

```bash
ailake compact docs.chunks --store ./lake --min-files 2 --format json
```
```json
{"files_compacted":1,"ok":true,"output_path":"data/compacted-1783973195154.parquet"}
```

| Flag | Default | Notes |
|---|---|---|
| `--target-size` | `536870912` (512 MiB) | target output file size |
| `--min-files` | `4` | minimum small-file count to trigger a merge |
| `--max-files-per-pass` | `20` | bounds peak RAM / HNSW rebuild cost per pass |
| `--deferred` | off | write merged Parquet immediately, rebuild HNSW in the background |

A table with any "foreign" file (written by a generic Iceberg engine, no AI-Lake
footer/centroid) has compaction prioritized regardless of the size/count thresholds —
one foreign file is enough to trigger a repair pass. See `CLAUDE.md` §5A.

---

## 7. `info` / `estimate` — inspect and plan

```bash
ailake info docs.chunks --store ./lake
```
```
table:       docs.chunks
location:    docs/chunks
vector:      col=embedding dim=32 metric=cosine
files:       1 (1 indexed)
rows:        20
size:        10665 B
snapshot:    1783973180894877
```

`--format json` gives the same fields machine-parseable, plus `foreign_files`/
`foreign_file_paths` (files without an AI-Lake index — see §6) and `failed_files`.

`estimate` is pure math, no I/O — plan storage before writing anything:

```bash
ailake estimate --rows 500K --dim 768 --format json
```
```json
{
  "dim": 768, "rows": 500000, "hnsw_m": 16, "pq_m": 24,
  "estimates": [
    {"mode": "F16 (default)", "vectors_bytes": 768000000, "index_bytes": 144000000,
     "total_bytes": 912000000, "reduction_factor": "1.8×", "recall_at_10": "~99%", "note": ""},
    {"mode": "PQ-only (--pq-only)", "vectors_bytes": 0, "index_bytes": 12000000,
     "total_bytes": 12000000, "reduction_factor": "140.0×", "recall_at_10": "~94%", "note": "no reranking"}
  ]
}
```
(`--rows` accepts `K`/`M`/`B` suffixes: `1M`, `500K`, `1B`.)

---

## 8. Schema evolution — `evolve`, `add-vector-column`, `backfill-vector-column`

`evolve` adds/renames plain (non-vector) columns without rewriting any data file —
old files return the initial-default (or null) for new columns at read time:

```bash
ailake evolve docs.chunks --store ./lake \
  --add "topic:string" --initial-default '"general"'
```
```
new_schema_id: 1
```
`--add` and `--rename` (`old:new`) may each be repeated for multiple changes in one call.

Adding a **vector** column is a two-step process, matching `TableWriter`'s add/backfill
split in the SDKs:

```bash
# Step 1: register the column in metadata.json — no data rewritten, existing files
# return null for it until backfilled.
ailake add-vector-column docs.chunks --store ./lake \
  --column image_embedding --dim 16 --metric cosine
```
```
vector column 'image_embedding' added — new_schema_id: 1
```

```bash
# Step 2: rewrite every existing file with the new column populated via an
# external embed command (same stdin/stdout JSON protocol as `migrate`, below).
ailake backfill-vector-column docs.chunks --store ./lake \
  --column image_embedding --text-column text \
  --embed-cmd "python3 embed_cmd.py 16"
```
```
backfill: 1/1 files done (0 skipped), 20 rows
backfill complete for column 'image_embedding'
```
Idempotent: files that already contain the column are skipped on re-run.

---

## 9. Deletes — `delete-where` vs `delete-rows`

Two different mechanisms for two different shapes of delete:

| | `delete-where` | `delete-rows` |
|---|---|---|
| Predicate shape | equality on one column, any table version | explicit row positions in one named file |
| Mechanism | Iceberg equality delete file + `Delete` snapshot | Iceberg **Deletion Vector** (V3 only) |
| Table requirement | any format version | **V3 only** — `rejects_v2_table` |
| Typical use | `document_id`, `agent_id`, `session_id` bulk deletes | precise row-level deletes when you already know row positions (e.g. from `ailake info`/`search --format json`'s `row_id`) |

```bash
ailake delete-where docs.chunks --store ./lake \
  --col document_id --vals "doc-abc,doc-def"
```
```
delete-where committed: 1 predicates on column 'text'
```

```bash
ailake delete-rows docs.chunks --store ./lake \
  --file data/part-00001.parquet --rows "0,5,42"
```

Both are logical deletes — matching rows are masked at scan time, no data file is
rewritten. Run `compact` afterward to physically reclaim space.

---

## 10. `migrate` — re-embed to a new model

Re-embeds every chunk's text via an external command and cuts over the vector column,
without you writing any Rust/Python glue:

```bash
ailake migrate docs.chunks --store ./lake \
  --old-column embedding --new-column embedding_v2 --text-column text \
  --embed-cmd "python3 embed_cmd.py 32" \
  --strategy atomic-replace --model-name demo-embed-v2
```
```
migration: 1/2 files done, 20 rows migrated
migration: 2/2 files done, 22 rows migrated
migration complete
```

`--embed-cmd` protocol (same for `backfill-vector-column`): the command reads a JSON
array of strings from stdin and writes a JSON array of float arrays to stdout. Minimal
example (`python3 embed_cmd.py <dim>`, deterministic hash-seeded unit vectors — see
`airbyte-destination-ailake/demo/embed_cmd.py` for the real reference implementation):

```python
import json, math, random, sys

def unit_vec(text, dim):
    rng = random.Random(hash(text) & 0xFFFFFFFF)
    v = [rng.gauss(0.0, 1.0) for _ in range(dim)]
    n = math.sqrt(sum(x * x for x in v)) or 1.0
    return [x / n for x in v]

texts = json.load(sys.stdin)
dim = int(sys.argv[1]) if len(sys.argv) > 1 else 32
print(json.dumps([unit_vec(t, dim) for t in texts]))
```

| Flag | Notes |
|---|---|
| `--strategy` | `atomic-replace` (lower storage, brief unavailability) or `dual-write-then-cutover` (default; zero downtime, temporarily 2× storage) |
| `--new-column` | may equal `--old-column` for a true in-place upgrade |
| `--batch-size` | texts per `--embed-cmd` invocation (default `512`) |
| `--model-name` / `--model-version` | stored as `ailake.embedding-model` (`"<name>@<version>"`) after cutover — both strategies also update the table's `ailake.vector-column` property to point at the new column |

---

## 11. `decay-memories` — agent memory recency

Recomputes `recency_weight = exp(-lambda * days_since_access)` from each row's
`last_accessed_at` column across every file in the table (Phase 9 agent memory —
see `CLAUDE.md` §"Fase 9"):

```bash
ailake decay-memories agent.memories --store ./lake --lambda 0.1 --format json
```
```json
{"files_updated":0,"ok":true}
```
(`files_updated: 0` here because the demo table has no `last_accessed_at` column —
tables written via `ailake-py`'s `Agent` class or `EpisodicMemorySchema` carry it as a
real `Timestamp(Nanosecond, UTC)` column; see `docs/guides/PYTHON_INTEGRATION.md` §14/§16
for `TimestampNs` — a plain `Int64` column is silently rejected.)

`--lambda` — higher decays faster. Typical range `0.05` (slow) to `0.5` (aggressive).

---

## 12. `serve` — HTTP API

Starts a minimal JSON HTTP server exposing search/write/compact/info for one table —
useful for a sidecar process fronting a single AI-Lake table without embedding the Rust
crate or an SDK.

```bash
ailake serve docs.chunks --store ./lake --port 7700
```
```
ailake server listening on http://0.0.0.0:7700
WARNING: no authentication — expose only on a trusted network or behind an authenticating proxy
```

> **Security**: no authentication. Localhost/VPC-internal/sidecar deployments only — put
> an authenticating reverse proxy (nginx + mTLS, API gateway) in front for anything else.

| Endpoint | Method | Request body | Response |
|---|---|---|---|
| `/search` | POST | `{"query":[f32...],"top_k":10,"pruning_threshold":0.8}` | `{"results":[{"rank","row_id","distance","file_path"}...]}` |
| `/write` | POST | `{"texts":["..."],"embeddings":[[f32...]],"batch_id":"..."}` | `{"snapshot_id","rows"}` |
| `/compact` | POST | `{"target_size":536870912,"min_files":4}` (all optional) | `{"message","compacted_files"}` |
| `/info` | GET | — | same fields as `ailake info --format json` |

Real example round-trip:

```bash
curl -s http://localhost:7700/info
# {"table":"docs.chunks","location":"docs/chunks","vector_column":"embedding","vector_dim":"32",...}

curl -s -X POST http://localhost:7700/search -H 'Content-Type: application/json' \
  -d '{"query":[0.1,0.1,...],"top_k":3}'
# {"results":[{"rank":1,"row_id":5,"distance":0.78134036,"file_path":"data/part-....parquet"},...]}

curl -s -X POST http://localhost:7700/write -H 'Content-Type: application/json' \
  -d '{"texts":["new chunk a","new chunk b"],"embeddings":[[0.05,...],[0.06,...]]}'
# {"snapshot_id":1783973839931261,"rows":2}
```
Request bodies are capped at 32 MB (`MAX_BODY_BYTES`); `top_k` is capped at 10,000
(`MAX_TOP_K`) regardless of what's requested.

---

## 13. Catalog backends

`--catalog hadoop` (default) — a filesystem/object-store-native Iceberg catalog, works
with any `--store` scheme. `--catalog ducklake` — a real DuckLake catalog via the
`ducklake` DuckDB extension, local filesystem only, requires the `catalog-ducklake`
build feature. See `docs/guides/DUCKLAKE_CATALOG.md` for the full writeup, including four
real bugs found wiring it up (cross-attachment transactions, retired-file visibility,
relative-path resolution, `allow_missing` schema evolution).

```bash
ailake create docs.chunks --dim 32 --catalog ducklake --store /local/warehouse
```

---

## 14. Full example — ingest → index → search pipeline

```bash
#!/usr/bin/env bash
set -euo pipefail

STORE=s3://my-lake/docs
TABLE=default.chunks

# One-time table setup with FTS on the chunk text.
ailake create "$TABLE" --dim 1536 --metric cosine --pre-normalize \
  --fts-columns chunk_text --store "$STORE"

# Ingest, high-throughput (index builds async in the background).
ailake insert "$TABLE" batch_001.parquet --embeddings embedding \
  --batch-id "$(date +%Y%m%d)-batch-001" --deferred --store "$STORE"

# Compact once enough small files have accumulated (e.g. nightly cron).
ailake compact "$TABLE" --min-files 8 --format json --store "$STORE"

# Search.
QUERY_VEC=$(python3 embed_query.py "what changed in the DuckLake catalog?")
ailake search "$TABLE" --hybrid-text "DuckLake catalog changes" \
  --query "$QUERY_VEC" --bm25-weight 0.3 --top-k 10 --format json --store "$STORE"
```

---

## 15. Command reference

| Command | Purpose |
|---|---|
| `create` | new table |
| `insert` | write a Parquet file (single or multi-column) |
| `search` | vector / FTS / hybrid search |
| `compact` | merge small files |
| `info` | table statistics |
| `estimate` | storage math, no I/O |
| `evolve` | add/rename non-vector columns |
| `add-vector-column` | register a new vector column (metadata only) |
| `backfill-vector-column` | populate a registered vector column across existing files |
| `delete-where` | equality-predicate logical delete (any format version) |
| `delete-rows` | positional logical delete via Deletion Vectors (V3 only) |
| `migrate` | re-embed to a new model, atomic or dual-write cutover |
| `decay-memories` | recompute agent-memory recency weights |
| `serve` | HTTP API for search/write/compact/info |

---

## Related docs

- `docs/guides/PYTHON_INTEGRATION.md`, `GO_INTEGRATION.md`, `CPP_INTEGRATION.md`,
  `JVM_INTEGRATION.md` — SDK bindings, several of which shell out to this same CLI
- `docs/guides/DUCKLAKE_CATALOG.md` — `--catalog ducklake` details
- `docs/guides/DBT_INTEGRATION.md` — dbt post-hook patterns (via Spark/Trino SQL
  functions, not this CLI)
- `docs/specs/FILE_FORMAT.md` — physical file layout referenced in §4's F16 encoding note
- `CLAUDE.md` — architecture overview, Fase roadmap
