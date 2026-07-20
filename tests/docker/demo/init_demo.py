"""
AI-Lake demo fixture generator.

Writes multiple AI-Lake tables to demonstrate all SDK features:
  - HNSW table        — 500 rows, dim=32 (main fixture, notebooks 01-05)
  - PQ-only table     — 500 rows, dim=32, no raw vectors stored
  - Deferred table    — 200 rows, write_batch_auto_deferred
  - Residual-PQ       — 500 rows, ivf_residual=True
  - Model-tracked     — 100 rows, embedding_model="synthetic-embed-v1"@1.0 (notebook 01 §18)
  - Multimodal        — 200 rows, text embedding (dim=32) + image embedding (dim=16)
  - Agent memory      — 100 rows across 2 agents (Phase 9 partition isolation demo)
  - Partitioned v3    — 200 rows, partition_fields=[topic_id:identity:int], format_version=3
  - Delete demo       — 100 rows, 10 pre-deleted via delete_where (notebook §29 demo)
  - Schema-evo demo   — 100 rows, add_column + rename + evolve_schema (notebook §30 demo)

Runs once at container startup via entrypoint.sh; skipped on restart if
version-hint.text already exists AND the stamp file at /data/.fixture-version
matches FIXTURE_VERSION below. Bump FIXTURE_VERSION whenever this script
changes what gets written (new table, new property, new arg) — otherwise a
container restart against a pre-existing `demo-data` volume silently keeps
serving fixtures from before the change, since the volume outlives image
rebuilds.
"""

import json
import math
import os
import pathlib
import random
import sys

# Bump on any change to what main() writes (new table, new property, new
# arg) — entrypoint.sh compares this against /data/.fixture-version and
# forces a regen on mismatch, even if version-hint.text already exists.
FIXTURE_VERSION     = "3"

TABLE_PATH          = os.environ.get("DEMO_TABLE_PATH", "/data/ailake_demo")
PQ_PATH             = str(pathlib.Path(TABLE_PATH).parent / "ailake_pq")
RESIDUAL_PATH       = str(pathlib.Path(TABLE_PATH).parent / "ailake_residual_pq")
DEFERRED_PATH       = str(pathlib.Path(TABLE_PATH).parent / "ailake_deferred")
MODEL_TRACKED_PATH  = str(pathlib.Path(TABLE_PATH).parent / "ailake_model_tracked")
MULTIMODAL_PATH     = str(pathlib.Path(TABLE_PATH).parent / "ailake_multimodal")
AGENT_PATH          = os.environ.get("DEMO_AGENT_PATH",
                          str(pathlib.Path(TABLE_PATH).parent / "ailake_agent"))
PARTITIONED_V3_PATH = str(pathlib.Path(TABLE_PATH).parent / "ailake_partitioned_v3")
DELETE_DEMO_PATH    = str(pathlib.Path(TABLE_PATH).parent / "ailake_delete_demo")
SCHEMA_EVO_PATH     = str(pathlib.Path(TABLE_PATH).parent / "ailake_schema_evo")
BM25_PATH           = str(pathlib.Path(TABLE_PATH).parent / "ailake_bm25")
FTS_PATH            = os.environ.get("DEMO_FTS_PATH",
                          str(pathlib.Path(TABLE_PATH).parent / "ailake_fts"))
DIM                 = int(os.environ.get("DEMO_DIM", "32"))
IMAGE_DIM           = 16   # synthetic "CLIP-like" image embeddings (half the text dim)
N_DOCS              = 500
N_DEFERRED          = 200
N_AGENT_DOCS        = 50   # per agent — 50 × 2 agents = 100 rows total
METRIC              = "cosine"

TOPICS = [
    "machine learning", "database systems", "vector search", "data lakes",
    "cloud storage", "Apache Iceberg", "embedding models", "RAG pipelines",
    "stream processing", "graph neural networks", "transformer architecture",
    "quantization", "approximate nearest neighbor", "columnar storage",
    "time-series analysis", "recommendation systems", "LLM inference",
    "distributed computing", "data versioning", "semantic search",
]

TEMPLATES = [
    "This document covers {topic}. It discusses key concepts, implementations, and real-world applications.",
    "An introduction to {topic}: fundamentals, algorithms, and best practices for production systems.",
    "Deep dive into {topic} — architecture decisions, performance trade-offs, and scaling strategies.",
    "Research notes on {topic}: state of the art, open problems, and future directions.",
    "Engineering guide for {topic}: setup, benchmarks, and operational considerations.",
]


def rand_unit_vec(dim: int, seed: int) -> list[float]:
    rng = random.Random(seed)
    v = [rng.gauss(0.0, 1.0) for _ in range(dim)]
    norm = math.sqrt(sum(x * x for x in v))
    if norm < 1e-9:
        v[0] = 1.0
        norm = 1.0
    return [x / norm for x in v]


def _build_corpus(n: int) -> tuple[list[str], list[list[float]]]:
    texts: list[str] = []
    embeddings: list[list[float]] = []
    for i in range(n):
        topic    = TOPICS[i % len(TOPICS)]
        template = TEMPLATES[(i // len(TOPICS)) % len(TEMPLATES)]
        texts.append(template.format(topic=topic) + f" (doc_id={i})")
        embeddings.append(rand_unit_vec(DIM, seed=i))
    return texts, embeddings


def _write_hnsw(texts: list[str], embeddings: list[list[float]]) -> None:
    """Main HNSW table — standard index, raw vectors kept for reranking."""
    import ailake
    os.makedirs(TABLE_PATH, exist_ok=True)
    table = ailake.open_table(
        TABLE_PATH,
        dim=DIM,
        metric=METRIC,
        embedding_model="synthetic-embed-v1",
        embedding_model_version="1.0",
    )
    table.insert(texts, embeddings)
    snap_id = table.commit()
    print(f"[HNSW]     Committed snapshot_id={snap_id}  rows={len(texts)}")


def _write_pq_only(texts: list[str], embeddings: list[list[float]]) -> None:
    """PQ-only table — raw vectors discarded after index build (maximum compression)."""
    import ailake
    os.makedirs(PQ_PATH, exist_ok=True)
    table = ailake.open_table(PQ_PATH, dim=DIM, metric=METRIC, pq_only=True)
    table.insert(texts, embeddings)
    snap_id = table.commit()
    print(f"[PQ-only]  Committed snapshot_id={snap_id}  rows={len(texts)}")


def _write_residual_pq(texts: list[str], embeddings: list[list[float]]) -> None:
    """Residual-PQ table — encodes residuals from cluster centroid (better recall)."""
    import ailake
    os.makedirs(RESIDUAL_PATH, exist_ok=True)
    table = ailake.open_table(RESIDUAL_PATH, dim=DIM, metric=METRIC, ivf_residual=True)
    table.insert(texts, embeddings)
    snap_id = table.commit()
    print(f"[Residual] Committed snapshot_id={snap_id}  rows={len(texts)}")


def _write_multimodal(texts: list[str], embeddings: list[list[float]]) -> None:
    """Multimodal table — text column (dim=DIM) + image column (dim=IMAGE_DIM)."""
    from ailake import TableWriter, VectorColSpec

    os.makedirs(MULTIMODAL_PATH, exist_ok=True)

    # Synthetic "image" embeddings — different dim to demonstrate cross-modal
    image_embs = [rand_unit_vec(IMAGE_DIM, seed=i + 10_000) for i in range(200)]

    text_spec  = VectorColSpec("embedding",       DIM,       "cosine", "text")
    image_spec = VectorColSpec("image_embedding", IMAGE_DIM, "cosine", "image")

    w = TableWriter(MULTIMODAL_PATH, dim=DIM, metric="cosine")
    w.write_batch_multi(
        texts[:200],
        [(text_spec, embeddings[:200]), (image_spec, image_embs)],
    )
    snap_id = w.commit()
    print(
        f"[Multimodal] Committed snapshot_id={snap_id}  rows=200"
        f"  cols=embedding(dim={DIM})+image_embedding(dim={IMAGE_DIM})"
    )


def _write_agent_memory(texts: list[str], embeddings: list[list[float]]) -> None:
    """Phase 9 — two agents writing to the same table with partition isolation.

    agent-A owns rows 0..N_AGENT_DOCS, agent-B owns rows N_AGENT_DOCS..2*N_AGENT_DOCS.
    Embeddings for each agent cluster around an orthogonal centroid so partition
    pruning is visibly effective in notebook §25.
    """
    import ailake
    os.makedirs(AGENT_PATH, exist_ok=True)

    # Agent-A: topics 0..N_AGENT_DOCS (first N rows of corpus)
    writer_a = ailake.TableWriter(
        AGENT_PATH, dim=DIM, metric=METRIC,
        partition_by="agent_id", partition_value="agent-A",
    )
    writer_a.write_batch(texts[:N_AGENT_DOCS], embeddings[:N_AGENT_DOCS])
    snap_a = writer_a.commit()
    print(
        f"[Agent-A]  Committed snapshot_id={snap_a}"
        f"  rows={N_AGENT_DOCS}  partition=agent_id/agent-A"
    )

    # Agent-B: topics N_AGENT_DOCS..2*N_AGENT_DOCS (next N rows of corpus)
    writer_b = ailake.TableWriter(
        AGENT_PATH, dim=DIM, metric=METRIC,
        partition_by="agent_id", partition_value="agent-B",
    )
    writer_b.write_batch(
        texts[N_AGENT_DOCS : N_AGENT_DOCS * 2],
        embeddings[N_AGENT_DOCS : N_AGENT_DOCS * 2],
    )
    snap_b = writer_b.commit()
    print(
        f"[Agent-B]  Committed snapshot_id={snap_b}"
        f"  rows={N_AGENT_DOCS}  partition=agent_id/agent-B"
    )


def _write_model_tracked(texts: list[str], embeddings: list[list[float]]) -> None:
    """HNSW table with embedding model metadata — demonstrates model tracking feature."""
    import ailake
    os.makedirs(MODEL_TRACKED_PATH, exist_ok=True)
    table = ailake.open_table(
        MODEL_TRACKED_PATH,
        dim=DIM,
        metric=METRIC,
        embedding_model="synthetic-embed-v1",
        embedding_model_version="1.0",
    )
    table.insert(texts[:100], embeddings[:100])
    snap_id = table.commit()
    print(f"[ModelTracked] Committed snapshot_id={snap_id}  rows=100  model=synthetic-embed-v1@1.0")


def _write_deferred(texts: list[str], embeddings: list[list[float]]) -> None:
    """Deferred write — Parquet immediate, index built in background."""
    from ailake import TableWriter
    os.makedirs(DEFERRED_PATH, exist_ok=True)
    w = TableWriter(DEFERRED_PATH, dim=DIM, metric=METRIC)
    w.write_batch_auto_deferred(texts[:N_DEFERRED], embeddings[:N_DEFERRED])
    snap_id = w.commit()
    print(f"[Deferred] Committed snapshot_id={snap_id}  rows={N_DEFERRED}  (index builds in bg)")


def _write_bm25(texts: list[str], embeddings: list[list[float]]) -> None:
    """BM25-indexed table — demonstrates hybrid vector+lexical search.

    Uses the first 200 rows. BM25 IDF stats are accumulated at write time
    and stored as metadata/ailake_bm25_stats.bin inside the table root.
    """
    from ailake import TableWriter
    os.makedirs(BM25_PATH, exist_ok=True)
    # write_batch stores texts as "text" column — use that as BM25 source column.
    w = TableWriter(BM25_PATH, dim=DIM, metric=METRIC, bm25_text_column="text")
    w.write_batch(texts[:200], embeddings[:200])
    snap_id = w.commit()
    print(f"[BM25]     Committed snapshot_id={snap_id}  rows=200  bm25_col=text")


def _write_fts(texts: list[str], embeddings: list[list[float]]) -> None:
    """Phase T — Tantivy per-file FTS table.

    Embeds an inverted Tantivy index in every Parquet file's AILK_FTS section.
    search_text() uses Tantivy O(log N) fast path when the blob is present;
    legacy files fall back to BM25 O(N) scan automatically.
    """
    from ailake import TableWriter
    os.makedirs(FTS_PATH, exist_ok=True)
    w = TableWriter(
        FTS_PATH,
        dim=DIM,
        metric=METRIC,
        fts_text_columns=["text"],
        fts_tokenizer="default",
    )
    w.write_batch(texts[:200], embeddings[:200])
    snap_id = w.commit()
    print(f"[FTS]      Committed snapshot_id={snap_id}  rows=200  fts_col=text  tokenizer=default")


def _write_partitioned_v3(texts: list[str], embeddings: list[list[float]]) -> None:
    """Iceberg format_version=3 + partition_fields demo (Phase L).

    Partitions by topic_id (int, identity transform) — each of the 20 topics
    gets its own Iceberg partition. Demonstrates geometric pruning at the
    partition level: queries about "machine learning" only scan that partition.
    """
    from ailake import TableWriter
    os.makedirs(PARTITIONED_V3_PATH, exist_ok=True)
    N = 200
    # Assign topic_id 0-19 round-robin across rows
    topic_ids = [i % len(TOPICS) for i in range(N)]
    w = TableWriter(
        PARTITIONED_V3_PATH,
        dim=DIM,
        metric=METRIC,
        partition_fields=[("topic_id", "identity", "int")],
        format_version=3,
    )
    w.write_batch(texts[:N], embeddings[:N], extra_columns={"topic_id": topic_ids})
    snap_id = w.commit()
    print(
        f"[PartV3]   Committed snapshot_id={snap_id}  rows={N}"
        f"  partition=topic_id/identity  format_version=3"
    )


def _write_delete_demo(texts: list[str], embeddings: list[list[float]]) -> None:
    """Delete-where demo table — 100 rows, rows 0-9 pre-deleted (notebook §29)."""
    import ailake
    os.makedirs(DELETE_DEMO_PATH, exist_ok=True)
    w = ailake.TableWriter(DELETE_DEMO_PATH, dim=DIM, metric=METRIC)
    w.write_batch(texts[:100], embeddings[:100],
                  extra_columns={"id": [str(i) for i in range(100)]})
    snap_id = w.commit()
    # Pre-delete rows 0-9 so notebook §29 shows "before/after" scan counts.
    ailake.delete_where(DELETE_DEMO_PATH, "id", [str(i) for i in range(10)])
    print(
        f"[DelDemo]  Committed snapshot_id={snap_id}  rows=100  pre-deleted=10"
    )


def _write_schema_evo(texts: list[str], embeddings: list[list[float]]) -> None:
    """Schema-evolution demo table — 100 rows, source_url column added (notebook §30)."""
    import ailake
    os.makedirs(SCHEMA_EVO_PATH, exist_ok=True)
    w = ailake.TableWriter(SCHEMA_EVO_PATH, dim=DIM, metric=METRIC)
    w.write_batch(texts[:100], embeddings[:100])
    snap_id = w.commit()
    # Add a new optional column — existing files unaffected (field-id stable).
    schema_id = ailake.add_column(
        SCHEMA_EVO_PATH, "source_url", "string",
        required=False, initial_default="",
    )
    print(
        f"[SchemaEvo] Committed snapshot_id={snap_id}  rows=100"
        f"  add_column=source_url  new_schema_id={schema_id}"
    )


def _save_query_payload(embeddings: list[list[float]], texts: list[str]) -> None:
    query_payload = {
        "query_vector":       embeddings[0],
        "expected_top1_text": texts[0],
        "dim":                DIM,
        "metric":             METRIC,
        "table_paths": {
            "hnsw":             TABLE_PATH,
            "pq_only":          PQ_PATH,
            "residual":         RESIDUAL_PATH,
            "deferred":         DEFERRED_PATH,
            "model_tracked":    MODEL_TRACKED_PATH,
            "multimodal":       MULTIMODAL_PATH,
            "agent":            AGENT_PATH,
            "bm25":             BM25_PATH,
            "fts":              FTS_PATH,
            "partitioned_v3":   PARTITIONED_V3_PATH,
            "delete_demo":      DELETE_DEMO_PATH,
            "schema_evo":       SCHEMA_EVO_PATH,
        },
        "multimodal": {
            "text_dim":       DIM,
            "image_dim":      IMAGE_DIM,
            "text_column":    "embedding",
            "image_column":   "image_embedding",
        },
        "agent": {
            "agent_ids":        ["agent-A", "agent-B"],
            "partition_column": "agent_id",
            "n_docs_per_agent": N_AGENT_DOCS,
        },
    }
    query_path = os.path.join(os.path.dirname(TABLE_PATH), "demo_query.json")
    with open(query_path, "w") as fh:
        json.dump(query_payload, fh, indent=2)
    print(f"Demo query vector saved to {query_path}")


def main() -> None:
    try:
        import ailake
    except ImportError as exc:
        print(f"ERROR: ailake module not available: {exc}", file=sys.stderr)
        sys.exit(1)

    print(f"Writing demo tables: n={N_DOCS}  dim={DIM}  metric={METRIC}")
    texts, embeddings = _build_corpus(N_DOCS)

    _write_hnsw(texts, embeddings)
    _write_pq_only(texts, embeddings)
    _write_residual_pq(texts, embeddings)
    _write_deferred(texts, embeddings)
    _write_model_tracked(texts, embeddings)
    _write_multimodal(texts, embeddings)
    _write_agent_memory(texts, embeddings)
    _write_bm25(texts, embeddings)
    _write_fts(texts, embeddings)
    _write_partitioned_v3(texts, embeddings)
    _write_delete_demo(texts, embeddings)
    _write_schema_evo(texts, embeddings)
    _save_query_payload(embeddings, texts)

    _maybe_register_nessie(TABLE_PATH)
    _maybe_register_nessie(PARTITIONED_V3_PATH, nessie_name="partitioned_v3")
    _maybe_register_nessie(DELETE_DEMO_PATH,    nessie_name="delete_demo")
    _maybe_register_nessie(SCHEMA_EVO_PATH,     nessie_name="schema_evo")

    stamp_path = pathlib.Path(TABLE_PATH).parent / ".fixture-version"
    stamp_path.write_text(FIXTURE_VERSION)

    print("All fixtures ready.")


def _maybe_register_nessie(table_path: str, *, nessie_name: str = "table") -> None:
    """Register an AI-Lake table in the Nessie catalog so Trino can discover it.

    Uses the Nessie REST API v1 directly (stdlib urllib — no extra deps).
    No-op when NESSIE_URI is unset.

    Trino 400+ dropped hadoop catalog type; Nessie is the catalog backend for
    the engines profile (--profile engines in compose-demo.yml).
    """
    import urllib.request

    nessie_uri = os.environ.get("NESSIE_URI")
    if not nessie_uri:
        return

    meta_dir  = pathlib.Path(table_path) / "default" / "table" / "metadata"
    hint_file = meta_dir / "version-hint.text"
    if not hint_file.exists():
        print("WARNING: version-hint.text missing, skipping Nessie registration", file=sys.stderr)
        return

    hint         = hint_file.read_text().strip()
    meta_location = f"file://{meta_dir}/v{hint}.metadata.json"

    def _nessie(method: str, path: str, body: dict | None = None) -> dict:
        url  = f"{nessie_uri.rstrip('/')}{path}"
        data = json.dumps(body).encode() if body is not None else None
        req  = urllib.request.Request(
            url, data=data,
            headers={"Content-Type": "application/json", "Accept": "application/json"},
            method=method,
        )
        with urllib.request.urlopen(req, timeout=10) as r:
            return json.loads(r.read())

    try:
        try:
            _nessie("PUT", "/namespaces/namespace/main/default", {
                "type": "NAMESPACE", "elements": ["default"], "properties": {},
            })
        except urllib.error.HTTPError as e:
            if e.code != 409:
                raise

        branch       = _nessie("GET", "/trees/tree/main")
        current_hash = branch["hash"]

        with open(meta_dir / f"v{hint}.metadata.json") as fh:
            meta_json = json.load(fh)
        snapshot_id   = meta_json.get("current-snapshot-id", -1) or -1
        schema_id     = meta_json.get("current-schema-id", 0)
        spec_id       = meta_json.get("default-spec-id", 0)
        sort_order_id = meta_json.get("default-sort-order-id", 0)

        _nessie("POST", f"/trees/branch/main/commit?expectedHash={current_hash}", {
            "commitMeta": {"message": f"register ailake demo table: {nessie_name}"},
            "operations": [{
                "type": "PUT",
                "key": {"elements": ["default", nessie_name]},
                "content": {
                    "type":             "ICEBERG_TABLE",
                    "metadataLocation": meta_location,
                    "snapshotId":       snapshot_id,
                    "schemaId":         schema_id,
                    "specId":           spec_id,
                    "sortOrderId":      sort_order_id,
                },
            }],
        })
        print(f"Table '{nessie_name}' registered in Nessie: {meta_location}")
    except Exception as e:
        print(f"WARNING: Nessie registration failed: {e}", file=sys.stderr)


if __name__ == "__main__":
    if "--nessie-only" in sys.argv:
        _maybe_register_nessie(os.environ.get("DEMO_TABLE_PATH", "/data/ailake_demo"))
    else:
        main()
