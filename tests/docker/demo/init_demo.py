"""
AI-Lake demo fixture generator.

Writes a local AI-Lake table to DEMO_TABLE_PATH (default /data/ailake_demo).
Runs once at container startup via entrypoint.sh; skipped on restart if
version-hint.text already exists.

Uses the fluent open_table() / Table API introduced in v0.0.14.
"""

import json
import math
import os
import pathlib
import random
import sys

TABLE_PATH = os.environ.get("DEMO_TABLE_PATH", "/data/ailake_demo")
DIM = int(os.environ.get("DEMO_DIM", "32"))
N_DOCS = 500
METRIC = "cosine"

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


def main() -> None:
    try:
        import ailake
    except ImportError as exc:
        print(f"ERROR: ailake module not available: {exc}", file=sys.stderr)
        sys.exit(1)

    print(f"Writing demo table: path={TABLE_PATH}  n={N_DOCS}  dim={DIM}  metric={METRIC}")
    os.makedirs(TABLE_PATH, exist_ok=True)

    texts: list[str] = []
    embeddings: list[list[float]] = []

    for i in range(N_DOCS):
        topic = TOPICS[i % len(TOPICS)]
        template = TEMPLATES[(i // len(TOPICS)) % len(TEMPLATES)]
        texts.append(template.format(topic=topic) + f" (doc_id={i})")
        embeddings.append(rand_unit_vec(DIM, seed=i))

    # Fluent API: open_table() + insert() + commit()
    table = ailake.open_table(TABLE_PATH, dim=DIM, metric=METRIC)
    table.insert(texts, embeddings)
    snap_id = table.commit()
    print(f"Committed snapshot_id={snap_id}  rows={N_DOCS}")

    # Persist the first document's embedding as a demo query vector so notebooks
    # don't need to re-derive it.
    query_payload = {
        "query_vector": embeddings[0],
        "expected_top1_text": texts[0],
        "dim": DIM,
        "metric": METRIC,
    }
    query_path = os.path.join(os.path.dirname(TABLE_PATH), "demo_query.json")
    with open(query_path, "w") as fh:
        json.dump(query_payload, fh, indent=2)
    print(f"Demo query vector saved to {query_path}")

    _maybe_register_nessie(TABLE_PATH)
    print("Fixture ready.")


def _maybe_register_nessie(table_path: str) -> None:
    """Register the AI-Lake table in the Nessie catalog so Trino can discover it.

    Uses the Nessie REST API v1 directly (stdlib urllib — no extra deps).
    No-op when NESSIE_URI is unset.

    Trino 400+ dropped hadoop catalog type; Nessie is the catalog backend for
    the engines profile (--profile engines in compose-demo.yml).
    """
    import urllib.request

    nessie_uri = os.environ.get("NESSIE_URI")
    if not nessie_uri:
        return

    meta_dir = pathlib.Path(table_path) / "default" / "table" / "metadata"
    hint_file = meta_dir / "version-hint.text"
    if not hint_file.exists():
        print("WARNING: version-hint.text missing, skipping Nessie registration", file=sys.stderr)
        return

    hint = hint_file.read_text().strip()
    meta_location = f"file://{meta_dir}/v{hint}.metadata.json"

    def _nessie(method: str, path: str, body: dict | None = None) -> dict:
        url = f"{nessie_uri.rstrip('/')}{path}"
        data = json.dumps(body).encode() if body is not None else None
        req = urllib.request.Request(
            url, data=data,
            headers={"Content-Type": "application/json", "Accept": "application/json"},
            method=method,
        )
        with urllib.request.urlopen(req, timeout=10) as r:
            return json.loads(r.read())

    try:
        # Create namespace (idempotent — 409 if exists)
        try:
            _nessie("PUT", "/namespaces/namespace/main/default", {
                "type": "NAMESPACE", "elements": ["default"], "properties": {},
            })
        except urllib.error.HTTPError as e:
            if e.code != 409:
                raise

        # Get current branch hash (required by Nessie optimistic locking)
        branch = _nessie("GET", "/trees/tree/main")
        current_hash = branch["hash"]

        # Read actual IDs from metadata — Trino validates these against the file.
        with open(meta_dir / f"v{hint}.metadata.json") as fh:
            meta_json = json.load(fh)
        snapshot_id = meta_json.get("current-snapshot-id", -1) or -1
        schema_id = meta_json.get("current-schema-id", 0)
        spec_id = meta_json.get("default-spec-id", 0)
        sort_order_id = meta_json.get("default-sort-order-id", 0)

        # Register the Iceberg table pointer (idempotent via commit)
        _nessie("POST", f"/trees/branch/main/commit?expectedHash={current_hash}", {
            "commitMeta": {"message": "register ailake demo table"},
            "operations": [{
                "type": "PUT",
                "key": {"elements": ["default", "table"]},
                "content": {
                    "type": "ICEBERG_TABLE",
                    "metadataLocation": meta_location,
                    "snapshotId": snapshot_id,
                    "schemaId": schema_id,
                    "specId": spec_id,
                    "sortOrderId": sort_order_id,
                },
            }],
        })
        print(f"Table registered in Nessie: {meta_location}")
    except Exception as e:
        print(f"WARNING: Nessie registration failed: {e}", file=sys.stderr)


if __name__ == "__main__":
    if "--nessie-only" in sys.argv:
        _maybe_register_nessie(os.environ.get("DEMO_TABLE_PATH", "/data/ailake_demo"))
    else:
        main()
