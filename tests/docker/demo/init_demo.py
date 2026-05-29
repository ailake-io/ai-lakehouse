"""
AI-Lake demo fixture generator.

Writes a local AI-Lake table to DEMO_TABLE_PATH (default /data/ailake_demo).
Runs once at container startup via entrypoint.sh; skipped on restart if
version-hint.text already exists.

No numpy required — uses stdlib math only.
"""

import json
import math
import os
import random
import sys

TABLE_PATH = os.environ.get("DEMO_TABLE_PATH", "/data/ailake_demo")
DIM = int(os.environ.get("DEMO_DIM", "16"))
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

    writer = ailake.TableWriter(TABLE_PATH, vector_column="embedding", dim=DIM, metric=METRIC)
    writer.write_batch(texts, embeddings)
    snap_id = writer.commit()
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
    print("Fixture ready.")


if __name__ == "__main__":
    main()
