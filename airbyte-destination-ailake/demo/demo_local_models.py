#!/usr/bin/env python3
"""
airbyte-destination-ailake local-model demo — real embeddings, no API key.

Unlike demo_local.py (which uses CmdEmbedder + embed_cmd.py's deterministic
random unit vectors — a stub, not a real model), this script runs two real,
free, locally-executing embedding models back to back:

  - fastembed             (ONNX Runtime, no PyTorch)
  - sentence-transformers (PyTorch, widest model selection)

Both default to BAAI/bge-small-en-v1.5 (dim=384) so results are comparable.
Neither requires an API key or external service — only a one-time model
download on first run (cached locally afterward).

Run:
    pip install -e "path/to/airbyte-destination-ailake[fastembed]"
    pip install -e "path/to/airbyte-destination-ailake[sentence-transformers]"
    cd airbyte-destination-ailake/demo
    python demo_local_models.py

Either extra may be installed alone — this script skips whichever backend's
package isn't available instead of failing outright.

Requires: ailake (Rust extension), numpy
"""

import pathlib
import sys
import tempfile

# --- make sure the package is importable from the repo root ---
REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "airbyte-destination-ailake"))

from airbyte_destination_ailake.config import AilakeDestinationConfig
from airbyte_destination_ailake.embedder import build_embedder
from airbyte_destination_ailake.writer import StreamWriter

DIM = 384  # BAAI/bge-small-en-v1.5 output dimension

RECORDS = [
    {"id": i, "content": text, "category": cat}
    for i, (text, cat) in enumerate(
        [
            ("Apache Iceberg provides ACID transactions for data lakes.", "databases"),
            ("HNSW graphs enable sub-linear approximate nearest neighbour search.", "algorithms"),
            ("Parquet columnar storage reduces I/O for analytical queries.", "storage"),
            ("Retrieval-augmented generation grounds LLMs in external knowledge.", "llm"),
            ("Product quantization compresses high-dimensional vectors by 32-256x.", "compression"),
            ("Vector databases power semantic search and recommendation systems.", "databases"),
            ("Rust zero-cost abstractions enable safe systems programming.", "languages"),
            ("MinIO provides S3-compatible object storage for on-premise clusters.", "infrastructure"),
            ("Embedding models convert text to dense vectors for semantic matching.", "ml"),
            ("DuckDB runs analytical SQL directly on Parquet files without a server.", "databases"),
            ("Apache Arrow in-memory columnar format enables zero-copy IPC.", "storage"),
            ("Cosine similarity measures the angle between two vectors, ignoring magnitude.", "algorithms"),
        ]
    )
]

QUERY_TEXT = "vector similarity for semantic retrieval"


def run_one(embed_mode: str, table_base: str) -> None:
    cfg_dict = {
        "table_base_path": table_base,
        "embed_mode": embed_mode,
        "embedding_dim": DIM,
        "embedding_metric": "cosine",
        "embedding_model": f"{embed_mode}-BAAI/bge-small-en-v1.5",
        "text_field": "content",
        "batch_size": 5,
    }
    cfg = AilakeDestinationConfig.from_dict(cfg_dict)

    errors = cfg.validate()
    if errors:
        print(f"[{embed_mode}] config errors: {errors}")
        return

    try:
        embedder = build_embedder(cfg)
    except ImportError as e:
        print(f"[{embed_mode}] SKIP — {e}")
        return

    print(f"=== {embed_mode} ===")
    stream_name = f"tech_docs_{embed_mode}"
    writer = StreamWriter(stream_name, cfg, embedder)

    print(f"Writing {len(RECORDS)} records via real {embed_mode} embeddings...")
    for rec in RECORDS:
        writer.add(rec)

    snap_id = writer.commit()
    table_path = cfg.table_path(stream_name)
    print(f"Committed snapshot_id={snap_id}  path={table_path}")

    try:
        import ailake

        query_vec = embedder.embed([QUERY_TEXT])[0].tolist()
        print(f'Searching: "{QUERY_TEXT}"')
        results = ailake.search(table_path, query_vec, top_k=3, fetch_data=True)
        df = results.to_pandas()
        print(df[["text", "_distance"]].to_string(index=False))
    except ImportError:
        print("ailake not installed — skipping search")

    print()


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="ailake_airbyte_local_models_demo_") as tmp:
        print("=== airbyte-destination-ailake local-model demo ===")
        print("Two real, free, in-process embedding backends — no API key.\n")

        run_one("fastembed", tmp)
        run_one("sentence_transformers", tmp)

        print("Done.")


if __name__ == "__main__":
    main()
