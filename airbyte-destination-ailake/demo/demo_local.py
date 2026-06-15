#!/usr/bin/env python3
"""
airbyte-destination-ailake local demo — no API keys, no Docker, no Airbyte platform.

Uses the connector classes directly with a CmdEmbedder backed by embed_cmd.py
(deterministic random unit vectors seeded from text hash).

Run:
    pip install -e "path/to/airbyte-destination-ailake"
    cd airbyte-destination-ailake/demo
    python demo_local.py

Requires: ailake (Rust extension), numpy
"""

import json
import pathlib
import sys
import tempfile

import numpy as np

# --- make sure the package is importable from the repo root ---
REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "airbyte-destination-ailake"))

from airbyte_destination_ailake.config import AilakeDestinationConfig
from airbyte_destination_ailake.embedder import CmdEmbedder
from airbyte_destination_ailake.writer import StreamWriter

EMBED_CMD = f"{sys.executable} {pathlib.Path(__file__).parent / 'embed_cmd.py'} 32"

RECORDS = [
    {"id": i, "content": text, "category": cat}
    for i, (text, cat) in enumerate(
        [
            ("Apache Iceberg provides ACID transactions for data lakes.", "databases"),
            ("HNSW graphs enable sub-linear approximate nearest neighbour search.", "algorithms"),
            ("Parquet columnar storage reduces I/O for analytical queries.", "storage"),
            ("Retrieval-augmented generation grounds LLMs in external knowledge.", "llm"),
            ("Product quantization compresses high-dimensional vectors by 32–256×.", "compression"),
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


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="ailake_airbyte_demo_") as tmp:
        table_base = tmp

        cfg = AilakeDestinationConfig.from_dict(
            {
                "table_base_path": table_base,
                "embed_mode": "cmd",
                "embed_cmd": EMBED_CMD,
                "embedding_dim": 32,
                "embedding_metric": "cosine",
                "embedding_model": "demo-embed-v1",
                "embedding_model_version": "1.0",
                "text_field": "content",
                "batch_size": 5,
            }
        )

        errors = cfg.validate()
        if errors:
            print("Config errors:", errors)
            sys.exit(1)

        print("=== airbyte-destination-ailake local demo ===")
        print(f"table_base_path : {cfg.table_base_path}")
        print(f"embed_mode      : {cfg.embed_mode}")
        print(f"embedding_model : {cfg.embedding_model}@{cfg.embedding_model_version}")
        print(f"batch_size      : {cfg.batch_size}")
        print()

        embedder = CmdEmbedder(cfg.embed_cmd)

        # --- simulate what Destination.write() does ---
        writer = StreamWriter("tech_docs", cfg, embedder)

        print(f"Writing {len(RECORDS)} records to stream 'tech_docs'...")
        for rec in RECORDS:
            writer.add(rec)

        snap_id = writer.commit()
        table_path = cfg.table_path("tech_docs")
        print(f"Committed snapshot_id={snap_id}  path={table_path}")
        print()

        # --- search ---
        try:
            import ailake

            query_text = "vector similarity for semantic retrieval"
            query_vec = embedder.embed([query_text])[0].tolist()

            print(f'Searching: "{query_text}"')
            results = ailake.search(table_path, query_vec, top_k=5, fetch_data=True)
            df = results.to_pandas()
            print(df[["text", "_distance"]].to_string(index=False))
            print()
        except ImportError:
            print("ailake not installed — skipping search (install the Rust extension to search)")
            print()

        # --- verify Iceberg metadata ---
        meta_dir = pathlib.Path(table_path) / "default" / "table" / "metadata"
        hint_file = meta_dir / "version-hint.text"
        if hint_file.exists():
            hint = hint_file.read_text().strip()
            meta = json.loads((meta_dir / f"v{hint}.metadata.json").read_text())
            props = meta.get("properties", {})
            print("Iceberg metadata — AI-Lake properties:")
            for k in sorted(props):
                if k.startswith("ailake."):
                    print(f"  {k:40s} = {props[k]}")
        else:
            print("(Iceberg metadata not found — ailake extension required for write)")

        print()
        print("Done.")


if __name__ == "__main__":
    main()
