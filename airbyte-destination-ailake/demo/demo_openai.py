#!/usr/bin/env python3
"""
airbyte-destination-ailake OpenAI demo — real embeddings via OpenAI API.

Run:
    pip install -e "path/to/airbyte-destination-ailake[openai]"
    export OPENAI_API_KEY=sk-...
    cd airbyte-destination-ailake/demo
    python demo_openai.py [--table-path /tmp/ailake_airbyte_openai]

Requires: ailake (Rust extension), openai, numpy
"""

import argparse
import json
import os
import pathlib
import sys
import tempfile

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "airbyte-destination-ailake"))

from airbyte_destination_ailake.config import AilakeDestinationConfig
from airbyte_destination_ailake.embedder import OpenAIEmbedder
from airbyte_destination_ailake.writer import StreamWriter

RECORDS = [
    {"id": i, "content": text, "source": "demo"}
    for i, text in enumerate(
        [
            "Apache Iceberg provides ACID transactions and time-travel for data lakes.",
            "HNSW graphs enable approximate nearest-neighbour search in sub-linear time.",
            "Parquet columnar format reduces I/O by scanning only needed columns.",
            "Retrieval-augmented generation (RAG) grounds LLMs with external knowledge.",
            "Product quantization compresses high-dimensional vectors by orders of magnitude.",
            "Vector databases enable semantic search and content-based recommendations.",
            "Rust zero-cost abstractions allow safe and fast systems programming.",
            "MinIO provides open-source S3-compatible object storage.",
            "Embedding models convert text into dense vectors capturing semantic meaning.",
            "DuckDB processes analytical SQL on Parquet files with no infrastructure.",
            "Apache Arrow enables zero-copy columnar data exchange across systems.",
            "Cosine similarity finds vectors that point in the same semantic direction.",
        ]
    )
]

QUERIES = [
    "vector similarity and semantic search",
    "ACID transactions for analytical workloads",
    "LLM grounding with retrieved documents",
]


def main() -> None:
    parser = argparse.ArgumentParser(description="AI-Lake Airbyte destination OpenAI demo")
    parser.add_argument("--table-path", default="", help="Base path for tables (default: temp dir)")
    parser.add_argument("--model", default="text-embedding-3-small")
    parser.add_argument("--dim", type=int, default=1536)
    args = parser.parse_args()

    api_key = os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        print("ERROR: OPENAI_API_KEY not set", file=sys.stderr)
        sys.exit(1)

    _run(api_key, args.model, args.dim, args.table_path)


def _run(api_key: str, model: str, dim: int, table_path_override: str) -> None:
    ctx = tempfile.TemporaryDirectory(prefix="ailake_airbyte_openai_") if not table_path_override else None
    table_base = table_path_override if table_path_override else ctx.name

    try:
        cfg = AilakeDestinationConfig.from_dict(
            {
                "table_base_path": table_base,
                "embed_mode": "openai",
                "openai_api_key": api_key,
                "openai_model": model,
                "embedding_dim": dim,
                "embedding_metric": "cosine",
                "embedding_model": model,
                "embedding_model_version": "1",
                "text_field": "content",
                "batch_size": 12,
                "pre_normalize": True,
            }
        )

        print("=== airbyte-destination-ailake OpenAI demo ===")
        print(f"model           : {model}  dim={dim}")
        print(f"table_base_path : {table_base}")
        print()

        embedder = OpenAIEmbedder(api_key=api_key, model=model)
        writer = StreamWriter("tech_docs", cfg, embedder)

        print(f"Writing {len(RECORDS)} records (1 embed API call, batch=12)...")
        for rec in RECORDS:
            writer.add(rec)
        snap_id = writer.commit()
        table_path = cfg.table_path("tech_docs")
        print(f"Committed snapshot_id={snap_id}")
        print()

        try:
            import ailake

            for query_text in QUERIES:
                import numpy as np
                q_vec = embedder.embed([query_text])[0].tolist()
                # normalize to match pre_normalize=True tables
                v = np.array(q_vec, dtype=np.float32)
                v = (v / np.linalg.norm(v)).tolist()

                results = ailake.search(table_path, v, top_k=3, fetch_data=True).to_pandas()
                print(f'Query: "{query_text}"')
                for _, row in results.iterrows():
                    print(f'  [{row["_distance"]:.4f}] {row["text"][:80]}')
                print()
        except ImportError:
            print("ailake not installed — skipping search")

        # Iceberg metadata
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

    finally:
        if ctx:
            ctx.cleanup()


if __name__ == "__main__":
    main()
