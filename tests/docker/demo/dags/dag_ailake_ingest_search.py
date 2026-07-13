"""
AI-Lake ingest + search demo DAG.

Pipeline:
  write_docs → vector_search → fts_search → assemble_context

Each task uses `import ailake` directly (Python SDK) via Airflow's PythonOperator
(TaskFlow API). This avoids requiring the ailake CLI binary inside the Airflow
container — useful for demo and for pure-Python pipeline environments.

For production use with the ailake CLI binary available, see:
  airflow-providers-ailake/airflow_providers_ailake/operators/ailake.py
  (AilakeWriteOperator, AilakeSearchOperator, AilakeFtsSearchOperator)

Schedule: @daily (runs once per day; catchup=False so only the latest run matters)
"""

from __future__ import annotations

from datetime import datetime

from airflow.decorators import dag, task


@dag(
    dag_id="ailake_ingest_search",
    schedule="@daily",
    start_date=datetime(2025, 1, 1),
    catchup=False,
    tags=["ailake", "demo"],
    doc_md=__doc__,
)
def ailake_ingest_search():

    @task
    def write_docs() -> str:
        """Write 50 synthetic documents with Tantivy FTS enabled."""
        import os
        import numpy as np
        import ailake

        path = os.environ.get("AILAKE_RAG_TABLE_PATH", "/data/ailake_rag_airflow")
        dim = int(os.environ.get("DEMO_DIM", "32"))
        rng = np.random.default_rng(7)

        topics = [
            "transformer architecture", "retrieval-augmented generation",
            "vector databases", "semantic search", "LLM fine-tuning",
            "approximate nearest neighbor", "columnar storage", "data versioning",
            "stream processing", "knowledge graphs",
        ]
        texts = [
            f"An introduction to {topics[i % len(topics)]}: "
            f"concepts and applications in modern AI data systems. "
            f"Document index: {i}."
            for i in range(50)
        ]
        embs = rng.standard_normal((50, dim)).astype("float32")
        embs /= np.linalg.norm(embs, axis=1, keepdims=True)

        table = ailake.open_table(
            path,
            dim=dim,
            metric="cosine",
            fts_text_columns=["text"],
        )
        table.insert(texts, embs)
        snap_id = table.commit()
        print(f"write_docs: {len(texts)} rows → snapshot {snap_id}")
        return str(snap_id)

    @task
    def vector_search(snap_id: str) -> list:
        """Top-5 nearest neighbours for a random query vector."""
        import os
        import numpy as np
        import ailake

        path = os.environ.get("AILAKE_RAG_TABLE_PATH", "/data/ailake_rag_airflow")
        dim = int(os.environ.get("DEMO_DIM", "32"))

        rng = np.random.default_rng(99)
        query = rng.standard_normal(dim).astype("float32")
        query /= np.linalg.norm(query)

        results = ailake.search(path, query.tolist(), top_k=5).to_list()
        for r in results:
            print(f"  row_id={r['row_id']:3d}  distance={r['distance']:.4f}")
        return results

    @task
    def fts_search(snap_id: str) -> list:
        """BM25 / Tantivy keyword search for 'transformer retrieval'."""
        import os
        import ailake

        path = os.environ.get("AILAKE_RAG_TABLE_PATH", "/data/ailake_rag_airflow")
        hits = ailake.search_text(path, "transformer retrieval", top_k=5)
        for h in hits:
            print(f"  row_id={h.get('row_id')}  score={h.get('score', 0):.4f}")
        return hits

    @task
    def assemble_context(snap_id: str, vector_results: list, fts_results: list) -> str:
        """Assemble LLM context XML from combined vector + FTS results."""
        import ailake

        # Build chunk list from vector results (fetch_data=False → no chunk_text available;
        # use synthesised text that mirrors what write_docs wrote).
        topics = [
            "transformer architecture", "retrieval-augmented generation",
            "vector databases", "semantic search", "LLM fine-tuning",
        ]
        chunks = [
            {
                "document_id": str(r["row_id"]),
                "chunk_index": 0,
                "chunk_text": f"Introduction to {topics[r['row_id'] % len(topics)]}.",
                "document_title": f"Doc {r['row_id']}",
                "distance": float(r["distance"]),
            }
            for r in vector_results[:5]
        ]

        # assemble_context() returns {"text": str, "chunk_count": int, "token_estimate": int}
        # (Fase 15) — not a plain string.
        result = ailake.assemble_context(chunks=chunks, max_tokens=512, dedup_threshold=0.1)
        context = result["text"]
        print(f"Context assembled: {len(context)} chars, chunk_count={result['chunk_count']}, "
              f"token_estimate={result['token_estimate']}")
        print(context[:400])
        return context[:400]

    snap = write_docs()
    vec = vector_search(snap)
    fts = fts_search(snap)
    assemble_context(snap, vec, fts)


ailake_ingest_search()
