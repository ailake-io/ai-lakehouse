"""
AI-Lake compaction DAG.

Pipeline:
  compact_table → table_info

Runs weekly. Merges small Parquet files in the RAG table produced by
dag_ailake_ingest_search and rebuilds the HNSW index for the merged files.

Each task uses `import ailake` directly (Python SDK) — no CLI binary required.
"""

from __future__ import annotations

from datetime import datetime

from airflow.decorators import dag, task


@dag(
    dag_id="ailake_compaction",
    schedule="@weekly",
    start_date=datetime(2025, 1, 1),
    catchup=False,
    tags=["ailake", "maintenance"],
    doc_md=__doc__,
)
def ailake_compaction():

    @task
    def compact_table() -> dict:
        """Compact small files in the RAG table. No-op when no files qualify."""
        import os
        import ailake

        path = os.environ.get("AILAKE_RAG_TABLE_PATH", "/data/ailake_rag_airflow")
        result = ailake.compact(path)
        files_compacted = result.get("files_compacted", 0)
        print(f"compact: {files_compacted} file(s) compacted at {path}")
        return result

    @task
    def table_info(compact_result: dict) -> None:
        """Log table metadata after compaction."""
        import os
        import json
        import pathlib

        path = os.environ.get("AILAKE_RAG_TABLE_PATH", "/data/ailake_rag_airflow")
        meta_dir = pathlib.Path(path) / "default" / "table" / "metadata"
        hint_file = meta_dir / "version-hint.text"

        if not hint_file.exists():
            print(f"table_info: no metadata at {meta_dir} — table not written yet")
            return

        hint = hint_file.read_text().strip()
        meta = json.loads((meta_dir / f"v{hint}.metadata.json").read_text())
        props = meta.get("properties", {})

        print(f"Table UUID       : {meta.get('table-uuid')}")
        print(f"Format version   : {meta.get('format-version')}")
        print(f"Snapshots        : {len(meta.get('snapshots', []))}")
        print(f"Current snapshot : {meta.get('current-snapshot-id')}")
        print(f"files_compacted  : {compact_result.get('files_compacted', 0)}")
        print()
        print("AI-Lake properties:")
        for k, v in sorted(props.items()):
            if k.startswith("ailake."):
                print(f"  {k:45s} = {v}")

    info = compact_table()
    table_info(info)


ailake_compaction()
