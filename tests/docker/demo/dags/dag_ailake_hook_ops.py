"""
AI-Lake CLI-based hook operations demo DAG.

Exercises the `AilakeHook` methods that wrap the `ailake` CLI binary directly —
distinct from `dag_ailake_ingest_search.py` / `dag_ailake_compaction.py`, which
use `import ailake` (Python SDK) and need no CLI binary on PATH. This DAG is
the pattern for production use of `airflow-providers-ailake`'s CLI-based
operators/hooks (see 12_airflow.ipynb §10).

Two independent table + connection pairs — originally kept separate because
chaining migrate/backfill_vector_column onto a table that decay_memories/
delete_rows also touches hit a real bug in `decay-memories` (found while
building this DAG): `migrate_embeddings()` never updated the table's
`ailake.vector-column` property on cutover, so every property-driven reader
(this hook's `decay_memories()` included) kept resolving to the pre-migration
column name, which the migrated file physically no longer has —
"vector dimension mismatch: expected N, got 0". Fixed at the root in
`ailake-query/src/migration.rs` (both migration strategies now update the
property on cutover — see CHANGELOG); the two-table split here is no longer
required for correctness, kept as-is since it's still a clean separation of
the delete/decay vs. schema-evolution demo concerns.

Pipeline:
  run_estimate()                                              (no table needed)
  setup_delete_decay_table() -> delete_some_rows() -> decay()
  setup_evolve_table() -> add_vector_column() -> backfill_vector_column() -> migrate_primary()

Requires two Airflow Connections (conn_type="ailake"), created by
12_airflow.ipynb via the REST API before triggering this DAG:
  ailake_hooks_delete  -> host = AILAKE_HOOKS_DELETE_PATH
  ailake_hooks_evolve  -> host = AILAKE_HOOKS_EVOLVE_PATH

Schedule: None (triggered manually from the notebook only).
"""

from __future__ import annotations

from datetime import datetime

from airflow.decorators import dag, task

EMBED_CMD = "python3 /opt/ailake/embed_cmd.py 32"


@dag(
    dag_id="ailake_hook_ops",
    schedule=None,
    start_date=datetime(2025, 1, 1),
    catchup=False,
    tags=["ailake", "demo", "hooks"],
    doc_md=__doc__,
)
def ailake_hook_ops():

    @task
    def run_estimate() -> dict:
        """Pure-math storage sizing — no I/O, no table required."""
        from airflow_providers_ailake.hooks.ailake import AilakeHook

        hook = AilakeHook(ailake_conn_id="ailake_hooks_evolve")
        result = hook.estimate("1M", 32, hnsw_m=16)
        print(f"estimate(1M rows, dim=32): {len(result.get('estimates', []))} modes returned")
        for e in result.get("estimates", []):
            print(f"  {e['mode']:<24} total_bytes={e['total_bytes']:>12,}  recall={e['recall_at_10']}")
        return result

    # ── delete-rows + decay-memories (format_version=3, last_accessed_at) ──────

    @task
    def setup_delete_decay_table() -> str:
        import os
        import ailake
        import numpy as np

        path = os.environ.get("AILAKE_HOOKS_DELETE_PATH", "/data/ailake_hooks_delete_demo")
        dim = int(os.environ.get("DEMO_DIM", "32"))
        rng = np.random.default_rng(3)

        texts = [f"Hook-ops delete/decay demo doc {i}." for i in range(20)]
        embs = rng.standard_normal((20, dim)).astype(np.float32)
        embs /= np.linalg.norm(embs, axis=1, keepdims=True)

        w = ailake.TableWriter(path, dim=dim, metric="cosine", format_version=3)
        w.write_batch(
            texts, embs.tolist(),
            extra_columns={"last_accessed_at": [ailake.TimestampNs(ailake.now_ns())] * 20},
        )
        snap_id = w.commit()
        print(f"setup_delete_decay_table: snapshot_id={snap_id}  rows=20  path={path}")
        return path

    @task
    def delete_some_rows(path: str) -> str:
        """Mark rows 0-2 deleted via Iceberg Deletion Vectors (V3-only)."""
        import pathlib
        from airflow_providers_ailake.hooks.ailake import AilakeHook

        data_file = next(pathlib.Path(path, "data").glob("*.parquet"))
        rel_file = f"data/{data_file.name}"

        hook = AilakeHook(ailake_conn_id="ailake_hooks_delete")
        hook.delete_rows("default.table", rel_file, [0, 1, 2])
        print(f"delete_rows: marked rows [0,1,2] deleted in {rel_file}")
        return path

    @task
    def decay(path: str) -> int:
        """Recompute recency_weight = exp(-lambda * days_since_access) across all files."""
        from airflow_providers_ailake.hooks.ailake import AilakeHook

        hook = AilakeHook(ailake_conn_id="ailake_hooks_delete")
        files_updated = hook.decay_memories("default.table", decay_lambda=0.15)
        print(f"decay_memories: files_updated={files_updated}")
        return files_updated

    # ── add_vector_column + backfill_vector_column + migrate ───────────────────

    @task
    def setup_evolve_table() -> str:
        import os
        import ailake
        import numpy as np

        path = os.environ.get("AILAKE_HOOKS_EVOLVE_PATH", "/data/ailake_hooks_evolve_demo")
        dim = int(os.environ.get("DEMO_DIM", "32"))
        rng = np.random.default_rng(5)

        texts = [f"Hook-ops evolve demo doc {i}." for i in range(20)]
        embs = rng.standard_normal((20, dim)).astype(np.float32)
        embs /= np.linalg.norm(embs, axis=1, keepdims=True)

        w = ailake.TableWriter(path, dim=dim, metric="cosine")
        w.write_batch(texts, embs.tolist())
        snap_id = w.commit()
        print(f"setup_evolve_table: snapshot_id={snap_id}  rows=20  path={path}")
        return path

    @task
    def add_vector_column(path: str) -> int:
        """Declare a second vector column — metadata-only, no files rewritten."""
        from airflow_providers_ailake.hooks.ailake import AilakeHook

        hook = AilakeHook(ailake_conn_id="ailake_hooks_evolve")
        schema_id = hook.add_vector_column(
            "default.table", "embedding_v2", 32, metric="cosine", precision="f32",
        )
        print(f"add_vector_column(embedding_v2): new_schema_id={schema_id}")
        return schema_id

    @task
    def backfill_vector_column(schema_id: int) -> None:
        """Rewrite every file, computing embedding_v2 via embed_cmd over text_column='text'."""
        from airflow_providers_ailake.hooks.ailake import AilakeHook

        hook = AilakeHook(ailake_conn_id="ailake_hooks_evolve")
        hook.backfill_vector_column(
            "default.table", "embedding_v2",
            embed_cmd=EMBED_CMD, text_column="text", batch_size=512,
        )
        print(f"backfill_vector_column(embedding_v2): done (schema_id was {schema_id})")

    @task
    def migrate_primary() -> None:
        """Re-embed the primary column into embedding_v3 via embed_cmd, dual-write-then-cutover."""
        from airflow_providers_ailake.hooks.ailake import AilakeHook

        hook = AilakeHook(ailake_conn_id="ailake_hooks_evolve")
        hook.migrate(
            "default.table",
            embed_cmd=EMBED_CMD,
            old_column="embedding",
            new_column="embedding_v3",
            text_column="text",
            strategy="dual-write-then-cutover",
            batch_size=512,
        )
        print("migrate(embedding -> embedding_v3): done")

    run_estimate()

    delete_decay_path = setup_delete_decay_table()
    after_delete = delete_some_rows(delete_decay_path)
    decay(after_delete)

    evolve_path = setup_evolve_table()
    new_schema_id = add_vector_column(evolve_path)
    backfill_vector_column(new_schema_id) >> migrate_primary()


ailake_hook_ops()
