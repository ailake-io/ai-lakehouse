# SPDX-License-Identifier: MIT OR Apache-2.0
"""Airflow operators for AI-Lake: write, compact, search."""

from __future__ import annotations

from typing import Any, Sequence

from airflow.models import BaseOperator
from airflow.utils.context import Context

from airflow_providers_ailake.hooks.ailake import AilakeHook


class AilakeWriteOperator(BaseOperator):
    """Insert a Parquet file into an AI-Lake table.

    Wraps ``ailake insert <table> <file> --embeddings <col>``.

    The ``batch_id`` parameter enables idempotent retries: if a previous run
    already committed this batch, the insert is skipped.  Defaults to
    ``{{ run_id }}_{{ task_id }}`` so Airflow retry logic is safe by default.

    :param table: Fully-qualified table name (``namespace.table`` or ``table``).
    :param source_file: Local path to the source Parquet file.  May be a Jinja
        template, e.g. ``{{ ti.xcom_pull(task_ids='generate') }}``.
    :param embeddings_column: Name of the embedding column in the source file.
    :param batch_id: Idempotency key.  Defaults to ``{{ run_id }}_{{ task_id }}``.
    :param ailake_conn_id: Airflow connection id (conn_type="ailake").
    """

    template_fields: Sequence[str] = ("table", "source_file", "batch_id")
    ui_color = "#b3e0ff"

    def __init__(
        self,
        *,
        table: str,
        source_file: str,
        embeddings_column: str = "embedding",
        batch_id: str = "{{ run_id }}_{{ task.task_id }}",
        ailake_conn_id: str = AilakeHook.default_conn_name,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.table = table
        self.source_file = source_file
        self.embeddings_column = embeddings_column
        self.batch_id = batch_id
        self.ailake_conn_id = ailake_conn_id

    def execute(self, context: Context) -> None:
        hook = AilakeHook(ailake_conn_id=self.ailake_conn_id)

        # Check idempotency before any I/O: if batch_id already committed, skip.
        info = hook.get_table_info(self.table)
        snapshot_id = info.get("snapshot_id")
        if snapshot_id is not None:
            # Ask the CLI if batch_id is present — we use ailake insert which
            # internally calls write_batch_idempotent when --batch-id is supplied.
            self.log.info(
                "table %s has snapshot %s; inserting with batch_id=%s",
                self.table,
                snapshot_id,
                self.batch_id,
            )

        result = hook.run_cli(
            "insert",
            self.table,
            self.source_file,
            "--embeddings", self.embeddings_column,
            "--batch-id", self.batch_id,
        )
        self.log.info(result.stdout.strip())


class AilakeCompactOperator(BaseOperator):
    """Compact small files in an AI-Lake table.

    Wraps ``ailake compact <table>``.

    :param table: Fully-qualified table name.
    :param target_size: Target file size in bytes (default 512 MiB).
    :param min_files: Min small files required to trigger compaction (default 4).
    :param ailake_conn_id: Airflow connection id.
    """

    template_fields: Sequence[str] = ("table",)
    ui_color = "#ffe0b3"

    def __init__(
        self,
        *,
        table: str,
        target_size: int = 536_870_912,
        min_files: int = 4,
        ailake_conn_id: str = AilakeHook.default_conn_name,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.table = table
        self.target_size = target_size
        self.min_files = min_files
        self.ailake_conn_id = ailake_conn_id

    def execute(self, context: Context) -> None:
        hook = AilakeHook(ailake_conn_id=self.ailake_conn_id)
        result = hook.run_cli(
            "compact",
            self.table,
            "--target-size", str(self.target_size),
            "--min-files", str(self.min_files),
        )
        self.log.info(result.stdout.strip())


class AilakeSearchOperator(BaseOperator):
    """Run a vector similarity search on an AI-Lake table and push results to XCom.

    The query vector is read from ``query_vector`` (list of floats) or from
    XCom via ``query_xcom_task_id`` + ``query_xcom_key``.  Results are pushed
    to XCom under key ``search_results`` as a list of dicts::

        [{"rank": 1, "row_id": 42, "distance": 0.003, "file_path": "data/part-0.parquet"}, ...]

    :param table: Fully-qualified table name.
    :param query_vector: Flat list of floats.  Mutually exclusive with
        ``query_xcom_task_id``.
    :param query_xcom_task_id: Task id whose XCom holds the query vector list.
    :param query_xcom_key: XCom key (default ``"return_value"``).
    :param top_k: Number of nearest neighbours (default 10).
    :param pruning_threshold: Geometric pruning threshold 0–1 (default 0.8).
    :param ailake_conn_id: Airflow connection id.
    """

    template_fields: Sequence[str] = ("table",)
    ui_color = "#d4f0c4"

    def __init__(
        self,
        *,
        table: str,
        query_vector: list[float] | None = None,
        query_xcom_task_id: str | None = None,
        query_xcom_key: str = "return_value",
        top_k: int = 10,
        pruning_threshold: float = 0.8,
        ailake_conn_id: str = AilakeHook.default_conn_name,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.table = table
        self.query_vector = query_vector
        self.query_xcom_task_id = query_xcom_task_id
        self.query_xcom_key = query_xcom_key
        self.top_k = top_k
        self.pruning_threshold = pruning_threshold
        self.ailake_conn_id = ailake_conn_id

    def execute(self, context: Context) -> list[dict[str, Any]]:
        if self.query_vector is not None:
            query = self.query_vector
        elif self.query_xcom_task_id is not None:
            query = context["ti"].xcom_pull(
                task_ids=self.query_xcom_task_id,
                key=self.query_xcom_key,
            )
            if query is None:
                raise ValueError(
                    f"XCom pull from task '{self.query_xcom_task_id}' "
                    f"key '{self.query_xcom_key}' returned None"
                )
        else:
            raise ValueError("Provide query_vector or query_xcom_task_id")

        hook = AilakeHook(ailake_conn_id=self.ailake_conn_id)
        results = hook.search(
            self.table,
            query=query,
            top_k=self.top_k,
            pruning_threshold=self.pruning_threshold,
        )
        self.log.info("search returned %d results", len(results))
        return results
