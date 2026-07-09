# SPDX-License-Identifier: MIT OR Apache-2.0
"""Airflow operators for AI-Lake: write, compact, search, delete_where, evolve_schema,
migrate, delete_rows, add_vector_column, backfill_vector_column."""

from __future__ import annotations

import json
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
    :param embeddings_column: Name of the embedding column in the source file
        (single-column mode; ignored when ``vector_cols`` is set).
    :param vector_cols: Multi-column (Phase 8 multimodal) mode — one dict per
        vector column with keys ``column``, ``dim``, and optionally ``metric``
        (default ``"cosine"``) and ``modality``. When set, ``embeddings_column``
        is ignored.
    :param batch_id: Idempotency key.  Defaults to ``{{ run_id }}_{{ task_id }}``.
    :param fts_columns: Text columns to index with Tantivy FTS (e.g. ``["chunk_text"]``).
        Empty or ``None`` disables FTS (default).
    :param fts_tokenizer: Tantivy tokenizer name (default ``"default"``).
    :param hnsw_m: HNSW graph connectivity (M). ``None`` = use table default.
    :param hnsw_ef_construction: HNSW ef_construction. ``None`` = use table default.
    :param pre_normalize: Normalize vectors to unit L2 at write time (recommended for cosine).
    :param deferred: Build index asynchronously. Parquet committed immediately.
        Not combinable with ``batch_id`` on the CLI — when ``True``, ``batch_id``
        is dropped (with a warning) rather than passed, since deferred writes
        don't carry an idempotency tag yet.
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
        vector_cols: list[dict[str, Any]] | None = None,
        batch_id: str = "{{ run_id }}_{{ task.task_id }}",
        partition_by: str | None = None,
        partition_value: str | None = None,
        partition_fields: list[dict[str, Any]] | None = None,
        format_version: int = 2,
        fts_columns: list[str] | None = None,
        fts_tokenizer: str = "default",
        hnsw_m: int | None = None,
        hnsw_ef_construction: int | None = None,
        pre_normalize: bool = False,
        deferred: bool = False,
        ailake_conn_id: str = AilakeHook.default_conn_name,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.table = table
        self.source_file = source_file
        self.embeddings_column = embeddings_column
        self.vector_cols = vector_cols
        self.batch_id = batch_id
        self.partition_by = partition_by
        self.partition_value = partition_value
        self.partition_fields = partition_fields
        self.format_version = format_version
        self.fts_columns = fts_columns or []
        self.fts_tokenizer = fts_tokenizer
        self.hnsw_m = hnsw_m
        self.hnsw_ef_construction = hnsw_ef_construction
        self.pre_normalize = pre_normalize
        self.deferred = deferred
        self.ailake_conn_id = ailake_conn_id

    def execute(self, context: Context) -> None:
        hook = AilakeHook(ailake_conn_id=self.ailake_conn_id)

        # Idempotency is enforced downstream by the CLI's --batch-id flag
        # (write_batch_idempotent): a retry with the same batch_id is a safe no-op.
        # No pre-check is done here — there is no CLI-exposed way to ask "was this
        # exact batch_id already committed" ahead of time, and get_table_info()'s
        # snapshot_id only reports whether the table has *any* snapshot, which says
        # nothing about this batch_id specifically.
        extra_args: list[str] = []
        if self.partition_by:
            extra_args += ["--partition-by", self.partition_by]
        if self.partition_value:
            extra_args += ["--partition-value", self.partition_value]
        if self.partition_fields:
            extra_args += ["--partition-fields", json.dumps(self.partition_fields)]
        if self.format_version != 2:
            extra_args += ["--format-version", str(self.format_version)]
        if self.fts_columns:
            extra_args += ["--fts-columns", ",".join(self.fts_columns)]
            if self.fts_tokenizer != "default":
                extra_args += ["--fts-tokenizer", self.fts_tokenizer]
        if self.hnsw_m is not None:
            extra_args += ["--hnsw-m", str(self.hnsw_m)]
        if self.hnsw_ef_construction is not None:
            extra_args += ["--hnsw-ef", str(self.hnsw_ef_construction)]
        if self.pre_normalize:
            extra_args += ["--pre-normalize"]
        if self.deferred:
            extra_args += ["--deferred"]

        if self.vector_cols:
            specs = []
            for vc in self.vector_cols:
                spec = f"{vc['column']}:{vc['dim']}:{vc.get('metric', 'cosine')}"
                if vc.get("modality"):
                    spec += f":{vc['modality']}"
                specs.append(spec)
            extra_args += ["--vector-cols", ",".join(specs)]
        else:
            extra_args += ["--embeddings", self.embeddings_column]

        # --deferred and --batch-id are mutually exclusive on the CLI (deferred
        # writes don't yet carry an idempotency tag) — drop the idempotency key
        # rather than crash when both are set, since deferred=True is an explicit
        # opt-in the caller made.
        if self.deferred:
            self.log.warning(
                "AilakeWriteOperator: deferred=True — dropping batch_id '%s', "
                "idempotency is not enforced for deferred writes",
                self.batch_id,
            )
        else:
            extra_args += ["--batch-id", self.batch_id]

        result = hook.run_cli(
            "insert",
            self.table,
            self.source_file,
            *extra_args,
        )
        self.log.info(result.stdout.strip())


class AilakeCompactOperator(BaseOperator):
    """Compact small files in an AI-Lake table.

    Wraps ``ailake compact <table>``.

    :param table: Fully-qualified table name.
    :param target_size: Target file size in bytes (default 512 MiB).
    :param min_files: Min small files required to trigger compaction (default 4).
    :param max_files_per_pass: Bounds peak RAM / HNSW rebuild cost (default 20).
    :param deferred: Build the HNSW index in the background instead of blocking.
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
        max_files_per_pass: int = 20,
        deferred: bool = False,
        ailake_conn_id: str = AilakeHook.default_conn_name,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.table = table
        self.target_size = target_size
        self.min_files = min_files
        self.max_files_per_pass = max_files_per_pass
        self.deferred = deferred
        self.ailake_conn_id = ailake_conn_id

    def execute(self, context: Context) -> int:
        hook = AilakeHook(ailake_conn_id=self.ailake_conn_id)
        files_compacted = hook.compact(
            self.table,
            target_size=self.target_size,
            min_files=self.min_files,
            max_files_per_pass=self.max_files_per_pass,
            deferred=self.deferred,
        )
        self.log.info("compact: table=%s files_compacted=%d", self.table, files_compacted)
        return files_compacted


class AilakeSearchOperator(BaseOperator):
    """Run a vector similarity search on an AI-Lake table and return the results.

    The query vector is read from ``query_vector`` (list of floats) or from
    XCom via ``query_xcom_task_id`` + ``query_xcom_key``.  Results are returned
    as a list of dicts and, via Airflow's default ``do_xcom_push`` behavior,
    land on XCom under the standard ``"return_value"`` key::

        [{"rank": 1, "row_id": 42, "distance": 0.003, "file_path": "data/part-0.parquet"}, ...]

    :param table: Fully-qualified table name.
    :param query_vector: Flat list of floats.  Mutually exclusive with
        ``query_xcom_task_id``.
    :param query_xcom_task_id: Task id whose XCom holds the query vector list.
    :param query_xcom_key: XCom key (default ``"return_value"``).
    :param top_k: Number of nearest neighbours (default 10).
    :param pruning_threshold: Geometric pruning threshold 0–1 (default 0.8).
    :param hybrid_text: When set, enables hybrid BM25+vector RRF fusion. The text is scored
        against ``text_column`` with BM25 and fused with vector results via RRF.
    :param text_column: Parquet column for BM25 scoring (default ``"chunk_text"``).
    :param bm25_weight: BM25 weight in RRF (0.0 = pure vector, 1.0 = pure BM25, default 0.5).
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
        partition_filter: str | None = None,
        hybrid_text: str | None = None,
        text_column: str = "chunk_text",
        bm25_weight: float = 0.5,
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
        self.partition_filter = partition_filter
        self.hybrid_text = hybrid_text
        self.text_column = text_column
        self.bm25_weight = bm25_weight
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
            partition_filter=self.partition_filter,
            hybrid_text=self.hybrid_text,
            text_column=self.text_column,
            bm25_weight=self.bm25_weight,
        )
        self.log.info("search returned %d results", len(results))
        return results


class AilakeDeleteWhereOperator(BaseOperator):
    """Logically delete rows matching a column equality predicate.

    Wraps ``ailake delete-where <table> --col <col> --vals <v1,v2,...>``.

    Deleted rows are masked at scan time via an Iceberg equality delete file —
    no data files are rewritten.  The operator is idempotent: deleting
    already-deleted rows is a no-op.

    :param table: Fully-qualified table name (``namespace.table``).
    :param column: Column to match on (equality predicate).
    :param values: List of values to delete.  May be a Jinja template that
        resolves to a list, or use ``values_xcom_task_id`` to read from XCom.
    :param values_xcom_task_id: Task id whose XCom holds the values list.
    :param values_xcom_key: XCom key (default ``"return_value"``).
    :param ailake_conn_id: Airflow connection id.
    """

    template_fields: Sequence[str] = ("table", "column")
    ui_color = "#ffd0d0"

    def __init__(
        self,
        *,
        table: str,
        column: str,
        values: list[str] | None = None,
        values_xcom_task_id: str | None = None,
        values_xcom_key: str = "return_value",
        ailake_conn_id: str = AilakeHook.default_conn_name,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.table = table
        self.column = column
        self.values = values
        self.values_xcom_task_id = values_xcom_task_id
        self.values_xcom_key = values_xcom_key
        self.ailake_conn_id = ailake_conn_id

    def execute(self, context: Context) -> None:
        if self.values is not None:
            values = self.values
        elif self.values_xcom_task_id is not None:
            values = context["ti"].xcom_pull(
                task_ids=self.values_xcom_task_id,
                key=self.values_xcom_key,
            )
            if values is None:
                raise ValueError(
                    f"XCom pull from task '{self.values_xcom_task_id}' "
                    f"key '{self.values_xcom_key}' returned None"
                )
        else:
            raise ValueError("Provide values or values_xcom_task_id")

        if not values:
            self.log.info("delete_where: values list is empty — no-op")
            return

        hook = AilakeHook(ailake_conn_id=self.ailake_conn_id)
        hook.delete_where(self.table, self.column, list(values))
        self.log.info(
            "delete_where: table=%s column=%s deleted %d value(s)",
            self.table,
            self.column,
            len(values),
        )


class AilakeEvolveSchemaOperator(BaseOperator):
    """Apply a metadata-only schema evolution to an AI-Lake table.

    Wraps ``ailake evolve <table> [--add name:type [--initial-default JSON]]
    [--rename old:new]``.

    No data files are rewritten.  The new ``schema_id`` is pushed to XCom
    under key ``"schema_id"`` for downstream tasks.

    :param table: Fully-qualified table name (``namespace.table``).
    :param add_columns: Columns to add.  Each entry must have ``name`` and
        ``type`` keys; ``initial_default`` is optional (a JSON literal:
        ``null``, ``0``, ``0.0``, ``"unknown"``).
    :param rename_columns: Columns to rename.  Each entry must have ``from``
        and ``to`` keys.
    :param ailake_conn_id: Airflow connection id.

    Example::

        AilakeEvolveSchemaOperator(
            task_id="add_score_col",
            table="default.docs",
            add_columns=[{"name": "score", "type": "float", "initial_default": "0.0"}],
            rename_columns=[{"from": "old_name", "to": "new_name"}],
        )
    """

    template_fields: Sequence[str] = ("table",)
    ui_color = "#e0d4f0"

    def __init__(
        self,
        *,
        table: str,
        add_columns: list[dict[str, Any]] | None = None,
        rename_columns: list[dict[str, Any]] | None = None,
        ailake_conn_id: str = AilakeHook.default_conn_name,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.table = table
        self.add_columns = add_columns or []
        self.rename_columns = rename_columns or []
        self.ailake_conn_id = ailake_conn_id

    def execute(self, context: Context) -> int:
        if not self.add_columns and not self.rename_columns:
            self.log.info("evolve_schema: nothing to evolve — no-op")
            return 0

        hook = AilakeHook(ailake_conn_id=self.ailake_conn_id)
        schema_id = hook.evolve_schema(
            self.table,
            add_columns=self.add_columns or None,
            rename_columns=self.rename_columns or None,
        )
        self.log.info(
            "evolve_schema: table=%s new_schema_id=%s",
            self.table,
            schema_id,
        )
        context["ti"].xcom_push(key="schema_id", value=schema_id)
        return schema_id


class AilakeFtsSearchOperator(BaseOperator):
    """Run a full-text (BM25 / Tantivy) search on an AI-Lake table.

    Uses the Tantivy per-file FTS index when present (O(log N)); falls back to
    BM25 brute-force for legacy files.  Results are pushed to XCom under key
    ``"fts_results"`` as a list of dicts::

        [{"rank": 1, "row_id": 42, "score": 4.21, "file_path": "data/part-0.parquet"}, ...]

    :param table: Fully-qualified table name (``namespace.table``).
    :param query_text: Free-text query string.  Supports Jinja templating.
    :param text_columns: Parquet columns to search (default ``["chunk_text"]``).
    :param top_k: Maximum results to return (default 10).
    :param partition_filter: Restrict search to files with this partition value.
    :param ailake_conn_id: Airflow connection id.

    Example::

        AilakeFtsSearchOperator(
            task_id="fts_search",
            table="default.docs",
            query_text="{{ dag_run.conf['query'] }}",
            text_columns=["chunk_text", "document_title"],
            top_k=20,
        )
    """

    template_fields: Sequence[str] = ("table", "query_text")
    ui_color = "#f0e4d4"

    def __init__(
        self,
        *,
        table: str,
        query_text: str,
        text_columns: list[str] | None = None,
        top_k: int = 10,
        partition_filter: str | None = None,
        ailake_conn_id: str = AilakeHook.default_conn_name,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.table = table
        self.query_text = query_text
        self.text_columns = text_columns or ["chunk_text"]
        self.top_k = top_k
        self.partition_filter = partition_filter
        self.ailake_conn_id = ailake_conn_id

    def execute(self, context: Context) -> list[dict[str, Any]]:
        if not self.query_text:
            self.log.info("fts_search: empty query_text — no-op")
            return []

        hook = AilakeHook(ailake_conn_id=self.ailake_conn_id)
        results = hook.search_text(
            self.table,
            query_text=self.query_text,
            text_columns=self.text_columns,
            top_k=self.top_k,
            partition_filter=self.partition_filter,
        )
        self.log.info("fts_search: table=%s returned %d results", self.table, len(results))
        context["ti"].xcom_push(key="fts_results", value=results)
        return results


class AilakeMigrateOperator(BaseOperator):
    """Re-embed a table's vector column via an external embed command.

    Wraps ``ailake migrate <table> --embed-cmd <cmd> [...]``.

    :param table: Fully-qualified table name (``namespace.table``).
    :param embed_cmd: Shell command that reads a JSON array of strings from
        stdin and writes a JSON array of float arrays to stdout.
    :param old_column: Name of the existing embedding column (default ``"embedding"``).
    :param new_column: Name for the migrated column; may equal ``old_column``
        for an in-place upgrade (default ``"embedding_v2"``).
    :param text_column: Parquet column holding the raw text to re-embed
        (default ``"chunk_text"``).
    :param strategy: ``"atomic_replace"`` (lower storage) or
        ``"dual-write-then-cutover"`` (zero downtime, default).
    :param batch_size: Number of texts per embed-cmd call (default 512).
    :param model_name: Model identifier stored in ``ailake.embedding-model``.
    :param model_version: Optional version tag appended to ``model_name``.
    :param ailake_conn_id: Airflow connection id.
    """

    template_fields: Sequence[str] = ("table",)
    ui_color = "#c4e0f0"

    def __init__(
        self,
        *,
        table: str,
        embed_cmd: str,
        old_column: str = "embedding",
        new_column: str = "embedding_v2",
        text_column: str = "chunk_text",
        strategy: str = "dual-write-then-cutover",
        batch_size: int = 512,
        model_name: str | None = None,
        model_version: str | None = None,
        ailake_conn_id: str = AilakeHook.default_conn_name,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.table = table
        self.embed_cmd = embed_cmd
        self.old_column = old_column
        self.new_column = new_column
        self.text_column = text_column
        self.strategy = strategy
        self.batch_size = batch_size
        self.model_name = model_name
        self.model_version = model_version
        self.ailake_conn_id = ailake_conn_id

    def execute(self, context: Context) -> None:
        hook = AilakeHook(ailake_conn_id=self.ailake_conn_id)
        hook.migrate(
            self.table,
            embed_cmd=self.embed_cmd,
            old_column=self.old_column,
            new_column=self.new_column,
            text_column=self.text_column,
            strategy=self.strategy,
            batch_size=self.batch_size,
            model_name=self.model_name,
            model_version=self.model_version,
        )
        self.log.info(
            "migrate: table=%s %s -> %s complete", self.table, self.old_column, self.new_column
        )


class AilakeDeleteRowsOperator(BaseOperator):
    """Mark rows as deleted in a V3 table using Iceberg Deletion Vectors.

    Distinct from :class:`AilakeDeleteWhereOperator` (equality-predicate delete):
    this deletes specific row *positions* within one data file.

    Wraps ``ailake delete-rows <table> --file <file> --rows <v1,v2,...>``.

    Requires the table to have been created with ``format_version=3``
    (Deletion Vectors are a V3-only Iceberg feature) — the CLI raises a clear
    error on a V2 table rather than corrupting it. For V2 tables, use
    :class:`AilakeDeleteWhereOperator` (equality predicate) instead.

    :param table: Fully-qualified table name (``namespace.table``).
    :param file: Parquet data file path as reported by ``ailake info``
        (e.g. ``"data/part-00001.parquet"``).
    :param row_positions: 0-based row positions to delete within ``file``.
    :param ailake_conn_id: Airflow connection id.
    """

    template_fields: Sequence[str] = ("table", "file")
    ui_color = "#ffb0b0"

    def __init__(
        self,
        *,
        table: str,
        file: str,
        row_positions: list[int],
        ailake_conn_id: str = AilakeHook.default_conn_name,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.table = table
        self.file = file
        self.row_positions = row_positions
        self.ailake_conn_id = ailake_conn_id

    def execute(self, context: Context) -> None:
        if not self.row_positions:
            self.log.info("delete_rows: row_positions is empty — no-op")
            return

        hook = AilakeHook(ailake_conn_id=self.ailake_conn_id)
        hook.delete_rows(self.table, self.file, list(self.row_positions))
        self.log.info(
            "delete_rows: table=%s file=%s deleted %d row(s)",
            self.table,
            self.file,
            len(self.row_positions),
        )


class AilakeAddVectorColumnOperator(BaseOperator):
    """Add a new vector column to an existing table schema (no data files rewritten).

    Old files return null for the new column until
    :class:`AilakeBackfillVectorColumnOperator` is run. Pushes the new
    ``schema_id`` to XCom under key ``"schema_id"``.

    Wraps ``ailake add-vector-column <table> --column <c> --dim <n> [...]``.

    :param table: Fully-qualified table name (``namespace.table``).
    :param column: New vector column name.
    :param dim: Vector dimensionality for the new column.
    :param metric: Distance metric (default ``"cosine"``).
    :param precision: Vector precision (default ``"f16"``).
    :param pre_normalize: Normalize vectors to unit L2 at write time.
    :param hnsw_m: HNSW M parameter (default: table default).
    :param hnsw_ef: HNSW ef_construction parameter (default: table default).
    :param ailake_conn_id: Airflow connection id.
    """

    template_fields: Sequence[str] = ("table",)
    ui_color = "#d4f0e0"

    def __init__(
        self,
        *,
        table: str,
        column: str,
        dim: int,
        metric: str = "cosine",
        precision: str = "f16",
        pre_normalize: bool = False,
        hnsw_m: int | None = None,
        hnsw_ef: int | None = None,
        ailake_conn_id: str = AilakeHook.default_conn_name,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.table = table
        self.column = column
        self.dim = dim
        self.metric = metric
        self.precision = precision
        self.pre_normalize = pre_normalize
        self.hnsw_m = hnsw_m
        self.hnsw_ef = hnsw_ef
        self.ailake_conn_id = ailake_conn_id

    def execute(self, context: Context) -> int:
        hook = AilakeHook(ailake_conn_id=self.ailake_conn_id)
        schema_id = hook.add_vector_column(
            self.table,
            self.column,
            self.dim,
            metric=self.metric,
            precision=self.precision,
            pre_normalize=self.pre_normalize,
            hnsw_m=self.hnsw_m,
            hnsw_ef=self.hnsw_ef,
        )
        self.log.info(
            "add_vector_column: table=%s column=%s new_schema_id=%s",
            self.table,
            self.column,
            schema_id,
        )
        context["ti"].xcom_push(key="schema_id", value=schema_id)
        return schema_id


class AilakeBackfillVectorColumnOperator(BaseOperator):
    """Backfill a new vector column in all existing files.

    Reads text from ``text_column``, calls ``embed_cmd`` for each batch, and
    rewrites files to include both the original vector column and the new
    one. Idempotent: files already containing the column are skipped.
    Requires :class:`AilakeAddVectorColumnOperator` to have been run first
    for ``column``.

    Wraps ``ailake backfill-vector-column <table> --column <c> --embed-cmd <cmd> [...]``.

    :param table: Fully-qualified table name (``namespace.table``).
    :param column: Vector column to backfill (must already exist via
        ``AilakeAddVectorColumnOperator``).
    :param embed_cmd: Shell command that reads a JSON array of strings from
        stdin and writes a JSON array of float arrays to stdout.
    :param text_column: Parquet column holding the raw text to embed
        (default ``"chunk_text"``).
    :param batch_size: Texts per embed-cmd call (default 512).
    :param ailake_conn_id: Airflow connection id.
    """

    template_fields: Sequence[str] = ("table",)
    ui_color = "#d4f0e0"

    def __init__(
        self,
        *,
        table: str,
        column: str,
        embed_cmd: str,
        text_column: str = "chunk_text",
        batch_size: int = 512,
        ailake_conn_id: str = AilakeHook.default_conn_name,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.table = table
        self.column = column
        self.embed_cmd = embed_cmd
        self.text_column = text_column
        self.batch_size = batch_size
        self.ailake_conn_id = ailake_conn_id

    def execute(self, context: Context) -> None:
        hook = AilakeHook(ailake_conn_id=self.ailake_conn_id)
        hook.backfill_vector_column(
            self.table,
            self.column,
            embed_cmd=self.embed_cmd,
            text_column=self.text_column,
            batch_size=self.batch_size,
        )
        self.log.info("backfill_vector_column: table=%s column=%s complete", self.table, self.column)
