# SPDX-License-Identifier: MIT OR Apache-2.0
"""AilakeHook — wraps the ailake CLI with credentials from an Airflow Connection."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from typing import Any

from airflow.hooks.base import BaseHook


class AilakeHook(BaseHook):
    """Manage connection to an AI-Lake warehouse and run the ailake CLI.

    Airflow Connection (conn_type="ailake"):
        host      — warehouse URI, e.g. ``s3://my-bucket/warehouse`` or ``/local/path``
        extra     — JSON object with optional fields:

            # AWS S3
            "aws_access_key_id":     "AKIA..."
            "aws_secret_access_key": "..."
            "aws_region":            "us-east-1"

            # Azure Blob
            "azure_account_name":    "myaccount"
            "azure_account_key":     "..."

            # GCS
            "google_application_credentials": "/path/to/sa.json"

            # CLI path override (default: "ailake" from PATH)
            "ailake_binary": "/opt/ailake/bin/ailake"

    Usage::

        hook = AilakeHook(ailake_conn_id="ailake_prod")
        info = hook.get_table_info("default.docs")
        results = hook.search("default.docs", query=[0.1, 0.2, ...], top_k=10)
    """

    conn_name_attr = "ailake_conn_id"
    default_conn_name = "ailake_default"
    conn_type = "ailake"
    hook_name = "AI-Lake"

    def __init__(self, ailake_conn_id: str = default_conn_name) -> None:
        super().__init__()
        self.ailake_conn_id = ailake_conn_id
        self._conn = None

    def get_conn(self):
        if self._conn is None:
            self._conn = self.get_connection(self.ailake_conn_id)
        return self._conn

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def get_warehouse_uri(self) -> str:
        """Return the warehouse URI from the connection host field."""
        conn = self.get_conn()
        host = conn.host or ""
        if not host:
            raise ValueError(
                f"Airflow connection '{self.ailake_conn_id}' has no host "
                "(expected warehouse URI, e.g. s3://bucket/warehouse)"
            )
        return host

    def _extra(self) -> dict:
        conn = self.get_conn()
        raw = conn.extra or "{}"
        if isinstance(raw, str):
            return json.loads(raw)
        return raw or {}

    def get_cli_env(self) -> dict[str, str]:
        """Build env dict for subprocess calls — injects cloud credentials."""
        env = os.environ.copy()
        extra = self._extra()

        if extra.get("aws_access_key_id"):
            env["AWS_ACCESS_KEY_ID"] = extra["aws_access_key_id"]
        if extra.get("aws_secret_access_key"):
            env["AWS_SECRET_ACCESS_KEY"] = extra["aws_secret_access_key"]
        if extra.get("aws_region"):
            env["AWS_DEFAULT_REGION"] = extra["aws_region"]

        if extra.get("azure_account_name"):
            env["AZURE_STORAGE_ACCOUNT_NAME"] = extra["azure_account_name"]
        if extra.get("azure_account_key"):
            env["AZURE_STORAGE_ACCOUNT_KEY"] = extra["azure_account_key"]

        if extra.get("google_application_credentials"):
            env["GOOGLE_APPLICATION_CREDENTIALS"] = extra["google_application_credentials"]

        return env

    def _binary(self) -> str:
        extra = self._extra()
        binary = extra.get("ailake_binary") or shutil.which("ailake") or "ailake"
        return binary

    def run_cli(self, *args: str, check: bool = True) -> subprocess.CompletedProcess:
        """Run ``ailake --store <warehouse> <args...>`` and return the result."""
        cmd = [self._binary(), "--store", self.get_warehouse_uri(), *args]
        self.log.info("ailake cli: %s", " ".join(cmd))
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            env=self.get_cli_env(),
            check=False,
        )
        if check and result.returncode != 0:
            # Truncate outputs to prevent cloud SDK verbose error messages (which may
            # include credential-adjacent context) from flooding Airflow task logs.
            stdout_snippet = result.stdout[:4096]
            stderr_snippet = result.stderr[:4096]
            raise RuntimeError(
                f"ailake CLI failed (exit {result.returncode}):\n"
                f"stdout: {stdout_snippet}\nstderr: {stderr_snippet}"
            )
        return result

    # ------------------------------------------------------------------
    # High-level operations
    # ------------------------------------------------------------------

    def get_table_info(self, table: str) -> dict[str, Any]:
        """Return table metadata as a dict, or {} if the table does not yet exist."""
        result = self.run_cli("info", table, "--format", "json", check=False)
        if result.returncode != 0:
            return {}
        try:
            return json.loads(result.stdout)
        except json.JSONDecodeError:
            return {}

    def get_current_snapshot_id(self, table: str) -> int | None:
        """Return the current snapshot_id for a table, or None if no snapshot exists."""
        info = self.get_table_info(table)
        return info.get("snapshot_id")

    def search(
        self,
        table: str,
        query: list[float],
        top_k: int = 10,
        pruning_threshold: float = 0.8,
        partition_filter: str | None = None,
        hybrid_text: str | None = None,
        text_column: str = "chunk_text",
        bm25_weight: float = 0.5,
    ) -> list[dict[str, Any]]:
        """Run vector search and return results as a list of dicts."""
        query_csv = ",".join(str(v) for v in query)
        extra: list[str] = []
        if partition_filter:
            extra += ["--partition-filter", partition_filter]
        if hybrid_text:
            extra += ["--hybrid-text", hybrid_text, "--text-column", text_column, "--bm25-weight", str(bm25_weight)]
        result = self.run_cli(
            "search",
            table,
            "--query", query_csv,
            "--top-k", str(top_k),
            "--pruning-threshold", str(pruning_threshold),
            "--format", "json",
            *extra,
        )
        return json.loads(result.stdout).get("results", [])

    def search_text(
        self,
        table: str,
        query_text: str,
        text_columns: list[str] | None = None,
        top_k: int = 10,
        partition_filter: str | None = None,
    ) -> list[dict[str, Any]]:
        """Run full-text search and return results as a list of dicts.

        Uses Tantivy FTS when present; falls back to BM25 brute-force.
        Wraps ``ailake search <table> --text <query> --text-columns <cols>
        --top-k <k> --format json``.
        """
        cols = ",".join(text_columns) if text_columns else "chunk_text"
        extra: list[str] = []
        if partition_filter:
            extra += ["--partition-filter", partition_filter]
        result = self.run_cli(
            "search",
            table,
            "--text", query_text,
            "--text-columns", cols,
            "--top-k", str(top_k),
            "--format", "json",
            *extra,
        )
        return json.loads(result.stdout).get("results", [])

    def delete_where(
        self,
        table: str,
        column: str,
        values: list[str],
    ) -> None:
        """Logically delete rows where ``column`` equals any value in ``values``.

        Wraps ``ailake delete-where <table> --col <col> --vals <v1,v2,...>``.
        No-op when ``values`` is empty.
        """
        if not values:
            return
        vals_csv = ",".join(values)
        self.run_cli("delete-where", table, "--col", column, "--vals", vals_csv)

    def compact(
        self,
        table: str,
        *,
        target_size: int = 536_870_912,
        min_files: int = 4,
        max_files_per_pass: int = 20,
        deferred: bool = False,
    ) -> int:
        """Compact small files in an AI-Lake table.

        Wraps ``ailake compact <table> --target-size <n> --min-files <n>
        --max-files-per-pass <n> --format json``.

        Args:
            table: Fully-qualified table name (``namespace.table``).
            target_size: Target output file size in bytes (default 512 MiB).
            min_files: Minimum eligible files required to trigger compaction (default 4).
            max_files_per_pass: Bounds peak RAM / HNSW rebuild cost (default 20).
            deferred: Build HNSW index in the background when ``True`` (default ``False``).

        Returns:
            Number of files compacted.  ``0`` when nothing qualified.
        """
        extra: list[str] = []
        if deferred:
            extra += ["--deferred"]
        result = self.run_cli(
            "compact",
            table,
            "--target-size", str(target_size),
            "--min-files", str(min_files),
            "--max-files-per-pass", str(max_files_per_pass),
            "--format", "json",
            *extra,
        )
        try:
            return json.loads(result.stdout).get("files_compacted", 0)
        except json.JSONDecodeError:
            return 0

    def decay_memories(
        self,
        table: str,
        *,
        decay_lambda: float = 0.1,
    ) -> int:
        """Recompute recency weights using exponential decay across all memory files.

        Wraps ``ailake decay-memories <table> --lambda <λ>``.

        Args:
            table: Fully-qualified table name (``namespace.table``).
            decay_lambda: Exponential decay rate λ (default 0.1, half-life ≈ 7 days).

        Returns:
            Number of files updated.  ``0`` when nothing changed.
        """
        result = self.run_cli(
            "decay-memories",
            table,
            "--lambda", str(decay_lambda),
        )
        for line in result.stdout.splitlines():
            if "files_updated:" in line:
                try:
                    return int(line.split("files_updated:")[1].strip().split()[0])
                except (ValueError, IndexError):
                    pass
        return 0

    def evolve_schema(
        self,
        table: str,
        add_columns: list[dict[str, Any]] | None = None,
        rename_columns: list[dict[str, Any]] | None = None,
    ) -> int:
        """Apply a metadata-only schema evolution to the table.

        Wraps ``ailake evolve <table> [--add name:type [--initial-default JSON]]
        [--rename old:new]``.

        Each entry in ``add_columns`` must have ``name`` and ``type`` keys, and
        optionally ``initial_default`` (a JSON literal: null, 0, 0.0, "unknown").
        Each entry in ``rename_columns`` must have ``from`` and ``to`` keys.

        Returns the new ``schema_id`` on success, ``-1`` when not parseable from
        CLI output, ``0`` when both lists are empty (no-op).
        """
        extra: list[str] = []
        for ac in (add_columns or []):
            extra += ["--add", f"{ac['name']}:{ac['type']}"]
            if ac.get("initial_default") is not None:
                extra += ["--initial-default", str(ac["initial_default"])]
        for rc in (rename_columns or []):
            extra += ["--rename", f"{rc['from']}:{rc['to']}"]
        if not extra:
            return 0
        result = self.run_cli("evolve", table, *extra)
        for line in result.stdout.splitlines():
            if "new_schema_id:" in line:
                try:
                    return int(line.split("new_schema_id:")[1].strip().split()[0])
                except (ValueError, IndexError):
                    pass
        return -1

    def migrate(
        self,
        table: str,
        *,
        embed_cmd: str,
        old_column: str = "embedding",
        new_column: str = "embedding_v2",
        text_column: str = "chunk_text",
        strategy: str = "dual-write-then-cutover",
        batch_size: int = 512,
        model_name: str | None = None,
        model_version: str | None = None,
    ) -> None:
        """Re-embed a table's vector column via an external embed command.

        Wraps ``ailake migrate <table> --old-column <c> --new-column <c>
        --text-column <c> --embed-cmd <cmd> --strategy <s> --batch-size <n>
        [--model-name <name>] [--model-version <v>]``.

        ``embed_cmd`` is a shell command that reads a JSON array of strings
        from stdin and writes a JSON array of float arrays to stdout.

        Raises ``RuntimeError`` on failure (see :meth:`run_cli`).
        """
        extra: list[str] = [
            "--old-column", old_column,
            "--new-column", new_column,
            "--text-column", text_column,
            "--embed-cmd", embed_cmd,
            "--strategy", strategy,
            "--batch-size", str(batch_size),
        ]
        if model_name:
            extra += ["--model-name", model_name]
        if model_version:
            extra += ["--model-version", model_version]
        self.run_cli("migrate", table, *extra)

    def delete_rows(self, table: str, file: str, row_positions: list[int]) -> None:
        """Mark rows as deleted in a V3 table using Iceberg Deletion Vectors.

        Wraps ``ailake delete-rows <table> --file <file> --rows <v1,v2,...>``.
        ``file`` is the Parquet data file path as reported by
        :meth:`get_table_info` (e.g. ``"data/part-00001.parquet"``).
        No-op when ``row_positions`` is empty.

        Requires the table to have been created with ``format_version=3``
        (Deletion Vectors are a V3-only Iceberg feature) — the CLI raises a
        clear error on a V2 table rather than corrupting it. For V2 tables,
        use :meth:`delete_where` (equality predicate) instead.
        """
        if not row_positions:
            return
        rows_csv = ",".join(str(r) for r in row_positions)
        self.run_cli("delete-rows", table, "--file", file, "--rows", rows_csv)

    def add_vector_column(
        self,
        table: str,
        column: str,
        dim: int,
        *,
        metric: str = "cosine",
        precision: str = "f16",
        pre_normalize: bool = False,
        hnsw_m: int | None = None,
        hnsw_ef: int | None = None,
    ) -> int:
        """Add a new vector column to an existing table schema (no data files rewritten).

        Wraps ``ailake add-vector-column <table> --column <c> --dim <n> [...]``.
        Old files return null for the new column until :meth:`backfill_vector_column`
        is run. Returns the new schema_id, or ``-1`` when not parseable from CLI output.
        """
        extra = [
            "--column", column,
            "--dim", str(dim),
            "--metric", metric,
            "--precision", precision,
        ]
        if pre_normalize:
            extra += ["--pre-normalize"]
        if hnsw_m is not None:
            extra += ["--hnsw-m", str(hnsw_m)]
        if hnsw_ef is not None:
            extra += ["--hnsw-ef", str(hnsw_ef)]
        result = self.run_cli("add-vector-column", table, *extra)
        for line in result.stdout.splitlines():
            if "new_schema_id:" in line:
                try:
                    return int(line.split("new_schema_id:")[1].strip().split()[0])
                except (ValueError, IndexError):
                    pass
        return -1

    def backfill_vector_column(
        self,
        table: str,
        column: str,
        *,
        embed_cmd: str,
        text_column: str = "chunk_text",
        batch_size: int = 512,
    ) -> None:
        """Backfill a new vector column in all existing files.

        Wraps ``ailake backfill-vector-column <table> --column <c>
        --text-column <c> --embed-cmd <cmd> --batch-size <n>``. Requires
        :meth:`add_vector_column` to have been run first for ``column``.
        Idempotent: files already containing the column are skipped.
        Raises ``RuntimeError`` on failure.
        """
        self.run_cli(
            "backfill-vector-column", table,
            "--column", column,
            "--text-column", text_column,
            "--embed-cmd", embed_cmd,
            "--batch-size", str(batch_size),
        )

    def estimate(
        self,
        rows: str,
        dim: int,
        *,
        hnsw_m: int = 16,
        pq_m: int | None = None,
    ) -> dict[str, Any]:
        """Estimate storage usage before writing (no I/O — pure math).

        Wraps ``ailake estimate --rows <n> --dim <d> --hnsw-m <m>
        [--pq-m <m>] --format json``. ``rows`` supports K/M/B suffixes
        (e.g. ``"1M"``, ``"500K"``).

        Returns ``{"rows", "dim", "hnsw_m", "pq_m", "estimates": [...]}``,
        or ``{}`` on parse failure.
        """
        extra = ["--rows", rows, "--dim", str(dim), "--hnsw-m", str(hnsw_m), "--format", "json"]
        if pq_m is not None:
            extra += ["--pq-m", str(pq_m)]
        result = self.run_cli("estimate", *extra)
        try:
            return json.loads(result.stdout)
        except json.JSONDecodeError:
            return {}
