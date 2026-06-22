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

        hook = AilakeHook(conn_id="ailake_prod")
        info = hook.get_table_info("default.docs")
        results = hook.search("default.docs", query_vector=[0.1, 0.2, ...], top_k=10)
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
            raise RuntimeError(
                f"ailake CLI failed (exit {result.returncode}):\n"
                f"stdout: {result.stdout}\nstderr: {result.stderr}"
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
        deferred: bool = False,
    ) -> int:
        """Compact small files in an AI-Lake table.

        Wraps ``ailake compact <table> --target-size <n> --min-files <n>``.

        Args:
            table: Fully-qualified table name (``namespace.table``).
            target_size: Target output file size in bytes (default 512 MiB).
            min_files: Minimum eligible files required to trigger compaction (default 4).
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
            *extra,
        )
        for line in result.stdout.splitlines():
            if "files_compacted:" in line:
                try:
                    return int(line.split("files_compacted:")[1].strip().split()[0])
                except (ValueError, IndexError):
                    pass
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
