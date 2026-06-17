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
        """Return table metadata as a dict (uses ``ailake info --format json``)."""
        result = self.run_cli("info", table, "--format", "json")
        return json.loads(result.stdout)

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
    ) -> list[dict[str, Any]]:
        """Run vector search and return results as a list of dicts."""
        query_csv = ",".join(str(v) for v in query)
        extra: list[str] = []
        if partition_filter:
            extra += ["--partition-filter", partition_filter]
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
