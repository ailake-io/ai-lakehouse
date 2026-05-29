# SPDX-License-Identifier: MIT OR Apache-2.0
"""Unit tests for the AI-Lake Airflow provider.

These tests mock subprocess and Airflow internals — no running Airflow or
ailake binary required.
"""

from __future__ import annotations

import json
import subprocess
from unittest.mock import MagicMock, patch

import pytest

from airflow_providers_ailake import get_provider_info
from airflow_providers_ailake.hooks.ailake import AilakeHook
from airflow_providers_ailake.operators.ailake import (
    AilakeCompactOperator,
    AilakeSearchOperator,
    AilakeWriteOperator,
)
from airflow_providers_ailake.sensors.ailake import AilakeSnapshotSensor


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_conn(host="s3://bucket/wh", extra=None):
    conn = MagicMock()
    conn.host = host
    conn.extra = json.dumps(extra or {})
    return conn


def _make_hook(host="s3://bucket/wh", extra=None):
    hook = AilakeHook.__new__(AilakeHook)
    hook.ailake_conn_id = "ailake_test"
    hook._conn = _make_conn(host, extra)
    return hook


def _completed(stdout="", returncode=0):
    r = MagicMock(spec=subprocess.CompletedProcess)
    r.stdout = stdout
    r.stderr = ""
    r.returncode = returncode
    return r


# ---------------------------------------------------------------------------
# Provider metadata
# ---------------------------------------------------------------------------

def test_get_provider_info_has_connection_type():
    info = get_provider_info()
    conn_types = [c["connection-type"] for c in info["connection-types"]]
    assert "ailake" in conn_types


# ---------------------------------------------------------------------------
# AilakeHook
# ---------------------------------------------------------------------------

class TestAilakeHook:
    def test_get_warehouse_uri(self):
        hook = _make_hook("s3://my-bucket/warehouse")
        assert hook.get_warehouse_uri() == "s3://my-bucket/warehouse"

    def test_get_warehouse_uri_missing_host_raises(self):
        hook = _make_hook(host="")
        with pytest.raises(ValueError, match="no host"):
            hook.get_warehouse_uri()

    def test_get_cli_env_injects_aws_creds(self):
        hook = _make_hook(extra={
            "aws_access_key_id": "AKIATEST",
            "aws_secret_access_key": "secret",
            "aws_region": "eu-west-1",
        })
        env = hook.get_cli_env()
        assert env["AWS_ACCESS_KEY_ID"] == "AKIATEST"
        assert env["AWS_SECRET_ACCESS_KEY"] == "secret"
        assert env["AWS_DEFAULT_REGION"] == "eu-west-1"

    def test_get_cli_env_injects_azure_creds(self):
        hook = _make_hook(extra={
            "azure_account_name": "myaccount",
            "azure_account_key": "key==",
        })
        env = hook.get_cli_env()
        assert env["AZURE_STORAGE_ACCOUNT_NAME"] == "myaccount"
        assert env["AZURE_STORAGE_ACCOUNT_KEY"] == "key=="

    def test_run_cli_raises_on_nonzero_exit(self):
        hook = _make_hook()
        with patch("subprocess.run", return_value=_completed(returncode=1, stdout="err")):
            with pytest.raises(RuntimeError, match="ailake CLI failed"):
                hook.run_cli("search", "default.docs")

    def test_get_table_info_parses_json(self):
        payload = json.dumps({"table": "docs", "snapshot_id": 999, "rows": 100})
        hook = _make_hook()
        with patch.object(hook, "run_cli", return_value=_completed(stdout=payload)):
            info = hook.get_table_info("default.docs")
        assert info["snapshot_id"] == 999

    def test_get_current_snapshot_id(self):
        hook = _make_hook()
        with patch.object(
            hook, "get_table_info", return_value={"snapshot_id": 42}
        ):
            assert hook.get_current_snapshot_id("default.docs") == 42

    def test_get_current_snapshot_id_none_when_missing(self):
        hook = _make_hook()
        with patch.object(hook, "get_table_info", return_value={}):
            assert hook.get_current_snapshot_id("default.docs") is None

    def test_search_returns_results(self):
        payload = json.dumps({"results": [{"rank": 1, "row_id": 7, "distance": 0.01, "file_path": "a.parquet"}]})
        hook = _make_hook()
        with patch.object(hook, "run_cli", return_value=_completed(stdout=payload)):
            results = hook.search("default.docs", query=[0.1, 0.2], top_k=5)
        assert len(results) == 1
        assert results[0]["row_id"] == 7


# ---------------------------------------------------------------------------
# AilakeWriteOperator
# ---------------------------------------------------------------------------

class TestAilakeWriteOperator:
    def _op(self, **kwargs):
        return AilakeWriteOperator(
            task_id="write",
            table="default.docs",
            source_file="/tmp/data.parquet",
            **kwargs,
        )

    def test_execute_calls_insert_with_batch_id(self):
        op = self._op(batch_id="run1_write")
        hook = _make_hook()
        with patch("airflow_providers_ailake.operators.ailake.AilakeHook", return_value=hook):
            with patch.object(
                hook, "get_table_info", return_value={"snapshot_id": 1}
            ):
                with patch.object(hook, "run_cli", return_value=_completed()) as mock_cli:
                    op.execute(context={})

        call_args = mock_cli.call_args[0]
        assert "insert" in call_args
        assert "--batch-id" in call_args
        assert "run1_write" in call_args

    def test_default_batch_id_is_templated(self):
        op = self._op()
        assert "run_id" in op.batch_id or "task" in op.batch_id


# ---------------------------------------------------------------------------
# AilakeCompactOperator
# ---------------------------------------------------------------------------

class TestAilakeCompactOperator:
    def test_execute_calls_compact(self):
        op = AilakeCompactOperator(task_id="compact", table="default.docs")
        hook = _make_hook()
        with patch("airflow_providers_ailake.operators.ailake.AilakeHook", return_value=hook):
            with patch.object(hook, "run_cli", return_value=_completed()) as mock_cli:
                op.execute(context={})
        call_args = mock_cli.call_args[0]
        assert "compact" in call_args
        assert "default.docs" in call_args

    def test_custom_target_size_passed(self):
        op = AilakeCompactOperator(task_id="compact", table="t", target_size=1024, min_files=2)
        hook = _make_hook()
        with patch("airflow_providers_ailake.operators.ailake.AilakeHook", return_value=hook):
            with patch.object(hook, "run_cli", return_value=_completed()) as mock_cli:
                op.execute(context={})
        args = mock_cli.call_args[0]
        assert "--target-size" in args
        assert "1024" in args
        assert "--min-files" in args
        assert "2" in args


# ---------------------------------------------------------------------------
# AilakeSearchOperator
# ---------------------------------------------------------------------------

class TestAilakeSearchOperator:
    def test_execute_with_query_vector(self):
        op = AilakeSearchOperator(
            task_id="search",
            table="default.docs",
            query_vector=[0.1, 0.2, 0.3],
            top_k=5,
        )
        hook = _make_hook()
        results = [{"rank": 1, "row_id": 0, "distance": 0.0, "file_path": "a.parquet"}]
        with patch("airflow_providers_ailake.operators.ailake.AilakeHook", return_value=hook):
            with patch.object(hook, "search", return_value=results) as mock_search:
                returned = op.execute(context={})

        mock_search.assert_called_once_with(
            "default.docs", query=[0.1, 0.2, 0.3], top_k=5, pruning_threshold=0.8
        )
        assert returned == results

    def test_execute_reads_query_from_xcom(self):
        op = AilakeSearchOperator(
            task_id="search",
            table="default.docs",
            query_xcom_task_id="embed",
            top_k=3,
        )
        query = [0.5, 0.6, 0.7]
        ti = MagicMock()
        ti.xcom_pull.return_value = query
        context = {"ti": ti}
        hook = _make_hook()
        with patch("airflow_providers_ailake.operators.ailake.AilakeHook", return_value=hook):
            with patch.object(hook, "search", return_value=[]) as mock_search:
                op.execute(context=context)
        mock_search.assert_called_once_with(
            "default.docs", query=query, top_k=3, pruning_threshold=0.8
        )

    def test_execute_raises_without_query(self):
        op = AilakeSearchOperator(task_id="search", table="default.docs")
        with pytest.raises(ValueError, match="query_vector or query_xcom_task_id"):
            op.execute(context={"ti": MagicMock()})


# ---------------------------------------------------------------------------
# AilakeSnapshotSensor
# ---------------------------------------------------------------------------

class TestAilakeSnapshotSensor:
    def _sensor(self, baseline=None):
        return AilakeSnapshotSensor(
            task_id="wait",
            table="default.docs",
            baseline_snapshot_id=baseline,
            poke_interval=1,
        )

    def _context(self, xcom_store=None):
        store = xcom_store if xcom_store is not None else {}
        ti = MagicMock()
        ti.task_id = "wait"
        ti.xcom_pull.side_effect = lambda task_ids, key: store.get(key)
        ti.xcom_push.side_effect = lambda key, value: store.update({key: value})
        return {"ti": ti}, store

    def test_first_poke_records_baseline_and_returns_false(self):
        sensor = self._sensor()
        context, store = self._context()
        hook = _make_hook()
        with patch("airflow_providers_ailake.sensors.ailake.AilakeHook", return_value=hook):
            with patch.object(hook, "get_current_snapshot_id", return_value=100):
                result = sensor.poke(context)
        assert result is False
        assert store["_ailake_baseline_snapshot_id"] == 100

    def test_second_poke_no_change_returns_false(self):
        sensor = self._sensor()
        context, store = self._context({"_ailake_baseline_snapshot_id": 100})
        hook = _make_hook()
        with patch("airflow_providers_ailake.sensors.ailake.AilakeHook", return_value=hook):
            with patch.object(hook, "get_current_snapshot_id", return_value=100):
                result = sensor.poke(context)
        assert result is False

    def test_poke_returns_true_on_new_snapshot(self):
        sensor = self._sensor()
        context, store = self._context({"_ailake_baseline_snapshot_id": 100})
        hook = _make_hook()
        with patch("airflow_providers_ailake.sensors.ailake.AilakeHook", return_value=hook):
            with patch.object(hook, "get_current_snapshot_id", return_value=200):
                result = sensor.poke(context)
        assert result is True
        assert store["snapshot_id"] == 200

    def test_explicit_baseline_skips_first_poke_recording(self):
        sensor = self._sensor(baseline=50)
        context, store = self._context()
        hook = _make_hook()
        with patch("airflow_providers_ailake.sensors.ailake.AilakeHook", return_value=hook):
            with patch.object(hook, "get_current_snapshot_id", return_value=50):
                result = sensor.poke(context)
        assert result is False

    def test_explicit_baseline_detects_new_snapshot(self):
        sensor = self._sensor(baseline=50)
        context, store = self._context()
        hook = _make_hook()
        with patch("airflow_providers_ailake.sensors.ailake.AilakeHook", return_value=hook):
            with patch.object(hook, "get_current_snapshot_id", return_value=99):
                result = sensor.poke(context)
        assert result is True
        assert store["snapshot_id"] == 99
