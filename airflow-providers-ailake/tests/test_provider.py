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
    AilakeDeleteWhereOperator,
    AilakeEvolveSchemaOperator,
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

    def test_search_partition_filter_passed_to_cli(self):
        payload = json.dumps({"results": []})
        hook = _make_hook()
        with patch.object(hook, "run_cli", return_value=_completed(stdout=payload)) as mock_cli:
            hook.search("default.docs", query=[0.1, 0.2], top_k=5, partition_filter="agent-A")
        args = mock_cli.call_args[0]
        assert "--partition-filter" in args, f"--partition-filter missing: {args}"
        assert "agent-A" in args, f"partition_filter value missing: {args}"

    def test_search_no_partition_filter_when_none(self):
        payload = json.dumps({"results": []})
        hook = _make_hook()
        with patch.object(hook, "run_cli", return_value=_completed(stdout=payload)) as mock_cli:
            hook.search("default.docs", query=[0.1], top_k=1, partition_filter=None)
        args = mock_cli.call_args[0]
        assert "--partition-filter" not in args, "--partition-filter should be absent when None"


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

    def test_write_operator_partition_by_passed_to_cli(self):
        op = self._op(partition_by="agent_id", partition_value="agent-A")
        hook = _make_hook()
        with patch("airflow_providers_ailake.operators.ailake.AilakeHook", return_value=hook):
            with patch.object(hook, "get_table_info", return_value={}):
                with patch.object(hook, "run_cli", return_value=_completed()) as mock_cli:
                    op.execute(context={})
        args = mock_cli.call_args[0]
        assert "--partition-by" in args, f"--partition-by missing from CLI args: {args}"
        assert "agent_id" in args, f"partition_by value missing from CLI args: {args}"
        assert "--partition-value" in args, f"--partition-value missing from CLI args: {args}"
        assert "agent-A" in args, f"partition_value missing from CLI args: {args}"

    def test_write_operator_no_partition_args_when_not_set(self):
        op = self._op()
        hook = _make_hook()
        with patch("airflow_providers_ailake.operators.ailake.AilakeHook", return_value=hook):
            with patch.object(hook, "get_table_info", return_value={}):
                with patch.object(hook, "run_cli", return_value=_completed()) as mock_cli:
                    op.execute(context={})
        args = mock_cli.call_args[0]
        assert "--partition-by" not in args, "--partition-by should be absent when not set"
        assert "--partition-value" not in args, "--partition-value should be absent when not set"


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
            "default.docs", query=[0.1, 0.2, 0.3], top_k=5, pruning_threshold=0.8,
            partition_filter=None,
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
            "default.docs", query=query, top_k=3, pruning_threshold=0.8,
            partition_filter=None,
        )

    def test_execute_raises_without_query(self):
        op = AilakeSearchOperator(task_id="search", table="default.docs")
        with pytest.raises(ValueError, match="query_vector or query_xcom_task_id"):
            op.execute(context={"ti": MagicMock()})

    def test_search_operator_partition_filter_passed_to_hook(self):
        op = AilakeSearchOperator(
            task_id="search",
            table="default.docs",
            query_vector=[0.1, 0.2],
            partition_filter="agent-A",
        )
        hook = _make_hook()
        with patch("airflow_providers_ailake.operators.ailake.AilakeHook", return_value=hook):
            with patch.object(hook, "search", return_value=[]) as mock_search:
                op.execute(context={})
        mock_search.assert_called_once_with(
            "default.docs", query=[0.1, 0.2], top_k=10, pruning_threshold=0.8,
            partition_filter="agent-A",
        )


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


# ---------------------------------------------------------------------------
# AilakeHook — delete_where / evolve_schema
# ---------------------------------------------------------------------------


class TestAilakeHookDeleteEvolve:
    def test_delete_where_calls_cli(self):
        hook = _make_hook()
        with patch.object(hook, "run_cli", return_value=_completed()) as mock_cli:
            hook.delete_where("default.docs", "doc_id", ["a", "b", "c"])
        args = mock_cli.call_args[0]
        assert "delete-where" in args
        assert "default.docs" in args
        assert "--col" in args
        assert "doc_id" in args
        assert "--vals" in args
        assert "a,b,c" in args

    def test_delete_where_noop_on_empty_values(self):
        hook = _make_hook()
        with patch.object(hook, "run_cli", return_value=_completed()) as mock_cli:
            hook.delete_where("default.docs", "doc_id", [])
        mock_cli.assert_not_called()

    def test_evolve_schema_add_column(self):
        hook = _make_hook()
        stdout = "Evolved schema. new_schema_id: 5\n"
        with patch.object(hook, "run_cli", return_value=_completed(stdout=stdout)) as mock_cli:
            schema_id = hook.evolve_schema(
                "default.docs",
                add_columns=[{"name": "score", "type": "float", "initial_default": "0.0"}],
            )
        args = mock_cli.call_args[0]
        assert "evolve" in args
        assert "--add" in args
        assert "score:float" in args
        assert "--initial-default" in args
        assert "0.0" in args
        assert schema_id == 5

    def test_evolve_schema_rename_column(self):
        hook = _make_hook()
        stdout = "new_schema_id: 7\n"
        with patch.object(hook, "run_cli", return_value=_completed(stdout=stdout)) as mock_cli:
            schema_id = hook.evolve_schema(
                "default.docs",
                rename_columns=[{"from": "old_col", "to": "new_col"}],
            )
        args = mock_cli.call_args[0]
        assert "--rename" in args
        assert "old_col:new_col" in args
        assert schema_id == 7

    def test_evolve_schema_noop_returns_zero(self):
        hook = _make_hook()
        with patch.object(hook, "run_cli", return_value=_completed()) as mock_cli:
            result = hook.evolve_schema("default.docs")
        mock_cli.assert_not_called()
        assert result == 0

    def test_evolve_schema_returns_minus_one_on_no_id_in_output(self):
        hook = _make_hook()
        with patch.object(hook, "run_cli", return_value=_completed(stdout="done\n")):
            result = hook.evolve_schema(
                "default.docs",
                add_columns=[{"name": "x", "type": "string"}],
            )
        assert result == -1


# ---------------------------------------------------------------------------
# AilakeWriteOperator — partition_fields / format_version (Phase Q)
# ---------------------------------------------------------------------------


class TestAilakeWriteOperatorPhaseQ:
    def _op(self, **kwargs):
        return AilakeWriteOperator(
            task_id="write",
            table="default.docs",
            source_file="/tmp/data.parquet",
            **kwargs,
        )

    def test_partition_fields_passed_as_json_to_cli(self):
        fields = [{"column": "agent_id", "transform": "identity", "column_type": "string"}]
        op = self._op(partition_fields=fields)
        hook = _make_hook()
        with patch("airflow_providers_ailake.operators.ailake.AilakeHook", return_value=hook):
            with patch.object(hook, "get_table_info", return_value={}):
                with patch.object(hook, "run_cli", return_value=_completed()) as mock_cli:
                    op.execute(context={})
        args = mock_cli.call_args[0]
        assert "--partition-fields" in args, f"--partition-fields missing: {args}"
        import json as _json
        pf_idx = list(args).index("--partition-fields") + 1
        parsed = _json.loads(args[pf_idx])
        assert parsed == fields

    def test_format_version_3_passed_to_cli(self):
        op = self._op(format_version=3)
        hook = _make_hook()
        with patch("airflow_providers_ailake.operators.ailake.AilakeHook", return_value=hook):
            with patch.object(hook, "get_table_info", return_value={}):
                with patch.object(hook, "run_cli", return_value=_completed()) as mock_cli:
                    op.execute(context={})
        args = mock_cli.call_args[0]
        assert "--format-version" in args
        assert "3" in args

    def test_format_version_2_not_passed_to_cli(self):
        op = self._op(format_version=2)
        hook = _make_hook()
        with patch("airflow_providers_ailake.operators.ailake.AilakeHook", return_value=hook):
            with patch.object(hook, "get_table_info", return_value={}):
                with patch.object(hook, "run_cli", return_value=_completed()) as mock_cli:
                    op.execute(context={})
        args = mock_cli.call_args[0]
        assert "--format-version" not in args, "--format-version should be absent for default v2"

    def test_partition_fields_absent_when_none(self):
        op = self._op()
        hook = _make_hook()
        with patch("airflow_providers_ailake.operators.ailake.AilakeHook", return_value=hook):
            with patch.object(hook, "get_table_info", return_value={}):
                with patch.object(hook, "run_cli", return_value=_completed()) as mock_cli:
                    op.execute(context={})
        args = mock_cli.call_args[0]
        assert "--partition-fields" not in args


# ---------------------------------------------------------------------------
# AilakeDeleteWhereOperator
# ---------------------------------------------------------------------------


class TestAilakeDeleteWhereOperator:
    def _op(self, **kwargs):
        return AilakeDeleteWhereOperator(
            task_id="delete",
            table="default.docs",
            column="doc_id",
            **kwargs,
        )

    def test_execute_calls_delete_where_on_hook(self):
        op = self._op(values=["id-1", "id-2"])
        hook = _make_hook()
        with patch("airflow_providers_ailake.operators.ailake.AilakeHook", return_value=hook):
            with patch.object(hook, "delete_where") as mock_dw:
                op.execute(context={})
        mock_dw.assert_called_once_with("default.docs", "doc_id", ["id-1", "id-2"])

    def test_execute_reads_values_from_xcom(self):
        op = self._op(values_xcom_task_id="upstream", values_xcom_key="ids")
        ti = MagicMock()
        ti.xcom_pull.return_value = ["x", "y"]
        context = {"ti": ti}
        hook = _make_hook()
        with patch("airflow_providers_ailake.operators.ailake.AilakeHook", return_value=hook):
            with patch.object(hook, "delete_where") as mock_dw:
                op.execute(context=context)
        mock_dw.assert_called_once_with("default.docs", "doc_id", ["x", "y"])

    def test_execute_raises_when_xcom_returns_none(self):
        op = self._op(values_xcom_task_id="upstream")
        ti = MagicMock()
        ti.xcom_pull.return_value = None
        hook = _make_hook()
        with patch("airflow_providers_ailake.operators.ailake.AilakeHook", return_value=hook):
            with pytest.raises(ValueError, match="XCom pull"):
                op.execute(context={"ti": ti})

    def test_execute_raises_without_values_or_xcom(self):
        op = self._op()
        with pytest.raises(ValueError, match="values or values_xcom_task_id"):
            op.execute(context={"ti": MagicMock()})

    def test_execute_noop_on_empty_values_list(self):
        op = self._op(values=[])
        hook = _make_hook()
        with patch("airflow_providers_ailake.operators.ailake.AilakeHook", return_value=hook):
            with patch.object(hook, "delete_where") as mock_dw:
                op.execute(context={})
        mock_dw.assert_not_called()


# ---------------------------------------------------------------------------
# AilakeEvolveSchemaOperator
# ---------------------------------------------------------------------------


class TestAilakeEvolveSchemaOperator:
    def _op(self, **kwargs):
        return AilakeEvolveSchemaOperator(
            task_id="evolve",
            table="default.docs",
            **kwargs,
        )

    def test_execute_calls_evolve_schema_on_hook(self):
        add_cols = [{"name": "score", "type": "float", "initial_default": "0.0"}]
        rename_cols = [{"from": "old", "to": "new"}]
        op = self._op(add_columns=add_cols, rename_columns=rename_cols)
        hook = _make_hook()
        ti = MagicMock()
        context = {"ti": ti}
        with patch("airflow_providers_ailake.operators.ailake.AilakeHook", return_value=hook):
            with patch.object(hook, "evolve_schema", return_value=5) as mock_es:
                result = op.execute(context=context)
        mock_es.assert_called_once_with(
            "default.docs",
            add_columns=add_cols,
            rename_columns=rename_cols,
        )
        assert result == 5
        ti.xcom_push.assert_called_once_with(key="schema_id", value=5)

    def test_execute_noop_on_empty_lists(self):
        op = self._op()
        hook = _make_hook()
        with patch("airflow_providers_ailake.operators.ailake.AilakeHook", return_value=hook):
            with patch.object(hook, "evolve_schema") as mock_es:
                result = op.execute(context={"ti": MagicMock()})
        mock_es.assert_not_called()
        assert result == 0

    def test_schema_id_pushed_to_xcom(self):
        op = self._op(add_columns=[{"name": "x", "type": "string"}])
        hook = _make_hook()
        ti = MagicMock()
        with patch("airflow_providers_ailake.operators.ailake.AilakeHook", return_value=hook):
            with patch.object(hook, "evolve_schema", return_value=9):
                op.execute(context={"ti": ti})
        ti.xcom_push.assert_called_once_with(key="schema_id", value=9)
