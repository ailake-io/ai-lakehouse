# SPDX-License-Identifier: MIT OR Apache-2.0
"""AilakeSnapshotSensor — waits for a new snapshot before releasing downstream tasks."""

from __future__ import annotations

from typing import Any, Sequence

from airflow.sensors.base import BaseSensorOperator
from airflow.utils.context import Context

from airflow_providers_ailake.hooks.ailake import AilakeHook

_XCOM_BASELINE_KEY = "_ailake_baseline_snapshot_id"


class AilakeSnapshotSensor(BaseSensorOperator):
    """Wait until the AI-Lake table has a snapshot newer than a baseline.

    **How it works**

    On the **first poke**, the sensor records the current snapshot id as the
    baseline (stored in XCom so it survives across rescheduled pokes).  On
    every subsequent poke it checks if the snapshot id has changed.  When a
    new snapshot is detected the sensor succeeds and pushes the new snapshot
    id to XCom under key ``"snapshot_id"``.

    Pass ``baseline_snapshot_id`` explicitly to watch for a snapshot newer
    than a *known* value — useful when the write task runs in a different DAG
    run and you already know the last-seen snapshot.

    **Retry safety**: using ``mode="reschedule"`` (recommended) the sensor
    releases the worker slot between pokes, which is critical for long waits.

    :param table: Fully-qualified table name (``namespace.table`` or ``table``).
    :param baseline_snapshot_id: Known snapshot id to compare against.  When
        ``None`` (default) the first poke captures the current snapshot as the
        baseline.
    :param ailake_conn_id: Airflow connection id.

    Example DAG::

        write = AilakeWriteOperator(task_id="write", table="default.docs", ...)

        wait = AilakeSnapshotSensor(
            task_id="wait_for_snapshot",
            table="default.docs",
            mode="reschedule",
            poke_interval=30,
            timeout=3600,
        )

        search = AilakeSearchOperator(task_id="search", table="default.docs", ...)

        write >> wait >> search
    """

    template_fields: Sequence[str] = ("table",)
    ui_color = "#f0d4f0"

    def __init__(
        self,
        *,
        table: str,
        baseline_snapshot_id: int | None = None,
        ailake_conn_id: str = AilakeHook.default_conn_name,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.table = table
        self.baseline_snapshot_id = baseline_snapshot_id
        self.ailake_conn_id = ailake_conn_id

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_baseline(self, context: Context) -> int | None:
        """Return the stored baseline snapshot id (from XCom or constructor arg)."""
        if self.baseline_snapshot_id is not None:
            return self.baseline_snapshot_id
        ti = context["ti"]
        return ti.xcom_pull(task_ids=ti.task_id, key=_XCOM_BASELINE_KEY)

    def _store_baseline(self, context: Context, snapshot_id: int | None) -> None:
        context["ti"].xcom_push(key=_XCOM_BASELINE_KEY, value=snapshot_id)

    # ------------------------------------------------------------------
    # Sensor contract
    # ------------------------------------------------------------------

    def poke(self, context: Context) -> bool:
        hook = AilakeHook(ailake_conn_id=self.ailake_conn_id)
        current_snapshot_id = hook.get_current_snapshot_id(self.table)

        baseline = self._get_baseline(context)

        if baseline is None:
            # First poke: record baseline and wait.
            self.log.info(
                "AilakeSnapshotSensor baseline set to snapshot_id=%s for table %s",
                current_snapshot_id,
                self.table,
            )
            self._store_baseline(context, current_snapshot_id)
            return False

        if current_snapshot_id != baseline:
            self.log.info(
                "New snapshot detected on table %s: %s → %s",
                self.table,
                baseline,
                current_snapshot_id,
            )
            context["ti"].xcom_push(key="snapshot_id", value=current_snapshot_id)
            return True

        self.log.info(
            "No new snapshot on table %s (snapshot_id=%s). Waiting…",
            self.table,
            current_snapshot_id,
        )
        return False
