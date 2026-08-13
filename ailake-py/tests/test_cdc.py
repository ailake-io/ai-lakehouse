# SPDX-License-Identifier: MIT OR Apache-2.0
# Copyright (c) 2026 Thiago Egon Lange
"""End-to-end tests for AI-Lake CDC via the Python binding.

Run against a ``maturin develop`` build::

    maturin develop --release
    pytest ailake-py/tests/test_cdc.py
"""
from __future__ import annotations

import random

import pyarrow as pa

import ailake


def _rand_vec(dim: int = 4) -> list[float]:
    return [random.random() for _ in range(dim)]


def test_read_changes_detects_inserts(tmp_path):
    path = str(tmp_path / "cdc_inserts")
    w = ailake.TableWriter(path, dim=4)
    w.write_batch(["hello", "world"], [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]])
    snap1 = w.commit()

    w2 = ailake.TableWriter(path, dim=4)
    w2.write_batch(["foo"], [[0.0, 0.0, 1.0, 0.0]])
    snap2 = w2.commit()

    table = ailake.read_changes(path, start_snapshot=snap1, end_snapshot=snap2)
    assert isinstance(table, pa.Table)
    assert table.num_rows == 1
    assert table.column("_change_type").to_pylist() == ["insert"]
    assert table.column("text").to_pylist() == ["foo"]


def test_read_changes_detects_equality_deletes(tmp_path):
    path = str(tmp_path / "cdc_deletes")
    w = ailake.TableWriter(path, dim=4)
    w.write_batch(
        ["hello", "world"],
        [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]],
        extra_columns={"doc_id": ["d1", "d2"]},
    )
    snap1 = w.commit()

    ailake.delete_where(path, "doc_id", ["d1"])
    snap2 = ailake.info(path)["snapshot_id"]

    table = ailake.read_changes(path, start_snapshot=snap1, end_snapshot=snap2)
    assert table.num_rows == 1
    assert table.column("_change_type").to_pylist() == ["delete"]
    assert table.column("doc_id").to_pylist() == ["d1"]


def test_read_changes_coalesces_update(tmp_path):
    path = str(tmp_path / "cdc_update")
    w = ailake.TableWriter(path, dim=4)
    w.write_batch(
        ["v1"],
        [[1.0, 0.0, 0.0, 0.0]],
        extra_columns={"doc_id": ["d1"]},
    )
    snap1 = w.commit()

    ailake.delete_where(path, "doc_id", ["d1"])

    w2 = ailake.TableWriter(path, dim=4)
    w2.write_batch(
        ["v2"],
        [[0.0, 1.0, 0.0, 0.0]],
        extra_columns={"doc_id": ["d1"]},
    )
    w2.commit()
    snap2 = ailake.info(path)["snapshot_id"]

    table = ailake.read_changes(
        path,
        start_snapshot=snap1,
        end_snapshot=snap2,
        pk_columns=["doc_id"],
        coalesce_updates=True,
    )
    assert table.num_rows == 2
    assert table.column("_change_type").to_pylist() == ["update_before", "update_after"]
    assert table.column("doc_id").to_pylist() == ["d1", "d1"]
    assert table.column("text").to_pylist() == ["v1", "v2"]
