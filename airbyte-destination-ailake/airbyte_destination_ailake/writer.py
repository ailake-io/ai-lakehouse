# SPDX-License-Identifier: MIT OR Apache-2.0
"""AI-Lake TableWriter wrapper with batching and text extraction."""

from __future__ import annotations

import logging
from typing import Any, Iterator

import numpy as np

from airbyte_destination_ailake.config import AilakeDestinationConfig
from airbyte_destination_ailake.embedder import Embedder

logger = logging.getLogger(__name__)


def _extract_text(record: dict[str, Any], field_path: str) -> str:
    """Resolve dot-separated field path from a record dict."""
    parts = field_path.split(".")
    val: Any = record
    for part in parts:
        if not isinstance(val, dict):
            return ""
        val = val.get(part, "")
    return str(val) if val is not None else ""


def _chunked(lst: list, size: int) -> Iterator[list]:
    for i in range(0, len(lst), size):
        yield lst[i : i + size]


class StreamWriter:
    """Buffers records for one Airbyte stream and writes to AI-Lake in batches."""

    def __init__(
        self,
        stream_name: str,
        cfg: AilakeDestinationConfig,
        embedder: Embedder,
    ) -> None:
        self._stream_name = stream_name
        self._cfg = cfg
        self._embedder = embedder
        self._table_path = cfg.table_path(stream_name)
        self._table = None
        self._buffer: list[dict[str, Any]] = []

    def _get_table(self):
        if self._table is None:
            import ailake

            kwargs: dict[str, Any] = {
                "dim": self._cfg.embedding_dim,
                "metric": self._cfg.embedding_metric,
                "pre_normalize": self._cfg.pre_normalize,
                "pq_only": self._cfg.pq_only,
            }
            if self._cfg.embedding_model:
                kwargs["embedding_model"] = self._cfg.embedding_model
            if self._cfg.embedding_model_version:
                kwargs["embedding_model_version"] = self._cfg.embedding_model_version
            if self._cfg.partition_by:
                kwargs["partition_by"] = self._cfg.partition_by
            if self._cfg.partition_fields:
                kwargs["partition_fields"] = self._cfg.partition_fields
            if self._cfg.format_version != 2:
                kwargs["format_version"] = self._cfg.format_version
            self._table = ailake.open_table(self._table_path, **kwargs)
        return self._table

    def add(self, record: dict[str, Any]) -> None:
        self._buffer.append(record)
        if len(self._buffer) >= self._cfg.batch_size:
            self._flush()

    def _flush(self) -> None:
        if not self._buffer:
            return
        records = self._buffer
        self._buffer = []

        texts = [_extract_text(r, self._cfg.text_field) for r in records]
        embeddings: np.ndarray = self._embedder.embed(texts)

        table = self._get_table()
        table.insert(texts, embeddings.tolist())
        logger.info(
            "stream=%s flushed batch size=%d path=%s",
            self._stream_name,
            len(records),
            self._table_path,
        )

    def commit(self) -> int:
        self._flush()
        table = self._get_table()
        snap_id = table.commit()
        logger.info("stream=%s committed snapshot_id=%d", self._stream_name, snap_id)
        return snap_id
