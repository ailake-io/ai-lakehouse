# SPDX-License-Identifier: MIT OR Apache-2.0
"""AI-Lake TableWriter wrapper with batching, text extraction, and partition routing."""

from __future__ import annotations

import logging
from typing import Any, Iterator

import numpy as np

from airbyte_destination_ailake.config import AilakeDestinationConfig
from airbyte_destination_ailake.embedder import Embedder

logger = logging.getLogger(__name__)

_SCALAR_TYPES = (str, int, float, bool)


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


def _partition_columns(cfg: AilakeDestinationConfig) -> list[str]:
    """Column name(s) whose per-record value determines the partition, if configured."""
    if cfg.partition_fields:
        return [pf["column"] for pf in cfg.partition_fields]
    if cfg.partition_by:
        return [cfg.partition_by]
    return []


def _partition_key(record: dict[str, Any], columns: list[str]) -> tuple[str, ...] | None:
    """Value tuple for `columns`, or None if any column is missing from the record.

    A None key means "no partition value derivable from this record" — it's written
    through a Table opened with only the partition *schema* (partition_by/partition_fields),
    matching the pre-existing behavior for records that don't carry the partition column.
    """
    values: list[str] = []
    for col in columns:
        if col not in record or record[col] is None:
            return None
        values.append(str(record[col]))
    return tuple(values)


def _extract_extra_columns(
    records: list[dict[str, Any]],
    exclude: set[str],
    allowlist: list[str] | None,
) -> dict[str, list]:
    """Build `{column: [values]}` for every record field except `exclude`.

    Non-scalar values (dict/list) aren't supported by the underlying Parquet writer's
    type inference — a field is skipped (with a warning) if any record's value for it
    isn't a str/int/float/bool/None.
    """
    if allowlist is not None:
        keys: list[str] = [k for k in allowlist if k not in exclude]
    else:
        keys = []
        seen: set[str] = set()
        for record in records:
            for k in record.keys():
                if k in exclude or k in seen:
                    continue
                seen.add(k)
                keys.append(k)

    extra_columns: dict[str, list] = {}
    for key in keys:
        values: list[Any] = []
        skip = False
        first_type: type | None = None
        warned_mixed_type = False
        for record in records:
            v = record.get(key)
            if v is not None and not isinstance(v, _SCALAR_TYPES):
                skip = True
                break
            if v is not None:
                if first_type is None:
                    first_type = type(v)
                elif type(v) is not first_type and not warned_mixed_type:
                    # The underlying column type is inferred from the first non-null
                    # value only (see ailake-py's write_batch); a later value of a
                    # different scalar type is silently coerced/nulled downstream with
                    # no other signal, unlike the non-scalar case above which does warn.
                    logger.warning(
                        "ailake destination: field '%s' has mixed scalar types "
                        "(%s then %s) — the column type is inferred from the first "
                        "value seen; later values of a different type may be "
                        "silently nulled",
                        key,
                        first_type.__name__,
                        type(v).__name__,
                    )
                    warned_mixed_type = True
            values.append(v)
        if skip:
            logger.warning(
                "ailake destination: skipping non-scalar field '%s' — "
                "only str/int/float/bool record fields become extra columns",
                key,
            )
            continue
        extra_columns[key] = values
    return extra_columns


class StreamWriter:
    """Buffers records for one Airbyte stream and writes to AI-Lake in batches.

    When `partition_fields`/`partition_by` is configured, records are grouped by the
    partition column(s)' per-record value and routed to a dedicated `Table` per distinct
    value — AI-Lake's partition value is fixed per `Table` instance (like a Hive
    partition directory), not derived per-row, so a single writer can't mix values.
    """

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
        self._partition_cols = _partition_columns(cfg)
        # Keyed by the partition value tuple (or None for "no partition value on this
        # record" / unpartitioned). Lazily opened per distinct value seen.
        self._tables: dict[tuple[str, ...] | None, Any] = {}
        self._buffer: list[dict[str, Any]] = []

    def _get_table(self, partition_key: tuple[str, ...] | None):
        table = self._tables.get(partition_key)
        if table is None:
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
            if self._cfg.fts_columns:
                kwargs["fts_text_columns"] = self._cfg.fts_columns
                kwargs["fts_tokenizer"] = self._cfg.fts_tokenizer
            if self._cfg.hnsw_m is not None:
                kwargs["hnsw_m"] = self._cfg.hnsw_m
            if self._cfg.hnsw_ef_construction is not None:
                kwargs["hnsw_ef_construction"] = self._cfg.hnsw_ef_construction

            # Only pass an explicit partition value when this record group actually
            # carried one — otherwise open_table() the same way it always was
            # (partition schema declared, no value pinned to this writer).
            if partition_key is not None:
                if self._cfg.partition_fields:
                    kwargs["partition_values"] = dict(zip(self._partition_cols, partition_key))
                elif self._cfg.partition_by:
                    kwargs["partition_value"] = partition_key[0]

            table = ailake.open_table(self._table_path, **kwargs)
            self._tables[partition_key] = table
        return table

    def add(self, record: dict[str, Any]) -> None:
        self._buffer.append(record)
        if len(self._buffer) >= self._cfg.batch_size:
            self._flush()

    def _flush(self) -> None:
        if not self._buffer:
            return
        records = self._buffer[:]

        # Group by partition value so each group can be routed to its own Table.
        groups: dict[tuple[str, ...] | None, list[dict[str, Any]]] = {}
        for record in records:
            key = _partition_key(record, self._partition_cols)
            groups.setdefault(key, []).append(record)

        exclude = {self._cfg.text_field, *self._partition_cols}
        for key, group_records in groups.items():
            texts = [_extract_text(r, self._cfg.text_field) for r in group_records]
            embeddings: np.ndarray = self._embedder.embed(texts)
            extra_columns = _extract_extra_columns(
                group_records, exclude, self._cfg.extra_columns or None
            )
            # Partition columns are still regular queryable data — carry them through
            # even though they also drove which Table this group was routed to.
            for col in self._partition_cols:
                extra_columns[col] = [r.get(col) for r in group_records]

            table = self._get_table(key)
            table.insert(texts, embeddings.tolist(), extra_columns=extra_columns or None)

        del self._buffer[: len(records)]
        logger.info(
            "stream=%s flushed batch size=%d groups=%d path=%s",
            self._stream_name,
            len(records),
            len(groups),
            self._table_path,
        )

    def commit(self) -> int:
        self._flush()
        snap_id = -1
        for table in self._tables.values():
            snap_id = table.commit()
        self._tables = {}  # reset so the next batch opens fresh Table instance(s)
        logger.info("stream=%s committed snapshot_id=%d", self._stream_name, snap_id)
        return snap_id
