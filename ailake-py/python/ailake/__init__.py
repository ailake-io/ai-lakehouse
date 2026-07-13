# SPDX-License-Identifier: MIT OR Apache-2.0
# Copyright (c) 2026 Thiago Egon Lange
"""AI-Lake Python SDK — fluent API over the Rust core."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any, Callable, Iterable, Optional, Sequence, Union

from ailake._ailake import (  # type: ignore[import]
    TableWriter as _TableWriter,
    TimestampNs,
    VectorColSpec,
    WorkingMemoryBuffer,
    add_column,
    add_vector_column,
    assemble_context as _assemble_context_raw,
    backfill_vector_column,
    compact as _compact_native,
    decay_memories,
    delete_rows,
    delete_where,
    estimate,
    hardware_info,
    migrate_embeddings,
    now_ns,
    rename_column,
    search as _search_raw,
    search_multimodal,
    search_text,
    search_with_data as _search_with_data,
)

# Expose search_with_data for callers that need raw IPC bytes (advanced use).
search_with_data = _search_with_data

# scan() is a naming-parity alias for search_with_data — same capability as
# ailake-go's Scan() and ailake-jni's ailake_scan_json (search + full-row
# fetch in one call, no JOIN needed against a separately-registered table).
scan = _search_with_data

if TYPE_CHECKING:
    import numpy as np
    import pandas as pd
    import polars as pl

# Accepted embedding input types — list, numpy array, or any array with .tolist()
_Embeddings = Union[Sequence[Sequence[float]], "np.ndarray"]
_Vector = Union[Sequence[float], "np.ndarray"]

__all__ = [
    "open_table",
    "search",
    "search_text",
    "search_multimodal",
    "search_with_data",
    "scan",
    "compact",
    "estimate",
    "Table",
    "SearchQuery",
    "TableWriter",
    "VectorColSpec",
    "TimestampNs",
    "WorkingMemoryBuffer",
    "Agent",
    "assemble_context",
    "migrate_embeddings",
    "decay_memories",
    "delete_where",
    "delete_rows",
    "evolve_schema",
    "add_column",
    "rename_column",
    "add_vector_column",
    "backfill_vector_column",
    "now_ns",
    "hardware_info",
]

# Backward-compat re-export: ailake.TableWriter still works.
TableWriter = _TableWriter

# ── HTML helpers ──────────────────────────────────────────────────────────────

_CARD_STYLE = (
    "font-family:monospace;border:1px solid #ddd;border-radius:6px;"
    "padding:14px 16px;max-width:520px;background:#fafafa;"
    "box-shadow:0 1px 3px rgba(0,0,0,.06)"
)
_LABEL_STYLE = "color:#888;padding:3px 12px 3px 0;white-space:nowrap"
_VALUE_STYLE = "padding:3px 0;word-break:break-all"
_TH_STYLE = (
    "text-align:left;color:#6c757d;padding:4px 10px 4px 0;"
    "border-bottom:1px solid #e0e0e0;font-weight:normal;font-size:12px"
)
_TD_STYLE = "padding:3px 10px 3px 0;font-size:13px"


def _kv_rows(items: list[tuple[str, object]]) -> str:
    return "".join(
        f'<tr><td style="{_LABEL_STYLE}">{k}</td>'
        f'<td style="{_VALUE_STYLE}">{v}</td></tr>'
        for k, v in items
    )


# ── SearchQuery ───────────────────────────────────────────────────────────────

class SearchQuery:
    """Lazy, chainable search result.

    Execute by calling ``.to_list()``, ``.to_arrow()``, ``.to_pandas()``,
    ``.to_polars()``, or iterating over the object.

    When ``fetch_data=True``, ``.to_arrow()`` / ``.to_pandas()`` / ``.to_polars()``
    return full row data (all Parquet columns + ``_distance``) instead of pointer-only
    dicts.  ``.to_list()`` always returns ``[{row_id, distance, file}]`` regardless.
    """

    def __init__(
        self,
        path: str,
        query: list[float],
        top_k: int,
        fetch_data: bool = False,
        partition_filter: "str | None" = None,
        score_fn: "Callable[[float, Any], float] | None" = None,
        hybrid_text: "str | None" = None,
        text_column: str = "chunk_text",
        bm25_weight: float = 0.5,
        pruning_threshold: "float | None" = None,
        ef_search: "int | None" = None,
        rerank_factor: "int | None" = None,
    ) -> None:
        self._path = path
        self._query = query
        self._top_k = top_k
        self._fetch_data = fetch_data
        self._partition_filter = partition_filter
        self._score_fn = score_fn
        self._hybrid_text = hybrid_text
        self._text_column = text_column
        self._bm25_weight = bm25_weight
        self._pruning_threshold = pruning_threshold
        self._ef_search = ef_search
        self._rerank_factor = rerank_factor
        self._results: list[dict] | None = None      # lazy — pointer-only
        self._arrow_batch: Any | None = None          # lazy — full RecordBatch

    # ── chain ─────────────────────────────────────────────────────────────────

    def limit(self, n: int) -> "SearchQuery":
        """Cap results to *n* nearest neighbours."""
        self._top_k = n
        self._results = None
        self._arrow_batch = None
        return self

    # ── materialise ───────────────────────────────────────────────────────────

    def _execute(self) -> list[dict]:
        if self._results is None:
            self._results = _search_raw(
                self._path, self._query, self._top_k, self._partition_filter,
                self._hybrid_text, self._text_column, self._bm25_weight,
                self._pruning_threshold, self._ef_search, self._rerank_factor,
            )
        return self._results

    def _execute_arrow(self):
        if self._arrow_batch is None:
            import io
            import pyarrow as pa  # noqa: PLC0415
            ipc_bytes: bytes = _search_with_data(
                self._path, self._query, self._top_k, self._partition_filter,
                self._hybrid_text, self._text_column, self._bm25_weight,
                self._pruning_threshold, self._ef_search, self._rerank_factor,
            )
            table = pa.ipc.open_file(io.BytesIO(ipc_bytes)).read_all()
            if self._score_fn is not None:
                table = _apply_score_fn(table, self._score_fn)
            self._arrow_batch = table
        return self._arrow_batch

    def to_list(self) -> list[dict]:
        """Return ``[{row_id, distance, file}]`` — pointer-only, regardless of fetch_data."""
        return self._execute()

    def to_arrow(self):
        """Return results as a ``pyarrow.Table``.

        When ``fetch_data=True``: all Parquet columns + ``_distance`` (Float32).
        When ``fetch_data=False``: pointer-only table with ``row_id``, ``_distance``,
        ``file`` columns (no I/O beyond the index search).
        """
        if self._fetch_data:
            return self._execute_arrow()
        import pyarrow as pa  # noqa: PLC0415

        data = self._execute()
        return pa.table({
            "row_id":   [r["row_id"] for r in data],
            "distance": [r["distance"] for r in data],
            "file":     [r["file"] for r in data],
        })

    def to_pandas(self) -> "pd.DataFrame":
        """Return results as a ``pandas.DataFrame``.

        Full row data when ``fetch_data=True``, pointer-only otherwise.
        """
        if self._fetch_data:
            return self._execute_arrow().to_pandas()
        import pandas as pd  # noqa: PLC0415

        return pd.DataFrame(self._execute())

    def to_polars(self) -> "pl.DataFrame":
        """Return results as a ``polars.DataFrame``.

        Full row data when ``fetch_data=True``, pointer-only otherwise.
        """
        if self._fetch_data:
            import polars as pl  # noqa: PLC0415

            return pl.from_arrow(self._execute_arrow())  # type: ignore[return-value]
        import polars as pl  # noqa: PLC0415

        return pl.DataFrame(self._execute())

    # ── protocol ──────────────────────────────────────────────────────────────

    def __iter__(self) -> Iterable[dict]:
        return iter(self._execute())

    def __len__(self) -> int:
        return len(self._execute())

    def __repr__(self) -> str:
        if self._results is None and self._arrow_batch is None:
            return f"SearchQuery(top_k={self._top_k}, pending)"
        n = self._arrow_batch.num_rows if self._fetch_data and self._arrow_batch is not None \
            else len(self._results or [])
        return f"SearchQuery({n} results, top_k={self._top_k})"

    # ── async ─────────────────────────────────────────────────────────────────

    async def to_list_async(self) -> list[dict]:
        """Async variant of :meth:`to_list` — runs search in a thread executor."""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self._execute)

    async def to_arrow_async(self):
        """Async variant of :meth:`to_arrow`."""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self.to_arrow)

    async def to_pandas_async(self) -> "pd.DataFrame":
        """Async variant of :meth:`to_pandas`."""
        if self._fetch_data:
            loop = asyncio.get_running_loop()
            arrow = await loop.run_in_executor(None, self._execute_arrow)
            return arrow.to_pandas()
        import pandas as pd  # noqa: PLC0415

        return pd.DataFrame(await self.to_list_async())

    async def to_polars_async(self) -> "pl.DataFrame":
        """Async variant of :meth:`to_polars`."""
        if self._fetch_data:
            import polars as pl  # noqa: PLC0415

            loop = asyncio.get_running_loop()
            arrow = await loop.run_in_executor(None, self._execute_arrow)
            return pl.from_arrow(arrow)  # type: ignore[return-value]
        import polars as pl  # noqa: PLC0415

        return pl.DataFrame(await self.to_list_async())

    # ── display ───────────────────────────────────────────────────────────────

    def _repr_html_(self) -> str:
        pending = self._results is None and self._arrow_batch is None
        if pending:
            mode = "full-data" if self._fetch_data else "pointers"
            return (
                f'<span style="font-family:monospace;color:#888">'
                f"SearchQuery(top_k={self._top_k}, {mode}, <em>not yet executed</em>)"
                f"</span>"
            )

        # Full-data mode: render all columns from the Arrow batch.
        if self._fetch_data and self._arrow_batch is not None:
            batch = self._arrow_batch
            col_names = batch.schema.names
            header = "".join(f'<th style="{_TH_STYLE}">{c}</th>' for c in col_names)
            header = f'<tr><th style="{_TH_STYLE}">#</th>{header}</tr>'
            body_rows = []
            for i in range(batch.num_rows):
                cells = "".join(
                    f'<td style="{_TD_STYLE}">{batch.column(j)[i].as_py()}</td>'
                    for j in range(len(col_names))
                )
                body_rows.append(f'<tr><td style="{_TD_STYLE};color:#aaa">{i}</td>{cells}</tr>')
            body = "".join(body_rows)
            n = batch.num_rows
            label = f"SearchQuery — {n} result{'s' if n != 1 else ''} (full data)"
        else:
            rows = self._results or []
            header = (
                f'<tr>'
                f'<th style="{_TH_STYLE}">#</th>'
                f'<th style="{_TH_STYLE}">row_id</th>'
                f'<th style="{_TH_STYLE}">distance</th>'
                f'<th style="{_TH_STYLE}">file</th>'
                f'</tr>'
            )
            body = "".join(
                f'<tr>'
                f'<td style="{_TD_STYLE};color:#aaa">{i}</td>'
                f'<td style="{_TD_STYLE}">{r["row_id"]}</td>'
                f'<td style="{_TD_STYLE}">{r["distance"]:.6f}</td>'
                f'<td style="{_TD_STYLE};color:#555;font-size:11px">{r["file"]}</td>'
                f'</tr>'
                for i, r in enumerate(rows)
            )
            n = len(rows)
            label = f"SearchQuery — {n} result{'s' if n != 1 else ''}"

        return (
            f'<div style="{_CARD_STYLE}">'
            f'<div style="color:#6c757d;font-size:11px;margin-bottom:8px">{label}</div>'
            f'<table style="border-collapse:collapse;width:100%">'
            f"{header}{body}"
            f"</table>"
            f"</div>"
        )


# ── Table ─────────────────────────────────────────────────────────────────────

class Table:
    """Handle to an AI-Lake table supporting write and vector search."""

    def __init__(
        self,
        path: str,
        vector_column: str = "embedding",
        dim: int = 1536,
        metric: str = "cosine",
        precision: str = "f16",
        pre_normalize: bool = False,
        hnsw_m: int | None = None,
        hnsw_ef_construction: int | None = None,
        pq_only: bool = False,
        ivf_residual: bool = False,
        embedding_model: str | None = None,
        embedding_model_version: str | None = None,
        embed_fn: Optional[Callable[[list[str]], list[list[float]]]] = None,
        bm25_text_column: str | None = None,
        fts_text_columns: list[str] | None = None,
        fts_tokenizer: str = "default",
        partition_by: str | None = None,
        partition_value: str | None = None,
        partition_column_type: str | None = None,
        partition_fields: list[tuple[str, str, str]] | None = None,
        partition_values: dict[str, str] | None = None,
        format_version: int = 2,
    ) -> None:
        self._path = path
        self._vector_column = vector_column
        self._dim = dim
        self._metric = metric
        self._precision = precision
        self._pre_normalize = pre_normalize
        self._hnsw_m = hnsw_m
        self._hnsw_ef = hnsw_ef_construction
        self._pq_only = pq_only
        self._ivf_residual = ivf_residual
        self._embedding_model = embedding_model
        self._embedding_model_version = embedding_model_version
        self._embed_fn = embed_fn
        self._writer = _TableWriter(
            path,
            vector_column=vector_column,
            dim=dim,
            metric=metric,
            precision=precision,
            pre_normalize=pre_normalize,
            hnsw_m=hnsw_m,
            hnsw_ef_construction=hnsw_ef_construction,
            pq_only=pq_only,
            ivf_residual=ivf_residual,
            embedding_model=embedding_model,
            embedding_model_version=embedding_model_version,
            embed_fn=embed_fn,
            bm25_text_column=bm25_text_column,
            fts_text_columns=fts_text_columns,
            fts_tokenizer=fts_tokenizer,
            partition_by=partition_by,
            partition_value=partition_value,
            partition_column_type=partition_column_type,
            partition_fields=partition_fields,
            partition_values=partition_values,
            format_version=format_version,
        )

    # ── write ─────────────────────────────────────────────────────────────────

    def insert(
        self,
        texts: list[str],
        embeddings: Optional[_Embeddings] = None,
        extra_columns: Optional[dict[str, list]] = None,
    ) -> "Table":
        """Buffer a batch for writing.  Call ``commit()`` to persist.

        Args:
            texts: one string per row.
            embeddings: ``list[list[float]]`` or any array with a ``.tolist()``
                        method (numpy, torch, etc.).  May be omitted when
                        *embed_fn* was passed to ``__init__``.
            extra_columns: additional tabular columns, e.g.
                           ``{"id": [...], "category": [...]}``. Column type is
                           inferred from the first element (bool/float/int/str).
        """
        if embeddings is not None:
            _emb: list[list[float]] | None = (
                embeddings.tolist()  # type: ignore[union-attr]
                if hasattr(embeddings, "tolist")
                else [list(row) for row in embeddings]
            )
        else:
            _emb = None
        self._writer.write_batch(texts, _emb, extra_columns)
        return self

    def commit(self) -> int:
        """Persist all buffered batches as a new Iceberg snapshot.

        Returns the new snapshot id.
        """
        return self._writer.commit()

    def write_batch_auto_deferred(
        self,
        texts: list[str],
        embeddings: _Embeddings,
        extra_columns: Optional[dict[str, list]] = None,
    ) -> "Table":
        """Deferred-index write — Parquet persisted immediately (~200k vec/s).

        Selects IVF-PQ when a GPU or ≥8 CPU cores are detected and the batch
        has ≥5 000 vectors; falls back to HNSW otherwise.  Index is built in a
        background thread — shard is served via flat scan until the index is ready.

        Args:
            texts: one string per row.
            embeddings: ``list[list[float]]`` or any array with a ``.tolist()`` method.
            extra_columns: additional tabular columns — see :meth:`insert`.
        """
        _emb: list[list[float]] = (
            embeddings.tolist()  # type: ignore[union-attr]
            if hasattr(embeddings, "tolist")
            else [list(row) for row in embeddings]
        )
        self._writer.write_batch_auto_deferred(texts, _emb, extra_columns)
        return self

    async def write_batch_auto_deferred_async(
        self,
        texts: list[str],
        embeddings: _Embeddings,
        extra_columns: Optional[dict[str, list]] = None,
    ) -> "Table":
        """Async variant of :meth:`write_batch_auto_deferred`."""
        _emb: list[list[float]] = (
            embeddings.tolist()  # type: ignore[union-attr]
            if hasattr(embeddings, "tolist")
            else [list(row) for row in embeddings]
        )
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(
            None, self._writer.write_batch_auto_deferred, texts, _emb, extra_columns
        )
        return self

    async def insert_async(
        self,
        texts: list[str],
        embeddings: _Embeddings,
        extra_columns: Optional[dict[str, list]] = None,
    ) -> "Table":
        """Async variant of :meth:`insert` — runs write_batch in a thread executor."""
        _emb: list[list[float]] = (
            embeddings.tolist()  # type: ignore[union-attr]
            if hasattr(embeddings, "tolist")
            else [list(row) for row in embeddings]
        )
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, self._writer.write_batch, texts, _emb, extra_columns)
        return self

    async def commit_async(self) -> int:
        """Async variant of :meth:`commit`."""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self._writer.commit)

    def write_batch_idempotent(
        self,
        texts: list[str],
        embeddings: _Embeddings,
        batch_id: str,
        extra_columns: Optional[dict[str, list]] = None,
    ) -> "Table":
        """Idempotent write — no-op if *batch_id* was already committed.

        Safe for retried Airflow tasks / at-least-once pipelines: calling this
        twice with the same *batch_id* only writes the batch once.

        Args:
            texts: one string per row.
            embeddings: ``list[list[float]]`` or any array with a ``.tolist()`` method.
            batch_id: unique key for this batch (e.g. Airflow run_id + task_id).
            extra_columns: additional tabular columns — see :meth:`insert`.
        """
        _emb: list[list[float]] = (
            embeddings.tolist()  # type: ignore[union-attr]
            if hasattr(embeddings, "tolist")
            else [list(row) for row in embeddings]
        )
        self._writer.write_batch_idempotent(texts, _emb, batch_id, extra_columns)
        return self

    def write_batch_multi(
        self,
        texts: list[str],
        columns: list[tuple["VectorColSpec", _Embeddings]],
        extra_columns: Optional[dict[str, list]] = None,
    ) -> "Table":
        """Write a batch with N independent vector columns (Phase 8 multimodal).

        Each column gets its own HNSW index in the file footer. Use for
        multimodal tables where the same row has embeddings from different
        modalities (text + image, etc.).

        Args:
            texts: one string per row (primary tabular column).
            columns: ``[(VectorColSpec, embeddings), ...]`` — each tuple pairs a
                     column spec with its embeddings (``list[list[float]]`` or
                     array with ``.tolist()``); embedding-list length must equal
                     ``len(texts)`` for every entry.
            extra_columns: additional tabular columns — see :meth:`insert`.
        """
        _columns = [
            (spec, emb.tolist() if hasattr(emb, "tolist") else [list(row) for row in emb])
            for spec, emb in columns
        ]
        self._writer.write_batch_multi(texts, _columns, extra_columns)
        return self

    def write_batch_multi_deferred(
        self,
        texts: list[str],
        columns: list[tuple["VectorColSpec", _Embeddings]],
        extra_columns: Optional[dict[str, list]] = None,
    ) -> "Table":
        """Deferred variant of :meth:`write_batch_multi`.

        Persists Parquet immediately and builds all N column HNSW indexes in
        a background task — see the native ``write_batch_multi_deferred``
        docstring for the index-status lifecycle.
        """
        _columns = [
            (spec, emb.tolist() if hasattr(emb, "tolist") else [list(row) for row in emb])
            for spec, emb in columns
        ]
        self._writer.write_batch_multi_deferred(texts, _columns, extra_columns)
        return self

    # ── search ────────────────────────────────────────────────────────────────

    def search(
        self,
        query: _Vector,
        top_k: int = 10,
        fetch_data: bool = False,
        partition_filter: "str | None" = None,
        score_fn: "Callable[[float, Any], float] | None" = None,
        hybrid_text: "str | None" = None,
        text_column: str = "chunk_text",
        bm25_weight: float = 0.5,
        pruning_threshold: "float | None" = None,
        ef_search: "int | None" = None,
        rerank_factor: "int | None" = None,
    ) -> SearchQuery:
        """Return a chainable :class:`SearchQuery`.

        Args:
            query: embedding vector — ``list[float]`` or array with ``.tolist()``.
            top_k: maximum neighbours to return.
            fetch_data: when ``True``, ``.to_arrow()`` / ``.to_pandas()`` / ``.to_polars()``
                        return full row data (all Parquet columns + ``_distance``).
                        When ``False`` (default), only ``row_id``, ``distance``, and
                        ``file`` are returned — matches the original behaviour.
            partition_filter: optional partition value to restrict search (e.g. agent_id).
            score_fn: optional re-ranking callable ``(distance, row) -> float``. Requires ``fetch_data=True``.
            hybrid_text: optional BM25 query for hybrid RRF search.
            text_column: Parquet column used for BM25 scoring (default ``"chunk_text"``).
            bm25_weight: BM25 weight in RRF fusion (default ``0.5``).
            pruning_threshold: geometric pruning distance. Files whose centroid is farther
                               than this from the query are skipped. Default ``None`` = no pruning.
            ef_search: HNSW beam width override. Default ``None`` = 50.
            rerank_factor: when set, fetches ``top_k * rerank_factor`` HNSW candidates and
                           reranks with exact F32 distances before truncating — corrects PQ
                           approximation error on IVF-PQ-indexed tables. Default ``None`` = off.
        """
        _q: list[float] = (
            query.tolist()  # type: ignore[union-attr]
            if hasattr(query, "tolist")
            else list(query)
        )
        return SearchQuery(
            self._path, _q, top_k,
            fetch_data=fetch_data,
            partition_filter=partition_filter,
            score_fn=score_fn,
            hybrid_text=hybrid_text,
            text_column=text_column,
            bm25_weight=bm25_weight,
            pruning_threshold=pruning_threshold,
            ef_search=ef_search,
            rerank_factor=rerank_factor,
        )

    # ── context manager ───────────────────────────────────────────────────────

    def __enter__(self) -> "Table":
        return self

    def __exit__(self, *_) -> None:
        pass

    # ── display ───────────────────────────────────────────────────────────────

    def __repr__(self) -> str:
        return (
            f"Table(path={self._path!r}, "
            f"vector_column={self._vector_column!r}, "
            f"dim={self._dim}, metric={self._metric!r})"
        )

    def _repr_html_(self) -> str:
        hnsw_extra = ""
        if self._hnsw_m is not None:
            hnsw_extra += f"<tr><td style='{_LABEL_STYLE}'>hnsw_m</td><td style='{_VALUE_STYLE}'>{self._hnsw_m}</td></tr>"
        if self._hnsw_ef is not None:
            hnsw_extra += f"<tr><td style='{_LABEL_STYLE}'>hnsw_ef_construction</td><td style='{_VALUE_STYLE}'>{self._hnsw_ef}</td></tr>"

        rows = _kv_rows([
            ("vector_column", self._vector_column),
            ("dim", self._dim),
            ("metric", self._metric),
            ("pre_normalize", self._pre_normalize),
        ])

        return (
            f'<div style="{_CARD_STYLE}">'
            f'<div style="color:#6c757d;font-size:11px;margin-bottom:6px">AI-Lake Table</div>'
            f'<div style="font-weight:bold;margin-bottom:10px;word-break:break-all;font-size:14px">'
            f"{self._path}"
            f"</div>"
            f'<table style="border-collapse:collapse;width:100%">'
            f"{rows}{hnsw_extra}"
            f"</table>"
            f"</div>"
        )


def _apply_score_fn(table, score_fn):
    """Re-rank a pyarrow Table using a Python-level score_fn(distance, row) -> float.

    score_fn receives (distance: float, row: dict) where row maps column names to
    scalar Python values for that row. Returns float; lower = better (matches
    distance semantics). Table is re-sorted by the new score; ``_score`` column appended.
    """
    import pyarrow as pa  # noqa: PLC0415
    import pyarrow.compute as pc  # noqa: PLC0415

    n = table.num_rows
    col_names = table.schema.names
    scores = []
    dist_col = table.column("_distance")
    for i in range(n):
        dist = dist_col[i].as_py()
        row = {name: table.column(name)[i].as_py() for name in col_names}
        scores.append(score_fn(dist, row))
    score_array = pa.array(scores, type=pa.float32())
    table = table.append_column("_score", score_array)
    order = pc.sort_indices(table, sort_keys=[("_score", "ascending")])
    return table.take(order)


# ── module-level helpers ──────────────────────────────────────────────────────

def open_table(
    path: str,
    *,
    vector_column: str = "embedding",
    dim: int = 1536,
    metric: str = "cosine",
    precision: str = "f16",
    pre_normalize: bool = False,
    hnsw_m: int | None = None,
    hnsw_ef_construction: int | None = None,
    pq_only: bool = False,
    ivf_residual: bool = False,
    embedding_model: str | None = None,
    embedding_model_version: str | None = None,
    embed_fn: Optional[Callable[[list[str]], list[list[float]]]] = None,
    bm25_text_column: str | None = None,
    fts_text_columns: list[str] | None = None,
    fts_tokenizer: str = "default",
    partition_by: str | None = None,
    partition_value: str | None = None,
    partition_column_type: str | None = None,
    partition_fields: list[tuple[str, str, str]] | None = None,
    partition_values: dict[str, str] | None = None,
    format_version: int = 2,
) -> Table:
    """Open or create an AI-Lake table at *path*.

    Args:
        path: Local filesystem path or object-storage URI (``s3://``, ``gs://``, ``az://``).
        vector_column: Name of the embedding column (default ``"embedding"``).
        dim: Embedding dimension (default 1536).
        metric: Distance metric — ``"cosine"``, ``"euclidean"``, ``"dot_product"``,
                ``"normalized_cosine"``.
        precision: Storage precision — ``"f16"`` (default) | ``"f32"`` | ``"i8"``.
        pre_normalize: Normalise vectors to unit-L2 at write time (~12-20 % search speedup).
        hnsw_m: HNSW graph degree *M* per layer.
        hnsw_ef_construction: HNSW build-time beam width.
        embedding_model: Model identifier stored in ``ailake.embedding-model`` Iceberg
                         property (e.g. ``"text-embedding-3-small"``).
        embedding_model_version: Optional version tag (e.g. ``"2024-01"``).
        embed_fn: ``Callable[[list[str]], list[list[float]]]`` — auto-embed callable.
                  When set, ``insert(texts)`` may be called without *embeddings*.
        bm25_text_column: Column name for BM25 scoring (Phase 5 hybrid search).
        fts_text_columns: Columns to index with Tantivy FTS (Phase T).
        fts_tokenizer: Tokenizer for Tantivy FTS (default ``"default"``).
        partition_by: Single-column Iceberg identity partition (e.g. ``"agent_id"``).
                      Every row written through this ``Table`` instance is tagged with
                      *partition_value* — open a separate ``Table`` per distinct value.
        partition_value: The partition value for this ``Table`` instance when
                         *partition_by* (or *partition_fields*) is set.
        partition_column_type: Iceberg type of the *partition_by* column (e.g. ``"string"``).
        partition_fields: Multi-column partition spec as a list of
                          ``(column, transform, column_type)`` tuples, e.g.
                          ``[("agent_id", "identity", "string")]``. Defines the table's
                          partition schema; takes precedence over *partition_by* at
                          table creation.
        partition_values: ``{column: value}`` for this ``Table`` instance's multi-column
                          partition value, matching *partition_fields*. Ignored when
                          *partition_value* is also set.
        format_version: Iceberg format version — ``2`` (default) or ``3`` (deletion
                        vectors, row lineage).
    """
    return Table(
        path,
        vector_column=vector_column,
        dim=dim,
        metric=metric,
        precision=precision,
        pre_normalize=pre_normalize,
        hnsw_m=hnsw_m,
        hnsw_ef_construction=hnsw_ef_construction,
        pq_only=pq_only,
        ivf_residual=ivf_residual,
        embedding_model=embedding_model,
        embedding_model_version=embedding_model_version,
        embed_fn=embed_fn,
        bm25_text_column=bm25_text_column,
        fts_text_columns=fts_text_columns,
        fts_tokenizer=fts_tokenizer,
        partition_by=partition_by,
        partition_value=partition_value,
        partition_column_type=partition_column_type,
        partition_fields=partition_fields,
        partition_values=partition_values,
        format_version=format_version,
    )


# ── Agent ─────────────────────────────────────────────────────────────────────

class Agent:
    """High-level agent memory helper — Phase 9.

    Wraps :class:`TableWriter` + vector search + :func:`assemble_context` into
    a single interface designed for agent frameworks (LangChain, CrewAI, AutoGen).

    Episodic memories and tool-call records are stored as vector-indexed rows
    in a shared AI-Lake table. Recall uses **hybrid scoring** — the HNSW distance
    is modulated by exponential recency decay and an agent-assigned importance
    score, so recent and important memories rank above older, less-important ones
    even when semantically equidistant.

    Args:
        table_path: Local path or object-storage URI for the agent memory table.
        embed_fn:   ``Callable[[list[str]], list[list[float]]]`` — text → embedding.
                    Must return one embedding per input string.
        agent_id:   Stable UUID string identifying this agent instance.
                    Auto-generated if omitted (new UUID each session).
        session_id: UUID for the current conversation/task session.
                    Auto-generated if omitted.
        metric:     Distance metric — ``"cosine"`` (default), ``"euclidean"``,
                    ``"dot_product"``, ``"normalized_cosine"``.
        lambda_:    Exponential decay rate for recency. Default ``0.099``
                    (half-life ≈ 7 days). Use ``0.693`` for daily decay,
                    ``0.023`` for monthly decay.

    Example::

        import ailake, openai

        def embed(texts):
            resp = openai.embeddings.create(model="text-embedding-3-small", input=texts)
            return [d.embedding for d in resp.data]

        agent = ailake.Agent("s3://my-lake/agent-memory/", embed_fn=embed,
                             agent_id="agent-42")

        # Store memories
        agent.remember("User prefers concise responses", importance=0.9)
        agent.remember("Last session topic: Rust async patterns", importance=0.6)
        agent.log_tool_call("web_search", {"q": "Rust tokio docs"}, {"hits": 5})
        agent.commit()

        # Recall with hybrid scoring
        memories = agent.recall(embed(["async programming"])[0], top_k=5)
        for m in memories:
            print(f"[score={m['score']:.3f}] {m['text']}")

        # Assemble LLM context
        context_xml = agent.assemble_context(embed(["concise answers"])[0])
    """

    def __init__(
        self,
        table_path: str,
        embed_fn: Callable[[list[str]], list[list[float]]],
        agent_id: Optional[str] = None,
        session_id: Optional[str] = None,
        metric: str = "cosine",
        lambda_: float = 0.099,
    ) -> None:
        import uuid as _uuid
        self._table_path = table_path
        self._embed_fn = embed_fn
        self._agent_id = agent_id or str(_uuid.uuid4())
        self._session_id = session_id or str(_uuid.uuid4())
        self._metric = metric
        self._lambda = lambda_
        # Writer is lazily initialised on first write (dim unknown until first embedding).
        self._writer: Optional[_TableWriter] = None
        # Pending (text, embedding, meta) triples not yet written. `meta` holds
        # real typed columns (agent_id, mem_type, last_accessed_at, ...) written
        # via extra_columns — NOT packed into the text column.
        self._pending: list[tuple[str, list[float], dict]] = []
        self._step_counter = 0

    # ── internal ──────────────────────────────────────────────────────────────

    def _ensure_writer(self, dim: int) -> _TableWriter:
        if self._writer is None:
            self._writer = _TableWriter(
                self._table_path,
                vector_column="embedding",
                dim=dim,
                metric=self._metric,
                partition_by="agent_id",
                partition_value=self._agent_id,
            )
        return self._writer

    def _embed_one(self, text: str) -> list[float]:
        results = self._embed_fn([text])
        return results[0] if results else []

    def _next_step(self) -> int:
        step = self._step_counter
        self._step_counter += 1
        return step

    def _flush_pending(self) -> None:
        if not self._pending:
            return
        texts = [t for t, _, _ in self._pending]
        embs = [e for _, e, _ in self._pending]
        metas = [m for _, _, m in self._pending]
        dim = len(embs[0])
        writer = self._ensure_writer(dim)
        extra_columns = {key: [m[key] for m in metas] for key in metas[0]}
        writer.write_batch(texts, embs, extra_columns)
        self._pending.clear()

    # ── write ─────────────────────────────────────────────────────────────────

    def remember(self, text: str, importance: float = 1.0) -> str:
        """Embed *text* and buffer it as an episodic memory.

        Writes real typed columns (``agent_id``, ``session_id``, ``mem_type``,
        ``record_id``, ``importance``, ``created_at``, ``last_accessed_at``,
        ``access_count``) matching ``EpisodicMemorySchema`` — the table stays
        queryable by any AI-Lake client (Spark/Trino/Flink/DuckDB/CLI), not just
        this SDK, and :func:`decay_memories` (which requires a real
        ``last_accessed_at`` Timestamp column) works against it directly.

        Args:
            text:       Natural-language memory to store.
            importance: Agent-assigned salience in ``[0.0, 1.0]`` (default 1.0).
                        Higher values make this memory harder to displace in recall
                        even as it ages.

        Returns:
            ``mem_id`` — a UUID string that uniquely identifies this memory.

        Note:
            Call :meth:`commit` to persist buffered memories to storage.
        """
        import uuid as _uuid
        mem_id = str(_uuid.uuid4())
        now = TimestampNs(now_ns())
        meta = {
            "agent_id": self._agent_id,
            "session_id": self._session_id,
            "step_index": self._next_step(),
            "mem_type": "memory",
            "record_id": mem_id,
            "importance": float(importance),
            "created_at": now,
            "last_accessed_at": now,
            "access_count": 0,
            "tool_name": "",
            "tool_input_json": "",
            "tool_output_json": "",
            "outcome": "",
            "latency_ms": 0,
        }
        emb = self._embed_one(text)
        self._pending.append((text, emb, meta))
        return mem_id

    def log_tool_call(
        self,
        name: str,
        input: object,
        output: object,
        outcome: str = "success",
        latency_ms: int = 0,
        importance: float = 0.5,
    ) -> str:
        """Record a tool invocation in the agent memory table.

        The embedding is computed from ``"{name}: {input_json}"`` so that
        semantic search can find past calls by intent rather than just name.
        Writes real typed columns matching ``ToolCallSchema`` (``agent_id``,
        ``session_id``, ``step_index``, ``tool_name``, ``tool_input_json``,
        ``tool_output_json``, ``outcome``, ``latency_ms``).

        Args:
            name:        Tool name (e.g. ``"web_search"``, ``"code_exec"``).
            input:       Tool input — any JSON-serialisable value or string.
            output:      Tool output — any JSON-serialisable value or string.
            outcome:     ``"success"`` | ``"failure"`` | ``"timeout"``.
            latency_ms:  Wall-clock latency in milliseconds.
            importance:  Salience of this tool call for future recall (default 0.5).

        Returns:
            ``call_id`` — a UUID string identifying this tool call record.
        """
        import json
        import uuid as _uuid
        call_id = str(_uuid.uuid4())
        input_json = json.dumps(input) if not isinstance(input, str) else input
        output_json = json.dumps(output) if not isinstance(output, str) else output
        embed_text = f"{name}: {input_json}"
        now = TimestampNs(now_ns())
        meta = {
            "agent_id": self._agent_id,
            "session_id": self._session_id,
            "step_index": self._next_step(),
            "mem_type": "tool_call",
            "record_id": call_id,
            "importance": float(importance),
            "created_at": now,
            "last_accessed_at": now,
            "access_count": 0,
            "tool_name": name,
            "tool_input_json": input_json,
            "tool_output_json": output_json,
            "outcome": outcome,
            "latency_ms": int(latency_ms),
        }
        emb = self._embed_one(embed_text)
        self._pending.append((embed_text, emb, meta))
        return call_id

    def commit(self) -> int:
        """Persist all buffered memories and tool calls as a new Iceberg snapshot.

        Returns the new snapshot id, or 0 if nothing was pending.
        """
        if not self._pending and self._writer is None:
            return 0
        self._flush_pending()
        return self._writer.commit() if self._writer else 0

    # ── read ──────────────────────────────────────────────────────────────────

    def recall(
        self,
        query: _Vector,
        top_k: int = 10,
        oversample: int = 3,
    ) -> list[dict]:
        """Retrieve the *top_k* most relevant memories with hybrid scoring.

        HNSW distance is modulated by recency decay and importance:
        ``score = distance / (recency_weight × importance)``
        Lower score = better rank.

        Args:
            query:      Query embedding — ``list[float]`` or array with ``.tolist()``.
            top_k:      Number of results to return (default 10).
            oversample: Fetch ``top_k × oversample`` HNSW candidates before
                        re-ranking (default 3, i.e. 3× oversampling).

        Returns:
            List of dicts sorted by hybrid score (best first), each containing:
            ``text``, ``distance``, ``score``, ``recency``, ``importance``,
            ``type`` (``"memory"`` | ``"tool_call"``), ``mem_id`` / ``call_id``,
            ``agent_id``, ``session_id``, ``created_at`` (int, Unix epoch nanoseconds).
        """
        import io
        import math

        import pyarrow as pa  # noqa: PLC0415

        if isinstance(query, str):
            if self._embed_fn is None:
                raise ValueError(
                    "Agent.recall() received a text string but no embed_fn was provided. "
                    "Pass embed_fn to Agent.__init__() or pass a pre-computed vector."
                )
            q = list(self._embed_fn([query])[0])
        else:
            q = query.tolist() if hasattr(query, "tolist") else list(query)
        candidate_k = top_k * max(1, oversample)

        raw_ipc: bytes = _search_with_data(self._table_path, q, candidate_k, self._agent_id)
        batch = pa.ipc.open_file(io.BytesIO(raw_ipc)).read_all()

        col_names = set(batch.schema.names)

        def col(name: str):
            return batch.column(name) if name in col_names else None

        dist_col = col("_distance")
        text_col = col("text")
        importance_col = col("importance")
        last_accessed_col = col("last_accessed_at")
        created_col = col("created_at")
        mem_type_col = col("mem_type")
        record_id_col = col("record_id")
        agent_id_col = col("agent_id")
        session_id_col = col("session_id")
        tool_name_col = col("tool_name")
        tool_input_col = col("tool_input_json")
        tool_output_col = col("tool_output_json")
        outcome_col = col("outcome")
        latency_col = col("latency_ms")

        now = now_ns()

        scored: list[dict] = []
        for i in range(batch.num_rows):
            dist = float(dist_col[i].as_py()) if dist_col is not None else 1.0
            text = str(text_col[i].as_py()) if text_col is not None else ""
            importance = float(importance_col[i].as_py()) if importance_col is not None else 1.0

            # Read raw ns via `.value` — TimestampScalar.as_py() at nanosecond
            # resolution requires pandas to be installed; `.value` doesn't.
            last_accessed_ns = last_accessed_col[i].value if last_accessed_col is not None else None
            days_since = (
                max(0.0, (now - last_accessed_ns) / 86_400_000_000_000.0)
                if last_accessed_ns is not None
                else 0.0
            )
            recency = math.exp(-self._lambda * days_since)
            denom = max(recency * importance, 1e-7)
            score = dist / denom

            rtype = str(mem_type_col[i].as_py()) if mem_type_col is not None else "memory"
            entry: dict = {
                "text": text,
                "distance": dist,
                "score": score,
                "recency": recency,
                "importance": importance,
                "type": rtype,
                "agent_id": agent_id_col[i].as_py() if agent_id_col is not None else None,
                "session_id": session_id_col[i].as_py() if session_id_col is not None else None,
                "created_at": created_col[i].value if created_col is not None else None,
            }
            if rtype == "tool_call":
                entry["call_id"] = record_id_col[i].as_py() if record_id_col is not None else None
                entry["tool_name"] = tool_name_col[i].as_py() if tool_name_col is not None else None
                entry["tool_input_json"] = (
                    tool_input_col[i].as_py() if tool_input_col is not None else None
                )
                entry["tool_output_json"] = (
                    tool_output_col[i].as_py() if tool_output_col is not None else None
                )
                entry["outcome"] = outcome_col[i].as_py() if outcome_col is not None else None
                entry["latency_ms"] = latency_col[i].as_py() if latency_col is not None else None
            else:
                entry["mem_id"] = record_id_col[i].as_py() if record_id_col is not None else None

            scored.append(entry)

        scored.sort(key=lambda x: x["score"])
        return scored[:top_k]

    def assemble_context(self, query: _Vector, max_tokens: int = 4096) -> str:
        """Retrieve memories and assemble structured XML context for an LLM.

        Calls :meth:`recall` with ``top_k=20`` and formats the results using
        the same :func:`assemble_context` pipeline as the rest of the SDK
        (deduplication, token-budget enforcement, XML structure).

        Args:
            query:      Query embedding (same space as memories).
            max_tokens: Token budget for the context block (default 4096).

        Returns:
            XML string suitable for inclusion in a Claude / GPT-4 prompt.
        """
        memories = self.recall(query, top_k=20)
        chunks = [
            {
                "document_id": m.get("mem_id") or m.get("call_id") or f"mem-{i}",
                "chunk_index": i,
                "chunk_text": m["text"],
                "distance": m["distance"],
                "document_title": (
                    f"Tool: {m.get('tool_name', 'unknown')} [{m.get('outcome', '')}]"
                    if m.get("type") == "tool_call"
                    else f"Memory (importance={m['importance']:.2f})"
                ),
                "source_uri": self._table_path,
            }
            for i, m in enumerate(memories)
        ]
        return assemble_context(chunks, max_tokens=max_tokens)["text"]

    # ── async variants ─────────────────────────────────────────────────────────

    async def remember_async(self, text: str, importance: float = 1.0) -> str:
        """Async variant of :meth:`remember`."""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self.remember, text, importance)

    async def recall_async(self, query: _Vector, top_k: int = 10) -> list[dict]:
        """Async variant of :meth:`recall`."""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self.recall, query, top_k)

    async def commit_async(self) -> int:
        """Async variant of :meth:`commit`."""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self.commit)

    # ── context manager ───────────────────────────────────────────────────────

    def __enter__(self) -> "Agent":
        return self

    def __exit__(self, *_) -> None:
        self.commit()

    # ── display ───────────────────────────────────────────────────────────────

    def __repr__(self) -> str:
        pending = len(self._pending)
        return (
            f"Agent(table={self._table_path!r}, "
            f"agent_id={self._agent_id!r}, "
            f"session_id={self._session_id!r}, "
            f"pending={pending})"
        )

    def _repr_html_(self) -> str:
        rows = _kv_rows([
            ("agent_id", self._agent_id),
            ("session_id", self._session_id),
            ("table", self._table_path),
            ("metric", self._metric),
            ("lambda (decay)", f"{self._lambda:.4f} (half-life ≈ {0.693/self._lambda:.1f} days)"),
            ("pending writes", len(self._pending)),
        ])
        return (
            f'<div style="{_CARD_STYLE}">'
            f'<div style="color:#6c757d;font-size:11px;margin-bottom:6px">ailake.Agent — Phase 9</div>'
            f'<table style="border-collapse:collapse;width:100%">{rows}</table>'
            f"</div>"
        )


# ── module-level helpers ──────────────────────────────────────────────────────

def assemble_context(
    chunks: list[dict],
    max_tokens: int = 4096,
    dedup_threshold: float = 0.05,
    group_by_document: bool = True,
    max_chunks_per_document: int = 10,
) -> dict:
    """Assemble a list of text chunks into structured XML context for LLM input.

    Args:
        chunks: list of dicts with keys ``document_id``, ``chunk_index``, ``chunk_text``,
                and optional ``document_title``, ``section_path``, ``source_uri``,
                ``distance``, ``embedding`` (``list[float]`` — enables cosine-distance
                near-duplicate removal via *dedup_threshold*; chunks without an
                ``"embedding"`` key are never deduplicated, matching the underlying
                Rust default when no embedding is supplied).
        max_tokens: token budget (4 chars ≈ 1 token).
        dedup_threshold: cosine distance below which two chunks (that both carry an
                         ``"embedding"``) are considered duplicates and one is dropped.
        group_by_document: group and sort chunks by ``document_id``/``chunk_index``
                           before rendering (default ``True``).
        max_chunks_per_document: cap chunks per document group (default 10).

    Returns:
        ``{"text": str, "chunk_count": int, "token_estimate": int}``.

    Example::

        ctx = ailake.assemble_context(chunks, max_tokens=2048,
                                       dedup_threshold=0.05)
        print(ctx["text"])           # XML ready for LLM input
        print(ctx["chunk_count"])    # how many chunks made it in
    """
    return _assemble_context_raw(
        chunks,
        max_tokens=max_tokens,
        dedup_threshold=dedup_threshold,
        group_by_document=group_by_document,
        max_chunks_per_document=max_chunks_per_document,
    )


def evolve_schema(
    path: str,
    *,
    add_columns: "list[dict] | None" = None,
    rename_columns: "list[dict] | None" = None,
) -> int:
    """Apply schema evolution to an AI-Lake table without rewriting data files.

    Combines :func:`add_column` and :func:`rename_column` into a single call.
    Each operation is applied in order; the final schema-id is returned.

    Args:
        path: Table root path or URI.
        add_columns: Columns to add.  Each entry must have ``"name"`` and
            ``"type"`` keys (Iceberg type string: ``"string"``, ``"int"``,
            ``"long"``, ``"float"``, ``"double"``, ``"boolean"``).
            Optional keys: ``"initial_default"`` (Python scalar — ``None``,
            ``0``, ``0.0``, ``"unknown"``), ``"doc"`` (string).
        rename_columns: Columns to rename.  Each entry must have ``"from"``
            and ``"to"`` keys.

    Returns:
        New schema-id (int), or ``0`` when both lists are empty (no-op).
        Returns ``-1`` when the operation could not be parsed from the output.

    Example::

        ailake.evolve_schema(
            "s3://my-lake/docs/",
            add_columns=[{"name": "score", "type": "float", "initial_default": 0.0}],
            rename_columns=[{"from": "old_text", "to": "chunk_text"}],
        )
    """
    schema_id = 0
    for ac in (add_columns or []):
        schema_id = add_column(
            path,
            ac["name"],
            ac["type"],
            ac.get("required", False),
            ac.get("initial_default"),
            ac.get("write_default"),
            ac.get("doc"),
        )
    for rc in (rename_columns or []):
        schema_id = rename_column(path, rc["from"], rc["to"])
    return schema_id


def search(
    path: str,
    query: _Vector,
    top_k: int = 10,
    fetch_data: bool = False,
    partition_filter: "str | None" = None,
    score_fn: "Callable[[float, Any], float] | None" = None,
    hybrid_text: "str | None" = None,
    text_column: str = "chunk_text",
    bm25_weight: float = 0.5,
    pruning_threshold: "float | None" = None,
    ef_search: "int | None" = None,
    rerank_factor: "int | None" = None,
) -> SearchQuery:
    """Module-level search returning a chainable :class:`SearchQuery`.

    Args:
        path: Table root path or URI.
        query: Query embedding — ``list[float]`` or array with ``.tolist()``.
        top_k: Maximum neighbours to return (default 10).
        fetch_data: When ``True``, ``.to_arrow()`` / ``.to_pandas()`` / ``.to_polars()``
                    return full row data (all Parquet columns + ``_distance``).
                    When ``False`` (default), only ``row_id``, ``distance``, ``file``.
        partition_filter: Optional partition value to restrict search (e.g. agent_id).
                          Pruned at manifest level — no files from other partitions opened.
        score_fn: Optional Python callable ``(distance: float, row: pyarrow.RecordBatch) -> float``
                  applied post-search to re-rank results. Requires ``fetch_data=True``.
                  Note: not applied during GPU deferred-build window (SearchSession flat-scan).
        hybrid_text: Optional text query for BM25 hybrid search. When set, HNSW retrieves
                     a larger candidate pool, BM25 scores each candidate, and results are
                     fused via RRF. Requires ``TableWriter(bm25_text_column=...)`` at write time.
        text_column: Parquet column containing document text for BM25 scoring
                     (default ``"chunk_text"``).
        bm25_weight: Weight for BM25 signal in RRF fusion — ``0.0`` = pure vector,
                     ``1.0`` = pure BM25 (default ``0.5``).
        pruning_threshold: Geometric pruning distance. Files whose centroid is more than
                           this distance from the query are skipped entirely. ``None`` (default)
                           disables pruning (scans all files). Set to a small value (e.g. ``0.5``)
                           to skip distant shards for a significant latency win on large tables.

    Example::

        # Pointer-only (default — backward-compatible)
        results = ailake.search("s3://my-lake/docs/", query_vec, top_k=20)
        df = results.to_pandas()  # columns: row_id, distance, file

        # Hybrid BM25+vector search
        results = ailake.search(
            "s3://my-lake/docs/", query_vec, top_k=20,
            hybrid_text="rust async programming", bm25_weight=0.4,
        )

        # Full row data with partition isolation
        results = ailake.search(
            "s3://my-lake/docs/", query_vec, top_k=20,
            fetch_data=True, partition_filter="agent-A",
        )
        df = results.to_pandas()  # columns: id, text, embedding, ..., _distance

        # Custom scoring (recency × distance)
        def hybrid_score(dist, row):
            recency = row.column("recency_weight")[0].as_py()
            return dist / (recency + 1e-6)

        results = ailake.search(
            "s3://my-lake/docs/", query_vec, top_k=20,
            fetch_data=True, score_fn=hybrid_score,
        )
    """
    _q: list[float] = (
        query.tolist()  # type: ignore[union-attr]
        if hasattr(query, "tolist")
        else list(query)
    )
    return SearchQuery(
        path, _q, top_k,
        fetch_data=fetch_data,
        partition_filter=partition_filter,
        score_fn=score_fn,
        hybrid_text=hybrid_text,
        text_column=text_column,
        bm25_weight=bm25_weight,
        pruning_threshold=pruning_threshold,
        ef_search=ef_search,
        rerank_factor=rerank_factor,
    )


def compact(
    path: str,
    *,
    min_files: int = 4,
    target_size_bytes: int = 536_870_912,
    max_files_per_pass: int = 20,
    deferred: bool = False,
) -> dict:
    """Compact small files in an AI-Lake table into a larger merged file.

    Native binding — calls ``ailake_query::compaction`` directly (no
    external ``ailake`` CLI binary required). Reads table metadata from
    ``path``, selects files smaller than ``target_size_bytes``, merges them
    into a single file with a rebuilt HNSW/IVF-PQ index, and commits the
    result as a new Iceberg snapshot.

    Args:
        path: Table root path or URI (same value passed to :class:`TableWriter`).
        min_files: Minimum number of eligible files required to trigger
                   compaction (default 4). No-op when fewer files qualify.
        target_size_bytes: Files smaller than this are candidates for merge
                           (default 512 MiB, matching ``ailake compact``'s CLI default).
        max_files_per_pass: Maximum files merged in one pass (default 20).
                            Bounds peak RAM and HNSW rebuild cost.
        deferred: When ``True``, writes the merged Parquet immediately and
                  builds the HNSW index in the background (~200k vec/s write
                  throughput). When ``False`` (default), blocks until the
                  index is fully built.

    Returns:
        ``{"ok": True, "files_compacted": N, "output_path": "..."}`` or
        ``{"ok": True, "files_compacted": 0, "output_path": None}`` when nothing to compact.

    Example::

        result = ailake.compact("s3://my-lake/docs/", min_files=5)
        print(result)  # {"ok": True, "files_compacted": 1, "output_path": "data/compacted-..."}
    """
    return _compact_native(
        path,
        min_files=min_files,
        target_size_bytes=target_size_bytes,
        max_files_per_pass=max_files_per_pass,
        deferred=deferred,
    )
