# SPDX-License-Identifier: MIT OR Apache-2.0
# Copyright (c) 2026 Thiago Egon Lange
"""AI-Lake Python SDK — fluent API over the Rust core."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Callable, Iterable, Optional, Sequence, Union

from ailake._ailake import (  # type: ignore[import]
    TableWriter as _TableWriter,
    VectorColSpec,
    assemble_context,
    migrate_embeddings,
    search as _search_raw,
    search_multimodal,
    search_with_data as _search_with_data,
)

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
    "search_multimodal",
    "Table",
    "SearchQuery",
    "TableWriter",
    "VectorColSpec",
    "assemble_context",
    "migrate_embeddings",
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

    def __init__(self, path: str, query: list[float], top_k: int, fetch_data: bool = False) -> None:
        self._path = path
        self._query = query
        self._top_k = top_k
        self._fetch_data = fetch_data
        self._results: list[dict] | None = None      # lazy — pointer-only
        self._arrow_batch = None                      # lazy — full RecordBatch

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
            self._results = _search_raw(self._path, self._query, self._top_k)
        return self._results

    def _execute_arrow(self):
        if self._arrow_batch is None:
            import io
            import pyarrow as pa  # noqa: PLC0415
            ipc_bytes: bytes = _search_with_data(self._path, self._query, self._top_k)
            self._arrow_batch = pa.ipc.open_file(io.BytesIO(ipc_bytes)).read_all()
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
        pre_normalize: bool = False,
        hnsw_m: int | None = None,
        hnsw_ef_construction: int | None = None,
        pq_only: bool = False,
        ivf_residual: bool = False,
        embedding_model: str | None = None,
        embedding_model_version: str | None = None,
        embed_fn: Optional[Callable[[list[str]], list[list[float]]]] = None,
    ) -> None:
        self._path = path
        self._vector_column = vector_column
        self._dim = dim
        self._metric = metric
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
            pre_normalize=pre_normalize,
            hnsw_m=hnsw_m,
            hnsw_ef_construction=hnsw_ef_construction,
            pq_only=pq_only,
            ivf_residual=ivf_residual,
            embedding_model=embedding_model,
            embedding_model_version=embedding_model_version,
            embed_fn=embed_fn,
        )

    # ── write ─────────────────────────────────────────────────────────────────

    def insert(
        self,
        texts: list[str],
        embeddings: Optional[_Embeddings] = None,
    ) -> "Table":
        """Buffer a batch for writing.  Call ``commit()`` to persist.

        Args:
            texts: one string per row.
            embeddings: ``list[list[float]]`` or any array with a ``.tolist()``
                        method (numpy, torch, etc.).  May be omitted when
                        *embed_fn* was passed to ``__init__``.
        """
        if embeddings is not None:
            _emb: list[list[float]] | None = (
                embeddings.tolist()  # type: ignore[union-attr]
                if hasattr(embeddings, "tolist")
                else [list(row) for row in embeddings]
            )
        else:
            _emb = None
        self._writer.write_batch(texts, _emb)
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
    ) -> "Table":
        """Deferred-index write — Parquet persisted immediately (~200k vec/s).

        Selects IVF-PQ when a GPU or ≥8 CPU cores are detected and the batch
        has ≥5 000 vectors; falls back to HNSW otherwise.  Index is built in a
        background thread — shard is served via flat scan until the index is ready.

        Args:
            texts: one string per row.
            embeddings: ``list[list[float]]`` or any array with a ``.tolist()`` method.
        """
        _emb: list[list[float]] = (
            embeddings.tolist()  # type: ignore[union-attr]
            if hasattr(embeddings, "tolist")
            else [list(row) for row in embeddings]
        )
        self._writer.write_batch_auto_deferred(texts, _emb)
        return self

    async def write_batch_auto_deferred_async(
        self,
        texts: list[str],
        embeddings: _Embeddings,
    ) -> "Table":
        """Async variant of :meth:`write_batch_auto_deferred`."""
        _emb: list[list[float]] = (
            embeddings.tolist()  # type: ignore[union-attr]
            if hasattr(embeddings, "tolist")
            else [list(row) for row in embeddings]
        )
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, self._writer.write_batch_auto_deferred, texts, _emb)
        return self

    async def insert_async(
        self,
        texts: list[str],
        embeddings: _Embeddings,
    ) -> "Table":
        """Async variant of :meth:`insert` — runs write_batch in a thread executor."""
        _emb: list[list[float]] = (
            embeddings.tolist()  # type: ignore[union-attr]
            if hasattr(embeddings, "tolist")
            else [list(row) for row in embeddings]
        )
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, self._writer.write_batch, texts, _emb)
        return self

    async def commit_async(self) -> int:
        """Async variant of :meth:`commit`."""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self._writer.commit)

    # ── search ────────────────────────────────────────────────────────────────

    def search(self, query: _Vector, top_k: int = 10, fetch_data: bool = False) -> SearchQuery:
        """Return a chainable :class:`SearchQuery`.

        Args:
            query: embedding vector — ``list[float]`` or array with ``.tolist()``.
            top_k: maximum neighbours to return.
            fetch_data: when ``True``, ``.to_arrow()`` / ``.to_pandas()`` / ``.to_polars()``
                        return full row data (all Parquet columns + ``_distance``).
                        When ``False`` (default), only ``row_id``, ``distance``, and
                        ``file`` are returned — matches the original behaviour.
        """
        _q: list[float] = (
            query.tolist()  # type: ignore[union-attr]
            if hasattr(query, "tolist")
            else list(query)
        )
        return SearchQuery(self._path, _q, top_k, fetch_data=fetch_data)

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


# ── module-level helpers ──────────────────────────────────────────────────────

def open_table(
    path: str,
    *,
    vector_column: str = "embedding",
    dim: int = 1536,
    metric: str = "cosine",
    pre_normalize: bool = False,
    hnsw_m: int | None = None,
    hnsw_ef_construction: int | None = None,
    pq_only: bool = False,
    ivf_residual: bool = False,
    embedding_model: str | None = None,
    embedding_model_version: str | None = None,
    embed_fn: Optional[Callable[[list[str]], list[list[float]]]] = None,
) -> Table:
    """Open or create an AI-Lake table at *path*.

    Args:
        path: Local filesystem path or object-storage URI (``s3://``, ``gs://``, ``az://``).
        vector_column: Name of the embedding column (default ``"embedding"``).
        dim: Embedding dimension (default 1536).
        metric: Distance metric — ``"cosine"``, ``"euclidean"``, ``"dot_product"``,
                ``"normalized_cosine"``.
        pre_normalize: Normalise vectors to unit-L2 at write time (~12-20 % search speedup).
        hnsw_m: HNSW graph degree *M* per layer.
        hnsw_ef_construction: HNSW build-time beam width.
        embedding_model: Model identifier stored in ``ailake.embedding-model`` Iceberg
                         property (e.g. ``"text-embedding-3-small"``).
        embedding_model_version: Optional version tag (e.g. ``"2024-01"``).
        embed_fn: ``Callable[[list[str]], list[list[float]]]`` — auto-embed callable.
                  When set, ``insert(texts)`` may be called without *embeddings*.
    """
    return Table(
        path,
        vector_column=vector_column,
        dim=dim,
        metric=metric,
        pre_normalize=pre_normalize,
        hnsw_m=hnsw_m,
        hnsw_ef_construction=hnsw_ef_construction,
        pq_only=pq_only,
        ivf_residual=ivf_residual,
        embedding_model=embedding_model,
        embedding_model_version=embedding_model_version,
        embed_fn=embed_fn,
    )


def search(path: str, query: _Vector, top_k: int = 10, fetch_data: bool = False) -> SearchQuery:
    """Module-level search returning a chainable :class:`SearchQuery`.

    Args:
        path: Table root path or URI.
        query: Query embedding — ``list[float]`` or array with ``.tolist()``.
        top_k: Maximum neighbours to return (default 10).
        fetch_data: When ``True``, ``.to_arrow()`` / ``.to_pandas()`` / ``.to_polars()``
                    return full row data (all Parquet columns + ``_distance``).
                    When ``False`` (default), only ``row_id``, ``distance``, ``file``.

    Example::

        # Pointer-only (default — backward-compatible)
        results = ailake.search("s3://my-lake/docs/", query_vec, top_k=20)
        df = results.to_pandas()  # columns: row_id, distance, file

        # Full row data
        results = ailake.search("s3://my-lake/docs/", query_vec, top_k=20, fetch_data=True)
        df = results.to_pandas()  # columns: id, text, embedding, ..., _distance
    """
    _q: list[float] = (
        query.tolist()  # type: ignore[union-attr]
        if hasattr(query, "tolist")
        else list(query)
    )
    return SearchQuery(path, _q, top_k, fetch_data=fetch_data)
