# SPDX-License-Identifier: MIT OR Apache-2.0
# Copyright (c) 2026 Thiago Egon Lange
"""AI-Lake Python SDK — fluent API over the Rust core."""

from __future__ import annotations

from typing import TYPE_CHECKING, Iterable, Sequence, Union

from ailake._ailake import (  # type: ignore[import]
    TableWriter as _TableWriter,
    assemble_context,
    search as _search_raw,
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
    "Table",
    "SearchQuery",
    "assemble_context",
]


# ── SearchQuery ───────────────────────────────────────────────────────────────

class SearchQuery:
    """Lazy, chainable search result.

    Execute by calling ``.to_list()``, ``.to_pandas()``, ``.to_polars()``,
    or iterating over the object.
    """

    def __init__(self, path: str, query: list[float], top_k: int) -> None:
        self._path = path
        self._query = query
        self._top_k = top_k
        self._results: list[dict] | None = None

    # ── chain ─────────────────────────────────────────────────────────────────

    def limit(self, n: int) -> "SearchQuery":
        """Cap results to *n* nearest neighbours."""
        self._top_k = n
        self._results = None  # invalidate cache if already executed
        return self

    # ── materialise ───────────────────────────────────────────────────────────

    def _execute(self) -> list[dict]:
        if self._results is None:
            self._results = _search_raw(self._path, self._query, self._top_k)
        return self._results

    def to_list(self) -> list[dict]:
        """Return results as ``list[dict]`` with keys row_id, distance, file."""
        return self._execute()

    def to_pandas(self) -> "pd.DataFrame":
        """Return results as a ``pandas.DataFrame``."""
        import pandas as pd  # noqa: PLC0415

        return pd.DataFrame(self._execute())

    def to_polars(self) -> "pl.DataFrame":
        """Return results as a ``polars.DataFrame``."""
        import polars as pl  # noqa: PLC0415

        return pl.DataFrame(self._execute())

    # ── protocol ──────────────────────────────────────────────────────────────

    def __iter__(self) -> Iterable[dict]:
        return iter(self._execute())

    def __len__(self) -> int:
        return len(self._execute())

    def __repr__(self) -> str:
        if self._results is None:
            return f"SearchQuery(top_k={self._top_k}, not yet executed)"
        return f"SearchQuery({len(self._results)} results, top_k={self._top_k})"


# ── Table ─────────────────────────────────────────────────────────────────────

class Table:
    """Handle to an AI-Lake table supporting write and vector search."""

    def __init__(self, path: str, **kwargs) -> None:
        self._path = path
        self._writer = _TableWriter(path, **kwargs)

    # ── write ─────────────────────────────────────────────────────────────────

    def insert(
        self,
        texts: list[str],
        embeddings: _Embeddings,
    ) -> "Table":
        """Buffer a batch for writing.  Call ``commit()`` to persist.

        Args:
            texts: one string per row.
            embeddings: ``list[list[float]]`` or any array with a ``.tolist()``
                        method (numpy, torch, etc.).
        """
        _emb: list[list[float]] = (
            embeddings.tolist()  # type: ignore[union-attr]
            if hasattr(embeddings, "tolist")
            else [list(row) for row in embeddings]
        )
        self._writer.write_batch(texts, _emb)
        return self

    def commit(self) -> int:
        """Persist all buffered batches as a new Iceberg snapshot.

        Returns the new snapshot id.
        """
        return self._writer.commit()

    # ── search ────────────────────────────────────────────────────────────────

    def search(self, query: _Vector, top_k: int = 10) -> SearchQuery:
        """Return a chainable :class:`SearchQuery`.

        Args:
            query: embedding vector — ``list[float]`` or array with ``.tolist()``.
            top_k: maximum neighbours to return.
        """
        _q: list[float] = (
            query.tolist()  # type: ignore[union-attr]
            if hasattr(query, "tolist")
            else list(query)
        )
        return SearchQuery(self._path, _q, top_k)

    # ── context manager ───────────────────────────────────────────────────────

    def __enter__(self) -> "Table":
        return self

    def __exit__(self, *_) -> None:
        pass

    def __repr__(self) -> str:
        return f"Table({self._path!r})"


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
) -> Table:
    """Open or create an AI-Lake table at *path*.

    Keyword arguments are forwarded to :class:`TableWriter`:
    ``vector_column``, ``dim``, ``metric``, ``pre_normalize``,
    ``hnsw_m``, ``hnsw_ef_construction``.
    """
    kwargs = dict(
        vector_column=vector_column,
        dim=dim,
        metric=metric,
        pre_normalize=pre_normalize,
        hnsw_m=hnsw_m,
        hnsw_ef_construction=hnsw_ef_construction,
    )
    return Table(path, **{k: v for k, v in kwargs.items() if v is not None or k in ("pre_normalize",)})


def search(path: str, query: _Vector, top_k: int = 10) -> SearchQuery:
    """Module-level search returning a chainable :class:`SearchQuery`.

    Example::

        results = ailake.search("s3://my-lake/docs/", query_vec, top_k=20)
        df = results.to_pandas()
    """
    _q: list[float] = (
        query.tolist()  # type: ignore[union-attr]
        if hasattr(query, "tolist")
        else list(query)
    )
    return SearchQuery(path, _q, top_k)
