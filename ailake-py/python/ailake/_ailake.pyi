# SPDX-License-Identifier: MIT OR Apache-2.0
# Copyright (c) 2026 Thiago Egon Lange
# Stubs for the compiled Rust extension ailake._ailake.
# This file is the authoritative type source for type checkers and IDEs.

from typing import Optional, Sequence

class TableWriter:
    """Python-facing table writer.  Wraps ``ailake_query::TableWriter``."""

    def __init__(
        self,
        path: str,
        vector_column: str = "embedding",
        dim: int = 1536,
        metric: str = "cosine",
        pre_normalize: bool = False,
        hnsw_m: Optional[int] = None,
        hnsw_ef_construction: Optional[int] = None,
    ) -> None:
        """Open or create an AI-Lake table at *path*.

        Args:
            path: Local filesystem path or ``s3://`` / ``gs://`` / ``az://`` URI.
            vector_column: Name of the embedding column (default ``"embedding"``).
            dim: Embedding dimension (default 1536).
            metric: Distance metric — one of ``"cosine"``, ``"euclidean"``,
                    ``"dot_product"``, ``"normalized_cosine"``.
            pre_normalize: Normalise vectors to unit-L2 at write time for a
                           ~12-20 % speedup on cosine search (default ``False``).
            hnsw_m: HNSW graph degree *M* per layer.  ``None`` uses the
                    per-table default stored in Iceberg metadata.
            hnsw_ef_construction: HNSW build-time beam width.  ``None`` uses
                                  the per-table default.
        """
        ...

    def write_batch(
        self,
        texts: Sequence[str],
        embeddings: Sequence[Sequence[float]],
    ) -> None:
        """Buffer a batch of rows.  Call :meth:`commit` to persist.

        Args:
            texts: One string per row.
            embeddings: One embedding (list of floats) per row; length must
                        match *texts*.
        """
        ...

    def write_batch_idempotent(
        self,
        texts: Sequence[str],
        embeddings: Sequence[Sequence[float]],
        batch_id: str,
    ) -> None:
        """Idempotent write — no-op if *batch_id* was already committed.

        Args:
            texts: One string per row.
            embeddings: One embedding per row.
            batch_id: Unique key for this batch (e.g. Airflow ``run_id + task_id``).
        """
        ...

    def commit(self) -> int:
        """Persist all buffered batches as a new Iceberg snapshot.

        Returns:
            The new snapshot id (positive integer).
        """
        ...


class SearchResult(dict):
    """Individual search result.  A plain ``dict`` with guaranteed keys."""
    row_id: int
    distance: float
    file: str


def search(
    path: str,
    query: Sequence[float],
    top_k: int = 10,
) -> list[dict[str, object]]:
    """Search a table for the top-*k* nearest vectors to *query*.

    Args:
        path: Table root — same value used when writing.
        query: Query embedding as a flat list of floats.
        top_k: Number of neighbours to return (default 10).

    Returns:
        List of dicts with keys ``row_id`` (int), ``distance`` (float),
        ``file`` (str — absolute path / URI of the Parquet file).
    """
    ...


def assemble_context(
    chunks: list[dict[str, object]],
    max_tokens: int = 4096,
    dedup_threshold: float = 0.05,
) -> str:
    """Assemble chunks into structured XML context for LLM input.

    Args:
        chunks: List of dicts with keys ``document_id`` (str),
                ``chunk_index`` (int), ``chunk_text`` (str), and optional
                ``document_title``, ``section_path``, ``source_uri``,
                ``distance``.
        max_tokens: Token budget — 4 chars ≈ 1 token (default 4096).
        dedup_threshold: Cosine distance below which near-duplicate chunks
                         are deduplicated (default 0.05).

    Returns:
        XML string ready to pass to an LLM as context.
    """
    ...
