# SPDX-License-Identifier: MIT OR Apache-2.0
# Copyright (c) 2026 Thiago Egon Lange
# Stubs for the compiled Rust extension ailake._ailake.
# This file is the authoritative type source for type checkers and IDEs.

from typing import Callable, Optional, Sequence

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
        pq_only: bool = False,
        ivf_residual: bool = False,
        embedding_model: Optional[str] = None,
        embedding_model_version: Optional[str] = None,
        embed_fn: Optional[Callable[[list[str]], list[list[float]]]] = None,
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
            pq_only: When ``True``, only PQ-compressed codes are stored —
                     raw F16 vectors are discarded after index build.  Saves
                     ~95-99 % vector storage at the cost of no exact reranking.
                     Default ``False`` (keep raw for reranking).
            ivf_residual: When ``True``, IVF-PQ encodes residuals from each
                          cluster centroid rather than raw vectors.  Improves
                          recall@10 by ~2-4 pp at the same PQ budget.
                          Default ``False``.
            embedding_model: Human-readable model identifier stored in Iceberg
                             properties as ``ailake.embedding-model`` (e.g.
                             ``"text-embedding-3-small"``).  Used to detect
                             incompatible model changes at write time.
                             Default ``None`` (no model tracking).
            embedding_model_version: Optional version tag appended to
                                     *embedding_model* (e.g. ``"2024-01"``).
                                     Stored as ``"<name>@<version>"``.
        """
        ...

    def write_batch(
        self,
        texts: Sequence[str],
        embeddings: Optional[Sequence[Sequence[float]]] = None,
    ) -> None:
        """Buffer a batch of rows.  Call :meth:`commit` to persist.

        Args:
            texts: One string per row.
            embeddings: One embedding (list of floats) per row; length must
                        match *texts*.  May be omitted when *embed_fn* was
                        passed to :meth:`__init__` — embeddings are generated
                        automatically.
        """
        ...

    def write_batch_auto_deferred(
        self,
        texts: Sequence[str],
        embeddings: Sequence[Sequence[float]],
    ) -> None:
        """Deferred-index write — Parquet persisted immediately (~200k vec/s).

        Selects IVF-PQ when a GPU or ≥8 CPU cores are detected and the batch
        has ≥5 000 vectors; falls back to HNSW otherwise.  Index is built in a
        background thread — shard is served via flat scan until the index is ready.

        Args:
            texts: One string per row.
            embeddings: One embedding (list of floats) per row.
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


def search_with_data(
    path: str,
    query: Sequence[float],
    top_k: int = 10,
) -> bytes:
    """Search and return full row data serialized as Arrow IPC bytes.

    Deserialize in Python with::

        import io, pyarrow as pa
        table = pa.ipc.open_file(io.BytesIO(search_with_data(...))).read_all()

    Args:
        path: Table root — same value used when writing.
        query: Query embedding as a flat list of floats.
        top_k: Number of neighbours to return (default 10).

    Returns:
        Arrow IPC file-format bytes.  Deserialize to a ``pyarrow.Table``
        containing all Parquet columns plus ``_distance: float32``,
        ordered by ascending distance.
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


def migrate_embeddings(
    path: str,
    old_column: str,
    new_column: str,
    embed_fn: Callable[[list[str]], list[list[float]]],
    text_column: str = "chunk_text",
    strategy: str = "dual_write_then_cutover",
    batch_size: int = 512,
    new_model: Optional[str] = None,
    new_model_version: Optional[str] = None,
    on_progress: Optional[Callable[..., None]] = None,
) -> None:
    """Migrate an embedding column to a new model.

    Reads all chunks from *path*, re-embeds them via *embed_fn*, and writes
    new files with the updated embedding column.  Commits an Iceberg snapshot
    when done.

    Args:
        path: Table root path or URI — same value used when writing.
        old_column: Name of the existing embedding column (e.g. ``"embedding"``).
        new_column: Name for the migrated column (e.g. ``"embedding_v2"``).
                    May equal *old_column* for an in-place model upgrade.
        embed_fn: ``Callable[[list[str]], list[list[float]]]`` — your embedding
                  model.  Called in batches of *batch_size* texts.
        text_column: Parquet column that holds the raw text (default
                     ``"chunk_text"``).
        strategy: ``"atomic_replace"`` — replace each file one at a time
                  (lower peak storage, brief mixed-model window); or
                  ``"dual_write_then_cutover"`` — write all new files first,
                  then atomically swap (2× peak storage, zero downtime).
                  Default ``"dual_write_then_cutover"``.
        batch_size: Number of texts per *embed_fn* call (default 512).
        new_model: Model identifier stored in ``ailake.embedding-model`` after
                   migration (e.g. ``"text-embedding-3-small"``).
        new_model_version: Optional version tag (e.g. ``"2024-01"``).
        on_progress: Optional ``Callable`` receiving keyword args
                     ``files_done`` (int), ``files_total`` (int),
                     ``rows_migrated`` (int) after each file completes.

    Raises:
        ValueError: On invalid strategy, missing text column, or embed_fn error.
    """
    ...
