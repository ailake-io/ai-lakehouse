# SPDX-License-Identifier: MIT OR Apache-2.0
# Copyright (c) 2026 Thiago Egon Lange
# Stubs for the compiled Rust extension ailake._ailake.
# This file is the authoritative type source for type checkers and IDEs.

from typing import Any, Callable, Optional, Sequence, Union


class VectorColSpec:
    """Specification for one vector column in a multimodal write or search.

    Args:
        column: Vector column name (e.g. ``"embedding"``, ``"image_embedding"``).
        dim: Dimensionality of vectors in this column.
        metric: ``"cosine"`` | ``"euclidean"`` | ``"dot_product"`` | ``"normalized_cosine"``.
        modality: Optional tag — ``"text"`` | ``"image"`` | ``"audio"`` | ``"video"``.
                  Stored as ``ailake.modality-<column>`` in Iceberg properties.
    """

    column: str
    dim: int
    metric: str
    modality: Optional[str]

    def __init__(
        self,
        column: str,
        dim: int,
        metric: str = "cosine",
        modality: Optional[str] = None,
    ) -> None: ...

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
        partition_by: Optional[str] = None,
        partition_value: Optional[str] = None,
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

    def write_batch_multi(
        self,
        texts: Sequence[str],
        columns: Sequence[tuple["VectorColSpec", Sequence[Sequence[float]]]],
    ) -> None:
        """Write a batch with N independent vector columns.

        Each column gets its own HNSW index in the AILK section of the file footer.
        Use this for multimodal tables where the same row has embeddings from
        different models or modalities (e.g. text + image).

        Args:
            texts: One string per row (primary tabular column).
            columns: List of ``(VectorColSpec, embeddings)`` tuples.
                     Each embedding list must have the same length as *texts*.
                     The first column determines the table-level HNSW policy.

        Example::

            text_spec  = ailake.VectorColSpec("embedding",       1536, "cosine", "text")
            image_spec = ailake.VectorColSpec("image_embedding",  512, "cosine", "image")
            writer.write_batch_multi(
                texts,
                [(text_spec, text_embs), (image_spec, image_embs)],
            )
        """
        ...

    def write_batch_multi_deferred(
        self,
        texts: Sequence[str],
        columns: Sequence[tuple["VectorColSpec", Sequence[Sequence[float]]]],
    ) -> None:
        """Deferred variant of ``write_batch_multi``.

        Persists Parquet immediately and builds all N column HNSW indexes in a
        background task. During the build window, search is served via flat scan
        (exact, GPU-accelerated when available). Transitions to HNSW-indexed
        search automatically once ``IndexStatus`` becomes ``Ready``.

        Use when ingest throughput matters more than immediate HNSW availability.

        Args:
            texts: One string per row (primary tabular column).
            columns: List of ``(VectorColSpec, embeddings)`` tuples — same format
                     as :meth:`write_batch_multi`.

        Example::

            text_spec  = ailake.VectorColSpec("embedding",       1536, "cosine", "text")
            image_spec = ailake.VectorColSpec("image_embedding",  512, "cosine", "image")
            writer.write_batch_multi_deferred(
                texts,
                [(text_spec, text_embs), (image_spec, image_embs)],
            )
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
    partition_value: Optional[str] = None,
) -> bytes:
    """Search and return full row data serialized as Arrow IPC bytes.

    Deserialize in Python with::

        import io, pyarrow as pa
        table = pa.ipc.open_file(io.BytesIO(search_with_data(...))).read_all()

    Args:
        path: Table root — same value used when writing.
        query: Query embedding as a flat list of floats.
        top_k: Number of neighbours to return (default 10).
        partition_value: When set, only files tagged with this partition value are
                         searched (manifest-level pruning). Pass ``agent_id`` for
                         per-agent isolated search without post-scan filtering.

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


def search_multimodal(
    path: str,
    queries: Sequence[tuple[str, Sequence[float], float]],
    top_k: int = 10,
    dim: Optional[int] = None,
) -> list[dict[str, object]]:
    """Cross-modal search: fuse results from N vector columns via Reciprocal Rank Fusion.

    Runs an independent HNSW search for each ``(column, query, weight)`` triple,
    then fuses ranked lists using RRF: ``score = Σ weight_i / (60 + rank_i)``.

    Args:
        path: Table root — same value used when writing.
        queries: List of ``(column_name, query_vec, weight)`` tuples.
                 *weight* is the relative importance of each column (1.0 = equal).
                 Typical: ``0.7`` for text, ``0.3`` for image.
        top_k: Number of fused results to return (default 10).
        dim: Vector dimension.  Auto-detected from Iceberg metadata when ``None``.

    Returns:
        List of dicts with keys ``row_id`` (int), ``rrf_score`` (float, higher = better),
        ``file`` (str).  Ordered by descending ``rrf_score``.

    Example::

        results = ailake.search_multimodal(
            "s3://my-lake/media/",
            queries=[
                ("embedding",       text_vec,  0.7),
                ("image_embedding", image_vec, 0.3),
            ],
            top_k=20,
        )
    """
    ...


# ── Agent (Phase 9) ────────────────────────────────────────────────────────────

_Vector = Union[Sequence[float], Any]  # list[float] or numpy/torch array with .tolist()

class Agent:
    """High-level agent memory helper — Phase 9.

    Wraps ``TableWriter`` + vector search + ``assemble_context`` for agent
    frameworks (LangChain, CrewAI, AutoGen).

    Args:
        table_path: Local path or object-storage URI for the memory table.
        embed_fn:   ``Callable[[list[str]], list[list[float]]]``.
        agent_id:   Stable UUID string (auto-generated if omitted).
        session_id: Current session UUID (auto-generated if omitted).
        metric:     Distance metric (default ``"cosine"``).
        lambda_:    Recency decay rate (default 0.099 ≈ weekly half-life).
    """

    def __init__(
        self,
        table_path: str,
        embed_fn: Callable[[list[str]], list[list[float]]],
        agent_id: Optional[str] = None,
        session_id: Optional[str] = None,
        metric: str = "cosine",
        lambda_: float = 0.099,
    ) -> None: ...

    @property
    def agent_id(self) -> str: ...

    @property
    def session_id(self) -> str: ...

    def remember(self, text: str, importance: float = 1.0) -> str:
        """Buffer *text* as an episodic memory.  Returns ``mem_id`` UUID.

        Call :meth:`commit` to persist.
        """
        ...

    def log_tool_call(
        self,
        name: str,
        input: object,
        output: object,
        outcome: str = "success",
        latency_ms: int = 0,
        importance: float = 0.5,
    ) -> str:
        """Buffer a tool-call record.  Returns ``call_id`` UUID.

        Call :meth:`commit` to persist.
        """
        ...

    def commit(self) -> int:
        """Persist buffered records as a new Iceberg snapshot.  Returns snapshot id."""
        ...

    def recall(
        self,
        query: _Vector,
        top_k: int = 10,
        oversample: int = 3,
    ) -> list[dict]:
        """Retrieve *top_k* memories with hybrid scoring.

        Uses manifest-level partition pruning: only files written by this agent
        (tagged with ``partition_value=agent_id``) are searched — no post-scan filter.

        Returns list of dicts sorted by hybrid score (lower = better), each with:
        ``text``, ``distance``, ``score``, ``recency``, ``importance``,
        ``type`` (``"memory"`` or ``"tool_call"``), ``agent_id``, ``session_id``,
        ``created_at``, and type-specific fields (``mem_id`` or ``call_id``,
        ``tool_name``, ``tool_input_json``, ``tool_output_json``, ``outcome``).
        """
        ...

    def assemble_context(self, query: _Vector, max_tokens: int = 4096) -> str:
        """Recall memories and format as XML context for an LLM.

        Returns XML string ready for inclusion in a Claude / GPT-4 prompt.
        """
        ...

    async def remember_async(self, text: str, importance: float = 1.0) -> str: ...
    async def recall_async(self, query: _Vector, top_k: int = 10) -> list[dict]: ...
    async def commit_async(self) -> int: ...

    def __enter__(self) -> "Agent": ...
    def __exit__(self, *_: Any) -> None: ...
