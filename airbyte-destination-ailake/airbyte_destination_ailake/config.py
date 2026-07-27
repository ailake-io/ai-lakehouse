# SPDX-License-Identifier: MIT OR Apache-2.0
"""Configuration schema and validation for the AI-Lake Airbyte destination."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal


@dataclass
class AilakeDestinationConfig:
    """Typed config from the Airbyte connector spec.

    Populated by ``AilakeDestination._parse_config(raw)``.
    """

    # --- Table storage ---
    table_base_path: str
    """Base path for all tables. Each stream lands at ``{table_base_path}/{stream_name}/``."""

    # --- Embedding ---
    embed_mode: Literal["cmd", "openai", "cohere", "http", "fastembed", "sentence_transformers"]
    """How to produce embeddings from text. One of: ``cmd``, ``openai``, ``cohere``, ``http``,
    ``fastembed``, ``sentence_transformers``."""

    embedding_dim: int = 1536
    """Vector dimensionality — must match the model output."""

    embedding_metric: str = "cosine"
    """Distance metric: ``cosine``, ``euclidean``, ``dot_product``, ``normalized_cosine``."""

    embedding_model: str = ""
    """Model identifier stored in ``ailake.embedding-model`` Iceberg property."""

    embedding_model_version: str = ""
    """Optional version suffix. Stored as ``<model>@<version>``."""

    # --- Text extraction ---
    text_field: str = "content"
    """Record field whose value is the text to embed. Nested paths: ``meta.body``."""

    # --- Extra tabular columns ---
    extra_columns: list[str] = field(default_factory=list)
    """Record fields (besides ``text_field``) written as queryable Parquet columns
    alongside the embedding. Empty (default) = every scalar (str/int/float/bool)
    top-level field the record has, other than ``text_field``. Non-scalar values
    (nested objects/arrays) are never included — flatten them upstream if needed."""

    # --- cmd mode ---
    embed_cmd: str = ""
    """Shell command. Receives JSON array of strings on stdin; writes JSON array of
    float arrays to stdout. Example: ``python embed.py``."""

    # --- openai mode ---
    openai_api_key: str = ""
    openai_model: str = "text-embedding-3-small"
    openai_base_url: str = ""

    # --- cohere mode ---
    cohere_api_key: str = ""
    cohere_model: str = "embed-english-v3.0"
    cohere_input_type: str = "search_document"

    # --- http mode (OpenAI-compatible endpoint) ---
    http_url: str = ""
    """POST endpoint. Body: ``{"model": "...", "input": [...]}``; response: ``{"data": [{"embedding": [...]}]}``."""

    http_model: str = ""
    """Model name sent in the request body. Leave empty to omit the field."""

    http_auth_header: str = ""
    """Value of the ``Authorization`` header. Example: ``Bearer sk-...``."""

    http_timeout: int = 60
    """Request timeout in seconds."""

    # --- fastembed mode (local ONNX, no API key) ---
    fastembed_model: str = "BAAI/bge-small-en-v1.5"
    """fastembed model name. Runs in-process, no API key, no PyTorch dependency."""

    # --- sentence_transformers mode (local PyTorch, no API key) ---
    sentence_transformers_model: str = "BAAI/bge-small-en-v1.5"
    """Any Hugging Face sentence-embedding model name."""

    sentence_transformers_device: str = ""
    """``"cpu"``/``"cuda"``/etc. Empty = sentence-transformers auto-detects."""

    # --- Write behaviour ---
    batch_size: int = 512
    """Records per embed call and per ``TableWriter.write_batch()`` call."""

    pre_normalize: bool = False
    """Normalize vectors to unit L2 at write time (recommended for cosine)."""

    pq_only: bool = False
    """Discard raw F16 vectors after index build — max compression, no reranking."""

    # --- Agent partitioning (Phase 9) ---
    partition_by: str = ""
    """Iceberg identity partition column (e.g. ``agent_id``). Empty = no partitioning."""

    # --- Multi-column partition specs + format version (Phase Q) ---
    partition_fields: list[dict[str, Any]] = field(default_factory=list)
    """Multi-column partition spec (Phase K).  Each entry must have ``column``,
    ``transform`` (``"identity"`` or ``"truncate[W]"``), and ``column_type``.
    When non-empty, takes precedence over ``partition_by`` at table creation."""

    format_version: int = 2
    """Iceberg format version.  ``2`` (default, full compatibility) or ``3``
    (V3 tables: deletion vectors, row lineage, variant type)."""

    # --- Tantivy FTS (Phase T) ---
    fts_columns: list[str] = field(default_factory=list)
    """Text columns to index with Tantivy FTS. Empty = no FTS (zero overhead).
    Example: ``["chunk_text", "document_title"]``."""

    fts_tokenizer: str = "default"
    """Tantivy tokenizer name.  ``"default"`` (whitespace + lowercase) in most cases."""

    # --- HNSW tuning ---
    hnsw_m: int | None = None
    """HNSW M parameter (graph connectivity). ``None`` = use table default."""

    hnsw_ef_construction: int | None = None
    """HNSW ef_construction. ``None`` = use table default."""

    # --- Async index build ---
    deferred: bool = False
    """Build HNSW/IVF-PQ index asynchronously. Parquet committed immediately (~200k vec/s);
    index attached in background. ``IndexStatus`` transitions ``Indexing → Ready``."""

    @classmethod
    def from_dict(cls, raw: dict) -> "AilakeDestinationConfig":
        embed_mode = raw.get("embed_mode", "cmd")
        valid_modes = ("cmd", "openai", "cohere", "http", "fastembed", "sentence_transformers")
        if embed_mode not in valid_modes:
            raise ValueError(
                f"embed_mode must be one of {'/'.join(valid_modes)}, got '{embed_mode}'"
            )
        return cls(
            table_base_path=raw["table_base_path"].rstrip("/"),
            embed_mode=embed_mode,
            embedding_dim=int(raw.get("embedding_dim", 1536)),
            embedding_metric=raw.get("embedding_metric", "cosine"),
            embedding_model=raw.get("embedding_model", ""),
            embedding_model_version=raw.get("embedding_model_version", ""),
            text_field=raw.get("text_field", "content"),
            extra_columns=raw.get("extra_columns", []),
            embed_cmd=raw.get("embed_cmd", ""),
            openai_api_key=raw.get("openai_api_key", ""),
            openai_model=raw.get("openai_model", "text-embedding-3-small"),
            openai_base_url=raw.get("openai_base_url", ""),
            cohere_api_key=raw.get("cohere_api_key", ""),
            cohere_model=raw.get("cohere_model", "embed-english-v3.0"),
            cohere_input_type=raw.get("cohere_input_type", "search_document"),
            http_url=raw.get("http_url", ""),
            http_model=raw.get("http_model", ""),
            http_auth_header=raw.get("http_auth_header", ""),
            http_timeout=int(raw.get("http_timeout", 60)),
            fastembed_model=raw.get("fastembed_model", "BAAI/bge-small-en-v1.5"),
            sentence_transformers_model=raw.get("sentence_transformers_model", "BAAI/bge-small-en-v1.5"),
            sentence_transformers_device=raw.get("sentence_transformers_device", ""),
            batch_size=int(raw.get("batch_size", 512)),
            pre_normalize=bool(raw.get("pre_normalize", False)),
            pq_only=bool(raw.get("pq_only", False)),
            partition_by=raw.get("partition_by", ""),
            partition_fields=raw.get("partition_fields", []),
            format_version=int(raw.get("format_version", 2)),
            fts_columns=raw.get("fts_columns", []),
            fts_tokenizer=raw.get("fts_tokenizer", "default"),
            hnsw_m=int(raw["hnsw_m"]) if raw.get("hnsw_m") is not None else None,
            hnsw_ef_construction=int(raw["hnsw_ef_construction"]) if raw.get("hnsw_ef_construction") is not None else None,
            deferred=bool(raw.get("deferred", False)),
        )

    def validate(self) -> list[str]:
        errors: list[str] = []
        if not self.table_base_path:
            errors.append("table_base_path is required")
        if self.embed_mode == "cmd" and not self.embed_cmd:
            errors.append("embed_cmd is required when embed_mode=cmd")
        if self.embed_mode == "openai" and not self.openai_api_key:
            errors.append("openai_api_key is required when embed_mode=openai")
        if self.embed_mode == "cohere" and not self.cohere_api_key:
            errors.append("cohere_api_key is required when embed_mode=cohere")
        if self.embed_mode == "http" and not self.http_url:
            errors.append("http_url is required when embed_mode=http")
        if self.embedding_dim <= 0:
            errors.append(f"embedding_dim must be > 0, got {self.embedding_dim}")
        if self.batch_size <= 0:
            errors.append(f"batch_size must be > 0, got {self.batch_size}")
        if self.format_version not in (2, 3):
            errors.append(f"format_version must be 2 or 3, got {self.format_version}")
        for i, pf in enumerate(self.partition_fields):
            for key in ("column", "transform", "column_type"):
                if not pf.get(key):
                    errors.append(
                        f"partition_fields[{i}] missing required key '{key}'"
                    )
        return errors

    def table_path(self, stream_name: str) -> str:
        return f"{self.table_base_path}/{stream_name}"
