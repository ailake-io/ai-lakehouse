# SPDX-License-Identifier: MIT OR Apache-2.0
"""Embedding backends for the AI-Lake Airbyte destination."""

from __future__ import annotations

import json
import subprocess
from typing import Protocol

import numpy as np


class Embedder(Protocol):
    def embed(self, texts: list[str]) -> np.ndarray:
        """Return shape (len(texts), dim) float32 array."""
        ...


class CmdEmbedder:
    """Delegates embedding to an external process.

    Protocol: process receives a JSON array of strings on stdin,
    writes a JSON array of float arrays to stdout.
    """

    def __init__(self, cmd: str) -> None:
        self._cmd = cmd

    def embed(self, texts: list[str]) -> np.ndarray:
        proc = subprocess.run(
            self._cmd,
            shell=True,
            input=json.dumps(texts).encode(),
            capture_output=True,
            timeout=300,
        )
        if proc.returncode != 0:
            raise RuntimeError(
                f"embed_cmd failed (exit {proc.returncode}): {proc.stderr.decode()[:500]}"
            )
        vecs = json.loads(proc.stdout)
        return np.array(vecs, dtype=np.float32)


class OpenAIEmbedder:
    """OpenAI Embeddings API."""

    def __init__(self, api_key: str, model: str, base_url: str = "") -> None:
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError(
                "openai package required: pip install airbyte-destination-ailake[openai]"
            )
        kwargs: dict = {"api_key": api_key}
        if base_url:
            kwargs["base_url"] = base_url
        self._client = OpenAI(**kwargs)
        self._model = model

    def embed(self, texts: list[str]) -> np.ndarray:
        response = self._client.embeddings.create(input=texts, model=self._model)
        vecs = [item.embedding for item in response.data]
        return np.array(vecs, dtype=np.float32)


class CohereEmbedder:
    """Cohere Embed API."""

    def __init__(self, api_key: str, model: str, input_type: str = "search_document") -> None:
        try:
            import cohere
        except ImportError:
            raise ImportError(
                "cohere package required: pip install airbyte-destination-ailake[cohere]"
            )
        self._client = cohere.Client(api_key)
        self._model = model
        self._input_type = input_type

    def embed(self, texts: list[str]) -> np.ndarray:
        response = self._client.embed(
            texts=texts,
            model=self._model,
            input_type=self._input_type,
        )
        return np.array(response.embeddings, dtype=np.float32)


def build_embedder(cfg: "AilakeDestinationConfig") -> Embedder:  # noqa: F821 — forward ref
    from airbyte_destination_ailake.config import AilakeDestinationConfig

    if cfg.embed_mode == "cmd":
        return CmdEmbedder(cfg.embed_cmd)
    if cfg.embed_mode == "openai":
        return OpenAIEmbedder(
            api_key=cfg.openai_api_key,
            model=cfg.openai_model,
            base_url=cfg.openai_base_url,
        )
    if cfg.embed_mode == "cohere":
        return CohereEmbedder(
            api_key=cfg.cohere_api_key,
            model=cfg.cohere_model,
            input_type=cfg.cohere_input_type,
        )
    raise ValueError(f"Unknown embed_mode: {cfg.embed_mode}")
