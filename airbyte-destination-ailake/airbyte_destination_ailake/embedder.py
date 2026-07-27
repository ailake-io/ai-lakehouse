# SPDX-License-Identifier: MIT OR Apache-2.0
"""Embedding backends for the AI-Lake Airbyte destination."""

from __future__ import annotations

import json
import shlex
import subprocess
import urllib.error
import urllib.request
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
            shlex.split(self._cmd),
            shell=False,
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


class FastEmbedEmbedder:
    """Local ONNX embeddings via fastembed — no PyTorch, no API key.

    Lightweight (no torch dependency), CPU-friendly, good default for demos
    and CI. Model runs entirely in-process, downloaded once and cached.
    """

    def __init__(self, model: str = "BAAI/bge-small-en-v1.5") -> None:
        try:
            from fastembed import TextEmbedding
        except ImportError:
            raise ImportError(
                "fastembed package required: pip install airbyte-destination-ailake[fastembed]"
            )
        self._model = TextEmbedding(model_name=model)

    def embed(self, texts: list[str]) -> np.ndarray:
        return np.array(list(self._model.embed(texts)), dtype=np.float32)


class SentenceTransformersEmbedder:
    """Local embeddings via sentence-transformers — no API key.

    Widest model selection (any Hugging Face sentence-embedding model),
    PyTorch-based, supports GPU via `device`.
    """

    def __init__(self, model: str = "BAAI/bge-small-en-v1.5", device: str = "") -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            raise ImportError(
                "sentence-transformers package required: "
                "pip install airbyte-destination-ailake[sentence-transformers]"
            )
        kwargs: dict = {"device": device} if device else {}
        self._model = SentenceTransformer(model, **kwargs)

    def embed(self, texts: list[str]) -> np.ndarray:
        vecs = self._model.encode(texts, convert_to_numpy=True, normalize_embeddings=False)
        return vecs.astype(np.float32)


class HttpEmbedder:
    """OpenAI-compatible HTTP embedding endpoint.

    Protocol (request)::

        POST {url}
        Authorization: {auth_header}          # omitted if empty
        Content-Type: application/json

        {"model": "{model}", "input": ["text1", "text2", ...]}

    Protocol (response)::

        {"data": [{"embedding": [...]}, ...]}

    Compatible with: Ollama (``/v1/embeddings``), vLLM, LM Studio,
    Together.ai, Anyscale, Azure OpenAI, any OpenAI-compatible server.
    """

    def __init__(self, url: str, model: str = "", auth_header: str = "", timeout: int = 60) -> None:
        self._url = url
        self._model = model
        self._auth_header = auth_header
        self._timeout = timeout

    def embed(self, texts: list[str]) -> np.ndarray:
        body: dict = {"input": texts}
        if self._model:
            body["model"] = self._model

        data = json.dumps(body).encode()
        headers = {"Content-Type": "application/json", "Accept": "application/json"}
        if self._auth_header:
            headers["Authorization"] = self._auth_header

        req = urllib.request.Request(self._url, data=data, headers=headers, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=self._timeout) as resp:
                payload = json.loads(resp.read())
        except urllib.error.HTTPError as exc:
            body_preview = exc.read(500).decode(errors="replace")
            raise RuntimeError(
                f"HTTP embedder: {exc.code} {exc.reason} — {body_preview}"
            ) from exc

        try:
            vecs = [item["embedding"] for item in payload["data"]]
        except (KeyError, TypeError) as exc:
            raise RuntimeError(
                f"HTTP embedder: unexpected response shape. "
                f"Expected {{\"data\": [{{\"embedding\": [...]}}]}}, got: {str(payload)[:200]}"
            ) from exc

        return np.array(vecs, dtype=np.float32)


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
    if cfg.embed_mode == "http":
        return HttpEmbedder(
            url=cfg.http_url,
            model=cfg.http_model,
            auth_header=cfg.http_auth_header,
            timeout=cfg.http_timeout,
        )
    if cfg.embed_mode == "fastembed":
        return FastEmbedEmbedder(model=cfg.fastembed_model)
    if cfg.embed_mode == "sentence_transformers":
        return SentenceTransformersEmbedder(
            model=cfg.sentence_transformers_model,
            device=cfg.sentence_transformers_device,
        )
    raise ValueError(f"Unknown embed_mode: {cfg.embed_mode}")
