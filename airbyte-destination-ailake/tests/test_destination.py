# SPDX-License-Identifier: MIT OR Apache-2.0
"""Unit tests for airbyte-destination-ailake — all AI-Lake I/O mocked."""

from __future__ import annotations

import json
import sys
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from airbyte_destination_ailake.config import AilakeDestinationConfig
from airbyte_destination_ailake.embedder import CmdEmbedder
from airbyte_destination_ailake.writer import StreamWriter, _extract_text


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


class TestAilakeDestinationConfig:
    def test_from_dict_minimal(self):
        cfg = AilakeDestinationConfig.from_dict(
            {"table_base_path": "s3://bucket/prefix", "embed_mode": "openai", "openai_api_key": "sk-x"}
        )
        assert cfg.table_base_path == "s3://bucket/prefix"
        assert cfg.embed_mode == "openai"
        assert cfg.embedding_dim == 1536
        assert cfg.batch_size == 512

    def test_trailing_slash_stripped(self):
        cfg = AilakeDestinationConfig.from_dict(
            {"table_base_path": "s3://bucket/prefix/", "embed_mode": "cmd", "embed_cmd": "echo []"}
        )
        assert not cfg.table_base_path.endswith("/")

    def test_invalid_embed_mode_raises(self):
        with pytest.raises(ValueError, match="embed_mode"):
            AilakeDestinationConfig.from_dict(
                {"table_base_path": "/tmp", "embed_mode": "invalid_mode"}
            )

    def test_validate_missing_openai_key(self):
        cfg = AilakeDestinationConfig.from_dict(
            {"table_base_path": "/tmp", "embed_mode": "openai"}
        )
        errors = cfg.validate()
        assert any("openai_api_key" in e for e in errors)

    def test_validate_missing_cohere_key(self):
        cfg = AilakeDestinationConfig.from_dict(
            {"table_base_path": "/tmp", "embed_mode": "cohere"}
        )
        errors = cfg.validate()
        assert any("cohere_api_key" in e for e in errors)

    def test_validate_missing_http_url(self):
        cfg = AilakeDestinationConfig.from_dict(
            {"table_base_path": "/tmp", "embed_mode": "http"}
        )
        errors = cfg.validate()
        assert any("http_url" in e for e in errors)

    def test_validate_missing_embed_cmd(self):
        cfg = AilakeDestinationConfig.from_dict(
            {"table_base_path": "/tmp", "embed_mode": "cmd"}
        )
        errors = cfg.validate()
        assert any("embed_cmd" in e for e in errors)

    def test_table_path(self):
        cfg = AilakeDestinationConfig.from_dict(
            {"table_base_path": "s3://bucket/lake", "embed_mode": "cmd", "embed_cmd": "x"}
        )
        assert cfg.table_path("my_stream") == "s3://bucket/lake/my_stream"


# ---------------------------------------------------------------------------
# Text extraction
# ---------------------------------------------------------------------------


class TestExtractText:
    def test_simple_field(self):
        assert _extract_text({"content": "hello"}, "content") == "hello"

    def test_nested_field(self):
        assert _extract_text({"meta": {"body": "world"}}, "meta.body") == "world"

    def test_missing_field_returns_empty(self):
        assert _extract_text({"other": "x"}, "content") == ""

    def test_none_value_returns_empty(self):
        assert _extract_text({"content": None}, "content") == ""


# ---------------------------------------------------------------------------
# CmdEmbedder
# ---------------------------------------------------------------------------


class TestHttpEmbedder:
    def test_valid_response(self):
        from airbyte_destination_ailake.embedder import HttpEmbedder
        from unittest.mock import patch, MagicMock
        import io

        payload = json.dumps({"data": [{"embedding": [0.1, 0.2, 0.3]}, {"embedding": [0.4, 0.5, 0.6]}]}).encode()
        mock_resp = MagicMock()
        mock_resp.__enter__ = lambda s: s
        mock_resp.__exit__ = MagicMock(return_value=False)
        mock_resp.read.return_value = payload

        with patch("urllib.request.urlopen", return_value=mock_resp):
            embedder = HttpEmbedder(url="http://localhost:11434/v1/embeddings", model="nomic-embed-text")
            vecs = embedder.embed(["hello", "world"])

        assert vecs.shape == (2, 3)
        assert vecs.dtype == np.float32

    def test_auth_header_sent(self):
        from airbyte_destination_ailake.embedder import HttpEmbedder
        from unittest.mock import patch, MagicMock, call
        import urllib.request as ur

        payload = json.dumps({"data": [{"embedding": [0.1]}]}).encode()
        mock_resp = MagicMock()
        mock_resp.__enter__ = lambda s: s
        mock_resp.__exit__ = MagicMock(return_value=False)
        mock_resp.read.return_value = payload

        captured_req = []

        def fake_urlopen(req, timeout=None):
            captured_req.append(req)
            return mock_resp

        with patch("urllib.request.urlopen", side_effect=fake_urlopen):
            embedder = HttpEmbedder(url="http://example.com/embed", auth_header="Bearer sk-test")
            embedder.embed(["text"])

        assert captured_req[0].get_header("Authorization") == "Bearer sk-test"

    def test_model_in_request_body(self):
        from airbyte_destination_ailake.embedder import HttpEmbedder
        from unittest.mock import patch, MagicMock

        payload = json.dumps({"data": [{"embedding": [0.5, 0.5]}]}).encode()
        mock_resp = MagicMock()
        mock_resp.__enter__ = lambda s: s
        mock_resp.__exit__ = MagicMock(return_value=False)
        mock_resp.read.return_value = payload

        sent_bodies = []

        def fake_urlopen(req, timeout=None):
            sent_bodies.append(json.loads(req.data))
            return mock_resp

        with patch("urllib.request.urlopen", side_effect=fake_urlopen):
            embedder = HttpEmbedder(url="http://x/embed", model="mxbai-embed-large")
            embedder.embed(["hello"])

        assert sent_bodies[0]["model"] == "mxbai-embed-large"
        assert sent_bodies[0]["input"] == ["hello"]

    def test_no_model_omits_field(self):
        from airbyte_destination_ailake.embedder import HttpEmbedder
        from unittest.mock import patch, MagicMock

        payload = json.dumps({"data": [{"embedding": [1.0]}]}).encode()
        mock_resp = MagicMock()
        mock_resp.__enter__ = lambda s: s
        mock_resp.__exit__ = MagicMock(return_value=False)
        mock_resp.read.return_value = payload

        sent_bodies = []

        def fake_urlopen(req, timeout=None):
            sent_bodies.append(json.loads(req.data))
            return mock_resp

        with patch("urllib.request.urlopen", side_effect=fake_urlopen):
            embedder = HttpEmbedder(url="http://x/embed", model="")
            embedder.embed(["hello"])

        assert "model" not in sent_bodies[0]

    def test_http_error_raises(self):
        from airbyte_destination_ailake.embedder import HttpEmbedder
        from unittest.mock import patch
        import urllib.error, io

        err = urllib.error.HTTPError(
            url="http://x", code=401, msg="Unauthorized",
            hdrs=None, fp=io.BytesIO(b"invalid api key"),
        )
        with patch("urllib.request.urlopen", side_effect=err):
            embedder = HttpEmbedder(url="http://x/embed")
            with pytest.raises(RuntimeError, match="401"):
                embedder.embed(["text"])

    def test_malformed_response_raises(self):
        from airbyte_destination_ailake.embedder import HttpEmbedder
        from unittest.mock import patch, MagicMock

        payload = json.dumps({"result": "oops"}).encode()
        mock_resp = MagicMock()
        mock_resp.__enter__ = lambda s: s
        mock_resp.__exit__ = MagicMock(return_value=False)
        mock_resp.read.return_value = payload

        with patch("urllib.request.urlopen", return_value=mock_resp):
            embedder = HttpEmbedder(url="http://x/embed")
            with pytest.raises(RuntimeError, match="unexpected response shape"):
                embedder.embed(["text"])

    def test_build_embedder_http(self):
        from airbyte_destination_ailake.embedder import build_embedder, HttpEmbedder

        cfg = _make_cfg(embed_mode="http", http_url="http://ollama:11434/v1/embeddings", http_model="nomic-embed-text")
        emb = build_embedder(cfg)
        assert isinstance(emb, HttpEmbedder)


class TestCmdEmbedder:
    def test_valid_output(self, tmp_path):
        script = tmp_path / "embed.py"
        script.write_text(
            "import sys, json\n"
            "texts = json.load(sys.stdin)\n"
            "print(json.dumps([[0.1] * 4 for _ in texts]))\n"
        )
        embedder = CmdEmbedder(f"{sys.executable} {script}")
        vecs = embedder.embed(["hello", "world"])
        assert vecs.shape == (2, 4)
        assert vecs.dtype == np.float32

    def test_nonzero_exit_raises(self):
        embedder = CmdEmbedder("exit 1")
        with pytest.raises(RuntimeError, match="embed_cmd failed"):
            embedder.embed(["text"])


# ---------------------------------------------------------------------------
# StreamWriter
# ---------------------------------------------------------------------------


def _make_cfg(**overrides) -> AilakeDestinationConfig:
    raw: dict = {
        "table_base_path": "/tmp/ailake_test",
        "embed_mode": "cmd",
        "embed_cmd": "unused",
        "embedding_dim": 4,
        "batch_size": 3,
    }
    raw.update(overrides)
    # http mode needs http_url, not embed_cmd
    if raw.get("embed_mode") == "http" and "embed_cmd" in raw and not raw.get("http_url"):
        raw.setdefault("http_url", "http://localhost/embed")
    return AilakeDestinationConfig.from_dict(raw)


class FakeEmbedder:
    def __init__(self, dim: int = 4):
        self._dim = dim

    def embed(self, texts: list[str]) -> np.ndarray:
        return np.zeros((len(texts), self._dim), dtype=np.float32)


class TestStreamWriter:
    def test_batches_flush_at_threshold(self):
        cfg = _make_cfg(batch_size=2)
        embedder = FakeEmbedder()
        flush_calls: list[int] = []

        mock_table = MagicMock()
        mock_table.commit.return_value = 1
        fake_ailake = MagicMock()
        fake_ailake.open_table.return_value = mock_table

        with patch.dict("sys.modules", {"ailake": fake_ailake}):
            writer = StreamWriter("test_stream", cfg, embedder)
            writer.add({"content": "a"})
            assert mock_table.insert.call_count == 0
            writer.add({"content": "b"})
            assert mock_table.insert.call_count == 1

    def test_commit_flushes_remaining(self):
        cfg = _make_cfg(batch_size=10)
        embedder = FakeEmbedder()
        mock_table = MagicMock()
        mock_table.commit.return_value = 42
        fake_ailake = MagicMock()
        fake_ailake.open_table.return_value = mock_table

        with patch.dict("sys.modules", {"ailake": fake_ailake}):
            writer = StreamWriter("test_stream", cfg, embedder)
            writer.add({"content": "hello"})
            snap_id = writer.commit()

        assert snap_id == 42
        assert mock_table.insert.call_count == 1

    def test_embedding_model_passed_to_open_table(self):
        cfg = _make_cfg(embedding_model="text-embedding-3-small", embedding_model_version="1")
        embedder = FakeEmbedder()
        mock_table = MagicMock()
        mock_table.commit.return_value = 1
        fake_ailake = MagicMock()
        fake_ailake.open_table.return_value = mock_table

        with patch.dict("sys.modules", {"ailake": fake_ailake}):
            writer = StreamWriter("s", cfg, embedder)
            writer.add({"content": "x"})
            writer.commit()

        call_kwargs = fake_ailake.open_table.call_args[1]
        assert call_kwargs.get("embedding_model") == "text-embedding-3-small"
        assert call_kwargs.get("embedding_model_version") == "1"


# ---------------------------------------------------------------------------
# Destination.check
# ---------------------------------------------------------------------------


class TestDestinationCheck:
    def test_check_succeeds(self):
        from airbyte_destination_ailake.destination import AilakeDestination

        fake_embedder = FakeEmbedder(dim=4)
        with patch(
            "airbyte_destination_ailake.destination.build_embedder",
            return_value=fake_embedder,
        ):
            dest = AilakeDestination()
            result = dest.check(
                MagicMock(),
                {
                    "table_base_path": "/tmp/x",
                    "embed_mode": "cmd",
                    "embed_cmd": "x",
                    "embedding_dim": 4,
                },
            )
        assert result.status.value == "SUCCEEDED"

    def test_check_fails_on_dim_mismatch(self):
        from airbyte_destination_ailake.destination import AilakeDestination

        fake_embedder = FakeEmbedder(dim=8)
        with patch(
            "airbyte_destination_ailake.destination.build_embedder",
            return_value=fake_embedder,
        ):
            dest = AilakeDestination()
            result = dest.check(
                MagicMock(),
                {
                    "table_base_path": "/tmp/x",
                    "embed_mode": "cmd",
                    "embed_cmd": "x",
                    "embedding_dim": 4,
                },
            )
        assert result.status.value == "FAILED"
        assert "shape" in result.message

    def test_check_fails_on_config_error(self):
        from airbyte_destination_ailake.destination import AilakeDestination

        dest = AilakeDestination()
        result = dest.check(MagicMock(), {"embed_mode": "cmd"})
        assert result.status.value == "FAILED"
