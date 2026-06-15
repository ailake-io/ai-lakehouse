# SPDX-License-Identifier: MIT OR Apache-2.0
"""Airbyte Destination implementation for AI-Lake Format."""

from __future__ import annotations

import logging
import sys
from typing import Any, Iterable, Mapping

from airbyte_cdk import AirbyteLogger
from airbyte_cdk.destinations import Destination
from airbyte_cdk.models import (
    AirbyteConnectionStatus,
    AirbyteMessage,
    AirbyteStateMessage,
    ConfiguredAirbyteCatalog,
    Status,
    Type,
)

from airbyte_destination_ailake.config import AilakeDestinationConfig
from airbyte_destination_ailake.embedder import build_embedder
from airbyte_destination_ailake.writer import StreamWriter

logger = logging.getLogger(__name__)


class AilakeDestination(Destination):
    def check(
        self,
        logger: AirbyteLogger,
        config: Mapping[str, Any],
    ) -> AirbyteConnectionStatus:
        try:
            cfg = AilakeDestinationConfig.from_dict(dict(config))
            errors = cfg.validate()
            if errors:
                return AirbyteConnectionStatus(
                    status=Status.FAILED,
                    message="; ".join(errors),
                )

            embedder = build_embedder(cfg)
            test_vecs = embedder.embed(["connection check"])
            if test_vecs.shape != (1, cfg.embedding_dim):
                return AirbyteConnectionStatus(
                    status=Status.FAILED,
                    message=(
                        f"Embedder returned shape {test_vecs.shape}, "
                        f"expected (1, {cfg.embedding_dim})"
                    ),
                )

            return AirbyteConnectionStatus(status=Status.SUCCEEDED)
        except Exception as exc:
            return AirbyteConnectionStatus(
                status=Status.FAILED,
                message=str(exc),
            )

    def write(
        self,
        config: Mapping[str, Any],
        configured_catalog: ConfiguredAirbyteCatalog,
        input_messages: Iterable[AirbyteMessage],
    ) -> Iterable[AirbyteMessage]:
        cfg = AilakeDestinationConfig.from_dict(dict(config))
        errors = cfg.validate()
        if errors:
            raise ValueError(f"Invalid config: {'; '.join(errors)}")

        embedder = build_embedder(cfg)
        writers: dict[str, StreamWriter] = {}

        for stream in configured_catalog.streams:
            name = stream.stream.name
            writers[name] = StreamWriter(name, cfg, embedder)

        for message in input_messages:
            if message.type == Type.RECORD:
                rec = message.record
                if rec.stream in writers:
                    writers[rec.stream].add(rec.data)
            elif message.type == Type.STATE:
                # Flush all streams before emitting state — guarantees durability.
                for writer in writers.values():
                    writer.commit()
                yield message

        # Final flush after all messages consumed.
        for writer in writers.values():
            writer.commit()


def main() -> None:
    destination = AilakeDestination()
    destination.run(sys.argv[1:])
