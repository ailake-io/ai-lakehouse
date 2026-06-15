#!/usr/bin/env python3
"""Minimal embed_cmd helper for demo_local.py.

Protocol: reads JSON string array from stdin, writes JSON float[][] to stdout.
Uses random unit vectors seeded from the text hash — deterministic, no API key needed.
"""
import json
import math
import random
import sys


def _unit_vec(text: str, dim: int) -> list[float]:
    rng = random.Random(hash(text) & 0xFFFFFFFF)
    v = [rng.gauss(0.0, 1.0) for _ in range(dim)]
    norm = math.sqrt(sum(x * x for x in v)) or 1.0
    return [x / norm for x in v]


def main() -> None:
    texts = json.load(sys.stdin)
    dim = int(sys.argv[1]) if len(sys.argv) > 1 else 32
    print(json.dumps([_unit_vec(t, dim) for t in texts]))


if __name__ == "__main__":
    main()
