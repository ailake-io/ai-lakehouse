#!/usr/bin/env python3
"""Validate a benchmark result against an optional same-runner baseline."""

from __future__ import annotations

import argparse
import json
import pathlib
import sys


def load(path: pathlib.Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("benchmark") != "ailake-vec-distance":
        raise ValueError(f"{path}: unexpected benchmark name")
    results = value.get("results")
    if not isinstance(results, list) or not results:
        raise ValueError(f"{path}: results must be a non-empty list")
    for row in results:
        if not isinstance(row.get("dim"), int) or row.get("nanos_per_op", 0) <= 0:
            raise ValueError(f"{path}: malformed benchmark result: {row!r}")
    return value


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("result", type=pathlib.Path)
    parser.add_argument("--baseline", type=pathlib.Path)
    parser.add_argument("--max-regression", type=float, default=0.25)
    args = parser.parse_args()

    try:
        current = load(args.result)
        baseline = load(args.baseline) if args.baseline else None
    except (OSError, ValueError, json.JSONDecodeError) as error:
        print(f"benchmark validation failed: {error}", file=sys.stderr)
        return 2

    current_by_dim = {row["dim"]: row["nanos_per_op"] for row in current["results"]}
    print(json.dumps({"benchmark": current["benchmark"], "results": current["results"]}))

    if baseline is None:
        print("No baseline supplied; benchmark schema validated.")
        return 0

    baseline_by_dim = {row["dim"]: row["nanos_per_op"] for row in baseline["results"]}
    missing = sorted(set(baseline_by_dim) - set(current_by_dim))
    if missing:
        print(f"benchmark dimensions missing from current result: {missing}", file=sys.stderr)
        return 2

    failures = []
    for dim, before in sorted(baseline_by_dim.items()):
        after = current_by_dim[dim]
        ratio = after / before
        print(f"dim={dim}: baseline={before:.2f} ns/op current={after:.2f} ns/op ratio={ratio:.3f}")
        if ratio > 1.0 + args.max_regression:
            failures.append((dim, ratio))

    if failures:
        print(
            f"benchmark regression exceeded {args.max_regression:.0%}: {failures}",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
