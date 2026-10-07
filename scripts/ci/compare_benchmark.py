#!/usr/bin/env python3
"""Validate a benchmark result against an optional same-runner baseline."""

from __future__ import annotations

import argparse
import json
import math
import pathlib
import sys


def load(path: pathlib.Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("benchmark") != "ailake-vec-distance":
        raise ValueError(f"{path}: unexpected benchmark name")
    results = value.get("results")
    if not isinstance(results, list) or not results:
        raise ValueError(f"{path}: results must be a non-empty list")
    seen: set[tuple[str, int]] = set()
    for row in results:
        kernel = row.get("kernel", "cosine")
        nanos = row.get("nanos_per_op", 0)
        if (
            not isinstance(kernel, str)
            or not kernel
            or not isinstance(row.get("dim"), int)
            or not isinstance(nanos, (int, float))
            or not math.isfinite(nanos)
            or nanos <= 0
        ):
            raise ValueError(f"{path}: malformed benchmark result: {row!r}")
        key = (kernel, row["dim"])
        if key in seen:
            raise ValueError(f"{path}: duplicate benchmark case: {key!r}")
        seen.add(key)
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

    case_key = lambda row: (row.get("kernel", "cosine"), row["dim"])
    current_by_case = {case_key(row): row["nanos_per_op"] for row in current["results"]}
    print(json.dumps({"benchmark": current["benchmark"], "results": current["results"]}))

    if baseline is None:
        print("No baseline supplied; benchmark schema validated.")
        return 0

    baseline_by_case = {case_key(row): row["nanos_per_op"] for row in baseline["results"]}
    missing = sorted(set(baseline_by_case) - set(current_by_case))
    if missing:
        print(f"benchmark cases missing from current result: {missing}", file=sys.stderr)
        return 2

    failures = []
    for case, before in sorted(baseline_by_case.items()):
        after = current_by_case[case]
        ratio = after / before
        print(f"kernel={case[0]} dim={case[1]}: baseline={before:.2f} ns/op current={after:.2f} ns/op ratio={ratio:.3f}")
        if ratio > 1.0 + args.max_regression:
            failures.append((case, ratio))

    if failures:
        print(
            f"benchmark regression exceeded {args.max_regression:.0%}: {failures}",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
