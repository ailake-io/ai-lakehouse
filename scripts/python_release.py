#!/usr/bin/env python3
"""Resolve a Python SDK release from the matching, immutable core tag."""

import json
import os
import re
import subprocess
import sys
import tomllib
from pathlib import Path


def git(root: Path, *args: str) -> str:
    return subprocess.check_output(
        ["git", *args], cwd=root, text=True, stderr=subprocess.PIPE
    ).strip()


def version_at(root: Path, sha: str, manifest: str) -> str:
    return tomllib.loads(git(root, "show", f"{sha}:{manifest}"))["package"]["version"]


def resolve(root: Path, version: str, ref: str) -> dict[str, str]:
    if not version:
        version = tomllib.loads((root / "ailake-core/Cargo.toml").read_text())["package"]["version"]
    if not re.fullmatch(r"(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)", version):
        raise ValueError("Python version must use MAJOR.MINOR.PATCH")
    core_tag = f"v{version}"
    if ref not in {"refs/heads/main", f"refs/tags/{core_tag}"}:
        raise ValueError("Python releases must run from main or the matching core tag")
    try:
        core_sha = git(root, "rev-parse", "--verify", f"refs/tags/{core_tag}^{{commit}}")
    except subprocess.CalledProcessError as error:
        raise ValueError(f"Matching core tag {core_tag} does not exist") from error
    git(root, "merge-base", "--is-ancestor", core_sha, "origin/main")
    if version_at(root, core_sha, "ailake-core/Cargo.toml") != version:
        raise ValueError(f"Core manifest does not match {core_tag}")

    tag = f"python-v{version}"
    try:
        source_sha = git(root, "rev-parse", "--verify", f"refs/tags/{tag}^{{commit}}")
    except subprocess.CalledProcessError:
        source_sha = core_sha
    else:
        for manifest in ("ailake-py/Cargo.toml", "ailake-core/Cargo.toml"):
            if version_at(root, source_sha, manifest) != version:
                raise ValueError(f"Existing {tag}: {manifest} does not match {version}")
    return {"version": version, "tag": tag, "source_sha": source_sha, "core_tag": core_tag}


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    try:
        result = resolve(root, os.getenv("VERSION_INPUT", ""), os.environ["GITHUB_REF"])
    except (KeyError, ValueError, subprocess.CalledProcessError) as error:
        print(f"Invalid Python release: {error}", file=sys.stderr)
        return 1
    print(json.dumps(result))
    if output := os.getenv("GITHUB_OUTPUT"):
        with Path(output).open("a") as stream:
            for key, value in result.items():
                stream.write(f"{key}={value}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
