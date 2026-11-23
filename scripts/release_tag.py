#!/usr/bin/env python3
"""Validate an immutable core release tag before any publication."""

import json
import re
import subprocess
import sys
import tomllib
from pathlib import Path


def validate_versions(root: Path, tag: str) -> str:
    if not re.fullmatch(r"v(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)", tag):
        raise ValueError("Core tags must use vMAJOR.MINOR.PATCH (e.g. v0.1.25)")
    version = tag[1:]
    workspace = tomllib.loads((root / "Cargo.toml").read_text())
    # Plugin package versions are independent from the core release.
    registry = json.loads((root / "scripts/plugins.json").read_text())
    independent = {
        target["file"]
        for plugin in registry["plugins"]
        if plugin.get("version_policy", "core") == "independent"
        for target in plugin["targets"]
    }
    packages = set()
    for member in workspace["workspace"]["members"]:
        for directory in root.glob(member):
            path = directory / "Cargo.toml"
            if path.relative_to(root).as_posix() in independent:
                continue
            package = tomllib.loads(path.read_text())["package"]
            if package["version"] != version:
                raise ValueError(f"{path.relative_to(root)}: version must be {version}")
            packages.add(package["name"])
    if not packages:
        raise ValueError("No core workspace packages found")
    for name in ("Cargo.lock", "fuzz/Cargo.lock"):
        lock = tomllib.loads((root / name).read_text())
        for package in lock["package"]:
            if package["name"] in packages and package["version"] != version:
                raise ValueError(f"{name}: stale version for {package['name']}")
    changelog = (root / "CHANGELOG.md").read_text()
    if not re.search(rf"^## \[{re.escape(version)}\] — \d{{4}}-\d{{2}}-\d{{2}}$", changelog, re.M):
        raise ValueError(f"CHANGELOG.md must contain a dated [{version}] section")
    return version


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    if len(sys.argv) != 2:
        print("Usage: python3 scripts/release_tag.py vMAJOR.MINOR.PATCH", file=sys.stderr)
        return 1
    try:
        version = validate_versions(root, sys.argv[1])
        subprocess.run(
            ["git", "merge-base", "--is-ancestor", "HEAD", "origin/main"],
            cwd=root,
            check=True,
        )
    except (ValueError, subprocess.CalledProcessError) as error:
        print(
            f"Invalid release: {error}. The tagged commit must be prepared and on main.",
            file=sys.stderr,
        )
        return 1
    print(f"Validated core release {version}; no files, commits or tags were changed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
