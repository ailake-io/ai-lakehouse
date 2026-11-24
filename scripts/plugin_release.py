#!/usr/bin/env python3
"""Resolve an independently versioned plugin from reviewed version metadata."""

import json
import os
import re
import subprocess
import sys
from pathlib import Path

if __package__:
    from .plugins import load_registry, plugin_map, plugin_versions
    from .python_release import git
else:
    from plugins import load_registry, plugin_map, plugin_versions
    from python_release import git


def resolve(root: Path, plugin_id: str, version_input: str, ref: str) -> dict[str, str]:
    if ref != "refs/heads/main":
        raise ValueError("Plugin releases must run from main")
    plugin = plugin_map(load_registry(root / "scripts/plugins.json"))[plugin_id]
    if plugin.get("version_policy", "core") != "independent":
        raise ValueError("Use the core/Python release workflow for this package")
    versions = {version for _, version in plugin_versions(root, plugin)}
    if len(versions) != 1:
        raise ValueError("Plugin manifests and documentation versions differ")
    prepared = versions.pop()
    version = version_input or prepared
    if not re.fullmatch(r"(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)", version):
        raise ValueError("Plugin version must use MAJOR.MINOR.PATCH")
    tag = plugin["tag_prefix"] + version
    try:
        source_sha = git(root, "rev-parse", "--verify", f"refs/tags/{tag}^{{commit}}")
    except subprocess.CalledProcessError:
        if version != prepared:
            raise ValueError("Prepare this version and its documentation in a PR before publishing")
        source_sha = git(root, "rev-parse", "HEAD")
    else:
        # Historical tags may predate documentation tracking. Preserve their
        # exact source when retrying; never overwrite them from newer main.
        target = plugin["targets"][0]
        text = git(root, "show", f"{source_sha}:{target['file']}")
        match = re.search(target["version_pattern"], text)
        if not match or match.group(1) != version:
            raise ValueError(f"Existing {tag} has inconsistent source metadata")
    return {"version": version, "tag": tag, "source_sha": source_sha}


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    try:
        result = resolve(root, os.environ["PLUGIN"], os.getenv("VERSION_INPUT", ""), os.environ["GITHUB_REF"])
    except (KeyError, ValueError, subprocess.CalledProcessError) as error:
        print(f"Invalid plugin release: {error}", file=sys.stderr)
        return 1
    print(json.dumps(result))
    if output := os.getenv("GITHUB_OUTPUT"):
        with Path(output).open("a") as stream:
            for key, value in result.items():
                stream.write(f"{key}={value}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
