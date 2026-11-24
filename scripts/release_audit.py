#!/usr/bin/env python3
"""Audit published plugin versions and distribution completeness (read only)."""

import argparse
import fnmatch
import json
import re
import subprocess
import sys
import urllib.request
from pathlib import Path

if __package__:
    from .plugins import check_registry, load_registry, plugin_versions
    from .python_release import git
else:
    from plugins import check_registry, load_registry, plugin_versions
    from python_release import git


def check_assets(plugin: dict, version: str, release: dict) -> list[str]:
    errors = []
    expected = plugin["tag_prefix"] + version
    if release.get("tag_name") != expected or release.get("draft") or release.get("prerelease"):
        errors.append(f"{plugin['id']}: expected a published stable {expected}")
    names = [asset["name"] for asset in release.get("assets", [])]
    for template in plugin["artifacts"]:
        pattern = template.format(version=version)
        if not any(fnmatch.fnmatchcase(name, pattern) for name in names):
            errors.append(f"{plugin['id']}: missing artifact {pattern}")
    return errors


def latest_version(plugin: dict, releases: list[dict]) -> str | None:
    versions = []
    for release in releases:
        match = re.fullmatch(re.escape(plugin["tag_prefix"]) + r"([0-9]+)\.([0-9]+)\.([0-9]+)", release["tag_name"])
        if match and not release.get("draft") and not release.get("prerelease"):
            versions.append(tuple(map(int, match.groups())))
    return ".".join(map(str, max(versions))) if versions else None


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    registry = load_registry(root / "scripts/plugins.json")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plugin', choices=[p['id'] for p in registry['plugins']])
    parser.add_argument('--version', help='explicit published version (requires --plugin)')
    args = parser.parse_args()
    if args.version and (not args.plugin or not re.fullmatch(r'[0-9]+\.[0-9]+\.[0-9]+', args.version)):
        parser.error('--version requires --plugin and MAJOR.MINOR.PATCH')
    errors = check_registry(root, registry)
    pages = json.loads(subprocess.check_output(
        ["gh", "api", "repos/ailake-io/ai-lakehouse/releases?per_page=100",
         "--paginate", "--slurp"], text=True
    ))
    releases = [release for page in pages for release in page]
    for plugin in registry["plugins"]:
        if args.plugin and plugin['id'] != args.plugin:
            continue
        versions = {version for _, version in plugin_versions(root, plugin)}
        if len(versions) != 1:
            errors.append(f"{plugin['id']}: inconsistent prepared versions")
            continue
        version = args.version or versions.pop()
        latest = latest_version(plugin, releases)
        if not args.version and latest and latest != version:
            errors.append(f"{plugin['id']}: prepared {version}, latest published {latest}")
        tag = plugin["tag_prefix"] + version
        release = next((r for r in releases if r["tag_name"] == tag), None)
        if release is None:
            errors.append(f"{plugin['id']}: {tag} has not been published")
            continue
        errors.extend(check_assets(plugin, version, release))
        # Check manifests in the published source, not just asset filenames.
        for target in plugin["targets"]:
            if target["file"].endswith(".md"):
                continue  # historical tags predate documentation tracking
            try:
                text = git(root, "show", f"refs/tags/{tag}:{target['file']}")
                match = re.search(target["version_pattern"], text)
                if not match or match.group(1) != version:
                    errors.append(f"{tag}: source version differs in {target['file']}")
            except subprocess.CalledProcessError:
                errors.append(f"{tag}: source unavailable; fetch the published tags")
        if package := plugin.get("pypi_name"):
            try:
                with urllib.request.urlopen(f"https://pypi.org/pypi/{package}/{version}/json", timeout=30) as response:
                    data = json.load(response)
                if data["info"]["version"] != version:
                    errors.append(f"{package}: PyPI metadata version differs")
                pypi = {"tag_name": tag, "assets": [{"name": item["filename"]} for item in data["urls"]]}
                errors.extend("PyPI " + error for error in check_assets(plugin, version, pypi))
            except (OSError, ValueError) as error:
                errors.append(f"{package} {version}: PyPI could not be verified ({error})")
        print(f"Checked {plugin['id']}: {tag}")
    if errors:
        print("\n".join(errors), file=sys.stderr)
        return 1
    print("All registered GitHub/PyPI releases, source versions and expected assets verified.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
