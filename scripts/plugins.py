#!/usr/bin/env python3
"""Inspect and update versioned AI-Lake plugins from one registry."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
REGISTRY = ROOT / "scripts" / "plugins.json"


def load_registry(path: Path = REGISTRY) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def plugin_map(registry: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {plugin["id"]: plugin for plugin in registry["plugins"]}


def set_plugin_version(root: Path, plugin: dict[str, Any], version: str) -> list[Path]:
    changed: list[Path] = []
    for target in plugin["targets"]:
        path = root / target["file"]
        source = path.read_text(encoding="utf-8")
        pattern = re.compile(target["pattern"])
        updated, count = pattern.subn(target["replacement"].format(version=version), source)
        if count != 1:
            raise ValueError(f"{target['file']}: expected one version declaration, found {count}")
        if updated != source:
            path.write_text(updated, encoding="utf-8")
            changed.append(path)
    return changed


def plugin_versions(root: Path, plugin: dict[str, Any]) -> list[tuple[str, str]]:
    versions = []
    for target in plugin["targets"]:
        path = root / target["file"]
        match = re.search(target["version_pattern"], path.read_text(encoding="utf-8"))
        if not match:
            raise ValueError(f"{target['file']}: version declaration not found")
        versions.append((target["file"], match.group(1)))
    return versions


def bump_core_cargo_versions(
    root: Path, old_version: str, new_version: str, registry: dict[str, Any]
) -> list[Path]:
    """Bump core Cargo manifests without changing independently versioned packages."""
    independent_targets = {
        Path(target["file"])
        for plugin in registry["plugins"]
        if plugin.get("version_policy", "core") == "independent"
        for target in plugin["targets"]
        if target["file"].endswith("Cargo.toml")
    }
    changed: list[Path] = []
    old = f'"{old_version}"'
    new = f'"{new_version}"'
    for path in root.rglob("Cargo.toml"):
        relative = path.relative_to(root)
        if "target" in relative.parts or relative in independent_targets:
            continue
        source = path.read_text(encoding="utf-8")
        updated = source.replace(old, new)
        if updated != source:
            path.write_text(updated, encoding="utf-8")
            changed.append(path)
    return changed


def check_registry(root: Path, registry: dict[str, Any]) -> list[str]:
    errors = []
    expected_match = re.search(
        r'(?m)^version\s*=\s*"([^"]+)"$',
        (root / "ailake-core" / "Cargo.toml").read_text(encoding="utf-8"),
    )
    expected = expected_match.group(1) if expected_match else None
    for plugin in registry["plugins"]:
        policy = plugin.get("version_policy", "core")
        if policy not in {"core", "independent"}:
            errors.append(f"{plugin['id']}: invalid version_policy: {policy}")
            continue
        if not (root / plugin["directory"]).is_dir():
            errors.append(f"{plugin['id']}: directory not found: {plugin['directory']}")
        try:
            versions = plugin_versions(root, plugin)
        except (OSError, ValueError) as error:
            errors.append(f"{plugin['id']}: {error}")
            continue
        for path, version in versions:
            if policy == "core" and expected and version != expected:
                errors.append(f"{path}: version {version} differs from core {expected}")
        if versions and len({version for _, version in versions}) != 1:
            errors.append(f"{plugin['id']}: version targets are out of sync")
    return errors


def core_version(root: Path) -> str | None:
    match = re.search(
        r'(?m)^version\s*=\s*"([^"]+)"$',
        (root / "ailake-core" / "Cargo.toml").read_text(encoding="utf-8"),
    )
    return match.group(1) if match else None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("list", help="list plugins and their source directories")
    subparsers.add_parser("check", help="check plugin version targets and policies")
    update = subparsers.add_parser("update", help="update plugin artifact versions")
    update.add_argument("--version", required=True, help="target SemVer, for example 0.1.13")
    selection = update.add_mutually_exclusive_group(required=True)
    selection.add_argument("--plugin", help="one plugin ID from `list`")
    selection.add_argument("--all", action="store_true", help="update every registered plugin")
    selection.add_argument("--policy", choices=("core", "independent"), help="update plugins with this version policy")
    cargo_bump = subparsers.add_parser(
        "bump-core-cargo", help="bump core Cargo versions while preserving independent packages"
    )
    cargo_bump.add_argument("--from-version", required=True, help="current core version")
    cargo_bump.add_argument("--version", required=True, help="new core version")
    args = parser.parse_args()

    if args.command in {"update", "bump-core-cargo"} and not re.fullmatch(r"\d+\.\d+\.\d+", args.version):
        parser.error("--version must use SemVer numeric form: MAJOR.MINOR.PATCH")

    registry = load_registry()
    if args.command == "bump-core-cargo":
        if not re.fullmatch(r"\d+\.\d+\.\d+", args.from_version):
            parser.error("--from-version must use SemVer numeric form: MAJOR.MINOR.PATCH")
        changed = bump_core_cargo_versions(ROOT, args.from_version, args.version, registry)
        for path in changed:
            print(path.relative_to(ROOT))
        print(f"Updated {len(changed)} core Cargo manifest(s) to {args.version}.")
        return 0
    plugins = plugin_map(registry)
    if args.command == "list":
        for plugin in registry["plugins"]:
            policy = plugin.get("version_policy", "core")
            print(f"{plugin['id']:<10} {plugin['name']} [{policy}] ({plugin['directory']})")
        return 0
    if args.command == "check":
        errors = check_registry(ROOT, registry)
        if errors:
            print("\n".join(errors), file=sys.stderr)
            return 1
        print(f"Plugin registry satisfies version policies (core: {core_version(ROOT)}).")
        return 0

    if args.all:
        selected = registry["plugins"]
    elif args.policy:
        selected = [p for p in registry["plugins"] if p.get("version_policy", "core") == args.policy]
    else:
        selected = [plugins.get(args.plugin)]
    if args.plugin and selected[0] is None:
        parser.error(f"unknown plugin ID: {args.plugin}")
    if not selected:
        print(f"No plugins use the {args.policy} version policy; nothing to update.")
        return 0
    changed = []
    for plugin in selected:
        changed.extend(set_plugin_version(ROOT, plugin, args.version))
    for path in changed:
        print(path.relative_to(ROOT))
    print(f"Updated {len(changed)} plugin version file(s) to {args.version}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
