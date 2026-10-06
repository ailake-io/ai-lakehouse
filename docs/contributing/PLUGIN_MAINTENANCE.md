# Plugin maintenance

Plugin artifact versions and their source files are registered in
[`scripts/plugins.json`](../../scripts/plugins.json). The release workflow reads
this registry, so adding a plugin version target requires one registry entry
instead of another hardcoded release loop.

## Version commands

From the repository root:

```sh
python3 scripts/plugins.py list
python3 scripts/plugins.py check
python3 scripts/plugins.py update --plugin spark --version 0.1.13
python3 scripts/plugins.py update --all --version 0.1.13
```

`check` verifies registered artifact versions against `ailake-core`. `update`
requires each target to match exactly once and reports changed files. Add every
version-bearing file for a plugin to its `targets` entry. Plugins without an
independent artifact version can use an empty target list.

## Shared JVM dependencies

Spark, Trino, and Flink read common build plugin and JNA versions from
[`gradle/ailake-plugins.versions.toml`](../../gradle/ailake-plugins.versions.toml).
Update Kotlin, Shadow, or JNA there once. Keep engine versions and dependencies
that must match the runtime cluster in the corresponding plugin build file.

## Tests

The registry tests run with the Python standard library:

```sh
python3 -m unittest scripts.test_plugins
```

Plugin-specific test commands are recorded in `scripts/plugins.json`; run them
from their registered project directory (or use the command as specified in
that project's CI workflow) when changing plugin code.
