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
python3 scripts/plugins.py update --policy core --version 0.1.13
```

`check` verifies that core-coupled artifact versions match `ailake-core` and
that every plugin's version targets agree with each other. The registry's
`version_policy` is either `core` or `independent`. The core release updates
only `core` entries; `--all` remains available for an intentional fleet wide
bump. `update` requires each target to match exactly once and reports changed
files. Add every version-bearing file for a plugin to its `targets` entry.

Spark, Trino, and Flink use independent release tags (`spark-vX.Y.Z`,
`trino-vX.Y.Z`, `flink-vX.Y.Z`). Run **Actions → Release JVM plugin**, choose
one plugin, and optionally enter a version; blank increments that plugin's
latest tag. The workflow builds JNI, runs that plugin's tests, and publishes
only its fat JAR plus the matching `libailake_jni.so`. Re-running the same
workflow commit and version replaces those assets; a tag already tied to a
different commit is rejected. Their Gradle project versions accept
`-PpluginVersion=X.Y.Z`; the checked-in version is the local default.

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
