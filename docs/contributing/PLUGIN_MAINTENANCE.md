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

`check` validates each plugin's registered version targets and policy. Every
registered integration or SDK has an independent SemVer and tag prefix. The
core release updates Rust core crates without changing the Python SDK or plugin
versions. `--all` remains available for an intentional fleet wide bump. `update` requires each
target to match exactly once and reports changed files. Add every
version-bearing file for a plugin to its `targets` entry and its independent
tag prefix to the registry.

| Plugin | Tag prefix | Published result |
|---|---|---|
| Spark, Trino, Flink | `<plugin>-v` | Fat JAR + JNI for Linux x86_64, macOS arm64, Windows x86_64 |
| Airflow provider | `airflow-v` | PyPI wheel + sdist |
| Airbyte destination | `airbyte-v` | PyPI wheel + sdist + GHCR image |
| DuckDB extension | `duckdb-v` | Linux x86_64 extension, statically linked to its Rust core and pinned DuckDB build |
| C++ SDK | `cpp-v` | Tested source archive |
| Go SDK | `ailake-go/v` | Tested source archive and Go module tag |
| Python SDK | `python-v` | Linux/macOS/Windows wheels + sdist on PyPI and GitHub Release |
| Kof integration | `kof-v` | Tested HTTP JVM/JS and Native C-ABI source archive |

Run the named workflow under **Actions** (for example, **Release Spark** or
**Release Python**) and optionally enter a version; blank increments that
component's latest tag. Most plugin entry points invoke the shared release
workflow. Python has its own multi-platform build because its PyPI release
contains Linux, macOS, and Windows wheels. Kof releases the HTTP client and
Native adapter together; the Native adapter requires the compatible C-ABI
contract version recorded in the Kof guide. Release tags are recorded in
`tag_prefix`.

DuckDB's extension embeds the Rust core statically, so its independent version
does not mean its binary can be mixed with arbitrary core builds: the release
asset is an atomic plugin+core build and is also tied to its pinned DuckDB ABI.
The C++ and Go SDKs version their own public APIs and retain format compatibility
checks against published AI-Lake files.

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
