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

`check` validates each plugin's registered version targets and policy. Each
registered integration or SDK has a tag prefix. Plugin versions are independent;
the Python SDK shares the core SemVer and uses a `python-v` tag. Its wheels are
published separately, and `release-python.yml` rejects a version that differs
from the core. `--all` remains available for an intentional fleet wide bump.
`update` requires each target to match exactly once and reports changed files.
Add every version-bearing file for a plugin to its `targets` entry and tag prefix
to the registry.

| Plugin | Tag prefix | Published result |
|---|---|---|
| Spark, Trino, Flink | `<plugin>-v` | Fat JAR + JNI for Linux x86_64, macOS arm64, Windows x86_64 |
| Airflow provider | `airflow-v` | PyPI wheel + sdist |
| Airbyte destination | `airbyte-v` | PyPI wheel + sdist + GHCR image |
| DuckDB extension | `duckdb-v` | Linux x86_64 extension, statically linked to its Rust core and pinned DuckDB build |
| C++ SDK | `cpp-v` | Tested source archive |
| Go SDK | `ailake-go/v` | Tested source archive and Go module tag |
| Python SDK | `python-v<core version>` | Linux/macOS/Windows wheels + sdist on PyPI and GitHub Release |
| Kof integration | `kof-v` | Tested HTTP JVM/JS and Native C-ABI source archive |

Run the named workflow under **Actions** (for example, **Release Spark** or
**Release Python**). Plugin workflows increment from their latest component tag.
Python uses the current core version; an entered version must match it. Most
plugin entry points invoke the shared release workflow. Python has its own
multi-platform build because its PyPI release contains Linux, macOS, and Windows
wheels. Kof releases the HTTP client and Native adapter together; the Native
adapter requires the compatible C-ABI contract version recorded in the Kof guide.
Release tags are recorded in `tag_prefix`.

DuckDB's extension embeds the Rust core statically, so its independent version
does not mean its binary can be mixed with arbitrary core builds: the release
asset is an atomic plugin+core build and is also tied to its pinned DuckDB ABI.
The C++ and Go SDKs version their own public APIs and retain format compatibility
checks against published AI-Lake files.

## Core changes: plugin impact check

Every core change (Rust crates, workspace dependencies, file format, C-ABI)
must be checked against every plugin before it is merged and again before each
core release. Plugins release independently, so nothing republishes them
automatically: a core fix only reaches a plugin's users after that plugin's own
release.

How each plugin consumes the core decides what a core change requires:

| Plugin | Embeds the core? | Consumes the core via | Needs a release when the core changes… |
|---|---|---|---|
| DuckDB extension | **Yes** | Rust core linked statically | always (code, dependencies, security fixes) |
| Python SDK | **Yes** | Rust core compiled into the wheel | always; version follows the core |
| Spark, Trino, Flink | **Yes** | JNI library built from `ailake-jni` and bundled in the JAR | `ailake-jni`/core code or dependency changes (incl. security); C-ABI changes also need wrapper updates |
| Kof | No | HTTP client + Native adapter over the C-ABI | C-ABI contract changes |
| Go SDK | No | Native Avro/Parquet reader + `ailake` CLI | file/manifest format changes, CLI flag changes, its own `go.mod` dependencies |
| C++ SDK | No | Native header-only reader + `ailake` CLI | file/manifest format changes, CLI flag changes |
| Airflow provider | No | `ailake` CLI | CLI flag or output-format changes |
| Airbyte destination | No (via Python SDK) | `ailake` Python SDK from PyPI | Python SDK API changes (dependency fixes arrive through the SDK release) |

Plugins that embed the core build it from `main` at release time, so release
them **after** the core release that carries the fix, never before: a plugin
released earlier ships the previous core and needs another release. Plugins
that do not embed the core can release whenever their own code changes.

Checklist for a core change:

1. Classify the change: internal code, dependency/security update, file or
   manifest format, C-ABI, CLI flags/output.
2. Use the table to list affected plugins. Check each one for required code
   changes (wrapper, flags, format readers) in the same PR.
3. Format changes: verify backward compatibility by writing a table with the
   previous release and reading/compacting it with the new build (vector, FTS,
   deletes), including the native Go/C++ readers.
4. After the core release, run the release workflow of every affected plugin
   and record it in `CHANGELOG.md`.

To find plugin code changed since its last release, compare content rather than
`git log` (squash merges make history misleading):

```sh
git diff --stat <plugin-tag> origin/develop -- <plugin-directory>
```

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
