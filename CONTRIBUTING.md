# Contributing to AI-Lake

Thank you for your interest in contributing. This guide covers everything needed to go from zero to a merged pull request.

---

## Table of contents

1. [Prerequisites](#prerequisites)
2. [Development environment setup](#development-environment-setup)
3. [Building the project](#building-the-project)
4. [Running tests](#running-tests)
5. [Code style and quality gates](#code-style-and-quality-gates)
6. [Branch and commit strategy](#branch-and-commit-strategy)
7. [Pull request workflow](#pull-request-workflow)
8. [Reporting issues](#reporting-issues)

---

## Prerequisites

| Tool | Minimum version | Install |
|---|---|---|
| **Rust** (stable) | 1.78+ | `curl https://sh.rustup.rs -sSf \| sh` |
| **JDK** | 17+ | `sudo apt install openjdk-17-jdk` or [Adoptium](https://adoptium.net) |
| **Gradle** | 8.7+ | Included as wrapper (`./gradlew`) in each JVM subproject |
| **Python** | 3.10+ | System Python or [pyenv](https://github.com/pyenv/pyenv) |
| **maturin** | 1.4+ | `pip install maturin` — required to build `ailake-py` |
| **cargo-deny** | latest | `cargo install cargo-deny` — license and advisory audits |

Optional (for full compat-heavy tests):

| Tool | Purpose |
|---|---|
| **Docker** | Runs Spark, Trino, and BigQuery emulator in `compat-heavy.yml` |
| **NVIDIA CUDA runtime** | Enables GPU search (`libcudart.so` + `libcublas.so`) |
| **AMD ROCm runtime** | Enables GPU search (`libamdhip64.so` + `libhipblas.so`) |

---

## Development environment setup

### 1. Clone and enter the repository

```bash
git clone https://github.com/ThiagoLange/ai-lakehouse.git
cd ai-lakehouse
git checkout develop   # all work goes here first
```

### 2. Rust workspace

```bash
# Install or update Rust stable
rustup update stable

# Install required components
rustup component add rustfmt clippy

# Install cargo-deny (license + advisory audit)
cargo install cargo-deny

# Build the entire workspace (debug)
cargo build --workspace

# Build release (required before JVM plugin tests)
cargo build --workspace --release
```

### 3. Python extension (ailake-py)

```bash
pip install maturin

# Build and install the wheel into the active Python environment
cd ailake-py
maturin develop --release
cd ..

# Verify
python -c "import ailake; print(ailake.__doc__)"
```

### 4. JVM plugins (Spark / Trino / Flink)

Each JVM subproject includes a Gradle wrapper. No system-wide Gradle installation is required.

```bash
# Native library (required by all JVM tests)
cargo build --release -p ailake-jni

# Flink connector
cd ailake-flink
./gradlew test -Dailake.native.lib=$(pwd)/../target/release/libailake_jni.so
cd ..

# Spark plugin
cd spark-plugin
LD_LIBRARY_PATH=$(pwd)/../target/release \
AILAKE_SPARK_TRINO_FIXTURE=$(pwd)/../spark-trino-fixture \
./gradlew test
cd ..

# Trino plugin
cd trino-plugin
LD_LIBRARY_PATH=$(pwd)/../target/release \
AILAKE_SPARK_TRINO_FIXTURE=$(pwd)/../spark-trino-fixture \
./gradlew test
cd ..
```

### 5. Go SDK (ailake-go)

```bash
cd ailake-go
go build ./...
go vet ./...
go test ./...
cd ..
```

### 6. C++ SDK (ailake-cpp)

```bash
cmake -S ailake-cpp -B ailake-cpp/build \
  -DCMAKE_BUILD_TYPE=Release \
  -DAILAKE_CUDA=OFF \
  -DAILAKE_TESTS=OFF \
  -DAILAKE_EXAMPLES=ON
cmake --build ailake-cpp/build --parallel
```

---

## Building the project

```bash
# All Rust crates (debug)
cargo build --workspace

# All Rust crates (release — needed for JVM tests)
cargo build --workspace --release

# Single crate
cargo build -p ailake-query

# Check without producing artifacts (faster)
cargo check --workspace
```

---

## Running tests

### Rust — unit and integration

```bash
# All unit tests
cargo test --workspace --lib --bins

# Full integration suite (local FS, no Docker required)
cargo test -p ailake-tests -- --test-threads=1

# Parquet spec compliance
cargo test -p ailake-tests --test parquet_trailing_bytes --test positional_invariant
```

### Python compat (requires PyArrow, DuckDB, PyIceberg)

```bash
pip install pyarrow duckdb "pyiceberg[pyarrow]"
cargo run --example write_fixture -p ailake-query
python tests/compat/check_pyarrow.py
python tests/compat/check_duckdb.py
python tests/compat/check_pyiceberg.py
```

### ailake-py SDK

```bash
cd ailake-py && maturin develop --release && cd ..
python tests/compat/check_ailake_py.py
```

### Airflow provider

```bash
cd airflow-providers-ailake
pip install "apache-airflow>=2.6" pytest
pytest tests/
cd ..
```

### JNI C-ABI

```bash
cargo build --release -p ailake-jni
AILAKE_NATIVE_LIB=$(pwd)/target/release/libailake_jni.so \
  python tests/compat/check_jni_cabi.py
```

### Compat Heavy (Docker required — Spark, Trino, BigQuery)

Trigger via GitHub Actions UI: **Actions → Compat Heavy → Run workflow**.
See [`docs/contributing/TESTING.md`](./docs/contributing/TESTING.md) for the full manual Actions trigger order.

---

## Code style and quality gates

All of the following must pass before opening a PR. CI runs them automatically on every push.

### Rust

```bash
# Formatting — zero tolerance for diffs
cargo fmt --all -- --check

# Lints — zero warnings policy (-D warnings)
cargo clippy --workspace --all-targets -- -D warnings

# License and advisory audit
cargo deny check licenses advisories sources
```

To auto-fix formatting:
```bash
cargo fmt --all
```

Key linting rules enforced by clippy:
- No `unwrap()` in production code without a comment explaining why it cannot fail
- No `eprintln!` — use `tracing::{error!, warn!, info!, debug!}` instead
- No dead code exported from library crates

### JVM (Kotlin / Scala)

```bash
# Flink — format + compile check
cd ailake-flink && ./gradlew build --no-daemon && cd ..

# Spark — format + compile check
cd spark-plugin && ./gradlew build --no-daemon && cd ..

# Trino — format + compile check
cd trino-plugin && ./gradlew build --no-daemon && cd ..
```

### Go

```bash
cd ailake-go

# Formatting — gofmt is the canonical Go formatter (no config file needed)
gofmt -l .          # list files with formatting issues
gofmt -w .          # auto-fix all files

# Lints
go vet ./...

cd ..
```

Key rules:
- All exported symbols must have a doc comment (`// FuncName ...`)
- No `fmt.Println` in library code — use `log/slog.Debug` at most
- Error strings must not be capitalized and must not end with punctuation

### C++ (`ailake-cpp`)

```bash
# Formatting — clang-format 14+ with the project's .clang-format config
find ailake-cpp -name "*.cpp" -o -name "*.hpp" -o -name "*.h" | \
  xargs clang-format --dry-run --Werror   # check only

find ailake-cpp -name "*.cpp" -o -name "*.hpp" -o -name "*.h" | \
  xargs clang-format -i                   # auto-fix

# Build check (CPU-only, no CUDA)
cmake -S ailake-cpp -B ailake-cpp/build \
  -DCMAKE_BUILD_TYPE=Release \
  -DAILAKE_CUDA=OFF \
  -DAILAKE_TESTS=ON
cmake --build ailake-cpp/build --parallel
ctest --test-dir ailake-cpp/build --output-on-failure
```

Unit tests live in `ailake-cpp/tests/`:

| File | What it covers |
|---|---|
| `test_footer.cpp` | AILK header parsing, flag bits (`is_hnsw`, `is_ivfpq`), bad magic/version rejection |
| `test_hnsw.cpp` | Flat HNSW search (Euclidean, Cosine, top-k capped, empty index) |
| `test_ivfpq.cpp` | IVF-PQ nearest-cell search, top-k limit, zero nprobe |
| `test_write.cpp` | `delete_where`, `evolve_schema`, `shell_quote` edge cases, `resolve_bin` env override; integration tests guarded by `AILAKE_BIN`/`AILAKE_FIXTURE` |

All test binaries are compiled via `foreach(_test footer hnsw ivfpq write)` in `CMakeLists.txt` and run via `ctest`. New index types require a corresponding `test_<name>.cpp`.

Key rules:
- Header-only where possible — implementations in `include/ailake/*.hpp`
- No exceptions in public API headers (`noexcept` on search + parse functions)
- SPDX header on every file (see §SPDX)

### Python (`ailake-py` tests + demo scripts)

```bash
# Formatting — ruff (fast, covers isort + pyflakes + pycodestyle)
pip install ruff
ruff check tests/ tests/docker/demo/    # lint
ruff format --check tests/ tests/docker/demo/   # format check
ruff format tests/ tests/docker/demo/           # auto-fix
```

Key rules:
- No bare `except:` — always catch a specific exception class
- No `print()` in library code — scripts under `tests/docker/demo/` may use `print` for progress output
- Type annotations required for all new public Python functions

### General

- No `System.err.println` in JVM production code — use SLF4J logger
- All new public Rust functions must have at least one unit test
- All new Python-facing APIs must have at least one test in `ailake-py/tests/` or `tests/compat/`

---

## Branch and commit strategy

### Branches

| Branch | Purpose |
|---|---|
| `develop` | Integration branch — all PRs target here |
| `main` | Stable / released — only receives merges from `develop` after CI passes |

**Never commit directly to `main`.** All changes go to `develop` first.

### Commit messages

Follow [Conventional Commits](https://www.conventionalcommits.org/):

```
<type>(<scope>): <short description>

[optional body — explain WHY, not WHAT]
```

Types: `feat`, `fix`, `chore`, `docs`, `ci`, `perf`, `refactor`, `test`

Scopes (optional): crate name (`ailake-query`, `ailake-py`, `spark-plugin`, etc.)

Examples:
```
feat(ailake-query): add reranking after PQ with configurable factor
fix(ailake-jni): prevent double-free of JNA pointer on parse error
docs: update JVM_PLUGINS.md with pre-built JAR download links
ci: add publish-jvm workflow for GitHub Release artifacts
```

Subject line: ≤ 72 characters, imperative mood, no trailing period.

### CHANGELOG

Update `CHANGELOG.md` under `[Unreleased]` for every user-visible change before pushing. Defer nothing — the changelog entry is part of the commit.

---

## Pull request workflow

### Before opening a PR

```bash
# 1. Ensure you're on develop and up to date
git checkout develop && git pull origin develop

# 2. Run the quality gates
cargo fmt --all -- --check
cargo clippy --workspace --all-targets -- -D warnings
cargo deny check licenses advisories sources
cargo test --workspace --lib --bins
cargo test -p ailake-tests -- --test-threads=1

# 3. Update CHANGELOG.md under [Unreleased]

# 4. Push to develop
git push origin develop
```

### PR title

Use the same Conventional Commits format as commit messages:
```
feat(ailake-index): add IVF-PQ adaptive nlist selection
```

### PR description checklist

- [ ] What changed and why (not just what)
- [ ] Test coverage added or existing tests updated
- [ ] `CHANGELOG.md` updated under `[Unreleased]`
- [ ] No `unwrap()` without justification comment
- [ ] No `eprintln!` / `System.err.println` / `fmt.Println` in production code

### Review process

1. One approval required from a maintainer.
2. CI must be green (fmt + clippy + deny + unit + integration + compat-python).
3. Compat Heavy (`compat-heavy.yml`) is run manually by maintainers before merging significant changes to storage, catalog, or index code.
4. Maintainer merges to `develop`. Periodic batches are merged `develop → main` as releases.

### What happens after merge

- `develop` receives the PR.
- When ready for release, maintainers update CHANGELOG, run CI and Compat Heavy, then merge to `main` — `release.yml` fires automatically, auto-bumps the patch version in all `Cargo.toml` files, creates the git tag, and publishes crates, JVM plugins, Airflow provider, and Python wheels in a sequential chain. No manual version edits needed (see [`docs/contributing/TESTING.md`](./docs/contributing/TESTING.md#manual-actions-trigger-order-pre-release)).

---

## Reporting issues

- **Bug**: use the [Bug Report](https://github.com/ThiagoLange/ai-lakehouse/issues/new?template=bug_report.yml) template.
- **Feature request**: use the [Feature Request](https://github.com/ThiagoLange/ai-lakehouse/issues/new?template=feature_request.yml) template.
- **Security vulnerability**: follow [`SECURITY.md`](./SECURITY.md) — do not open a public issue.
- **Questions and design discussions**: use [GitHub Discussions](https://github.com/ThiagoLange/ai-lakehouse/discussions).

---

## Detailed references

| Document | What it covers |
|---|---|
| [`docs/WHY_AILAKE.md`](./docs/WHY_AILAKE.md) | Why AI-Lake — technical case vs Iceberg alone, LanceDB, and external vector DBs |
| [`docs/contributing/TESTING.md`](./docs/contributing/TESTING.md) | Test categories, fixtures, CI matrix, manual Actions trigger order |
| [`docs/contributing/CODING_STANDARDS.md`](./docs/contributing/CODING_STANDARDS.md) | Rust conventions, error handling, unsafe policy |
| [`docs/contributing/DECISIONS.md`](./docs/contributing/DECISIONS.md) | ADR log — why key architectural choices were made |
| [`docs/architecture/WORKSPACE.md`](./docs/architecture/WORKSPACE.md) | Crate map, dependency graph, build phases |
| [`docs/specs/FILE_FORMAT.md`](./docs/specs/FILE_FORMAT.md) | Binary spec of the AI-Lake `.parquet` file |

---

## License

By contributing, you agree that your changes are licensed under [MIT OR Apache-2.0](./LICENSE-MIT).
