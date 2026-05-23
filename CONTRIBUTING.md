# Contributing to AI-Lake

Thanks for your interest. This guide gets you from zero to a merged PR.

---

## Prerequisites

- Rust stable (≥ 1.78) — `rustup update stable`
- Python 3.10+ (for compat tests)
- `cargo`, `clippy`, `rustfmt` (included with Rust toolchain)

---

## Build

```bash
cargo build --workspace
```

For the Python extension:

```bash
cd ailake-py
pip install maturin
maturin build --release
pip install target/wheels/*.whl
```

---

## Tests

```bash
# All unit + integration tests
cargo test --workspace

# Compat tests (require Python deps)
pip install pyarrow duckdb pyiceberg
python tests/compat/check_pyarrow.py
python tests/compat/check_duckdb.py
python tests/compat/check_pyiceberg.py
python tests/compat/check_ailake_py.py

# JNI C-ABI test (requires release build)
cargo build --release -p ailake-jni
AILAKE_NATIVE_LIB=target/release/libailake_jni.so \
  python tests/compat/check_jni_cabi.py
```

---

## Before opening a PR

```bash
cargo fmt --all              # format
cargo clippy --workspace     # lint (zero warnings policy)
cargo test --workspace       # all tests green
```

---

## PR guidelines

- One logical change per PR.
- Branch off `develop`; target `develop` (not `main`).
- PR title follows [Conventional Commits](https://www.conventionalcommits.org/): `feat:`, `fix:`, `chore:`, `docs:`, `perf:`, `ci:`.
- Tests required for new public APIs and bug fixes.
- Update `CHANGELOG.md` under `[Unreleased]` for user-visible changes.

---

## Detailed references

| Document | What it covers |
|---|---|
| [`docs/contributing/CODING_STANDARDS.md`](./docs/contributing/CODING_STANDARDS.md) | Rust conventions, error handling, unsafe policy |
| [`docs/contributing/TESTING.md`](./docs/contributing/TESTING.md) | Test categories, fixtures, CI matrix |
| [`docs/contributing/DECISIONS.md`](./docs/contributing/DECISIONS.md) | ADR log — why key architectural choices were made |
| [`docs/architecture/WORKSPACE.md`](./docs/architecture/WORKSPACE.md) | Crate map, dependency graph, build instructions |
| [`docs/specs/FILE_FORMAT.md`](./docs/specs/FILE_FORMAT.md) | Binary spec of the AI-Lake `.parquet` file |

---

## Reporting issues

Open a [GitHub Issue](https://github.com/ThiagoLange/iceberg-ai-deltalakehouse/issues).
For security vulnerabilities, see [`SECURITY.md`](./SECURITY.md).

---

## License

By contributing, you agree your changes are licensed under [MIT OR Apache-2.0](./LICENSE-MIT).
