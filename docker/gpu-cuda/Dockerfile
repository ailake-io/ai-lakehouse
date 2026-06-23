# AI-Lake GPU test image — NVIDIA CUDA
#
# Uses the slim CUDA runtime (not devel) because ailake-index loads
# libcudart + libcublas at runtime via libloading — no compile-time GPU SDK.
#
# Local usage (requires nvidia-container-toolkit on the host):
#   docker build -t ailake-gpu-cuda -f docker/gpu-cuda/Dockerfile .
#   docker run --gpus all --rm ailake-gpu-cuda
#
# To run only the data integration tests:
#   docker run --gpus all --rm ailake-gpu-cuda \
#     cargo test -p ailake-index --test gpu_data -- --nocapture
#
# CUDA version matrix:
#   12.6.0-runtime-ubuntu22.04  →  libcudart.so.12 + libcublas.so.12  (default)
#   11.8.0-runtime-ubuntu22.04  →  libcudart.so.11 + libcublas.so.11
# Change the FROM line to switch CUDA versions.

FROM nvidia/cuda:12.6.0-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV AILAKE_GPU_BACKEND=cuda

# System deps: Rust toolchain prerequisites + cuBLAS runtime
RUN apt-get update && apt-get install -y --no-install-recommends \
        curl \
        build-essential \
        pkg-config \
        libssl-dev \
        ca-certificates \
        libcublas-12-6 \
    && rm -rf /var/lib/apt/lists/*

# Install Rust stable (minimal profile — no docs, no clippy)
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs \
    | sh -s -- -y --default-toolchain stable --profile minimal
ENV PATH="/root/.cargo/bin:${PATH}"

WORKDIR /workspace

# Copy only Cargo manifests first for dependency caching.
# Full source is copied later so a code change doesn't invalidate the dep layer.
COPY Cargo.toml Cargo.lock ./
COPY ailake-core/Cargo.toml         ailake-core/Cargo.toml
COPY ailake-vec/Cargo.toml          ailake-vec/Cargo.toml
COPY ailake-parquet/Cargo.toml      ailake-parquet/Cargo.toml
COPY ailake-index/Cargo.toml        ailake-index/Cargo.toml
COPY ailake-file/Cargo.toml         ailake-file/Cargo.toml
COPY ailake-catalog/Cargo.toml      ailake-catalog/Cargo.toml
COPY ailake-store/Cargo.toml        ailake-store/Cargo.toml
COPY ailake-query/Cargo.toml        ailake-query/Cargo.toml
COPY ailake-py/Cargo.toml           ailake-py/Cargo.toml
COPY ailake-cli/Cargo.toml          ailake-cli/Cargo.toml
COPY ailake-jni/Cargo.toml          ailake-jni/Cargo.toml
COPY ailake-fts/Cargo.toml          ailake-fts/Cargo.toml
COPY ailake-bench/Cargo.toml        ailake-bench/Cargo.toml
COPY tests/Cargo.toml               tests/Cargo.toml

# Stub src files so `cargo fetch` can resolve the workspace without full source
RUN for crate in ailake-core ailake-vec ailake-parquet ailake-index ailake-file \
        ailake-catalog ailake-store ailake-query ailake-py ailake-cli ailake-jni \
        ailake-fts ailake-bench; do \
        mkdir -p /workspace/${crate}/src && \
        echo "fn main() {}" > /workspace/${crate}/src/lib.rs; \
    done && \
    mkdir -p /workspace/tests/src && echo "" > /workspace/tests/src/lib.rs && \
    mkdir -p /workspace/tests/tests

RUN cargo fetch

# Copy full source (invalidates only when code changes, not deps)
COPY ailake-core     ailake-core/
COPY ailake-vec      ailake-vec/
COPY ailake-parquet  ailake-parquet/
COPY ailake-index    ailake-index/
COPY ailake-file     ailake-file/
COPY ailake-catalog  ailake-catalog/
COPY ailake-store    ailake-store/
COPY ailake-query    ailake-query/
COPY ailake-py       ailake-py/
COPY ailake-cli      ailake-cli/
COPY ailake-jni      ailake-jni/
COPY ailake-fts      ailake-fts/
COPY ailake-bench    ailake-bench/
COPY tests           tests/

# Pre-build ailake-index and its test harness (GPU tests compile without GPU)
RUN cargo build -p ailake-index && \
    cargo test -p ailake-index --no-run

# Default: run all ailake-index tests (unit + GPU data integration).
# GPU tests auto-skip when AILAKE_GPU_BACKEND=none.
CMD ["cargo", "test", "-p", "ailake-index", "--", "--nocapture"]
