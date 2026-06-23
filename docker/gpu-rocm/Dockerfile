# AI-Lake GPU test image — AMD ROCm
#
# ailake-index loads libamdhip64 + libhipblas at runtime via libloading —
# no compile-time ROCm SDK required. The image provides only the HIP runtime.
#
# Local usage (requires ROCm-capable host with /dev/kfd and /dev/dri):
#   docker build -t ailake-gpu-rocm -f docker/gpu-rocm/Dockerfile .
#   docker run --device /dev/kfd --device /dev/dri --group-add video --rm ailake-gpu-rocm
#
# ROCm version: 6.2 (Ubuntu 22.04). Change the tag to switch versions.
# Available tags: https://hub.docker.com/r/rocm/dev-ubuntu-22.04/tags

FROM rocm/dev-ubuntu-22.04:6.2

ENV DEBIAN_FRONTEND=noninteractive
ENV AILAKE_GPU_BACKEND=rocm
# hipBLAS is discovered via ldconfig — ensure ROCm libs are on the search path
ENV LD_LIBRARY_PATH="/opt/rocm/lib:${LD_LIBRARY_PATH}"

# System deps: Rust prerequisites. ROCm HIP runtime already in base image.
RUN apt-get update && apt-get install -y --no-install-recommends \
        curl \
        build-essential \
        pkg-config \
        libssl-dev \
        ca-certificates \
        hipblas \
    && rm -rf /var/lib/apt/lists/*

# Install Rust stable
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs \
    | sh -s -- -y --default-toolchain stable --profile minimal
ENV PATH="/root/.cargo/bin:${PATH}"

WORKDIR /workspace

# Manifest-only copy for dependency caching
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

RUN for crate in ailake-core ailake-vec ailake-parquet ailake-index ailake-file \
        ailake-catalog ailake-store ailake-query ailake-py ailake-cli ailake-jni \
        ailake-fts ailake-bench; do \
        mkdir -p /workspace/${crate}/src && \
        echo "fn main() {}" > /workspace/${crate}/src/lib.rs; \
    done && \
    mkdir -p /workspace/tests/src && echo "" > /workspace/tests/src/lib.rs && \
    mkdir -p /workspace/tests/tests

RUN cargo fetch

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

RUN cargo build -p ailake-index && \
    cargo test -p ailake-index --no-run

CMD ["cargo", "test", "-p", "ailake-index", "--", "--nocapture"]
