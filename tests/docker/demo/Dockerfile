# AI-Lake Demo Image
#
# Two-stage build:
#   Stage 1 (builder): installs Rust + maturin, compiles ailake-py wheel
#   Stage 2 (runtime): JupyterLab with the wheel + Python data stack
#
# Build context must be the repository root (set in compose-demo.yml).

# ── Stage 1: build ailake-py wheel ───────────────────────────────────────────
FROM python:3.12-slim AS builder

RUN apt-get update && apt-get install -y --no-install-recommends \
        curl \
        build-essential \
        pkg-config \
        libssl-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Rust stable (cached in a separate layer)
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs \
    | sh -s -- -y --default-toolchain stable --profile minimal
ENV PATH="/root/.cargo/bin:$PATH"

RUN pip install --no-cache-dir maturin

# Copy full workspace (Docker layer cache: invalidated only on source changes)
COPY Cargo.toml Cargo.lock ./
COPY ailake-core      ailake-core/
COPY ailake-vec       ailake-vec/
COPY ailake-parquet   ailake-parquet/
COPY ailake-index     ailake-index/
COPY ailake-file      ailake-file/
COPY ailake-catalog   ailake-catalog/
COPY ailake-store     ailake-store/
COPY ailake-query     ailake-query/
COPY ailake-py        ailake-py/
# Workspace members not needed for the wheel build, but cargo metadata requires
# all members listed in the root Cargo.toml to exist in the build context.
COPY ailake-cli       ailake-cli/
COPY ailake-jni       ailake-jni/
COPY ailake-bench     ailake-bench/
COPY tests            tests/

WORKDIR /ailake-py
RUN maturin build --release --out /wheels

# ── Stage 2: JupyterLab runtime ───────────────────────────────────────────────
FROM python:3.12-slim

# Java is required by PySpark (notebook 03_spark.ipynb uses local[*] mode).
# default-jre-headless installs OpenJDK 17 and creates /usr/lib/jvm/default-java.
RUN apt-get update && apt-get install -y --no-install-recommends \
        default-jre-headless \
    && rm -rf /var/lib/apt/lists/*

ENV JAVA_HOME=/usr/lib/jvm/default-java
ENV PATH="${JAVA_HOME}/bin:${PATH}"

COPY --from=builder /wheels /wheels

RUN pip install --no-cache-dir \
        /wheels/*.whl \
        jupyterlab \
        "pyiceberg[pyarrow,s3fs,sql-sqlite]" \
        duckdb \
        "pyspark<4.0" \
        trino \
        "google-cloud-bigquery>=3.13" \
        google-auth \
        pandas \
        numpy \
        pyarrow \
        boto3 \
    && rm -rf /wheels

COPY tests/docker/demo/init_demo.py   /opt/init_demo.py
COPY tests/docker/demo/entrypoint.sh  /entrypoint.sh
COPY tests/docker/demo/notebooks      /notebooks

RUN chmod +x /entrypoint.sh

WORKDIR /notebooks
EXPOSE 8888
ENTRYPOINT ["/entrypoint.sh"]
