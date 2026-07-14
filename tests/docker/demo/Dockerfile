# AI-Lake Demo Image
#
# Three-stage build:
#   Stage 1 (builder): installs Rust + maturin, compiles ailake-py wheel +
#                       the ailake-cli binary (with catalog-ducklake, for
#                       notebook 13_ducklake.ipynb) + ailake-jni cdylib +
#                       the duckdb-ailake C++ extension
#   Stage 2 (jvm-builder): Gradle/JDK image, builds the ailake Spark plugin
#                       shadowJar (notebook 03_spark.ipynb) and the ailake
#                       Flink connector shadowJar (notebook 14_flink.ipynb —
#                       needed here for the Flink *client*; the Flink *server*
#                       gets its own copy via Dockerfile.flink)
#   Stage 3 (runtime): JupyterLab with the wheel + CLI binary + Python data
#                       stack + a remote-only Flink SQL Client install
#
# Build context must be the repository root (set in compose-demo.yml).

# ── Stage 1: build ailake-py wheel ───────────────────────────────────────────
FROM python:3.12-slim AS builder

# cmake is required by the `duckdb` crate's bundled C++ build, pulled in by
# ailake-cli's catalog-ducklake feature (docs/guides/DUCKLAKE_CATALOG.md) —
# adds several minutes to a cold-cache build.
RUN apt-get update && apt-get install -y --no-install-recommends \
        curl \
        build-essential \
        cmake \
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
COPY ailake-fts       ailake-fts/
# Workspace members not needed for the wheel build, but cargo metadata requires
# all members listed in the root Cargo.toml to exist in the build context.
COPY ailake-cli       ailake-cli/
COPY ailake-jni       ailake-jni/
COPY tests            tests/

WORKDIR /ailake-py
RUN maturin build --release --out /wheels

# ailake-cli with catalog-ducklake — local-fs-only DuckLake catalog backend,
# CLI-only surface (no ailake-py binding, see docs/guides/DUCKLAKE_CATALOG.md).
# Built separately from the wheel above since it's a distinct binary target.
RUN cargo build --release --manifest-path /Cargo.toml -p ailake-cli --features catalog-ducklake

# ── duckdb-ailake C++ extension (notebook 02_duckdb.ipynb) ────────────────────
# ailake-jni cdylib — same C-ABI bridge used by the Spark/Trino/Flink plugins.
RUN cargo build --release --manifest-path /Cargo.toml -p ailake-jni

COPY duckdb-ailake /duckdb-ailake

# DUCKDB_VERSION must match the `duckdb` pip package pinned in stage 2 below —
# see duckdb-ailake/README.md "DuckDB version" and .github/workflows/ci-duckdb.yml
# (this is the version CI actually tests against; v1.1.3 from an older README
# revision does not build against current duckdb-ailake sources).
RUN cmake -S /duckdb-ailake -B /duckdb-ailake/build \
        -DCMAKE_BUILD_TYPE=Release -DDUCKDB_VERSION=v1.5.0 \
    && cmake --build /duckdb-ailake/build --parallel

# ── Stage 2: ailake Spark + Flink plugins (Gradle/Scala/Kotlin — separate JDK base,
#    decoupled from the Rust/Python builder above) ─────────────────────────────
FROM gradle:8.7-jdk17 AS jvm-builder

COPY spark-plugin /spark-plugin
WORKDIR /spark-plugin
# shadowJar bundles JNA (Spark doesn't provide it); excludes Spark/Scala itself
# (provided by the cluster at runtime). See spark-plugin/build.gradle.kts.
RUN gradle shadowJar -x test --no-daemon --no-configuration-cache

# ailake-flink connector — needed here (not just in Dockerfile.flink's server image)
# because Flink's SQL Client resolves DynamicTableFactory implementations via
# ServiceLoader against its *own* classpath during query planning, which happens
# client-side before submission — confirmed live: the jar must be in the *client's*
# lib/ dir too, or every CREATE TABLE fails with "Could not find any factory for
# identifier 'ailake'" even though the connector is correctly installed on the
# (remote) server. See 14_flink.ipynb intro.
COPY ailake-flink /ailake-flink
WORKDIR /ailake-flink
RUN gradle shadowJar -x test --no-daemon --no-configuration-cache

# ── Stage 3: JupyterLab runtime ───────────────────────────────────────────────
FROM python:3.12-slim

# Java is required by PySpark (notebook 03_spark.ipynb uses local[*] mode).
# default-jre-headless installs OpenJDK 17 and creates /usr/lib/jvm/default-java.
RUN apt-get update && apt-get install -y --no-install-recommends \
        default-jre-headless \
    && rm -rf /var/lib/apt/lists/*

ENV JAVA_HOME=/usr/lib/jvm/default-java
ENV PATH="${JAVA_HOME}/bin:${PATH}"

COPY --from=builder /wheels /wheels
COPY --from=builder /target/release/ailake /usr/local/bin/ailake
RUN chmod +x /usr/local/bin/ailake && ailake --version

# duckdb-ailake extension + its native lib — pinned duckdb==1.5.0 below must match
# -DDUCKDB_VERSION=v1.5.0 in the builder stage (see duckdb-ailake/README.md).
COPY --from=builder /target/release/libailake_jni.so /usr/local/lib/libailake_jni.so
COPY --from=builder /duckdb-ailake/build/ailake.duckdb_extension /usr/local/lib/ailake.duckdb_extension
ENV AILAKE_JNI_LIB=/usr/local/lib/libailake_jni.so
ENV AILAKE_DUCKDB_EXT=/usr/local/lib/ailake.duckdb_extension

# ailake Spark plugin — glob-copied since the shadowJar filename embeds the
# project version (spark-plugin-<version>-plugin.jar).
COPY --from=jvm-builder /spark-plugin/build/libs/spark-plugin-*-plugin.jar /usr/local/lib/ailake-spark-plugin.jar
ENV AILAKE_SPARK_JAR=/usr/local/lib/ailake-spark-plugin.jar

# Flink SQL Client — a plain client install (no local cluster started here), configured
# to submit to the `flink` compose service over the network. Notebook 14_flink.ipynb
# needs it because query vectors are passed as Flink *job parameters*
# (`-Dpipeline.global-job-parameters.ailake.query.vector=...`, a process-launch flag —
# see Dockerfile.flink), which no lightweight REST/DB-API client can set; PyFlink was
# considered and rejected — apache-flink==1.18.1 pins numpy==1.21.4 via its apache-beam
# dependency, which requires Python <3.11 and cannot install on this image's Python 3.12.
# execution.target=remote is the fixed baseline (this client never starts its own
# minicluster); rest.address/rest.port are passed per-invocation as -D flags from the
# notebook (FLINK_HOST/FLINK_PORT env vars — same convention as TRINO_HOST/TRINO_PORT),
# not baked in here, so the image doesn't hardcode the compose service name.
RUN curl -fsSL https://archive.apache.org/dist/flink/flink-1.18.1/flink-1.18.1-bin-scala_2.12.tgz \
        -o /tmp/flink.tgz \
    && mkdir -p /opt/flink-client \
    && tar -xzf /tmp/flink.tgz -C /opt/flink-client --strip-components=1 \
    && rm /tmp/flink.tgz \
    && printf 'execution.target: remote\n' >> /opt/flink-client/conf/flink-conf.yaml
COPY --from=jvm-builder /ailake-flink/build/libs/ailake-flink-*-plugin.jar /opt/flink-client/lib/ailake-flink-plugin.jar
ENV FLINK_CLIENT_HOME=/opt/flink-client
ENV PATH="${FLINK_CLIENT_HOME}/bin:${PATH}"

RUN pip install --no-cache-dir \
        /wheels/*.whl \
        jupyterlab \
        "pyiceberg[pyarrow,sql-sqlite]" \
        duckdb==1.5.0 \
        "pyspark<4.0" \
        trino \
        "google-cloud-bigquery>=3.13" \
        google-auth \
        pandas \
        numpy \
        pyarrow \
        polars \
        boto3 \
    && rm -rf /wheels

# Pre-download Iceberg Spark runtime JAR — avoids Ivy/Maven download at notebook runtime.
# Placed in /opt/iceberg/ so spark.jars can reference a deterministic path.
RUN python3 -c "\
import urllib.request, pathlib; \
jar_url='https://repo1.maven.org/maven2/org/apache/iceberg/iceberg-spark-runtime-3.5_2.12/1.7.0/iceberg-spark-runtime-3.5_2.12-1.7.0.jar'; \
dest=pathlib.Path('/opt/iceberg/iceberg-spark-runtime-3.5_2.12-1.7.0.jar'); \
dest.parent.mkdir(parents=True, exist_ok=True); \
urllib.request.urlretrieve(jar_url, str(dest)); \
print(f'Iceberg JAR: {dest.stat().st_size} bytes', flush=True)"

COPY tests/docker/demo/init_demo.py              /opt/init_demo.py
COPY tests/docker/demo/entrypoint.sh             /entrypoint.sh
COPY tests/docker/demo/notebooks                 /notebooks
COPY airbyte-destination-ailake                  /opt/airbyte-destination-ailake
COPY airbyte-destination-ailake/demo/embed_cmd.py /opt/demo/embed_cmd.py

RUN pip install --no-cache-dir \
        "airbyte-cdk>=1.0" \
        /opt/airbyte-destination-ailake \
    && rm -rf /opt/airbyte-destination-ailake

RUN chmod +x /entrypoint.sh

WORKDIR /notebooks
EXPOSE 8888
ENTRYPOINT ["/entrypoint.sh"]
