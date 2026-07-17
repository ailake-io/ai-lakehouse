FROM gcr.io/oss-fuzz-base/base-builder-rust:latest

RUN git clone --depth 1 https://github.com/anomalyco/ai-lakehouse.git ai-lakehouse

WORKDIR ai-lakehouse
COPY .oss-fuzz/build.sh /srvc/build.sh
COPY fuzz/ fuzz/
