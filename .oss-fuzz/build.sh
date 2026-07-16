#!/bin/bash -eu

cd "$SRC/ai-lakehouse/fuzz"

# Build all 7 fuzz targets
for target in \
    fuzz_ailake_header \
    fuzz_ailake_file_reader \
    fuzz_avro_manifest \
    fuzz_bincode_hnsw \
    fuzz_parquet_reader \
    fuzz_fts_blob \
    fuzz_json_apis
do
    cargo fuzz build "$target"
    cp "target/x86_64-unknown-linux-gnu/release/$target" "$OUT/"
done
