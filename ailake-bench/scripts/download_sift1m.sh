#!/usr/bin/env bash
# Download SIFT-1M dataset from the texmex corpus.
#
# Usage: ./scripts/download_sift1m.sh [OUTPUT_DIR]
#   OUTPUT_DIR defaults to ./data/sift1m
#
# Produces:
#   sift_base.fvecs       — 1,000,000 × 128-dim float32 vectors (~500 MB)
#   sift_query.fvecs      — 10,000 × 128-dim float32 vectors
#   sift_groundtruth.ivecs — 10,000 × 100 ground truth neighbors (int32)
#   sift_learn.fvecs      — 100,000 training vectors (not used by bench)
#
# Source: http://corpus-texmex.irisa.fr/

set -euo pipefail

OUTPUT_DIR="${1:-./data/sift1m}"
ARCHIVE="sift.tar.gz"
FTP_URL="ftp://ftp.irisa.fr/local/texmex/corpus/${ARCHIVE}"

mkdir -p "${OUTPUT_DIR}"
cd "${OUTPUT_DIR}"

if [[ -f "sift_base.fvecs" && -f "sift_query.fvecs" && -f "sift_groundtruth.ivecs" ]]; then
    echo "Dataset already present in ${OUTPUT_DIR}. Nothing to do."
    exit 0
fi

echo "Downloading SIFT-1M (~161 MB compressed) …"
if command -v curl &>/dev/null; then
    curl -L --retry 3 -o "${ARCHIVE}" "${FTP_URL}"
elif command -v wget &>/dev/null; then
    wget -c "${FTP_URL}" -O "${ARCHIVE}"
else
    echo "Error: neither curl nor wget found." >&2
    exit 1
fi

echo "Extracting …"
tar -xzf "${ARCHIVE}" --strip-components=1
rm -f "${ARCHIVE}"

echo ""
echo "Dataset ready in ${OUTPUT_DIR}:"
ls -lh sift_base.fvecs sift_query.fvecs sift_groundtruth.ivecs
