#!/usr/bin/env bash
set -euo pipefail

TABLE_PATH="${DEMO_TABLE_PATH:-/data/ailake_demo}"
VERSION_HINT="${TABLE_PATH}/default/table/metadata/version-hint.text"

if [ -f "${VERSION_HINT}" ]; then
    echo "=== Demo data already present — skipping init ==="
else
    echo "=== AI-Lake Demo: generating fixture data ==="
    python /opt/init_demo.py
    echo "=== Fixture data ready ==="
fi

echo "=== Starting JupyterLab at http://localhost:8888 ==="
exec jupyter lab \
    --ip=0.0.0.0 \
    --port=8888 \
    --no-browser \
    --allow-root \
    --IdentityProvider.token='' \
    --ServerApp.password='' \
    --notebook-dir=/notebooks
