#!/usr/bin/env bash
set -euo pipefail

TABLE_PATH="${DEMO_TABLE_PATH:-/data/ailake_demo}"
VERSION_HINT="${TABLE_PATH}/default/table/metadata/version-hint.text"
STAMP_FILE="$(dirname "${TABLE_PATH}")/.fixture-version"

# The `demo-data` named volume outlives `docker compose build` — an image
# rebuild that changes init_demo.py (new table, new property, new arg)
# does NOT touch fixtures already on disk. Without this check the container
# would silently keep serving stale fixtures forever after a code change,
# since VERSION_HINT alone only proves *some* fixture was generated once,
# not that it matches the current init_demo.py. We additionally compare a
# version stamp written by init_demo.py (FIXTURE_VERSION) against the one
# baked into this image, and force a regen on mismatch.
IMAGE_FIXTURE_VERSION="$(python -c "import sys; sys.path.insert(0, '/opt'); import init_demo; print(init_demo.FIXTURE_VERSION)")"
DISK_FIXTURE_VERSION="$(cat "${STAMP_FILE}" 2>/dev/null || true)"

if [ -f "${VERSION_HINT}" ] && [ "${DISK_FIXTURE_VERSION}" = "${IMAGE_FIXTURE_VERSION}" ]; then
    echo "=== Demo data already present (fixture v${DISK_FIXTURE_VERSION}) — skipping fixture generation ==="
    # Re-register in Nessie on every startup (Nessie is in-memory; loses state on restart)
    python /opt/init_demo.py --nessie-only 2>&1 || true
else
    if [ -f "${VERSION_HINT}" ]; then
        echo "=== Stale demo data on volume (disk=${DISK_FIXTURE_VERSION:-<none>} image=${IMAGE_FIXTURE_VERSION}) — wiping and regenerating ==="
        # init_demo.py only ever inserts/commits — it never truncates a table.
        # Regenerating against leftover directories would append on top of
        # the stale rows instead of replacing them, so wipe first.
        rm -rf "$(dirname "${TABLE_PATH}")"/*
    else
        echo "=== AI-Lake Demo: generating fixture data ==="
    fi
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
