#!/usr/bin/env bash
set -euo pipefail

SRC_ROOT="/data/ziyi/multitask"
REMOTE_USER="${REMOTE_USER:-ziyi}"
REMOTE_HOST="${REMOTE_HOST:-10.147.20.230}"
REMOTE_ROOT="/mnt/DATA/ziyi/multitask"

echo "[INFO] Source footprint:"
du -sh "${SRC_ROOT}"

echo "[INFO] Creating remote directory if missing..."
ssh "${REMOTE_USER}@${REMOTE_HOST}" "mkdir -p '${REMOTE_ROOT}'"

echo "[INFO] Running incremental rsync from ${SRC_ROOT} to ${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_ROOT}"
rsync -a --partial --update --compress --info=progress2 \
    "${SRC_ROOT}/" \
    "${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_ROOT}/"

echo "[DONE] Sync complete."
