#!/usr/bin/env bash
# Aggregate depth histograms for every cache_pt dataset and plot them.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
OUTPUT_DIR="${REPO_ROOT}/plot"
mkdir -p "${OUTPUT_DIR}"

CSV_PATH="${OUTPUT_DIR}/depth_histograms.csv"
PLOT_PATH="${OUTPUT_DIR}/depth_violin.png"
CACHE_ROOT="/data/ziyi/multitask/data"

echo "[1/2] Computing histograms into ${CSV_PATH}"
python "${REPO_ROOT}/tools/depth_distribution/compute_depth_histograms.py" \
  --cache-root "${CACHE_ROOT}" \
  --allow-missing \
  --output "${CSV_PATH}"

echo "[2/2] Rendering violin plot into ${PLOT_PATH}"
python "${REPO_ROOT}/tools/depth_distribution/plot_violin_from_csv.py" \
  --csv "${CSV_PATH}" \
  --output "${PLOT_PATH}"

echo "Done. Outputs stored in ${OUTPUT_DIR}"
