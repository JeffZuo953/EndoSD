#!/usr/bin/env bash
set -euo pipefail

# Launch StereoMIS LS cache jobs on GPUs 0-5 in two waves (6 + 2 jobs).

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/.." && pwd)
cd "${REPO_ROOT}"

JOB_FILE="${REPO_ROOT}/configs/cache_jobs.json"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

BATCH_SIZE="${BATCH_SIZE:-4}"
NUM_WORKERS="${NUM_WORKERS:-32}"

GPU_POOL=(0 1 2 3 4 5)

run_wave() {
  local -n jobs_ref=$1
  echo "Launching wave: ${jobs_ref[*]}"
  for idx in "${!jobs_ref[@]}"; do
    local job="${jobs_ref[$idx]}"
    local gpu="${GPU_POOL[$idx]}"
    echo "  [GPU ${gpu}] ${job}"
    CUDA_VISIBLE_DEVICES="${gpu}" CACHE_RESIZE_DEVICE="cuda:${gpu}" \
      python -m dataset.cache_utils_data \
        --job "${job}" \
        --job-file "${JOB_FILE}" \
        --batch-size "${BATCH_SIZE}" \
        --num-workers "${NUM_WORKERS}" &
  done
  wait
}

WAVE1=(stereomis_ls_part01 stereomis_ls_part02 stereomis_ls_part03 stereomis_ls_part04 stereomis_ls_part05 stereomis_ls_part06)
WAVE2=(stereomis_ls_part07 stereomis_ls_part08)

run_wave WAVE1
run_wave WAVE2

echo "All StereoMIS LS cache jobs finished."
