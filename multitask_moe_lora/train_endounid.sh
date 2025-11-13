#!/bin/bash
set -euo pipefail

# Convenience launcher for endounid (mtlga + adaptive tokens + GA loss).

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

DEFAULT_CUDA_DEVICES="2,3,4,5"
DEFAULT_NUM_GPUS=4
DEFAULT_BASE_DATA_PATH="/data/ziyi/multitask"
DEFAULT_BATCH_SIZE=3
DEFAULT_SEG_BATCH_SIZE=3
FIXED_LEARNING_RATE="5e-5"
FIXED_LR_DEPTH="5e-5"
FIXED_LR_SEG="5e-4"
FIXED_LR_CAMERA="5e-5"

CUDA_DEVICES="${CUDA_DEVICES:-${DEFAULT_CUDA_DEVICES}}"
DATA_PROFILE="NO"
BASE_DATA_PATH_CLI=""
BATCH_SIZE_CLI=""
SEG_BATCH_SIZE_CLI=""
TOKEN_COUNT_CLI=""
PASS_ARGS=()

usage() {
    cat <<'EOF'
Usage: bash train_endounid.sh [--cuda DEVICES] [--profile LS|NO] [--base-path PATH] [--batch-size N] [--seg-bs N] [--tokens N] [extra args...]

Adds GA loss (weight=0.1, start_epoch=15) and semantic tokens (default count=10) while running MODE=endounid.
Arguments after "--" are forwarded to train_lora.sh.
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --cuda)
            [[ $# -ge 2 ]] || { echo "--cuda requires DEVICES" >&2; exit 1; }
            CUDA_DEVICES="$2"
            shift 2
            ;;
        --profile)
            [[ $# -ge 2 ]] || { echo "--profile requires LS or NO" >&2; exit 1; }
            DATA_PROFILE="$(echo "$2" | tr '[:lower:]' '[:upper:]')"
            shift 2
            ;;
        --base-path)
            [[ $# -ge 2 ]] || { echo "--base-path requires a path" >&2; exit 1; }
            BASE_DATA_PATH_CLI="$2"
            shift 2
            ;;
        --batch-size)
            [[ $# -ge 2 ]] || { echo "--batch-size requires an integer" >&2; exit 1; }
            BATCH_SIZE_CLI="$2"
            shift 2
            ;;
        --seg-bs|--seg-batch-size)
            [[ $# -ge 2 ]] || { echo "--seg-bs requires an integer" >&2; exit 1; }
            SEG_BATCH_SIZE_CLI="$2"
            shift 2
            ;;
        --tokens)
            [[ $# -ge 2 ]] || { echo "--tokens requires an integer" >&2; exit 1; }
            TOKEN_COUNT_CLI="$2"
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        --)
            shift
            PASS_ARGS+=("$@")
            break
            ;;
        *)
            PASS_ARGS+=("$1")
            shift
            ;;
    esac
done

case "${DATA_PROFILE}" in
    LS|NO)
        ;;
    *)
        echo "Unsupported --profile '${DATA_PROFILE}'. Use LS or NO." >&2
        exit 1
        ;;
esac

if [[ -n "${BASE_DATA_PATH_CLI}" ]]; then
    export BASE_DATA_PATH="${BASE_DATA_PATH_CLI}"
else
    export BASE_DATA_PATH="${DEFAULT_BASE_DATA_PATH}"
fi

if [[ -z "${NUM_GPUS:-}" ]]; then
    IFS=',' read -ra __cuda_list <<<"${CUDA_DEVICES}"
    NUM_GPUS=${#__cuda_list[@]}
fi

DEFAULT_PRETRAIN="/home/ziyi/checkpoint_best_absrel_combined.pth"
if [[ -z "${PRETRAINED_WEIGHTS:-}" && -f "${DEFAULT_PRETRAIN}" ]]; then
    export PRETRAINED_WEIGHTS="${DEFAULT_PRETRAIN}"
fi

export CUDA_DEVICES
export NUM_GPUS=${NUM_GPUS:-${DEFAULT_NUM_GPUS}}
export MODE="endounid"
export GA_LOSS_WEIGHT="${GA_LOSS_WEIGHT:-0.05}"
export GA_LOSS_START_EPOCH="${GA_LOSS_START_EPOCH:-15}"
export LEARNING_RATE="${FIXED_LEARNING_RATE}"
export LR_DEPTH="${FIXED_LR_DEPTH}"
export LR_SEG="${FIXED_LR_SEG}"
export LR_CAMERA="${FIXED_LR_CAMERA}"

if [[ -n "${BATCH_SIZE_CLI}" ]]; then
    export BATCH_SIZE="${BATCH_SIZE_CLI}"
elif [[ -z "${BATCH_SIZE:-}" ]]; then
    export BATCH_SIZE="${DEFAULT_BATCH_SIZE}"
fi
if [[ -n "${SEG_BATCH_SIZE_CLI}" ]]; then
    export SEG_BATCH_SIZE="${SEG_BATCH_SIZE_CLI}"
elif [[ -z "${SEG_BATCH_SIZE:-}" ]]; then
    export SEG_BATCH_SIZE="${DEFAULT_SEG_BATCH_SIZE}"
fi
if [[ -n "${TOKEN_COUNT_CLI}" ]]; then
    export SEMANTIC_TOKEN_COUNT="${TOKEN_COUNT_CLI}"
elif [[ -z "${SEMANTIC_TOKEN_COUNT:-}" ]]; then
    export SEMANTIC_TOKEN_COUNT=10
fi

exec bash "${SCRIPT_DIR}/train_lora.sh" --profile "${DATA_PROFILE}" "${PASS_ARGS[@]}"
