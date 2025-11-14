#!/bin/bash
set -euo pipefail

# Sequential launcher for two modes:
#   1) Full fine-tuning (MODE=original, i.e. no LoRA adapters)
#   2) MTOaT (MODE=mtoat, semantic tokens enabled)
#
# Each stage reuses the same CLI options and enforces 50 training epochs
# unless explicitly overridden via --epochs.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

DEFAULT_CUDA_DEVICES="0,1,2,3"
DEFAULT_NUM_GPUS=4
DEFAULT_BASE_DATA_PATH="/mnt/DATA/ziyi/multitask"
DEFAULT_BATCH_SIZE=3
DEFAULT_SEG_BATCH_SIZE=3
DEFAULT_TOKEN_COUNT=10
DEFAULT_EPOCHS=50

CUDA_DEVICES="${CUDA_DEVICES:-${DEFAULT_CUDA_DEVICES}}"
DATA_PROFILE="NO"
BASE_DATA_PATH_CLI=""
BATCH_SIZE_CLI=""
SEG_BATCH_SIZE_CLI=""
TOKEN_COUNT_CLI=""
EPOCHS_CLI=""
PASS_ARGS=()

usage() {
    cat <<'EOF'
Usage: bash train_dual_full_mtoat.sh [--cuda DEVICES] [--profile NO|LS] [--base-path PATH]
                                     [--batch-size N] [--seg-bs N] [--tokens N] [--epochs N]
                                     [extra args...]

Runs two trainings back-to-back (full finetune -> MTOaT) using train_lora.sh.
Arguments after "--" are forwarded verbatim to train_lora.sh.
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
            [[ $# -ge 2 ]] || { echo "--profile requires NO or LS" >&2; exit 1; }
            DATA_PROFILE="$(echo "$2" | tr '[:lower:]' '[:upper:]')"
            shift 2
            ;;
        --base-path)
            [[ $# -ge 2 ]] || { echo "--base-path requires PATH" >&2; exit 1; }
            BASE_DATA_PATH_CLI="$2"
            shift 2
            ;;
        --batch-size)
            [[ $# -ge 2 ]] || { echo "--batch-size requires INT" >&2; exit 1; }
            BATCH_SIZE_CLI="$2"
            shift 2
            ;;
        --seg-bs|--seg-batch-size)
            [[ $# -ge 2 ]] || { echo "--seg-bs requires INT" >&2; exit 1; }
            SEG_BATCH_SIZE_CLI="$2"
            shift 2
            ;;
        --tokens)
            [[ $# -ge 2 ]] || { echo "--tokens requires INT" >&2; exit 1; }
            TOKEN_COUNT_CLI="$2"
            shift 2
            ;;
        --epochs)
            [[ $# -ge 2 ]] || { echo "--epochs requires INT" >&2; exit 1; }
            EPOCHS_CLI="$2"
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
    NO|LS)
        ;;
    *)
        echo "Unsupported --profile '${DATA_PROFILE}'. Use NO or LS." >&2
        exit 1
        ;;
esac

if [[ -n "${BASE_DATA_PATH_CLI}" ]]; then
    export BASE_DATA_PATH="${BASE_DATA_PATH_CLI}"
elif [[ -z "${BASE_DATA_PATH:-}" ]]; then
    if [[ -d "${DEFAULT_BASE_DATA_PATH}" ]]; then
        export BASE_DATA_PATH="${DEFAULT_BASE_DATA_PATH}"
    else
        export BASE_DATA_PATH="/data/ziyi/multitask"
    fi
fi

if [[ -z "${NUM_GPUS:-}" ]]; then
    IFS=',' read -ra __cuda_list <<<"${CUDA_DEVICES}"
    NUM_GPUS=${#__cuda_list[@]}
fi

DEFAULT_PRETRAIN="/home/ziyi/checkpoint_best_absrel_combined.pth"
if [[ -z "${PRETRAINED_WEIGHTS:-}" && -f "${DEFAULT_PRETRAIN}" ]]; then
    export PRETRAINED_WEIGHTS="${DEFAULT_PRETRAIN}"
fi

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

TOKEN_COUNT="${TOKEN_COUNT_CLI:-${SEMANTIC_TOKEN_COUNT:-${DEFAULT_TOKEN_COUNT}}}"
EPOCH_TARGET="${EPOCHS_CLI:-${EPOCHS:-${DEFAULT_EPOCHS}}}"

export CUDA_DEVICES
export NUM_GPUS=${NUM_GPUS:-${DEFAULT_NUM_GPUS}}

run_mode() {
    local mode="$1"
    local label="$2"
    shift 2
    echo "============================================================"
    echo "Launching ${label} (MODE=${mode})"
    echo "============================================================"
    (
        export MODE="${mode}"
        export EPOCHS="${EPOCH_TARGET}"
        if [[ "${mode}" == "mtoat" ]]; then
            export SEMANTIC_TOKEN_COUNT="${TOKEN_COUNT}"
        else
            unset SEMANTIC_TOKEN_COUNT || true
        fi
        bash "${SCRIPT_DIR}/train_lora.sh" --profile "${DATA_PROFILE}" "${PASS_ARGS[@]}"
    )
}

run_mode "original" "Full Finetune (LoRA none)"
run_mode "mtoat" "MTOaT"
