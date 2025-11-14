#!/bin/bash
set -euo pipefail

# Sequential launcher for:
#   1) Legacy LoRA (MODE=legacy-lora)
#   2) MTLGA (MODE=mtlga, Gram-alignment enabled)
# Both stages share CLI knobs and default to 50 training epochs.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

DEFAULT_CUDA_DEVICES="0,1,2,3"
DEFAULT_NUM_GPUS=4
DEFAULT_BASE_DATA_PATH="/mnt/DATA/ziyi/multitask"
DEFAULT_BATCH_SIZE=3
DEFAULT_SEG_BATCH_SIZE=3
DEFAULT_EPOCHS=50
DEFAULT_GA_WEIGHT=0.05
DEFAULT_GA_START_EPOCH=15

CUDA_DEVICES="${CUDA_DEVICES:-${DEFAULT_CUDA_DEVICES}}"
DATA_PROFILE="NO"
BASE_DATA_PATH_CLI=""
BATCH_SIZE_CLI=""
SEG_BATCH_SIZE_CLI=""
EPOCHS_CLI=""
GA_WEIGHT_CLI=""
GA_START_CLI=""
PASS_ARGS=()

usage() {
    cat <<'EOF'
Usage: bash train_dual_legacy_mtlga.sh [--cuda DEVICES] [--profile NO|LS] [--base-path PATH]
                                       [--batch-size N] [--seg-bs N] [--epochs N]
                                       [--ga-weight FLOAT] [--ga-start EPOCH]
                                       [extra args...]

Runs legacy LoRA then MTLGA sequentially via train_lora.sh.
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
        --epochs)
            [[ $# -ge 2 ]] || { echo "--epochs requires INT" >&2; exit 1; }
            EPOCHS_CLI="$2"
            shift 2
            ;;
        --ga-weight)
            [[ $# -ge 2 ]] || { echo "--ga-weight requires FLOAT" >&2; exit 1; }
            GA_WEIGHT_CLI="$2"
            shift 2
            ;;
        --ga-start)
            [[ $# -ge 2 ]] || { echo "--ga-start requires EPOCH" >&2; exit 1; }
            GA_START_CLI="$2"
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

EPOCH_TARGET="${EPOCHS_CLI:-${EPOCHS:-${DEFAULT_EPOCHS}}}"
GA_WEIGHT="${GA_WEIGHT_CLI:-${GA_LOSS_WEIGHT:-${DEFAULT_GA_WEIGHT}}}"
GA_START="${GA_START_CLI:-${GA_LOSS_START_EPOCH:-${DEFAULT_GA_START_EPOCH}}}"

export CUDA_DEVICES
export NUM_GPUS=${NUM_GPUS:-${DEFAULT_NUM_GPUS}}

run_mode() {
    local mode="$1"
    local label="$2"
    local enable_ga="$3"
    echo "============================================================"
    echo "Launching ${label} (MODE=${mode})"
    echo "============================================================"
    (
        export MODE="${mode}"
        export EPOCHS="${EPOCH_TARGET}"
        if [[ "${enable_ga}" == "true" ]]; then
            export GA_LOSS_WEIGHT="${GA_WEIGHT}"
            export GA_LOSS_START_EPOCH="${GA_START}"
        else
            unset GA_LOSS_WEIGHT GA_LOSS_START_EPOCH || true
        fi
        bash "${SCRIPT_DIR}/train_lora.sh" --profile "${DATA_PROFILE}" "${PASS_ARGS[@]}"
    )
}

run_mode "legacy-lora" "Legacy LoRA" "false"
run_mode "mtlga" "MTLGA" "true"
