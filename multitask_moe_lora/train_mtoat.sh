#!/bin/bash
set -euo pipefail

# Convenience launcher for mtoat (mtlora + adaptive tokens).

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

CUDA_DEVICES="0"
DATA_PROFILE="LS"
BASE_DATA_PATH_CLI=""
BATCH_SIZE_CLI=""
TOKEN_COUNT_CLI=""
PASS_ARGS=()

usage() {
    cat <<'EOF'
Usage: bash train_mtoat.sh [--cuda DEVICES] [--profile LS|NO] [--base-path PATH] [--batch-size N] [--tokens N] [extra args...]

Adds semantic tokens automatically (default count=10) and runs MODE=mtoat.
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
elif [[ -z "${BASE_DATA_PATH:-}" ]]; then
    if [[ -d "/mnt/DATA" ]]; then
        export BASE_DATA_PATH="/mnt/DATA/ziyi/multitask"
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

export CUDA_DEVICES
export NUM_GPUS=${NUM_GPUS:-1}
export MODE="mtoat"

if [[ -n "${BATCH_SIZE_CLI}" ]]; then
    export BATCH_SIZE="${BATCH_SIZE_CLI}"
fi

if [[ -n "${TOKEN_COUNT_CLI}" ]]; then
    export SEMANTIC_TOKEN_COUNT="${TOKEN_COUNT_CLI}"
elif [[ -z "${SEMANTIC_TOKEN_COUNT:-}" ]]; then
    export SEMANTIC_TOKEN_COUNT=10
fi

exec bash "${SCRIPT_DIR}/train_lora.sh" --profile "${DATA_PROFILE}" "${PASS_ARGS[@]}"
