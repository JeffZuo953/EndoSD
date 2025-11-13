#!/bin/bash
set -euo pipefail

# Convenience launcher for mtlga (mtlora + Gram alignment).
# Key knobs:
#   --cuda    : CUDA_VISIBLE_DEVICES string (default: 0)
#   --profile : Dataset profile (LS or NO, default: NO)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

CUDA_DEVICES="0"
DATA_PROFILE="NO"
PASS_ARGS=()

usage() {
    cat <<'EOF'
Usage: bash train_mtlga.sh [--cuda DEVICES] [--profile LS|NO] [extra args...]

Adds GA loss automatically with weight=0.1 starting from epoch 15.
Additional arguments are forwarded to train_lora.sh.
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --cuda)
            [[ $# -ge 2 ]] || { echo "--cuda requires an argument" >&2; exit 1; }
            CUDA_DEVICES="$2"
            shift 2
            ;;
        --profile)
            [[ $# -ge 2 ]] || { echo "--profile requires LS or NO" >&2; exit 1; }
            DATA_PROFILE="$(echo "$2" | tr '[:lower:]' '[:upper:]')"
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

if [[ -z "${NUM_GPUS:-}" ]]; then
    IFS=',' read -ra __cuda_list <<<"${CUDA_DEVICES}"
    NUM_GPUS=${#__cuda_list[@]}
fi

export CUDA_DEVICES
export NUM_GPUS=${NUM_GPUS:-1}
export MODE="mtlga"
export GA_LOSS_WEIGHT=0.1
export GA_LOSS_START_EPOCH=15

exec bash "${SCRIPT_DIR}/train_lora.sh" --profile "${DATA_PROFILE}" "${PASS_ARGS[@]}"
