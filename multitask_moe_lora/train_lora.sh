#!/bin/bash
set -euo pipefail

usage() {
    cat <<'EOF'
Usage: bash train_lora [DATA_PROFILE] [--debug] [--profile NAME]

DATA_PROFILE can be ENDO / EndoSynth / NO / LS (case-insensitive).
--debug        : enable debug mode (also honored via DEBUG=1/true), switches save path to .../train_lora_debug.
--profile NAME : same as providing DATA_PROFILE positionally.
EOF
}

pick_master_port() {
    python - <<'PY'
import socket
with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    s.bind(("", 0))
    print(s.getsockname()[1])
PY
}

# =============================================
# Generic Multi-Mode LoRA training launcher
# Supports NO / LS / EndoSynth datasets + 6 PEFT modes
# =============================================

export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1
export PYTHONPATH="$(dirname "$(pwd)"):${PYTHONPATH:-}"

###############################################
# CLI parsing
###############################################
DEBUG_MODE=false
DATA_PROFILE_CLI=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --debug)
            DEBUG_MODE=true
            shift
            ;;
        --profile)
            if [[ $# -lt 2 ]]; then
                echo "--profile expects an argument" >&2
                usage
                exit 1
            fi
            DATA_PROFILE_CLI="$2"
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        ENDO|endo|Endo|EndoSynth|endosynth)
            DATA_PROFILE_CLI="ENDO"
            shift
            ;;
        NO|no)
            DATA_PROFILE_CLI="NO"
            shift
            ;;
        LS|ls)
            DATA_PROFILE_CLI="LS"
            shift
            ;;
        *)
            echo "Unknown argument: $1" >&2
            usage
            exit 1
            ;;
    esac
done

if [[ "${DEBUG_MODE}" != "true" ]]; then
    case "${DEBUG:-}" in
        1|true|TRUE|True|yes|YES)
            DEBUG_MODE=true
            ;;
        *)
            DEBUG_MODE=false
            ;;
    esac
fi

###############################################
# Shared knobs (override via environment vars)
###############################################
NUM_GPUS=${NUM_GPUS:-1}
CUDA_DEVICES=${CUDA_DEVICES:-"5"}
ENCODER=${ENCODER:-"vitb"}
FEATURES=${FEATURES:-64}
NUM_CLASSES=${NUM_CLASSES:-10}
MAX_DEPTH=${MAX_DEPTH:-0.3}
# Use depth feature taps for segmentation to ensure identical layers
SEG_INPUT_TYPE=${SEG_INPUT_TYPE:-"from_depth"}
SEG_HEAD_TYPE=${SEG_HEAD_TYPE:-"linear"}

EPOCHS=${EPOCHS:-50}
BATCH_SIZE=${BATCH_SIZE:-12}
SEG_BATCH_SIZE=${SEG_BATCH_SIZE:-12}
VAL_BATCH_SIZE=${VAL_BATCH_SIZE:-36}
LEARNING_RATE=${LEARNING_RATE:-5e-5}
WEIGHT_DECAY=${WEIGHT_DECAY:-0.01}
IMG_SIZE=${IMG_SIZE:-518}
USE_MIXED_PRECISION=${USE_MIXED_PRECISION:-true}
FROZEN_BACKBONE=${FROZEN_BACKBONE:-false}

# Mode selection: original | lora-only | legacy-lora | endo-unid | mtlora | mtlga
MODE=${MODE:-"legacy-lora"}

# Dataset profile: ENDO (EndoSynth-only), NO (multi-domain no_bundle), LS (ls_bundle)
DATA_PROFILE=${DATA_PROFILE_CLI:-${DATA_PROFILE:-"ENDO"}}
DATA_PROFILE=$(echo "${DATA_PROFILE}" | tr '[:lower:]' '[:upper:]')
if [[ "${DATA_PROFILE}" == "ENDOSYNTH" ]]; then
    DATA_PROFILE="ENDO"
fi

###############################################
# Dataset-specific switches
###############################################
case "${DATA_PROFILE}" in
    ENDO)
        DATASET_CONFIG_NAME="endosynth_only"
        PATH_TRANSFORM_NAME="none"
        DATASET_MODALITY="mt"
        TRAIN_DATASET_INCLUDE="EndoSynth"
        VAL_DATASET_INCLUDE="EndoSynth"
        PROFILE_NUM_CLASSES=10
        ;;
    NO)
        DATASET_CONFIG_NAME="no_bundle"
        PATH_TRANSFORM_NAME="none"
        DATASET_MODALITY="mt"
        TRAIN_DATASET_INCLUDE=""
        VAL_DATASET_INCLUDE=""
        PROFILE_NUM_CLASSES=4
        ;;
    LS)
        DATASET_CONFIG_NAME="ls_bundle"
        PATH_TRANSFORM_NAME="none"
        DATASET_MODALITY="mt"
        TRAIN_DATASET_INCLUDE=""
        VAL_DATASET_INCLUDE=""
        PROFILE_NUM_CLASSES=10
        ;;
    *)
        echo "Unsupported DATA_PROFILE='${DATA_PROFILE}'. Choose ENDO | NO | LS."
        exit 1
        ;;
esac

NUM_CLASSES=${NUM_CLASSES:-${PROFILE_NUM_CLASSES}}

###############################################
# Mode-specific defaults / extras
###############################################
GA_LOSS_WEIGHT=${GA_LOSS_WEIGHT:-0.02}
GA_LOSS_START_EPOCH=${GA_LOSS_START_EPOCH:-50}

# EndoUniD adapter defaults
ENDO_SHARED_SHARDS=${ENDO_SHARED_SHARDS:-2}
ENDO_SHARED_R=${ENDO_SHARED_R:-4}
ENDO_SHARED_ALPHA=${ENDO_SHARED_ALPHA:-8}
ENDO_DEPTH_R=${ENDO_DEPTH_R:-8}
ENDO_DEPTH_ALPHA=${ENDO_DEPTH_ALPHA:-16}
ENDO_SEG_R=${ENDO_SEG_R:-8}
ENDO_SEG_ALPHA=${ENDO_SEG_ALPHA:-16}
ENDO_CAMERA_R=${ENDO_CAMERA_R:-4}
ENDO_CAMERA_ALPHA=${ENDO_CAMERA_ALPHA:-8}
ENDO_DROPOUT=${ENDO_DROPOUT:-0.0}

EXTRA_MODE_ARGS=()
case "${MODE}" in
    original)
        ;;
    lora-only|legacy-lora|mtlora|mtlga)
        EXTRA_MODE_ARGS+=(--lora-r "${LORA_R:-4}" --lora-alpha "${LORA_ALPHA:-8}")
        ;;
    endo-unid)
        EXTRA_MODE_ARGS+=(
            --endo-unid-shared-shards "${ENDO_SHARED_SHARDS}"
            --endo-unid-shared-r "${ENDO_SHARED_R}"
            --endo-unid-shared-alpha "${ENDO_SHARED_ALPHA}"
            --endo-unid-depth-r "${ENDO_DEPTH_R}"
            --endo-unid-depth-alpha "${ENDO_DEPTH_ALPHA}"
            --endo-unid-seg-r "${ENDO_SEG_R}"
            --endo-unid-seg-alpha "${ENDO_SEG_ALPHA}"
            --endo-unid-camera-r "${ENDO_CAMERA_R}"
            --endo-unid-camera-alpha "${ENDO_CAMERA_ALPHA}"
            --endo-unid-dropout "${ENDO_DROPOUT}"
        )
        ;;
    *)
        echo "Unsupported MODE='${MODE}'."
        exit 1
        ;;
esac

if [[ "${MODE}" == "mtlga" ]]; then
    EXTRA_MODE_ARGS+=(--ga-loss-weight "${GA_LOSS_WEIGHT}")
    EXTRA_MODE_ARGS+=(--ga-loss-start-epoch "${GA_LOSS_START_EPOCH}")
fi

###############################################
# Paths / logging
###############################################
BASE_DATA_PATH=${BASE_DATA_PATH:-"/data/ziyi/multitask"}
PRETRAINED_WEIGHTS=${PRETRAINED_WEIGHTS:-"${BASE_DATA_PATH}/save/FM/fd_vitb_fd_depth_fm_v1_camera_simple_train1_20251111_131329/checkpoint_epoch_30.pth"}
RESUME_CHECKPOINT=${RESUME_CHECKPOINT:-""}

DEBUG_SUFFIX=""
if [[ "${DEBUG_MODE}" == "true" ]]; then
    DEBUG_SUFFIX="_debug"
fi

BASE_SAVE_PATH="${BASE_DATA_PATH}/save/train_lora${DEBUG_SUFFIX}"
RUN_TAG="${DATA_PROFILE}_${ENCODER}_${MODE}_$(date +%Y%m%d_%H%M%S)"
SAVE_PATH="${BASE_SAVE_PATH}/multitask_${RUN_TAG}"
mkdir -p "${SAVE_PATH}"
exec > >(tee -a "${SAVE_PATH}/training.log") 2>&1

echo "============================================================"
echo " Multi-mode LoRA Training Launcher"
echo "------------------------------------------------------------"
echo "  GPUs:                ${NUM_GPUS} (CUDA_VISIBLE_DEVICES=${CUDA_DEVICES})"
echo "  Encoder:             ${ENCODER} | Mode: ${MODE}"
echo "  Dataset Profile:     ${DATA_PROFILE} (${DATASET_CONFIG_NAME})"
echo "  Batch Sizes:         depth=${BATCH_SIZE}, seg=${SEG_BATCH_SIZE}, val=${VAL_BATCH_SIZE}"
echo "  Image Size:          ${IMG_SIZE}"
echo "  Save Path:           ${SAVE_PATH}"
if [[ "${MODE}" == "mtlga" ]]; then
    echo "  GA Loss:             weight=${GA_LOSS_WEIGHT}, start_epoch=${GA_LOSS_START_EPOCH}"
fi
echo "============================================================"

if [ -n "${PRETRAINED_WEIGHTS}" ] && [ ! -f "${PRETRAINED_WEIGHTS}" ]; then
    echo "Warning: pretrained weights not found (${PRETRAINED_WEIGHTS}), fallback to random init."
    PRETRAINED_WEIGHTS=""
fi

export CUDA_VISIBLE_DEVICES=${CUDA_DEVICES}
export WORLD_SIZE=${NUM_GPUS}
export MASTER_ADDR=localhost
if [[ -z "${MASTER_PORT:-}" ]]; then
    MASTER_PORT=$(pick_master_port 2>/dev/null || true)
fi
if [[ -z "${MASTER_PORT}" ]]; then
    MASTER_PORT=20700
fi
export MASTER_PORT

###############################################
# Build command
###############################################
TRAIN_CMD=(
    python -m torch.distributed.run
    --nproc_per_node=${NUM_GPUS}
    --master_port=${MASTER_PORT}
    -m multitask_moe_lora.train_multitask_depth_seg
    --encoder "${ENCODER}"
    --features "${FEATURES}"
    --num-classes "${NUM_CLASSES}"
    --max-depth "${MAX_DEPTH}"
    --seg-input-type "${SEG_INPUT_TYPE}"
    --seg-head-type "${SEG_HEAD_TYPE}"
    --epochs "${EPOCHS}"
    --bs "${BATCH_SIZE}"
    --seg-bs "${SEG_BATCH_SIZE}"
    --val-bs "${VAL_BATCH_SIZE}"
    --lr "${LEARNING_RATE}"
    --weight-decay "${WEIGHT_DECAY}"
    --img-size "${IMG_SIZE}"
    --mode "${MODE}"
    --dataset-config-name "${DATASET_CONFIG_NAME}"
    --path-transform-name "${PATH_TRANSFORM_NAME}"
    --dataset-modality "${DATASET_MODALITY}"
    --save-path "${SAVE_PATH}"
    --checkpoint-policy "latest-only"
)

if [[ -n "${TRAIN_DATASET_INCLUDE}" ]]; then
    TRAIN_CMD+=(--train-dataset-include "${TRAIN_DATASET_INCLUDE}")
fi
if [[ -n "${VAL_DATASET_INCLUDE}" ]]; then
    TRAIN_CMD+=(--val-dataset-include "${VAL_DATASET_INCLUDE}")
fi

if [[ "${USE_MIXED_PRECISION}" == "true" ]]; then
    TRAIN_CMD+=(--mixed-precision)
fi
if [[ "${FROZEN_BACKBONE}" == "true" ]]; then
    TRAIN_CMD+=(--frozen-backbone)
fi
if [[ -n "${PRETRAINED_WEIGHTS}" ]]; then
    TRAIN_CMD+=(--resume-from "${PRETRAINED_WEIGHTS}")
fi
if [[ -n "${RESUME_CHECKPOINT}" ]]; then
    TRAIN_CMD+=(--resume-from "${RESUME_CHECKPOINT}" --resume-full-state)
fi

TRAIN_CMD+=("${EXTRA_MODE_ARGS[@]}")

echo "Launching command:"
printf ' %q' "${TRAIN_CMD[@]}"
echo
eval "${TRAIN_CMD[@]}"
