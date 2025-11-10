#!/bin/bash
# ==============================================================================
# Foundation Depth Camera Training (Simple head) - StereoMIS debug capture
# ==============================================================================
set -euo pipefail

export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}
export NCCL_DEBUG=${NCCL_DEBUG:-INFO}
export TORCH_DISTRIBUTED_DEBUG=${TORCH_DISTRIBUTED_DEBUG:-INFO}
export FM_DEBUG_SAVE_BAD_BATCH=1
export FM_DEBUG_SAVE_BAD_SNAPSHOT=1
export FM_DEBUG_MODE=1
PARENT_DIR="$(dirname "$(pwd)")"
export PYTHONPATH="$(pwd):${PARENT_DIR}:${PYTHONPATH:-}"
export FM_FILTER_SEG_HEAD=${FM_FILTER_SEG_HEAD:-1}

# ------------------------------------------------------------------------------
# Hardware / distributed configuration
# ------------------------------------------------------------------------------
NUM_GPUS=${NUM_GPUS:-3}
CUDA_DEVICES=${CUDA_DEVICES:-"3,4,5"}
MASTER_PORT=${MASTER_PORT:-21766}

# ------------------------------------------------------------------------------
# Core training hyper-parameters
# ------------------------------------------------------------------------------
ENCODER=${ENCODER:-"vits"}
FEATURES=${FEATURES:-64}
EPOCHS=${EPOCHS:-120}
BATCH_SIZE=${BATCH_SIZE:-24}
VAL_BATCH_SIZE=${VAL_BATCH_SIZE:-48}
LEARNING_RATE=${LEARNING_RATE:-3e-6}
WEIGHT_DECAY=${WEIGHT_DECAY:-0.01}
IMG_SIZE=${IMG_SIZE:-518}
MAX_DEPTH=${MAX_DEPTH:-0.3}
MIN_DEPTH=${MIN_DEPTH:-1e-3}
MIXED_PRECISION=${MIXED_PRECISION:-true}
FROZEN_BACKBONE=${FROZEN_BACKBONE:-false}
CAMERA_HEAD_MODE=${CAMERA_HEAD_MODE:-"none"}
CAMERA_LOSS_WEIGHT=${CAMERA_LOSS_WEIGHT:-0.3}
CAMERA_LOSS_TYPE=${CAMERA_LOSS_TYPE:-"l2"}
CAMERA_LR=${CAMERA_LR:-5e-4}
CLIP_GRAD_NORM=${CLIP_GRAD_NORM:-5.0}

TRAIN_SAMPLE_STEP=${TRAIN_SAMPLE_STEP:-1}
VAL_SAMPLE_STEP=${VAL_SAMPLE_STEP:--1}
VAL_MIN_SAMPLES_PER_DATASET=${VAL_MIN_SAMPLES_PER_DATASET:-100}

# ------------------------------------------------------------------------------
# Dataset configuration
# ------------------------------------------------------------------------------
DATASET_CONFIG_NAME=${DATASET_CONFIG_NAME:-"fd_depth_fm_v1"}
DATASET_MODALITY=${DATASET_MODALITY:-"fd"}
PATH_TRANSFORM_NAME=${PATH_TRANSFORM_NAME:-"none"}

TRAIN_DATASET_INCLUDE=${TRAIN_DATASET_INCLUDE:-"StereoMIS"}
VAL_DATASET_INCLUDE=${VAL_DATASET_INCLUDE:-"EndoVis2017"}

# ------------------------------------------------------------------------------
# Checkpoint configuration
# ------------------------------------------------------------------------------
BASE_DATA_PATH=${BASE_DATA_PATH:-"/data/ziyi/multitask"}
PRETRAINED_WEIGHTS=${PRETRAINED_WEIGHTS:-"${BASE_DATA_PATH}/pretained/depth_anything_v2_vits.pth"}
RESUME_CHECKPOINT=${RESUME_CHECKPOINT:-""}

# ------------------------------------------------------------------------------
# Output logging
# ------------------------------------------------------------------------------
SAVE_ROOT=${SAVE_ROOT:-"${BASE_DATA_PATH}/save/FM_debug_bad_batches"}
RUN_ID=$(date +%Y%m%d_%H%M%S)
SAMPLE_TAG="camera_${CAMERA_HEAD_MODE}_train${TRAIN_SAMPLE_STEP}"
SAVE_PATH="${SAVE_ROOT}/fd_${ENCODER}_${DATASET_CONFIG_NAME}_${SAMPLE_TAG}_${RUN_ID}"
mkdir -p "${SAVE_PATH}"

exec > >(tee -a "${SAVE_PATH}/train.log") 2>&1

# ------------------------------------------------------------------------------
# Sanity checks
# ------------------------------------------------------------------------------
if [[ -n "${CUDA_DEVICES}" ]]; then
    export CUDA_VISIBLE_DEVICES=${CUDA_DEVICES}
fi

if [[ -n "${PRETRAINED_WEIGHTS}" && ! -f "${PRETRAINED_WEIGHTS}" ]]; then
    echo "[WARN] Pretrained weights not found at ${PRETRAINED_WEIGHTS}, training will start from scratch."
    PRETRAINED_WEIGHTS=""
fi

# ------------------------------------------------------------------------------
# Assemble training command
# ------------------------------------------------------------------------------
BASE_CMD=(
    python -m torch.distributed.run
    --nproc_per_node="${NUM_GPUS}"
    --master_port="${MASTER_PORT}"
    --module multitask_moe_lora.train_multitask_depth_seg
    --
    --encoder "${ENCODER}"
    --features "${FEATURES}"
    --num-classes 1
    --min-depth "${MIN_DEPTH}"
    --max-depth "${MAX_DEPTH}"
    --epochs "${EPOCHS}"
    --bs "${BATCH_SIZE}"
    --val-bs "${VAL_BATCH_SIZE}"
    --lr "${LEARNING_RATE}"
    --weight-decay "${WEIGHT_DECAY}"
    --img-size "${IMG_SIZE}"
    --dataset-config-name "${DATASET_CONFIG_NAME}"
    --dataset-modality "${DATASET_MODALITY}"
    --path-transform-name "${PATH_TRANSFORM_NAME}"
    --train-sample-step "${TRAIN_SAMPLE_STEP}"
    --val-sample-step "${VAL_SAMPLE_STEP}"
    --val-min-samples-per-dataset "${VAL_MIN_SAMPLES_PER_DATASET}"
    --mode original
    --save-path "${SAVE_PATH}"
    --camera-head-mode "${CAMERA_HEAD_MODE}"
    --camera-loss-weight "${CAMERA_LOSS_WEIGHT}"
    --camera-loss-type "${CAMERA_LOSS_TYPE}"
    --lr-camera "${CAMERA_LR}"
    --clip-grad-norm "${CLIP_GRAD_NORM}"
)

if [[ -n "${TRAIN_DATASET_INCLUDE}" ]]; then
    BASE_CMD+=(--train-dataset-include "${TRAIN_DATASET_INCLUDE}")
fi
if [[ -n "${VAL_DATASET_INCLUDE}" ]]; then
    BASE_CMD+=(--val-dataset-include "${VAL_DATASET_INCLUDE}")
fi
if [[ "${MIXED_PRECISION}" == "true" ]]; then
    BASE_CMD+=(--mixed-precision)
fi
if [[ "${FROZEN_BACKBONE}" == "true" ]]; then
    BASE_CMD+=(--frozen-backbone)
fi
if [[ -n "${PRETRAINED_WEIGHTS}" ]]; then
    BASE_CMD+=(--resume-from "${PRETRAINED_WEIGHTS}")
fi
if [[ -n "${RESUME_CHECKPOINT}" ]]; then
    BASE_CMD+=(--resume-from "${RESUME_CHECKPOINT}" --resume-full-state)
fi

echo "[INFO] Launch command:"
printf ' %q' "${BASE_CMD[@]}"
echo ""

rsync -a --exclude='.git' --exclude='tmp_runs' ./ "${SAVE_PATH}/code_snapshot" >/dev/null 2>&1 || true

"${BASE_CMD[@]}"
