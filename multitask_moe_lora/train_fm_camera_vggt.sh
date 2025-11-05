#!/bin/bash
# ==============================================================================
# Foundation Depth Camera Training (VGGT-like head)
# ==============================================================================
set -euo pipefail

export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}
PARENT_DIR="$(dirname "$(pwd)")"
export PYTHONPATH="$(pwd):${PARENT_DIR}:${PYTHONPATH:-}"
export FM_FILTER_SEG_HEAD=${FM_FILTER_SEG_HEAD:-1}

# ------------------------------------------------------------------------------
# Hardware / distributed configuration
# ------------------------------------------------------------------------------
NUM_GPUS=3
CUDA_DEVICES="5,4,2"
MASTER_PORT=20763

# ------------------------------------------------------------------------------
# Core training hyper-parameters
# ------------------------------------------------------------------------------
ENCODER="vits"        # {vits, vitb, vitl, dinov3_*}
FEATURES=64
EPOCHS=120
BATCH_SIZE=24
VAL_BATCH_SIZE=96
LEARNING_RATE=5e-6
WEIGHT_DECAY=0.01
IMG_SIZE=518
MAX_DEPTH=0.3
MIN_DEPTH=1e-8
MIXED_PRECISION=true
FROZEN_BACKBONE=false
CAMERA_HEAD_MODE="vggtlike"
CAMERA_LOSS_WEIGHT=5.0
CAMERA_LR=1e-3

FM_SAMPLE_MODE="full"   # full | sample
FM_SAMPLE_SIZE=10
TRAIN_SAMPLE_STEP=150
VAL_SAMPLE_STEP=20
MAX_SAMPLES_PER_DATASET=""

# ------------------------------------------------------------------------------
# Dataset configuration
# NOTE: define `fd_depth_fm_v1` in util/data_utils.DATASET_PATHS to point to
#       the exact caches/filelists for the datasets enumerated below.
# ------------------------------------------------------------------------------
DATASET_CONFIG_NAME="fd_depth_fm_v1"
DATASET_MODALITY="fd"       # depth-only foundation mode
PATH_TRANSFORM_NAME="none"
MAX_SAMPLES_PER_DATASET=${MAX_SAMPLES_PER_DATASET}

TRAIN_DATASET_INCLUDE="SCARED,StereoMIS,dVPN,C3VDv2,SimCol,Kidney3D"
VAL_DATASET_INCLUDE="EndoNeRF,C3VD,EndoMapper"

# ------------------------------------------------------------------------------
# Checkpoint configuration
# ------------------------------------------------------------------------------
BASE_DATA_PATH="/data/ziyi/multitask"
PRETRAINED_WEIGHTS="${BASE_DATA_PATH}/pretained/depth_anything_v2_vits.pth"
RESUME_CHECKPOINT=""

# ------------------------------------------------------------------------------
# Output logging
# ------------------------------------------------------------------------------
SAVE_ROOT=${SAVE_ROOT:-"/data/ziyi/multitask/save/FM"}
RUN_ID=$(date +%Y%m%d_%H%M%S)
SAMPLE_TAG="camera_${CAMERA_HEAD_MODE}_train${TRAIN_SAMPLE_STEP}"
SAVE_PATH="${SAVE_ROOT}/fd_${ENCODER}_${DATASET_CONFIG_NAME}_${SAMPLE_TAG}_${RUN_ID}"
mkdir -p "${SAVE_PATH}"

exec > >(tee -a "${SAVE_PATH}/train.log") 2>&1

echo "=============================================================================="
echo "FD Depth Training (Legacy, No-LoRA)"
echo "------------------------------------------------------------------------------"
echo "  Encoder:               ${ENCODER}"
echo "  Features:              ${FEATURES}"
echo "  Epochs:                ${EPOCHS}"
echo "  Batch Size (train):    ${BATCH_SIZE}"
echo "  Batch Size (val):      ${VAL_BATCH_SIZE}"
echo "  LR / WD:               ${LEARNING_RATE} / ${WEIGHT_DECAY}"
echo "  Depth range:           [${MIN_DEPTH}, ${MAX_DEPTH}]"
echo "  Mixed precision:       ${MIXED_PRECISION}"
echo "  Frozen backbone:       ${FROZEN_BACKBONE}"
echo "  Camera head:           ${CAMERA_HEAD_MODE} (weight=${CAMERA_LOSS_WEIGHT})"
echo "  Camera LR:            ${CAMERA_LR}"
echo "  Dataset config:        ${DATASET_CONFIG_NAME}"
echo "  Dataset modality:      ${DATASET_MODALITY}"
echo "  Train include list:    ${TRAIN_DATASET_INCLUDE}"
echo "  Val include list:      ${VAL_DATASET_INCLUDE}"
echo "  Path transform:        ${PATH_TRANSFORM_NAME}"
if [[ -n "${MAX_SAMPLES_PER_DATASET}" ]]; then
    echo "  Max samples / dataset: ${MAX_SAMPLES_PER_DATASET}"
else
    echo "  Max samples / dataset: all"
fi
echo "  Sample mode:           ${FM_SAMPLE_MODE} (size=${FM_SAMPLE_SIZE})"
echo "  Train sample step:     ${TRAIN_SAMPLE_STEP}"
echo "  Save path:             ${SAVE_PATH}"
echo "=============================================================================="
echo ""

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
    --mode original
    --save-path "${SAVE_PATH}"
    --camera-head-mode "${CAMERA_HEAD_MODE}"
    --camera-loss-weight "${CAMERA_LOSS_WEIGHT}"
    --lr-camera "${CAMERA_LR}"
)
if [[ -n "${MAX_SAMPLES_PER_DATASET}" ]]; then
    BASE_CMD+=(--max-samples-per-dataset "${MAX_SAMPLES_PER_DATASET}")
fi

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

# Optional backup of current repo snapshot
rsync -a --exclude='.git' --exclude='tmp_runs' ./ "${SAVE_PATH}/code_snapshot" >/dev/null 2>&1 || true

"${BASE_CMD[@]}"
