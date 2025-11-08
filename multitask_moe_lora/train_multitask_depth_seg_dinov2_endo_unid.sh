#!/bin/bash
set -euo pipefail

export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1
export PYTHONPATH="$(dirname "$(pwd)"):${PYTHONPATH:-}"

# ==============================================================================
# Multi-Task Training (Depth + Segmentation) — DINOv2 EndoUniD Split LoRA
# ==============================================================================

NUM_GPUS=4
CUDA_DEVICES="0,1,2,3"
ENCODER="vits"           # Supported: vits | vitb
FEATURES=64
NUM_CLASSES=4
MAX_DEPTH=0.2
SEG_INPUT_TYPE="last_four"
CAMERA_HEAD_MODE="simple"

EPOCHS=400
BATCH_SIZE=32
SEG_BATCH_SIZE=32
VAL_BATCH_SIZE=64
LEARNING_RATE=3e-5
LR_DEPTH=""
LR_SEG=""
LR_CAMERA=""
WEIGHT_DECAY=0.01
IMG_SIZE=518
USE_MIXED_PRECISION=true

MODE="endo-unid"
NUM_EXPERTS=8
TOP_K=2

# EndoUniD adapter layout
ENDO_SHARED_SHARDS=2
ENDO_SHARED_R=4
ENDO_SHARED_ALPHA=8
ENDO_DEPTH_R=8
ENDO_DEPTH_ALPHA=16
ENDO_SEG_R=8
ENDO_SEG_ALPHA=16
ENDO_CAMERA_R=4
ENDO_CAMERA_ALPHA=8
ENDO_DROPOUT=0.05

DATASET_CONFIG_NAME="server_hk_01"
PATH_TRANSFORM_NAME="sz_to_hk"
DATASET_MODALITY="mt"
MAX_SAMPLES_PER_DATASET=""

LOSS_WEIGHTING_STRATEGY="uwl"
DEPTH_LOSS_WEIGHT=1.0
SEG_LOSS_WEIGHT=1.0
DWA_TEMPERATURE=2.0
CAMERA_LOSS_WEIGHT=1.0

BASE_DATA_PATH="/data/ziyi/multitask"
PRETRAINED_WEIGHTS="${BASE_DATA_PATH}/pretained/depth_anything_v2_metric_hypersim_vits.pth"
RESUME_CHECKPOINT=""

BASE_SAVE_PATH="${BASE_DATA_PATH}/save/train_EndoUniD"
SAVE_PATH="${BASE_SAVE_PATH}/multitask_${ENCODER}_${MODE}_$(date +%Y%m%d_%H%M%S)"
mkdir -p "${SAVE_PATH}"
exec > >(tee -a "${SAVE_PATH}/training.log") 2>&1

echo "=============================================================================="
echo "EndoUniD multi-task training"
echo "------------------------------------------------------------------------------"
echo "  GPUs:                ${NUM_GPUS} (CUDA_VISIBLE_DEVICES=${CUDA_DEVICES})"
echo "  Encoder/Features:    ${ENCODER} / ${FEATURES}"
echo "  Classes:             ${NUM_CLASSES}"
echo "  Max Depth:           ${MAX_DEPTH}"
echo "  Mode:                ${MODE}"
echo "  Dataset Config:      ${DATASET_CONFIG_NAME}"
echo "  Path Transform:      ${PATH_TRANSFORM_NAME}"
echo "  Save Path:           ${SAVE_PATH}"
echo "=============================================================================="

if [ -n "${PRETRAINED_WEIGHTS}" ] && [ ! -f "${PRETRAINED_WEIGHTS}" ]; then
    echo "Warning: pretrained weights not found (${PRETRAINED_WEIGHTS}), fallback to random init."
    PRETRAINED_WEIGHTS=""
fi

export CUDA_VISIBLE_DEVICES=${CUDA_DEVICES}
export WORLD_SIZE=${NUM_GPUS}
export MASTER_ADDR=localhost
export MASTER_PORT=20710

TRAIN_CMD="python -m torch.distributed.run --nproc_per_node=${NUM_GPUS} --master_port=${MASTER_PORT} -m multitask_moe_lora.train_multitask_depth_seg \
--encoder ${ENCODER} \
--features ${FEATURES} \
--num-classes ${NUM_CLASSES} \
--max-depth ${MAX_DEPTH} \
--seg-input-type ${SEG_INPUT_TYPE} \
--camera-head-mode ${CAMERA_HEAD_MODE} \
--epochs ${EPOCHS} \
--bs ${BATCH_SIZE} \
--seg-bs ${SEG_BATCH_SIZE} \
--val-bs ${VAL_BATCH_SIZE} \
--lr ${LEARNING_RATE} \
--weight-decay ${WEIGHT_DECAY} \
--img-size ${IMG_SIZE} \
--camera-loss-weight ${CAMERA_LOSS_WEIGHT} \
--save-path \"${SAVE_PATH}\" \
--dataset-config-name ${DATASET_CONFIG_NAME} \
--path-transform-name ${PATH_TRANSFORM_NAME} \
--dataset-modality ${DATASET_MODALITY} \
--mode ${MODE} \
--num-experts ${NUM_EXPERTS} \
--top-k ${TOP_K} \
--loss-weighting-strategy ${LOSS_WEIGHTING_STRATEGY} \
--depth-loss-weight ${DEPTH_LOSS_WEIGHT} \
--seg-loss-weight ${SEG_LOSS_WEIGHT} \
--dwa-temperature ${DWA_TEMPERATURE} \
--endo-unid-shared-shards ${ENDO_SHARED_SHARDS} \
--endo-unid-shared-r ${ENDO_SHARED_R} \
--endo-unid-shared-alpha ${ENDO_SHARED_ALPHA} \
--endo-unid-depth-r ${ENDO_DEPTH_R} \
--endo-unid-depth-alpha ${ENDO_DEPTH_ALPHA} \
--endo-unid-seg-r ${ENDO_SEG_R} \
--endo-unid-seg-alpha ${ENDO_SEG_ALPHA} \
--endo-unid-camera-r ${ENDO_CAMERA_R} \
--endo-unid-camera-alpha ${ENDO_CAMERA_ALPHA} \
--endo-unid-dropout ${ENDO_DROPOUT}"

if [ -n "${MAX_SAMPLES_PER_DATASET}" ]; then
    TRAIN_CMD="${TRAIN_CMD} --max-samples-per-dataset ${MAX_SAMPLES_PER_DATASET}"
fi

if [ -n "${LR_DEPTH}" ]; then
    TRAIN_CMD="${TRAIN_CMD} --lr-depth ${LR_DEPTH}"
fi
if [ -n "${LR_SEG}" ]; then
    TRAIN_CMD="${TRAIN_CMD} --lr-seg ${LR_SEG}"
fi
if [ -n "${LR_CAMERA}" ]; then
    TRAIN_CMD="${TRAIN_CMD} --lr-camera ${LR_CAMERA}"
fi

if [ "${USE_MIXED_PRECISION}" = "true" ]; then
    TRAIN_CMD="${TRAIN_CMD} --mixed-precision"
fi

if [ -n "${RESUME_CHECKPOINT}" ]; then
    TRAIN_CMD="${TRAIN_CMD} --resume-from \"${RESUME_CHECKPOINT}\" --resume-full-state"
elif [ -n "${PRETRAINED_WEIGHTS}" ]; then
    TRAIN_CMD="${TRAIN_CMD} --resume-from \"${PRETRAINED_WEIGHTS}\""
fi

echo "Launching training command:"
echo "${TRAIN_CMD}"
eval "${TRAIN_CMD}"
