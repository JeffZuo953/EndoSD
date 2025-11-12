#!/bin/bash
set -euo pipefail

export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1
export PYTHONPATH="$(dirname "$(pwd)"):${PYTHONPATH:-}"

# ==============================================================================
# Multi-Task Training (Depth + Segmentation) — DINOv2 Legacy LoRA (LS dataset bundle)
# ==============================================================================

NUM_GPUS=1
CUDA_DEVICES="5"
ENCODER="vits"
FEATURES=64
NUM_CLASSES=10
MAX_DEPTH=0.3
SEG_INPUT_TYPE="last_four"

EPOCHS=500
BATCH_SIZE=40
SEG_BATCH_SIZE=40
VAL_BATCH_SIZE=100
LEARNING_RATE=5e-5
LR_DEPTH=""
LR_SEG=""
WEIGHT_DECAY=0.01
IMG_SIZE=518
FROZEN_BACKBONE="true"
USE_MIXED_PRECISION=true

MODE="legacy-lora"
NUM_EXPERTS=8
TOP_K=2
LORA_R=4
LORA_ALPHA=8

DATASET_CONFIG_NAME="ls_bundle"
PATH_TRANSFORM_NAME="none"
DATASET_MODALITY="mt"  # mt (multi-task) or fd (depth-focused) for LS bundle
MAX_SAMPLES_PER_DATASET=""

LOSS_WEIGHTING_STRATEGY="uwl"
DEPTH_LOSS_WEIGHT=1.0
SEG_LOSS_WEIGHT=1.0
DWA_TEMPERATURE=2.0

BASE_DATA_PATH=${BASE_DATA_PATH:-"/data/ziyi/multitask"}
HOME_SSD_PATH=${HOME_SSD_PATH:-"$HOME/ssde"}
PRETRAINED_WEIGHTS="${BASE_DATA_PATH}/pretained/depth_anything_v2_metric_hypersim_vits.pth"
RESUME_CHECKPOINT="/data/ziyi/multitask/save/train_lesion/multitask_LS_vits_legacy-lora_uwl_20251030_124733/checkpoint_full_epoch_400.pth"

BASE_SAVE_PATH="${BASE_DATA_PATH}/save/train_lesion"
# SAVE_PATH="${BASE_SAVE_PATH}/multitask_LS_${ENCODER}_${MODE}_${LOSS_WEIGHTING_STRATEGY}_$(date +%Y%m%d_%H%M%S)"
SAVE_PATH="/data/ziyi/multitask/save/train_lesion/multitask_LS_vits_legacy-lora_uwl_20251030_124733"
mkdir -p "${SAVE_PATH}"
exec > >(tee -a "${SAVE_PATH}/training.log") 2>&1

echo "=============================================================================="
echo "LS bundle training — DINOv2 Legacy LoRA"
echo "------------------------------------------------------------------------------"
echo "  GPUs:                ${NUM_GPUS} (CUDA_VISIBLE_DEVICES=${CUDA_DEVICES})"
echo "  Encoder/Features:    ${ENCODER} / ${FEATURES}"
echo "  Classes:             ${NUM_CLASSES}"
echo "  Max Depth:           ${MAX_DEPTH}"
echo "  Mode:                ${MODE} (LoRA r=${LORA_R}, alpha=${LORA_ALPHA})"
echo "  Dataset Config:      ${DATASET_CONFIG_NAME}"
echo "  Path Transform:      ${PATH_TRANSFORM_NAME}"
echo "  Max Samples/DS:      ${MAX_SAMPLES_PER_DATASET}"
echo "  Save Path:           ${SAVE_PATH}"
echo "=============================================================================="

if [ -n "${PRETRAINED_WEIGHTS}" ] && [ ! -f "${PRETRAINED_WEIGHTS}" ]; then
    echo "Warning: pretrained weights not found (${PRETRAINED_WEIGHTS}), fallback to random init."
    PRETRAINED_WEIGHTS=""
fi

export CUDA_VISIBLE_DEVICES=${CUDA_DEVICES}
export WORLD_SIZE=${NUM_GPUS}
export MASTER_ADDR=localhost
export MASTER_PORT=20610

TRAIN_CMD="python -m torch.distributed.run --nproc_per_node=${NUM_GPUS} --master_port=${MASTER_PORT} -m multitask_moe_lora.train_multitask_depth_seg \
--encoder ${ENCODER} \
--features ${FEATURES} \
--num-classes ${NUM_CLASSES} \
--max-depth ${MAX_DEPTH} \
--seg-input-type ${SEG_INPUT_TYPE} \
--epochs ${EPOCHS} \
--bs ${BATCH_SIZE} \
--seg-bs ${SEG_BATCH_SIZE} \
--val-bs ${VAL_BATCH_SIZE} \
--lr ${LEARNING_RATE} \
--weight-decay ${WEIGHT_DECAY} \
--img-size ${IMG_SIZE} \
--save-path \"${SAVE_PATH}\" \
--dataset-config-name ${DATASET_CONFIG_NAME} \
--path-transform-name ${PATH_TRANSFORM_NAME}"

TRAIN_CMD="${TRAIN_CMD} --dataset-modality ${DATASET_MODALITY}"

if [ -n "${MAX_SAMPLES_PER_DATASET}" ]; then
    TRAIN_CMD="${TRAIN_CMD} --max-samples-per-dataset ${MAX_SAMPLES_PER_DATASET}"
fi

if [ -n "${LR_DEPTH}" ]; then
    TRAIN_CMD="${TRAIN_CMD} --lr-depth ${LR_DEPTH}"
fi
if [ -n "${LR_SEG}" ]; then
    TRAIN_CMD="${TRAIN_CMD} --lr-seg ${LR_SEG}"
fi

if [ "${USE_MIXED_PRECISION}" = "true" ]; then
    TRAIN_CMD="${TRAIN_CMD} --mixed-precision"
fi

if [ -n "${RESUME_CHECKPOINT}" ]; then
    TRAIN_CMD="${TRAIN_CMD} --resume-from \"${RESUME_CHECKPOINT}\" --resume-full-state"
elif [ -n "${PRETRAINED_WEIGHTS}" ]; then
    TRAIN_CMD="${TRAIN_CMD} --resume-from \"${PRETRAINED_WEIGHTS}\""
fi

if [ "${FROZEN_BACKBONE}" = "true" ]; then
    TRAIN_CMD="${TRAIN_CMD} --frozen-backbone"
fi

TRAIN_CMD="${TRAIN_CMD} --mode ${MODE} \
--lora-r ${LORA_R} \
--lora-alpha ${LORA_ALPHA} \
--num-experts ${NUM_EXPERTS} \
--top-k ${TOP_K} \
--loss-weighting-strategy ${LOSS_WEIGHTING_STRATEGY} \
--depth-loss-weight ${DEPTH_LOSS_WEIGHT} \
--seg-loss-weight ${SEG_LOSS_WEIGHT} \
--dwa-temperature ${DWA_TEMPERATURE}"

echo "Launching training command:"
echo "${TRAIN_CMD}"
eval "${TRAIN_CMD}"
