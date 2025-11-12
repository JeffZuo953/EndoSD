#!/bin/bash
set -euo pipefail

export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1

# ==============================================================================
# Multi-Task Training (Depth + Segmentation) — DINOv2 unfrozen
# ==============================================================================

# Basic Configuration
NUM_GPUS=3
# CUDA_DEVICES="0,1,2"
CUDA_DEVICES="3,4,5"
ENCODER="vits"              # vits | vitb | vitl | vitg | dinov3_vits16 | ...
FEATURES=64
NUM_CLASSES=4
MAX_DEPTH=0.2
SEG_INPUT_TYPE="last_four"  # last | last_four | from_depth

# Training Parameters
EPOCHS=500
BATCH_SIZE=20
SEG_BATCH_SIZE=20
VAL_BATCH_SIZE=100
LEARNING_RATE=5e-5          # Base LR; use LR_DEPTH/LR_SEG below to override
# To override task-specific LRs, set these to non-empty values; leave empty to skip
LR_DEPTH=""                 # e.g., 5e-6 (overrides --lr for depth)
LR_SEG=""                   # e.g., 1.5e-5 (overrides --lr*10 for seg)
WEIGHT_DECAY=0.01
IMG_SIZE=518
SAVE_INTERVAL=${SAVE_INTERVAL:-5}
FROZEN_BACKBONE="ture"
USE_MIXED_PRECISION=true  # disabled for stability.

# Mode: original | lora-only | legacy-lora | endo-unid | mtlora | mtlga
MODE="lora-only"
LORA_R=4
LORA_ALPHA=8

# Data source configuration
DATASET_CONFIG_NAME="server_hk_01"  # server_sz | server_hk_01
PATH_TRANSFORM_NAME="sz_to_hk"      # sz_to_hk | none

# Loss weighting strategy: fixed | uwl | dwa
LOSS_WEIGHTING_STRATEGY="uwl"
DEPTH_LOSS_WEIGHT=1.0
SEG_LOSS_WEIGHT=1.0
DWA_TEMPERATURE=2.0

# Data paths
BASE_DATA_PATH=${BASE_DATA_PATH:-"/data/ziyi/multitask"}
HOME_SSD_PATH=${HOME_SSD_PATH:-"$HOME/ssde"}
# PRETRAINED_WEIGHTS="${BASE_DATA_PATH}/pretained/dav2-f.pth"
PRETRAINED_WEIGHTS="${BASE_DATA_PATH}/pretained/train_4_dataset__same_maxdepth_full_20250613_210808__latest.pth"
RESUME_CHECKPOINT=""

# Save path
BASE_SAVE_PATH="${BASE_DATA_PATH}/save/train_lesion"
SAVE_PATH="${BASE_SAVE_PATH}/multitask_${ENCODER}_${MODE}_${LOSS_WEIGHTING_STRATEGY}_$(date +%Y%m%d_%H%M%S)"
mkdir -p "${SAVE_PATH}"
exec > >(tee -a "${SAVE_PATH}/training.log") 2>&1

# ==============================================================================
# Validation echo
# ==============================================================================
echo "=============================================================================="
echo "Multi-Task Training (Depth Estimation + Semantic Segmentation)"
echo "------------------------------------------------------------------------------"
echo "  Encoder:              ${ENCODER}"
echo "  Features:             ${FEATURES}"
echo "  Num Classes:          ${NUM_CLASSES}"
echo "  Max Depth:            ${MAX_DEPTH}"
echo "  Seg Input Type:       ${SEG_INPUT_TYPE}"
echo "  Epochs:               ${EPOCHS}"
echo "  Batch Size (Depth):   ${BATCH_SIZE}"
echo "  Batch Size (Seg):     ${SEG_BATCH_SIZE}"
echo "  Batch Size (Val):     ${VAL_BATCH_SIZE}"
echo "  Learning Rate (Base): ${LEARNING_RATE}"
if [ -n "${LR_DEPTH}" ]; then
    echo "  LR Depth:            ${LR_DEPTH}"
else
    echo "  LR Depth:            ${LEARNING_RATE} (same as base)"
fi
if [ -n "${LR_SEG}" ]; then
    echo "  LR Seg:              ${LR_SEG}"
else
    echo "  LR Seg:              base * 10 (no override set)"
fi
echo "  Weight Decay:        ${WEIGHT_DECAY}"
echo "  Image Size:          ${IMG_SIZE}"
echo "  Frozen Backbone:     ${FROZEN_BACKBONE}"
echo "  --- PEFT Mode ---"
echo "  Mode:                ${MODE}"
if [[ "${MODE}" == "lora-only" || "${MODE}" == "legacy-lora" || "${MODE}" == "endo-unid" || "${MODE}" == "mtlora" || "${MODE}" == "mtlga" ]]; then
    echo "  LoRA Rank (r):       ${LORA_R}"
    echo "  LoRA Alpha:          ${LORA_ALPHA}"
fi
echo "  --- Loss Weighting ---"
echo "  Strategy:            ${LOSS_WEIGHTING_STRATEGY}"
if [ "${LOSS_WEIGHTING_STRATEGY}" = "fixed" ]; then
    echo "  Depth Loss Weight:   ${DEPTH_LOSS_WEIGHT}"
    echo "  Seg Loss Weight:     ${SEG_LOSS_WEIGHT}"
fi
if [ "${LOSS_WEIGHTING_STRATEGY}" = "dwa" ]; then
    echo "  DWA Temperature:     ${DWA_TEMPERATURE}"
fi
echo "  --- Hardware Configuration ---"
echo "  Number of GPUs:      ${NUM_GPUS}"
echo "  GPU Devices:         ${CUDA_DEVICES}"
echo "  Mixed Precision:     ${USE_MIXED_PRECISION}"
echo "  OMP Threads:         ${OMP_NUM_THREADS}"
echo "  --- Data Source ---"
echo "  Dataset Config:      ${DATASET_CONFIG_NAME}"
echo "  Path Transform:      ${PATH_TRANSFORM_NAME}"
echo "  Pretrained Weights:  ${PRETRAINED_WEIGHTS}"
echo "  Save Path:           ${SAVE_PATH}"
echo "=============================================================================="
echo ""

# Verify necessary files
echo "Verifying data files..."
if [ -n "${PRETRAINED_WEIGHTS}" ] && [ ! -f "${PRETRAINED_WEIGHTS}" ]; then
    echo "Warning: Pretrained weights not found: ${PRETRAINED_WEIGHTS}"
    echo "Starting training from random initialization."
    PRETRAINED_WEIGHTS=""
fi
echo "Data file verification complete."
echo ""

# CUDA devices
if [ -n "${CUDA_DEVICES}" ]; then
    export CUDA_VISIBLE_DEVICES=${CUDA_DEVICES}
fi

# Distributed env
export WORLD_SIZE=${NUM_GPUS}
export MASTER_ADDR=localhost
# export MASTER_PORT=20599
export MASTER_PORT=20600
echo "Setting distributed env: WORLD_SIZE=${WORLD_SIZE}, MASTER=${MASTER_ADDR}:${MASTER_PORT}"

# Build training command
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
--save-interval ${SAVE_INTERVAL} \
--save-path \"${SAVE_PATH}\""

# Conditionally add task-specific LRs only when set
if [ -n "${LR_DEPTH}" ]; then
    TRAIN_CMD="${TRAIN_CMD} --lr-depth ${LR_DEPTH}"
fi
if [ -n "${LR_SEG}" ]; then
    TRAIN_CMD="${TRAIN_CMD} --lr-seg ${LR_SEG}"
fi

# Mixed precision
if [ "${USE_MIXED_PRECISION}" = "true" ]; then
    TRAIN_CMD="${TRAIN_CMD} --mixed-precision"
fi

# Resume or pretrained
if [ -n "${RESUME_CHECKPOINT}" ]; then
    echo "  >>>>> Resume Mode: ${RESUME_CHECKPOINT} <<<<<"
    TRAIN_CMD="${TRAIN_CMD} --resume-from \"${RESUME_CHECKPOINT}\" --resume-full-state"
    elif [ -n "${PRETRAINED_WEIGHTS}" ]; then
    echo "  >>>>> Pretrain Mode: ${PRETRAINED_WEIGHTS} <<<<<"
    TRAIN_CMD="${TRAIN_CMD} --resume-from \"${PRETRAINED_WEIGHTS}\""
fi

# Frozen backbone
if [ "${FROZEN_BACKBONE}" = "true" ]; then
    TRAIN_CMD="${TRAIN_CMD} --frozen-backbone"
fi

# Mode-specific flags
TRAIN_CMD="${TRAIN_CMD} --mode ${MODE} \
--lora-r ${LORA_R} \
--lora-alpha ${LORA_ALPHA} \
--loss-weighting-strategy ${LOSS_WEIGHTING_STRATEGY} \
--depth-loss-weight ${DEPTH_LOSS_WEIGHT} \
--seg-loss-weight ${SEG_LOSS_WEIGHT} \
--dwa-temperature ${DWA_TEMPERATURE} \
--dataset-config-name ${DATASET_CONFIG_NAME} \
--path-transform-name ${PATH_TRANSFORM_NAME} \
--checkpoint-policy full"

echo "Starting multi-task training..."
echo "Training command: ${TRAIN_CMD}"

# Backup code snapshot (optional; path may need adjustment)
cp -r /data/ziyi/multitask/code/DepthAnythingV2/multitask_moe_lora "${SAVE_PATH}/code" 2>/dev/null || true
echo "Code snapshot saved to ${SAVE_PATH}/code (if source existed)"

# Execute training
eval ${TRAIN_CMD}
