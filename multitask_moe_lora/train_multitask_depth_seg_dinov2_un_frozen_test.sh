#!/bin/bash
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1

# ==============================================================================
# Multi-Task Training Configuration Script (Depth Estimation + Semantic Segmentation)
# ==============================================================================

# Basic Configuration
NUM_GPUS=2
CUDA_DEVICES="2,3" # Specify GPU devices
ENCODER="vits"   # Model encoder: vits, vitb, vitl, vitg, dinov3_vits16, etc.
FEATURES=64      # Feature dimension
NUM_CLASSES=4    # Number of segmentation classes
MAX_DEPTH=0.2    # Maximum depth value
SEG_INPUT_TYPE="last_four" # Segmentation head input type: "last", "last_four", "from_depth"

# Training Parameters
EPOCHS=500
BATCH_SIZE=20        # Reduced from 40 to 20 to save memory
SEG_BATCH_SIZE=20    # Reduced from 40 to 20 to save memory
VAL_BATCH_SIZE=100    # Reduced from 100 to 50 to save memory
# BATCH_SIZE=80
LEARNING_RATE=5e-6   # Base learning rate (deprecated, use lr_depth and lr_seg instead)
LR_DEPTH=""          # Depth task learning rate (overrides base learning rate if set)
LR_SEG=""            # Segmentation task learning rate (overrides base learning rate * 10 if set)
WEIGHT_DECAY=0.01
IMG_SIZE=518
FROZEN_BACKBONE="true" # Backbone weights frozen for stability

# Training mode parameters (supports: original, lora-only, legacy-lora, endo-unid, mtlora, mtlga)
MODE="original"

# Parameters effective only in LoRA-based modes
LORA_R=4
LORA_ALPHA=8

# Data source configuration
DATASET_CONFIG_NAME="server_sz"  # 'server_sz' or 'server_hk_01'
PATH_TRANSFORM_NAME="none"      # 'sz_to_hk' or 'none'

# Loss weighting strategy
# Options: "fixed", "uwl", "dwa"
LOSS_WEIGHTING_STRATEGY="uwl"
DEPTH_LOSS_WEIGHT=1.0
SEG_LOSS_WEIGHT=1.0 # For 'fixed' strategy
DWA_TEMPERATURE=2.0   # For 'dwa' strategy

# Data Path Configuration
BASE_DATA_PATH=${BASE_DATA_PATH:-"/data/ziyi/multitask"}
HOME_SSD_PATH=${HOME_SSD_PATH:-"$HOME/ssde"}

# Pretrained Weights Path
# PRETRAINED_WEIGHTS="${BASE_DATA_PATH}/dinov3/dinov3_vits16_pretrain_lvd1689m-08c60483.pth"
# PRETRAINED_WEIGHTS="${BASE_DATA_PATH}/dinov2/dinov2_vits14_reg4_pretrain.pth"
# PRETRAINED_WEIGHTS="${BASE_DATA_PATH}/dinov2/dinov2_vits14_reg4_pretrain.pth"

PRETRAINED_WEIGHTS="${BASE_DATA_PATH}/train_4_dataset/same_maxdepth_full_20250613_210808/latest.pth"
RESUME_CHECKPOINT=""


# Save Path
BASE_SAVE_PATH="${BASE_DATA_PATH}/train_lesion"
SAVE_PATH="${BASE_SAVE_PATH}/multitask_${ENCODER}_${MODE}_${LOSS_WEIGHTING_STRATEGY}_$(date +%Y%m%d_%H%M%S)"

# Create save directory
mkdir -p "${SAVE_PATH}"
exec > >(tee -a "${SAVE_PATH}/training.log") 2>&1

# ==============================================================================
# Configuration Validation
# ==============================================================================

echo "=============================================================================="
echo "Multi-Task Training Configuration (Depth Estimation + Semantic Segmentation)"
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
if [ -n "$LR_DEPTH" ]; then
    echo "  LR Depth:            ${LR_DEPTH}"
else
    echo "  LR Depth:            ${LEARNING_RATE} (same as base)"
fi
if [ -n "$LR_SEG" ]; then
    echo "  LR Seg:              ${LR_SEG}"
else
    echo "  LR Seg:              $(echo "${LEARNING_RATE} * 3" | bc -l) (base * 3)"
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
echo "  -----------------------"
echo "  Pretrained Weights:  ${PRETRAINED_WEIGHTS}"
echo "  Save Path:           ${SAVE_PATH}"
echo "=============================================================================="
echo ""

# Verify necessary files
echo "Verifying data files..."

if [ ! -f "$PRETRAINED_WEIGHTS" ]; then
    echo "⚠️  Warning: Pretrained weights file not found: $PRETRAINED_WEIGHTS"
    echo "Starting training from random initialization."
    PRETRAINED_WEIGHTS=""
fi

echo "✅ Data file verification complete."
echo ""

# Set CUDA devices
if [ -n "$CUDA_DEVICES" ]; then
    export CUDA_VISIBLE_DEVICES=${CUDA_DEVICES}
fi

# torch.distributed.run automatically handles RANK, WORLD_SIZE, MASTER_ADDR, MASTER_PORT etc.
# Manual setting is not required, ensuring better automation and compatibility.

# Build training command (multi-GPU mode)
TRAIN_CMD="python -m torch.distributed.run --nproc_per_node=${NUM_GPUS} -m multitask_moe_lora.train_multitask_depth_seg \
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

# Add optional parameters
# Prioritize resume training logic
if [ -n "${RESUME_CHECKPOINT}" ]; then
    echo "  >>>>> Resume Mode: Resuming full training state from ${RESUME_CHECKPOINT} <<<<<"
    TRAIN_CMD="${TRAIN_CMD} --resume-from \"${RESUME_CHECKPOINT}\" --resume-full-state"
    # Handle loading pretrained weights logic
    elif [ -n "${PRETRAINED_WEIGHTS}" ]; then
    echo "  >>>>> Pretrain Mode: Loading model weights from ${PRETRAINED_WEIGHTS} <<<<<"
    TRAIN_CMD="${TRAIN_CMD} --resume-from \"${PRETRAINED_WEIGHTS}\""
fi

if [ "$FROZEN_BACKBONE" = "true" ]; then
    TRAIN_CMD="${TRAIN_CMD} --frozen-backbone"
fi

# Add mode-specific parameters
TRAIN_CMD="${TRAIN_CMD} --mode ${MODE} \
--lora-r ${LORA_R} \
--lora-alpha ${LORA_ALPHA} \
--loss-weighting-strategy ${LOSS_WEIGHTING_STRATEGY} \
--depth-loss-weight ${DEPTH_LOSS_WEIGHT} \
--seg-loss-weight ${SEG_LOSS_WEIGHT} \
--dwa-temperature ${DWA_TEMPERATURE} \
--dataset-config-name ${DATASET_CONFIG_NAME} \
--path-transform-name ${PATH_TRANSFORM_NAME}"

echo "Starting multi-task training..."
echo "Training command:"
echo "$TRAIN_CMD"
echo ""

cp -r /media/ExtHDD1/jianfu/depth/DepthAnythingV2/multitask_moe_lora "${SAVE_PATH}/code"
echo "code backed up to ${SAVE_PATH}/code"
echo ""

# Execute training
eval $TRAIN_CMD
