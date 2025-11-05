#!/bin/bash

# ==============================================================================
# General Configuration
# ==============================================================================
# Number of GPUs to use for distributed training
NUM_GPUS=1

# Specify CUDA devices to use (e.g., "0,1,2,3" for devices 0, 1, 2, and 3)
# If left empty, all visible GPUs will be used by default.
CUDA_DEVICES="2" # <--- NEW VARIABLE

# Encoder model type: "vits", "vitb", "vitl", "vitg"
ENCODER="vits"

# Number of classes for segmentation (e.g., 2 for binary, 150 for ADE20K)
NUM_CLASSES=3

# Image size for training (e.g., 518 for Depth Anything, 512 for ADE20K)
IMG_SIZE=518

# Number of training epochs
EPOCHS=500

# Batch size per GPU
BATCH_SIZE_PER_GPU=96

# Learning rate
LEARNING_RATE=5e-5

# Path to the FULL DINOv2 pretrained weights (.pth)
# This is crucial for all training strategies.
# !!! IMPORTANT: Update this path to your actual DINOv2 weights !!!
PRETRAINED_WEIGHTS="/media/ExtHDD1/jianfu/data/train_4_dataset/same_maxdepth_full_20250613_210808/latest.pth"

# Path to the pretrained segmentation head weights (.pth), leave empty to ignore
SEG_HEAD_WEIGHTS="/media/ExtHDD1/jianfu/data/dinov2/dinov2_vits14_ade20k_linear_head.pth"
# SEG_HEAD_WEIGHTS="/media/ExtHDD1/jianfu/data/dinov2/dinov2_vitg14_voc2012_linear_head.pth"

# Base directory to save checkpoints and logs
# A timestamped subdirectory will be created within this path.
BASE_SAVE_PATH="/media/ExtHDD1/jianfu/data/train_inhouse_seg_multitask_with_pretrained_head"

# ==============================================================================
# Training Specific Configuration
# = ============================================================================
# Segmentation Head Type: "BNHead" or "Mask2FormerHead"
# Choose "Mask2FormerHead" only if you have fully implemented it in seg_heads.py
HEAD_TYPE="BNHead"

# Training Strategy: "full_param", "lora", or "frozen"
# - "full_param": Trains all parameters of the model (backbone + head).
# - "lora": Only trains LoRA layers (if integrated in the model) and the head.
# - "frozen": Freezes the backbone and only trains the head.
TRAINING_STRATEGY="frozen"

# Segmentation input type: "last", "last_four", "from_depth"
SEG_INPUT_TYPE="last_four"

# ==============================================================================
# Script Logic (No need to modify below this line unless you know what you're doing)
# ==============================================================================

# Validate HEAD_TYPE
if [[ "$HEAD_TYPE" != "BNHead" && "$HEAD_TYPE" != "Mask2FormerHead" ]]; then
    echo "Error: Invalid HEAD_TYPE. Must be 'BNHead' or 'Mask2FormerHead'."
    exit 1
fi

# Validate TRAINING_STRATEGY
if [[ "$TRAINING_STRATEGY" != "full_param" && "$TRAINING_STRATEGY" != "lora" && "$TRAINING_STRATEGY" != "frozen" ]]; then
    echo "Error: Invalid TRAINING_STRATEGY. Must be 'full_param', 'lora', or 'frozen'."
    exit 1
fi

# Construct the save path with a timestamp and configuration details
SAVE_PATH="${BASE_SAVE_PATH}/${HEAD_TYPE}_${TRAINING_STRATEGY}_${ENCODER}_$(date +%Y%m%d_%H%M%S)"

echo "=============================================================================="
echo "Starting Training with the following configuration:"
echo "------------------------------------------------------------------------------"
echo "  Head Type:          ${HEAD_TYPE}"
echo "  Training Strategy:  ${TRAINING_STRATEGY}"
echo "  Seg Input Type:     ${SEG_INPUT_TYPE}"
echo "  Encoder:            ${ENCODER}"
echo "  Num Classes:        ${NUM_CLASSES}"
echo "  Image Size:         ${IMG_SIZE}"
echo "  Epochs:             ${EPOCHS}"
echo "  Batch Size/GPU:     ${BATCH_SIZE_PER_GPU}"
echo "  Learning Rate:      ${LEARNING_RATE}"
echo "  Pretrained Weights: ${PRETRAINED_WEIGHTS}"
if [ -n "$SEG_HEAD_WEIGHTS" ]; then
    echo "  Seg Head Weights:   ${SEG_HEAD_WEIGHTS}"
fi
if [ -n "$CUDA_DEVICES" ]; then # Only show if specified
    echo "  CUDA Devices:       ${CUDA_DEVICES}"
fi
echo "  Save Path:          ${SAVE_PATH}"
echo "=============================================================================="
echo ""

# Determine which training script to use
TRAIN_SCRIPT=""
if [ "$TRAINING_STRATEGY" == "full_param" ]; then
    TRAIN_SCRIPT="train_seg.py"
elif [ "$TRAINING_STRATEGY" == "lora" ]; then
    TRAIN_SCRIPT="train_seg_lora.py"
elif [ "$TRAINING_STRATEGY" == "frozen" ]; then
    TRAIN_SCRIPT="train_seg_frozen.py"
fi

# Prepend CUDA_VISIBLE_DEVICES if specified
CUDA_ENV=""
if [ -n "$CUDA_DEVICES" ]; then
    CUDA_ENV="CUDA_VISIBLE_DEVICES=${CUDA_DEVICES}"
fi

# Execute the training command
# Assuming the scripts are in the current directory (mutitask)
# Use torchrun for distributed training
# Set CUDA_VISIBLE_DEVICES if specified
if [ -n "$CUDA_DEVICES" ]; then
    export CUDA_VISIBLE_DEVICES=${CUDA_DEVICES}
fi

mkdir -p "${SAVE_PATH}"
exec > >(tee -a "${SAVE_PATH}/log") 2>&1

# Execute the training command
# Scripts are now in the segmentation_only directory
# Use torchrun for distributed training

# Build arguments in an array
TRAIN_ARGS=(
    --encoder "${ENCODER}"
    --dataset inhouse
    --num-classes "${NUM_CLASSES}"
    --img-size "${IMG_SIZE}"
    --epochs "${EPOCHS}"
    --bs "${BATCH_SIZE_PER_GPU}"
    --lr "${LEARNING_RATE}"
    --seg-head-type "${HEAD_TYPE}"
    --pretrained-from "${PRETRAINED_WEIGHTS}"
    --save-path "${SAVE_PATH}"
    --port 20598
)

# Add seg head weights argument if the path is provided
if [ -n "$SEG_HEAD_WEIGHTS" ]; then
    TRAIN_ARGS+=("--seg-head-weights" "${SEG_HEAD_WEIGHTS}")
fi

# Add seg input type argument
TRAIN_ARGS+=("--seg-input-type" "${SEG_INPUT_TYPE}")

python -m torch.distributed.run --master-port=29501 \
    --nproc_per_node=${NUM_GPUS} \
    "multitask/segmentation_only/${TRAIN_SCRIPT}" \
    "${TRAIN_ARGS[@]}"

echo ""
echo "Training script finished for ${HEAD_TYPE} with ${TRAINING_STRATEGY} strategy."
echo "Logs and checkpoints are saved to: ${SAVE_PATH}"
