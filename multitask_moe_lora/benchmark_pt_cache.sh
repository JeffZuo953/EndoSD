#!/bin/bash
set -euo pipefail

export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1
export PYTHONPATH="$(dirname "$(pwd)"):${PYTHONPATH:-}"

NUM_GPUS=3
CUDA_DEVICES="3,4,5"
ENCODER="vits"
FEATURES=64
NUM_CLASSES=10
MAX_DEPTH=0.3
SEG_INPUT_TYPE="last_four"

BATCH_SIZE=20
SEG_BATCH_SIZE=20
VAL_BATCH_SIZE=100
IMG_SIZE=518

MODE="legacy-lora"
NUM_EXPERTS=8
TOP_K=2
LORA_R=4
LORA_ALPHA=8

DATASET_CONFIG_NAME="ls_only_v1"
PATH_TRANSFORM_NAME="ls_default"
DATASET_MODALITY="mt"
TRAIN_DATASET_INCLUDE="SCARED,StereoMIS,dVPN,C3VDv2,SimCol,Kidney3D,EndoSynth"
VAL_DATASET_INCLUDE="hamlyn,EndoNeRF,C3VD,EndoMapper,Kidney3D"

BASE_DATA_PATH="/data/ziyi/multitask"
SAVE_PATH="${BASE_DATA_PATH}/save/benchmark_pt_cache_$(date +%Y%m%d_%H%M%S)"
mkdir -p "${SAVE_PATH}"

export CUDA_VISIBLE_DEVICES=${CUDA_DEVICES}
export WORLD_SIZE=${NUM_GPUS}
export MASTER_ADDR=localhost
export MASTER_PORT=20621

echo "=============================================================================="
echo "Benchmarking PT-Cache Data Loading (3 GPUs)"
echo "=============================================================================="

python -m torch.distributed.run --nproc_per_node=${NUM_GPUS} --master_port=${MASTER_PORT} \
    benchmark_data_loading.py \
    --encoder ${ENCODER} \
    --features ${FEATURES} \
    --num-classes ${NUM_CLASSES} \
    --max-depth ${MAX_DEPTH} \
    --seg-input-type ${SEG_INPUT_TYPE} \
    --bs ${BATCH_SIZE} \
    --seg-bs ${SEG_BATCH_SIZE} \
    --val-bs ${VAL_BATCH_SIZE} \
    --img-size ${IMG_SIZE} \
    --save-path "${SAVE_PATH}" \
    --dataset-config-name ${DATASET_CONFIG_NAME} \
    --path-transform-name ${PATH_TRANSFORM_NAME} \
    --train-dataset-include "${TRAIN_DATASET_INCLUDE}" \
    --val-dataset-include "${VAL_DATASET_INCLUDE}" \
    --dataset-modality ${DATASET_MODALITY} \
    --mode ${MODE} \
    --lora-r ${LORA_R} \
    --lora-alpha ${LORA_ALPHA} \
    --num-experts ${NUM_EXPERTS} \
    --top-k ${TOP_K} \
    --num-batches 100
