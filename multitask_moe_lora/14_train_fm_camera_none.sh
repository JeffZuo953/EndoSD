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
NUM_GPUS=${NUM_GPUS:-5}
CUDA_DEVICES=${CUDA_DEVICES:-"0,3,4,5,6"}
MASTER_PORT=${MASTER_PORT:-20863}

# ------------------------------------------------------------------------------
# Core training hyper-parameters
# ------------------------------------------------------------------------------
ENCODER=${ENCODER:-"vits"}        # {vits, vitb, vitl, dinov3_*}
FEATURES=${FEATURES:-64}
EPOCHS=${EPOCHS:-120}
BATCH_SIZE=${BATCH_SIZE:-24}
VAL_BATCH_SIZE=${VAL_BATCH_SIZE:-96}
LEARNING_RATE=${LEARNING_RATE:-5e-6}
WEIGHT_DECAY=${WEIGHT_DECAY:-0.01}
IMG_SIZE=${IMG_SIZE:-518}
MAX_DEPTH=${MAX_DEPTH:-0.3}
MIN_DEPTH=${MIN_DEPTH:-1e-6}
MIXED_PRECISION=${MIXED_PRECISION:-true}
FROZEN_BACKBONE=${FROZEN_BACKBONE:-false}
VAL_MIN_SAMPLES_PER_DATASET=${VAL_MIN_SAMPLES_PER_DATASET:-100}
# CAMERA_HEAD_MODE=${CAMERA_HEAD_MODE:-"vggtlike"}
CAMERA_HEAD_MODE=${CAMERA_HEAD_MODE:-"none"}
CAMERA_LOSS_WEIGHT=${CAMERA_LOSS_WEIGHT:-0.0}
CAMERA_LR=${CAMERA_LR:-1e-3}
TOLERATE_VALIDATION_ERRORS=${TOLERATE_VALIDATION_ERRORS:-true}

FM_SAMPLE_MODE=${FM_SAMPLE_MODE:-"sample"}   # full | sample
FM_SAMPLE_SIZE=${FM_SAMPLE_SIZE:-10}
TRAIN_SAMPLE_STEP=${TRAIN_SAMPLE_STEP:-200}
# TRAIN_SAMPLE_STEP=${TRAIN_SAMPLE_STEP:-1}
VAL_SAMPLE_STEP=${VAL_SAMPLE_STEP:-20}
# VAL_SAMPLE_STEP=${VAL_SAMPLE_STEP:-1}
MAX_SAMPLES_PER_DATASET=${MAX_SAMPLES_PER_DATASET:-}

# ------------------------------------------------------------------------------
# Dataset configuration
# NOTE: define `fd_depth_fm_v1` in util/data_utils.DATASET_PATHS to point to
#       the exact caches/filelists for the datasets enumerated below.
# ------------------------------------------------------------------------------
DATASET_CONFIG_NAME=${DATASET_CONFIG_NAME:-"fd_depth_fm_v1"}
DATASET_MODALITY=${DATASET_MODALITY:-"fd"}       # depth-only foundation mode
PATH_TRANSFORM_NAME=${PATH_TRANSFORM_NAME:-"none"}
MAX_SAMPLES_PER_DATASET=${MAX_SAMPLES_PER_DATASET}

TRAIN_DATASET_INCLUDE=${TRAIN_DATASET_INCLUDE:-"SCARED,StereoMIS,Endovis2017,EndoVis2018,EndoSynth,dVPN,C3VDv2,SimCol,Kidney3D"}
VAL_DATASET_INCLUDE=${VAL_DATASET_INCLUDE:-"hamlyn,EndoNeRF,C3VD,EndoMapper,Kidney3D,Endovis2017"}

# ------------------------------------------------------------------------------
# Checkpoint configuration
# ------------------------------------------------------------------------------
BASE_DATA_PATH=${BASE_DATA_PATH:-"/data/ziyi/multitask"}
HOME_SSD_PATH=${HOME_SSD_PATH:-"$HOME/ssde"}
export BASE_DATA_PATH
export HOME_SSD_PATH
LOCAL_CACHE_DIR=${LOCAL_CACHE_DIR:-"/data/ziyi/cache"}
if [[ -n "${LOCAL_CACHE_DIR}" ]]; then
    mkdir -p "${LOCAL_CACHE_DIR}"
    export LOCAL_CACHE_DIR
fi
PRETRAINED_WEIGHTS=${PRETRAINED_WEIGHTS:-"${BASE_DATA_PATH}/pretained/depth_anything_v2_vits.pth"}
RESUME_CHECKPOINT=${RESUME_CHECKPOINT:-""}

# ------------------------------------------------------------------------------
# Output logging
# ------------------------------------------------------------------------------
SAVE_ROOT=${SAVE_ROOT:-"/data/ziyi/save/FM"}
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
echo "  Camera LR:             ${CAMERA_LR}"
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
echo "  Val min samples:       ${VAL_MIN_SAMPLES_PER_DATASET}"
echo "  Save path:             ${SAVE_PATH}"
echo "=============================================================================="
echo ""

# ------------------------------------------------------------------------------
# Sanity checks / communication settings
# ------------------------------------------------------------------------------
if [[ -n "${CUDA_DEVICES}" ]]; then
    export CUDA_VISIBLE_DEVICES=${CUDA_DEVICES}
fi

NCCL_COMM_TIMEOUT=${NCCL_COMM_TIMEOUT:-1800}
export NCCL_TIMEOUT=${NCCL_TIMEOUT:-$NCCL_COMM_TIMEOUT}
export TORCH_NCCL_TIMEOUT=${TORCH_NCCL_TIMEOUT:-$NCCL_COMM_TIMEOUT}
export NCCL_BLOCKING_WAIT=${NCCL_BLOCKING_WAIT:-1}
export NCCL_ASYNC_ERROR_HANDLING=${NCCL_ASYNC_ERROR_HANDLING:-1}
export TORCH_NCCL_BLOCKING_WAIT=${TORCH_NCCL_BLOCKING_WAIT:-$NCCL_BLOCKING_WAIT}
export TORCH_NCCL_ASYNC_ERROR_HANDLING=${TORCH_NCCL_ASYNC_ERROR_HANDLING:-$NCCL_ASYNC_ERROR_HANDLING}

if [[ -n "${PRETRAINED_WEIGHTS}" && ! -f "${PRETRAINED_WEIGHTS}" ]]; then
    echo "[WARN] Pretrained weights not found at ${PRETRAINED_WEIGHTS}, training will start from scratch."
    PRETRAINED_WEIGHTS=""
fi
if [[ -n "${PRETRAINED_WEIGHTS}" && -f "${PRETRAINED_WEIGHTS}" && -n "${LOCAL_CACHE_DIR}" ]]; then
    CACHE_WEIGHTS_DIR="${LOCAL_CACHE_DIR}/pretrained_weights"
    mkdir -p "${CACHE_WEIGHTS_DIR}"
    CACHE_WEIGHTS_PATH="${CACHE_WEIGHTS_DIR}/$(basename "${PRETRAINED_WEIGHTS}")"
    if [[ ! -f "${CACHE_WEIGHTS_PATH}" || "${PRETRAINED_WEIGHTS}" -nt "${CACHE_WEIGHTS_PATH}" ]]; then
        echo "[INFO] Mirroring pretrained weights to local cache: ${CACHE_WEIGHTS_PATH}"
        rsync -a "${PRETRAINED_WEIGHTS}" "${CACHE_WEIGHTS_PATH}"
    fi
    PRETRAINED_WEIGHTS="${CACHE_WEIGHTS_PATH}"
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
if [[ -n "${LOCAL_CACHE_DIR}" ]]; then
    BASE_CMD+=(--local-cache-dir "${LOCAL_CACHE_DIR}")
fi
if [[ "${MIXED_PRECISION}" == "true" ]]; then
    BASE_CMD+=(--mixed-precision)
fi
if [[ "${FROZEN_BACKBONE}" == "true" ]]; then
    BASE_CMD+=(--frozen-backbone)
fi
if [[ "${TOLERATE_VALIDATION_ERRORS}" == "true" ]]; then
    BASE_CMD+=(--tolerate-validation-errors)
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
