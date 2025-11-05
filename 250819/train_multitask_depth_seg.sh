#!/bin/bash
export PYTHONDONTWRITEBYTECODE=1

# ==============================================================================
# 多任务训练配置脚本 (深度估计 + 语义分割)
# ==============================================================================

# 基础配置
NUM_GPUS=1
CUDA_DEVICES="0" # 指定使用的GPU设备
ENCODER="vits"   # 模型编码器: vits, vitb, vitl, vitg
FEATURES=64      # 特征维度
NUM_CLASSES=3    # 分割类别数
MAX_DEPTH=0.2    # 最大深度值
SEG_INPUT_TYPE="last_four" # 分割头输入类型: "last", "last_four", "from_depth"

# 训练参数
EPOCHS=250
BATCH_SIZE=80
SEG_BATCH_SIZE=80  # 分割任务的批处理大小
VAL_BATCH_SIZE=100  # 验证集的批处理大小
LEARNING_RATE=5e-6
WEIGHT_DECAY=0.01
IMG_SIZE=518
FROZEN_BACKBONE="true" # 是否冻结backbone: "true" 或 "false"



# 数据路径配置
BASE_DATA_PATH="/media/ExtHDD1/jianfu/data"

# 深度数据缓存路径
TRAIN_DEPTH_CACHE="${BASE_DATA_PATH}/inhouse/cache/train_cache.txt"
VAL_DEPTH_CACHE="${BASE_DATA_PATH}/inhouse/cache/val_cache.txt"

# 分割数据缓存路径
TRAIN_SEG_CACHE="${BASE_DATA_PATH}/seg_inhouse/cache/train_cache.txt"
VAL_SEG_CACHE="${BASE_DATA_PATH}/seg_inhouse/cache/val_cache.txt"

# 预训练权重路径
PRETRAINED_WEIGHTS="${BASE_DATA_PATH}/train_4_dataset/same_maxdepth_full_20250613_210808/latest.pth"

# 保存路径
BASE_SAVE_PATH="${BASE_DATA_PATH}/train_multitask_depth_seg"
SAVE_PATH="${BASE_SAVE_PATH}/multitask_${ENCODER}_$(date +%Y%m%d_%H%M%S)"

# 创建保存目录
mkdir -p "${SAVE_PATH}"
exec > >(tee -a "${SAVE_PATH}/training.log") 2>&1

# ==============================================================================
# 验证配置
# ==============================================================================

echo "=============================================================================="
echo "多任务训练配置 (深度估计 + 语义分割)"
echo "------------------------------------------------------------------------------"
echo "  编码器:              ${ENCODER}"
echo "  特征维度:            ${FEATURES}"
echo "  分割类别数:          ${NUM_CLASSES}"
echo "  最大深度:            ${MAX_DEPTH}"
echo "  分割输入类型:        ${SEG_INPUT_TYPE}"
echo "  训练轮数:            ${EPOCHS}"
echo "  批处理大小 (深度): ${BATCH_SIZE}"
echo "  批处理大小 (分割): ${SEG_BATCH_SIZE}"
echo "  批处理大小 (验证): ${VAL_BATCH_SIZE}"
echo "  学习率:              ${LEARNING_RATE}"
echo "  权重衰减:            ${WEIGHT_DECAY}"
echo "  图像尺寸:            ${IMG_SIZE}"
echo "  冻结Backbone:        ${FROZEN_BACKBONE}"
echo "  GPU设备:             ${CUDA_DEVICES}"
echo "  训练深度缓存:        ${TRAIN_DEPTH_CACHE}"
echo "  验证深度缓存:        ${VAL_DEPTH_CACHE}"
echo "  训练分割缓存:        ${TRAIN_SEG_CACHE}"
echo "  验证分割缓存:        ${VAL_SEG_CACHE}"
echo "  预训练权重:          ${PRETRAINED_WEIGHTS}"
echo "  保存路径:            ${SAVE_PATH}"
echo "=============================================================================="
echo ""

# 验证必要文件是否存在
echo "验证数据文件..."

if [ ! -f "$TRAIN_DEPTH_CACHE" ]; then
    echo "❌ 错误: 训练深度缓存文件不存在: $TRAIN_DEPTH_CACHE"
    echo "请先生成深度数据缓存或修改路径配置"
    exit 1
fi

if [ ! -f "$VAL_DEPTH_CACHE" ]; then
    echo "❌ 错误: 验证深度缓存文件不存在: $VAL_DEPTH_CACHE"
    echo "请先生成深度数据缓存或修改路径配置"
    exit 1
fi

if [ ! -f "$TRAIN_SEG_CACHE" ]; then
    echo "❌ 错误: 训练分割缓存文件不存在: $TRAIN_SEG_CACHE"
    echo "请先生成分割数据缓存或修改路径配置"
    exit 1
fi

if [ ! -f "$VAL_SEG_CACHE" ]; then
    echo "❌ 错误: 验证分割缓存文件不存在: $VAL_SEG_CACHE"
    echo "请先生成分割数据缓存或修改路径配置"
    exit 1
fi

if [ ! -f "$PRETRAINED_WEIGHTS" ]; then
    echo "⚠️  警告: 预训练权重文件不存在: $PRETRAINED_WEIGHTS"
    echo "将从随机初始化开始训练"
    PRETRAINED_WEIGHTS=""
fi

echo "✅ 数据文件验证完成"
echo ""

# 创建保存目录
mkdir -p "${SAVE_PATH}"
exec > >(tee -a "${SAVE_PATH}/training.log") 2>&1

# 设置CUDA设备
if [ -n "$CUDA_DEVICES" ]; then
    export CUDA_VISIBLE_DEVICES=${CUDA_DEVICES}
fi

# 构建训练命令（DDP模式）
TRAIN_CMD="python -m torch.distributed.run --nproc_per_node=${NUM_GPUS} --master_port=20599 multitask/train_multitask_depth_seg.py \
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
    --train-depth-cache \"${TRAIN_DEPTH_CACHE}\" \
    --val-depth-cache \"${VAL_DEPTH_CACHE}\" \
    --train-seg-cache \"${TRAIN_SEG_CACHE}\" \
    --val-seg-cache \"${VAL_SEG_CACHE}\" \
    --mixed-precision" \
    --save-path \"${SAVE_PATH}\" 

# 添加可选参数
if [ -n "$PRETRAINED_WEIGHTS" ]; then
    TRAIN_CMD="${TRAIN_CMD} --pretrained-from \"${PRETRAINED_WEIGHTS}\""
fi


if [ "$FROZEN_BACKBONE" = "true" ]; then
    TRAIN_CMD="${TRAIN_CMD} --frozen-backbone"
fi

echo "开始多任务训练..."
echo "训练命令:"
echo "$TRAIN_CMD"
echo ""

cp -r /media/ExtHDD1/jianfu/depth/DepthAnythingV2/multitask "${SAVE_PATH}/code"
echo "code backed up to ${SAVE_PATH}/code"
echo ""

# 执行训练
eval $TRAIN_CMD
