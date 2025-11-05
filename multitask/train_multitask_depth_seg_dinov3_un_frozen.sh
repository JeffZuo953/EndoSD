#!/bin/bash
echo "等待60秒，以错开训练任务启动时间，防止瞬时功耗过高..."
sleep 60
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1

# ==============================================================================
# 多任务训练配置脚本 (深度估计 + 语义分割)
# ==============================================================================

# 基础配置
NUM_GPUS=2
CUDA_DEVICES="2,3" # 指定使用的GPU设备
ENCODER="dinov3_vits16plus"   # 模型编码器: vits, vitb, vitl, vitg, dinov3_vits16, etc.
FEATURES=64      # 特征维度
NUM_CLASSES=4    # 分割类别数
MAX_DEPTH=0.2    # 最大深度值
SEG_INPUT_TYPE="last_four" # 分割头输入类型: "last", "last_four", "from_depth"

# 训练参数
EPOCHS=500
BATCH_SIZE=25
SEG_BATCH_SIZE=25  # 分割任务的批处理大小
VAL_BATCH_SIZE=50  # 验证集的批处理大小
# BATCH_SIZE=80
LEARNING_RATE=5e-6
WEIGHT_DECAY=0.01
IMG_SIZE=518
FROZEN_BACKBONE="false" # 是否冻结backbone: "true" 或 "false"



# 数据路径配置
BASE_DATA_PATH="/media/ExtHDD1/jianfu/data"

# 预训练权重路径
# PRETRAINED_WEIGHTS="${BASE_DATA_PATH}/dinov3/dinov3_vits16_pretrain_lvd1689m-08c60483.pth"
PRETRAINED_WEIGHTS="${BASE_DATA_PATH}/dinov3/dinov3_vits16plus_pretrain_lvd1689m-4057cbaa.pth"

# 保存路径
BASE_SAVE_PATH="${BASE_DATA_PATH}/train_lesion"
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
echo "  预训练权重:          ${PRETRAINED_WEIGHTS}"
echo "  保存路径:            ${SAVE_PATH}"
echo "=============================================================================="
echo ""

# 验证必要文件是否存在
echo "验证数据文件..."

if [ ! -f "$PRETRAINED_WEIGHTS" ]; then
    echo "⚠️  警告: 预训练权重文件不存在: $PRETRAINED_WEIGHTS"
    echo "将从随机初始化开始训练"
    PRETRAINED_WEIGHTS=""
fi

echo "✅ 数据文件验证完成"
echo ""

# 设置CUDA设备
if [ -n "$CUDA_DEVICES" ]; then
    export CUDA_VISIBLE_DEVICES=${CUDA_DEVICES}
fi

# 设置分布式环境变量（多GPU模式）
# export RANK=0
# export LOCAL_RANK=0
export WORLD_SIZE=${NUM_GPUS}
export MASTER_ADDR=localhost
export MASTER_PORT=20597

echo "设置分布式环境变量..."
echo "WORLD_SIZE=${WORLD_SIZE}, MASTER_ADDR=${MASTER_ADDR}:${MASTER_PORT}"
echo ""

# 构建训练命令（多GPU模式）
TRAIN_CMD="python -m torch.distributed.run --nproc_per_node=${NUM_GPUS} --master_port=${MASTER_PORT} multitask/train_multitask_depth_seg.py \
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
    --mixed-precision"

# 添加可选参数
if [ -n "$PRETRAINED_WEIGHTS" ]; then
    TRAIN_CMD="${TRAIN_CMD} --resume-from \"${PRETRAINED_WEIGHTS}\""
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
