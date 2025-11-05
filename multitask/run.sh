#!/bin/bash

# ==============================================================================
# 多任务推理配置脚本 (深度估计 + 语义分割)
# ==============================================================================

# 任务配置
TASK="all" # "depth", "seg", 或 "all"

# 模型配置
ENCODER="dinov3_vits16plus" # dinov3_vits16 dinov3_vits16plus vits
FEATURES=64
NUM_CLASSES=4
MAX_DEPTH=0.2
SEG_INPUT_TYPE="from_depth" # "last", "last_four", "from_depth"
# **重要**: 修改为你的模型权重路径
# 343 294 
# 397 350
EPOCH=343
# PRETRAINED_WEIGHTS="/media/ExtHDD1/jianfu/data/train_lesion/multitask_vits_20250829_204046/checkpoint_epoch_${EPOCH}.pth"
PRETRAINED_WEIGHTS="/media/ExtHDD1/jianfu/data/train_lesion/multitask_dinov3_vits16plus_20250830_071130/checkpoint_epoch_${EPOCH}.pth"
# GPU 配置
CUDA_DEVICES="2" # 指定要使用的GPU, e.g., "0", "0,1"

# 数据配置
# **重要**: 修改为你的输入路径 (单个图片, .pt 文件, 或 .txt 列表)
# INPUT_PATH="/media/ExtHDD1/jianfu/data/seg_inhouse/cache/val_cache.txt"
INPUT_PATH="/media/ExtHDD1/jianfu/depth/DepthAnythingV2/multitask/1.txt"
# **重要**: 修改为你的输出路径
BASE_DATA_PATH="/media/ExtHDD1/jianfu/data/seg_inhouse/cache" # 如果INPUT_PATH中的文件列表是相对路径，请设置此项；否则留空
BASE_OUTPUT_PATH="/media/ExtHDD1/jianfu/data/inference_lesion"
# 从预训练权重路径中提取模型子目录名（如 multitask_vits_20250716_194333）
MODEL_DIR_NAME=$(basename "$(dirname "${PRETRAINED_WEIGHTS}")")
# 从预训练权重的路径中提取模型名称，作为输出子目录
MODEL_NAME=$(basename "${PRETRAINED_WEIGHTS}" .pth)

OUTPUT_PATH="${BASE_OUTPUT_PATH}/$(date +%Y%m%d_%H%M%S)_${MODEL_DIR_NAME}_${MODEL_NAME}"

# 预处理与输入类型
IMG_SIZE=518
# 是否对输入图像进行预处理: "true" 或 "false"
PREPROCESS="true"  

# 输出配置
SAVE_IMAGE="true"   # 保存可视化图片: "true" 或 "false"
SAVE_PT="true"      # 保存为.pt文件: "true" 或 "false"
SAVE_NPZ="false"    # 保存为.npz文件: "true" 或 "false"
NORMALIZATION="min-max" # 深度图归一化方式: "min-max" 或 "max"
COLORMAP="gray"      # 深度图颜色映射, e.g., "gray", "viridis", "inferno"

# ==============================================================================
# 验证与执行
# ==============================================================================

echo "=============================================================================="
echo "多任务推理配置"
echo "------------------------------------------------------------------------------"
echo "  任务类型:            ${TASK}"
echo "  编码器:              ${ENCODER}"
echo "  分割输入类型:        ${SEG_INPUT_TYPE}"
echo "  预训练权重:          ${PRETRAINED_WEIGHTS}"
echo "  输入路径:            ${INPUT_PATH}"
echo "  输出路径:            ${OUTPUT_PATH}"
echo "  基础数据路径:        ${BASE_DATA_PATH}"
echo "  GPU设备:             ${CUDA_DEVICES}"
echo "  预处理:              ${PREPROCESS}"
echo "  保存图片:            ${SAVE_IMAGE}"
echo "  保存PT:              ${SAVE_PT}"
echo "  保存NPZ:             ${SAVE_NPZ}"
echo "=============================================================================="
echo ""

# 验证必要文件是否存在
if [ ! -f "$PRETRAINED_WEIGHTS" ]; then
    echo "❌ 错误: 预训练权重文件不存在: $PRETRAINED_WEIGHTS"
    exit 1
fi

if [ ! -f "$INPUT_PATH" ]; then
    echo "❌ 错误: 输入文件/列表不存在: $INPUT_PATH"
    exit 1
fi

# 创建保存目录
mkdir -p "${OUTPUT_PATH}"
# 将所有输出重定向到日志文件和控制台
exec > >(tee -a "${OUTPUT_PATH}/inference.log") 2>&1

# 设置CUDA设备
if [ -n "$CUDA_DEVICES" ]; then
    export CUDA_VISIBLE_DEVICES=${CUDA_DEVICES}
fi

# 构建推理命令
CMD="python run.py \
    --task ${TASK} \
    --encoder ${ENCODER} \
    --features ${FEATURES} \
    --num-classes ${NUM_CLASSES} \
    --max-depth ${MAX_DEPTH} \
    --seg-input-type ${SEG_INPUT_TYPE} \
    --pretrained-from \"${PRETRAINED_WEIGHTS}\" \
    --input-path \"${INPUT_PATH}\" \
    --output-path \"${OUTPUT_PATH}\" \
    --img-size ${IMG_SIZE} \
    --normalization ${NORMALIZATION} \
    --colormap ${COLORMAP}"

# 添加可选的布尔标志
if [ "$PREPROCESS" = "true" ]; then
    CMD="${CMD} --preprocess"
fi

if [ "$SAVE_IMAGE" = "true" ]; then
    CMD="${CMD} --save-image"
fi

if [ "$SAVE_PT" = "true" ]; then
    CMD="${CMD} --save-pt"
fi

if [ "$SAVE_NPZ" = "true" ]; then
    CMD="${CMD} --save-npz"
fi

# 添加可选的基础数据路径参数
if [ -n "$BASE_DATA_PATH" ]; then
    CMD="${CMD} --base-data-path \"${BASE_DATA_PATH}\""
fi

echo "开始推理..."
echo "执行命令:"
echo "$CMD"
echo ""

# 执行命令
eval "$CMD"

echo "推理完成, 结果保存在: ${OUTPUT_PATH}"

# --- 生成文件列表 ---
echo "正在生成深度图和分割图的文件列表..."
find "${OUTPUT_PATH}/depth" -name "*.pt" > "${OUTPUT_PATH}/depth_files.txt"
find "${OUTPUT_PATH}/seg" -name "*.pt" > "${OUTPUT_PATH}/seg_files.txt"
echo "文件列表生成完毕: "
echo "  - ${OUTPUT_PATH}/depth_files.txt"
echo "  - ${OUTPUT_PATH}/seg_files.txt"

# --- 性能评估 ---
# **重要**: 修改为你的真实数据 .txt 列表路径
GT_LIST_PATH="/media/ExtHDD1/jianfu/data/seg_inhouse/cache/val_cache.txt"

if [ -f "$GT_LIST_PATH" ]; then
    echo ""
    echo "=============================================================================="
    echo "开始性能评估..."
    echo "=============================================================================="

    # 评估深度估计
    if [ "$TASK" = "all" ] || [ "$TASK" = "depth" ]; then
        python multitask/evaluate_from_files.py \
            --pred-list "${OUTPUT_PATH}/depth_files.txt" \
            --gt-list "${GT_LIST_PATH}" \
            --task depth
    fi

    # 评估语义分割
    if [ "$TASK" = "all" ] || [ "$TASK" = "seg" ]; then
        python multitask/evaluate_from_files.py \
            --pred-list "${OUTPUT_PATH}/seg_files.txt" \
            --gt-list "${GT_LIST_PATH}" \
            --task seg \
            --num-classes ${NUM_CLASSES}
    fi
    
    echo "性能评估完成。"
else
    echo ""
    echo "警告: 未找到真实数据列表文件: ${GT_LIST_PATH}"
    echo "跳过性能评估步骤。"
fi
