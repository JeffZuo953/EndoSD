#!/bin/bash

# ==============================================================================
# 分割真值 (GT) 可视化脚本
# ==============================================================================

# **重要**: 修改为你的输入GT掩码路径
INPUT_PATH="/media/ssd2t/jianfu/data/polyp/CVC-EndoScene/ValidationDataset/cache/train_cache.txt"
# INPUT_PATH="/media/ssd2t/jianfu/data/polyp/ETIS-LaribPolypDB/cache/train_cache.txt"

# **重要**: 修改为你的输出路径
BASE_OUTPUT_PATH="/media/ExtHDD1/jianfu/data/polyp/CVC-EndoScene/ValidationDataset/gt_vis"

# BASE_OUTPUT_PATH="/media/ExtHDD1/jianfu/data/polyp/ETIS-LaribPolypDB/gt_vis"
# 创建一个基于输入文件名和时间戳的总输出目录
OUTPUT_DIR_NAME=$(basename "${INPUT_PATH}" .txt)_gt_vis_$(date +%Y%m%d_%H%M%S)
OUTPUT_PATH="${BASE_OUTPUT_PATH}/${OUTPUT_DIR_NAME}"

# **可选**: 设置此项以导出.pt文件中的原始RGB图像
ORIGIN_IMAGE_DIR="${BASE_OUTPUT_PATH}/origin_images" # 留空则不导出

# ==============================================================================
# 验证与执行
# ==============================================================================

echo "=============================================================================="
echo "分割真值可视化配置"
echo "------------------------------------------------------------------------------"
echo "  输入GT路径:        ${INPUT_PATH}"
echo "  输出可视化路径:    ${OUTPUT_PATH}"
echo "  输出原图路径:      ${ORIGIN_IMAGE_DIR}"
echo "=============================================================================="
echo ""

# 验证输入文件是否存在
if [ ! -f "$INPUT_PATH" ]; then
    echo "❌ 错误: 输入GT文件不存在: $INPUT_PATH"
    exit 1
fi

# 创建保存目录
mkdir -p "${OUTPUT_PATH}"
if [ -n "$ORIGIN_IMAGE_DIR" ]; then
    mkdir -p "$ORIGIN_IMAGE_DIR"
fi

# 构建命令
CMD="python gen_seg_gt.py \
    --input-path \"${INPUT_PATH}\" \
    --output-path \"${OUTPUT_PATH}\""

# 如果设置了原图导出目录，则添加到命令中
if [ -n "$ORIGIN_IMAGE_DIR" ]; then
    CMD="${CMD} --origin-image-dir \"${ORIGIN_IMAGE_DIR}\""
fi

echo "开始生成可视化..."
echo "执行命令:"
echo "$CMD"
echo ""

# 执行命令
eval "$CMD"

echo "可视化完成, 结果保存在: ${OUTPUT_PATH}"