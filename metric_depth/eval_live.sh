#!/bin/bash
# 定义模型编码器
ENCODER="vits"
# 定义输入文本文件路径，其中包含图像路径
# 请根据您的实际情况修改此路径

CASE=2

INPUT_TXT="/data/inhouse/case$CASE.txt"

# 定义输出目录
OUTPUT_DIR="/data/eval/vda/inhouse/$ENCODER/case$CASE"

# 定义图像宽度和高度
IMAGE_WIDTH=960
IMAGE_HEIGHT=540

# LOAD_FROM="/data/train_combined_with_dino_20250506_235908/latest.pth"
LOAD_FROM="/data/train_combined_with_dino_20250506_235908/best_abs_rel.pth"

MAX_DEPTH=50

# 创建输出目录（如果不存在）
mkdir -p $OUTPUT_DIR

echo "Starting depth inference using run_live.py..."
echo "LOAD_FROM: $LOAD_FROM"
echo "Input text file: $INPUT_TXT"
echo "Output directory: $OUTPUT_DIR"
echo "Image dimensions: ${IMAGE_WIDTH}x${IMAGE_HEIGHT}"

# 执行run_live.py脚本
python run_live.py \
    --input_txt "$INPUT_TXT" \
    --image_width "$IMAGE_WIDTH" \
    --image_height "$IMAGE_HEIGHT" \
    --save-path "$OUTPUT_DIR" \
    --encoder "$ENCODER" \
    --load-from "$LOAD_FROM" \
    --max-depth "$MAX_DEPTH"

echo "Depth inference completed. Results are saved in $OUTPUT_DIR"
