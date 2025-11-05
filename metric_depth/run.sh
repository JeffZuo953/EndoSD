#!/bin/bash

# =================================================================
# 运行Depth Anything V2时序平滑处理的Bash脚本 (最终版)
#
# 这个脚本为每个指定的 SIGMA 值设置了独立的执行块。
# 在每个块中, Python脚本会处理所有case文件和所有方法的组合,
# 从而保证模型加载次数最少, 运行效率最高。
# =================================================================

# ----------------- 变量设置 (请根据您的环境修改) -----------------

# 输入目录: 包含所有 case*.txt 文件的文件夹
INPUT_DIR="$JIANFU/data/inhouse/eval_paths/"

# 基础输出目录: Python脚本会在此目录下创建 sigma/init/scale/caseN 子目录
# OUTPUT_DIR_ROOT="/data/refined/4/raw"
OUTPUT_DIR_ROOT="$JIANFU/data/refined"

# 在这里列出所有要测试的初始化方法
INIT_METHODS="fixed anneal dynamic"

# 在这里列出所有要测试的缩放方法
SCALE_METHODS="least_squares mean median"

# 模型编码器类型: vits, vitb, vitl, vitg
ENCODER_TYPE="vits"

# 模型权重文件路径
# CHECKPOINT="/data/train_4_dataset/full_20250612_195847/latest.pth"
# CHECKPOINT="/data/train_combined_with_dino_20250506_235908/latest.pth"
CHECKPOINT="$JIANFU/data/train_4_dataset/same_maxdepth_full_20250613_210808/latest.pth"

INPUT_SIZE=400
MAX_DEPTH=0.2
WINDOW_SIZE=30

# # =================================================================
# #  执行块 1: SIGMA = 0.01
# # =================================================================
# echo "================================================================="
# echo "Starting batch processing for SIGMA = 0.01"
# echo "================================================================="

# python run.py \
#     --img-path "$INPUT_DIR" \
#     --outdir "$OUTPUT_DIR_ROOT" \
#     --encoder "$ENCODER_TYPE" \
#     --load-from "$CHECKPOINT" \
#     --max-depth "$MAX_DEPTH" \
#     --sigma "0.01" \
#     --temporal-smoothing \
#     --init-methods $INIT_METHODS \
#     --scale-methods $SCALE_METHODS \
#     --window-size "$WINDOW_SIZE" \
#     --input-size "$INPUT_SIZE" \
#     --save-npz \
#     --grayscale


# =================================================================
#  执行块 2: raw
# =================================================================
echo "================================================================="
echo "Starting batch processing for raw Image"
echo "================================================================="

python run.py \
    --img-path "$INPUT_DIR" \
    --outdir "$OUTPUT_DIR_ROOT" \
    --encoder "$ENCODER_TYPE" \
    --load-from "$CHECKPOINT" \
    --max-depth "$MAX_DEPTH" \
    --sigma "0.1" \
    --init-methods "fixed" \
    --scale-methods "mean" \
    --window-size "$WINDOW_SIZE" \
    --input-size "$INPUT_SIZE" \
    --save-npz \
    --grayscale


echo "================================================================="
echo "All cases and methods for all sigma values have been processed."
echo "================================================================="


# #!/bin/bash

# # =================================================================
# # 运行Depth Anything V2时序平滑处理的Bash脚本 (最终版)
# #
# # 这个脚本为每个指定的 SIGMA 值设置了独立的执行块。
# # 在每个块中, Python脚本会处理所有case文件和所有方法的组合,
# # 从而保证模型加载次数最少, 运行效率最高。
# # =================================================================

# # ----------------- 变量设置 (请根据您的环境修改) -----------------

# # 输入目录: 包含所有 case*.txt 文件的文件夹
# INPUT_DIR="/data/inhouse/eval_paths/"

# # 基础输出目录: Python脚本会在此目录下创建 sigma/init/scale/caseN 子目录
# # OUTPUT_DIR_ROOT="/data/refined/4/raw"
# OUTPUT_DIR_ROOT="/data/refined/3_same_02/raw"

# # 在这里列出所有要测试的初始化方法
# INIT_METHODS="fixed anneal dynamic"

# # 在这里列出所有要测试的缩放方法
# SCALE_METHODS="least_squares mean median"

# # 模型编码器类型: vits, vitb, vitl, vitg
# ENCODER_TYPE="vits"

# # 模型权重文件路径
# # CHECKPOINT="/data/train_4_dataset/full_20250612_195847/latest.pth"
# CHECKPOINT="/data/train_combined_with_dino_20250506_235908/latest.pth"
# # CHECKPOINT="/data/train_4_dataset/same_maxdepth_full_20250613_210808/latest.pth"

# INPUT_SIZE=400
# MAX_DEPTH=0.2
# WINDOW_SIZE=30

# # # =================================================================
# # #  执行块 1: SIGMA = 0.01
# # # =================================================================
# # echo "================================================================="
# # echo "Starting batch processing for SIGMA = 0.01"
# # echo "================================================================="

# # python run.py \
# #     --img-path "$INPUT_DIR" \
# #     --outdir "$OUTPUT_DIR_ROOT" \
# #     --encoder "$ENCODER_TYPE" \
# #     --load-from "$CHECKPOINT" \
# #     --max-depth "$MAX_DEPTH" \
# #     --sigma "0.01" \
# #     --temporal-smoothing \
# #     --init-methods $INIT_METHODS \
# #     --scale-methods $SCALE_METHODS \
# #     --window-size "$WINDOW_SIZE" \
# #     --input-size "$INPUT_SIZE" \
# #     --save-npz \
# #     --grayscale


# # =================================================================
# #  执行块 2: raw
# # =================================================================
# echo "================================================================="
# echo "Starting batch processing for raw Image"
# echo "================================================================="

# python run.py \
#     --img-path "$INPUT_DIR" \
#     --outdir "$OUTPUT_DIR_ROOT" \
#     --encoder "$ENCODER_TYPE" \
#     --load-from "$CHECKPOINT" \
#     --max-depth "$MAX_DEPTH" \
#     --sigma "0.1" \
#     --init-methods "fixed" \
#     --scale-methods "mean" \
#     --window-size "$WINDOW_SIZE" \
#     --input-size "$INPUT_SIZE" \
#     --save-npz \
#     --grayscale


# echo "================================================================="
# echo "All cases and methods for all sigma values have been processed."
# echo "================================================================="

