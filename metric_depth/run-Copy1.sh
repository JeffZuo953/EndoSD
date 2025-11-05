#!/bin/bash

# =================================================================
# 运行Depth Anything V2时序平滑处理的Bash脚本
# =================================================================

# ----------------- 变量设置 (请根据您的环境修改) -----------------
CASE="case1"
# 输入路径: 可以是单个图像, 包含图像路径的txt文件, 或者一个包含图像的文件夹
IMAGE_PATH="/data/inhouse/$CASE.txt"

# 模型编码器类型: vits, vitb, vitl, vitg
ENCODER_TYPE="vits"

# 模型权重文件路径
CHECKPOINT="/data/train_combined_with_dino_20250506_235908/latest.pth"

INPUT_SIZE=400

MAX_DEPTH=0.05

WINDOW_SIZE=30

SIGMA=0.01

# =================================================================
# 场景1: 动态初始化时序平滑 (推荐)
# 使用动态方法(逐帧对比)进行预热, 这是最稳健的选项。
# =================================================================
echo "================================================================="
echo "Running Scenario 1: Temporal Smoothing with DYNAMIC Initialization"
echo "================================================================="

INIT_METHOD="dynamic"
OUTPUT_DIR="/data/refined/$INIT_METHOD/$CASE"

python run.py \
    --img-path "$IMAGE_PATH" \
    --outdir "$OUTPUT_DIR" \
    --encoder "$ENCODER_TYPE" \
    --load-from "$CHECKPOINT" \
    --max-depth "$MAX_DEPTH" \
    --temporal-smoothing \
    --init-method "$INIT_METHOD" \
    --window-size "$WINDOW_SIZE" \
    --sigma "$SIGMA" \
    --input-size "$INPUT_SIZE" \
    --save-npz \
    --grayscale

echo "Scenario 1 finished. Results are in $OUTPUT_DIR"

# =================================================================
# 场景2: 退火初始化时序平滑
# 预热阶段的权重从低到高平滑过渡, 适合模型初始几帧不稳定的情况。
# 要运行此场景, 请注释掉上面的场景1, 并取消下面的注释。
# =================================================================
echo "================================================================="
echo "Running Scenario 2: Temporal Smoothing with ANNEAL Initialization"
echo "================================================================="

INIT_METHOD="anneal"
OUTPUT_DIR="/data/refined/$INIT_METHOD/$CASE"

python run.py \
    --img-path "$IMAGE_PATH" \
    --outdir "$OUTPUT_DIR" \
    --encoder "$ENCODER_TYPE" \
    --load-from "$CHECKPOINT" \
    --max-depth "$MAX_DEPTH" \
    --temporal-smoothing \
    --init-method "$INIT_METHOD" \
    --window-size "$WINDOW_SIZE" \
    --input-size "$INPUT_SIZE" \
    --save-npz \
    --grayscale

echo "Scenario 2 finished. Results are in $OUTPUT_DIR"

# =================================================================
# 场景3: 固定值初始化时序平滑
# 预热阶段所有帧的权重都为1.0, 这是最简单的预热方式。
# =================================================================
echo "================================================================="
echo "Running Scenario 3: Temporal Smoothing with FIXED Initialization"
echo "================================================================="

INIT_METHOD="fixed"
OUTPUT_DIR="/data/refined/$INIT_METHOD/$CASE"

python run.py \
    --img-path "$IMAGE_PATH" \
    --outdir "$OUTPUT_DIR" \
    --encoder "$ENCODER_TYPE" \
    --load-from "$CHECKPOINT" \
    --max-depth "$MAX_DEPTH" \
    --temporal-smoothing \
    --init-method "$INIT_METHOD" \
    --window-size "$WINDOW_SIZE" \
    --input-size "$INPUT_SIZE" \
    --save-npz \
    --grayscale

echo "Scenario 3 finished. Results are in $OUTPUT_DIR"

# =================================================================
# 场景4: 不使用时序平滑 (作为对比基线)
# 直接逐帧预测并保存结果。
# =================================================================
# echo "================================================================="
# echo "Running Scenario 4: No Temporal Smoothing (Baseline)"
# echo "================================================================="
#
# OUTPUT_DIR="./output_baseline"
#
# python run.py \
#     --img-path "$IMAGE_PATH" \
#     --outdir "$OUTPUT_DIR" \
#     --encoder "$ENCODER_TYPE" \
#     --load-from "$CHECKPOINT" \
#     \
#     --save-npz \
#     --grayscale
#
# echo "Scenario 4 finished. Results are in $OUTPUT_DIR"

#!/bin/bash

# =================================================================
# 运行Depth Anything V2时序平滑处理的Bash脚本
# =================================================================

# ----------------- 变量设置 (请根据您的环境修改) -----------------
CASE="case2"
# 输入路径: 可以是单个图像, 包含图像路径的txt文件, 或者一个包含图像的文件夹
IMAGE_PATH="/data/inhouse/$CASE.txt"

# 模型编码器类型: vits, vitb, vitl, vitg
ENCODER_TYPE="vits"

# 模型权重文件路径
CHECKPOINT="/data/train_combined_with_dino_20250506_235908/latest.pth"

# =================================================================
# 场景1: 动态初始化时序平滑 (推荐)
# 使用动态方法(逐帧对比)进行预热, 这是最稳健的选项。
# =================================================================
echo "================================================================="
echo "Running Scenario 1: Temporal Smoothing with DYNAMIC Initialization"
echo "================================================================="

INIT_METHOD="dynamic"
OUTPUT_DIR="/data/refined/$INIT_METHOD/$CASE"

python run.py \
    --img-path "$IMAGE_PATH" \
    --outdir "$OUTPUT_DIR" \
    --encoder "$ENCODER_TYPE" \
    --load-from "$CHECKPOINT" \
    --max-depth "$MAX_DEPTH" \
    --temporal-smoothing \
    --init-method "$INIT_METHOD" \
    --window-size "$WINDOW_SIZE" \
    --sigma "$SIGMA" \
    --input-size "$INPUT_SIZE" \
    --save-npz \
    --grayscale

echo "Scenario 1 finished. Results are in $OUTPUT_DIR"

# =================================================================
# 场景2: 退火初始化时序平滑
# 预热阶段的权重从低到高平滑过渡, 适合模型初始几帧不稳定的情况。
# 要运行此场景, 请注释掉上面的场景1, 并取消下面的注释。
# =================================================================
echo "================================================================="
echo "Running Scenario 2: Temporal Smoothing with ANNEAL Initialization"
echo "================================================================="

INIT_METHOD="anneal"
OUTPUT_DIR="/data/refined/$INIT_METHOD/$CASE"

python run.py \
    --img-path "$IMAGE_PATH" \
    --outdir "$OUTPUT_DIR" \
    --encoder "$ENCODER_TYPE" \
    --load-from "$CHECKPOINT" \
    --max-depth "$MAX_DEPTH" \
    --temporal-smoothing \
    --init-method "$INIT_METHOD" \
    --window-size "$WINDOW_SIZE" \
    --input-size "$INPUT_SIZE" \
    --save-npz \
    --grayscale

echo "Scenario 2 finished. Results are in $OUTPUT_DIR"

# =================================================================
# 场景3: 固定值初始化时序平滑
# 预热阶段所有帧的权重都为1.0, 这是最简单的预热方式。
# =================================================================
echo "================================================================="
echo "Running Scenario 3: Temporal Smoothing with FIXED Initialization"
echo "================================================================="

INIT_METHOD="fixed"
OUTPUT_DIR="/data/refined/$INIT_METHOD/$CASE"

python run.py \
    --img-path "$IMAGE_PATH" \
    --outdir "$OUTPUT_DIR" \
    --encoder "$ENCODER_TYPE" \
    --load-from "$CHECKPOINT" \
    --max-depth "$MAX_DEPTH" \
    --temporal-smoothing \
    --init-method "$INIT_METHOD" \
    --window-size "$WINDOW_SIZE" \
    --input-size "$INPUT_SIZE" \
    --save-npz \
    --grayscale

echo "Scenario 3 finished. Results are in $OUTPUT_DIR"

# =================================================================
# 场景4: 不使用时序平滑 (作为对比基线)
# 直接逐帧预测并保存结果。
# =================================================================
# echo "================================================================="
# echo "Running Scenario 4: No Temporal Smoothing (Baseline)"
# echo "================================================================="
#
# OUTPUT_DIR="./output_baseline"
#
# python run.py \
#     --img-path "$IMAGE_PATH" \
#     --outdir "$OUTPUT_DIR" \
#     --encoder "$ENCODER_TYPE" \
#     --load-from "$CHECKPOINT" \
#     \
#     --save-npz \
#     --grayscale
#
# echo "Scenario 4 finished. Results are in $OUTPUT_DIR"

#!/bin/bash

# =================================================================
# 运行Depth Anything V2时序平滑处理的Bash脚本
# =================================================================

# ----------------- 变量设置 (请根据您的环境修改) -----------------
CASE="case3"
# 输入路径: 可以是单个图像, 包含图像路径的txt文件, 或者一个包含图像的文件夹
IMAGE_PATH="/data/inhouse/$CASE.txt"

# 模型编码器类型: vits, vitb, vitl, vitg
ENCODER_TYPE="vits"

# 模型权重文件路径
CHECKPOINT="/data/train_combined_with_dino_20250506_235908/latest.pth"

# =================================================================
# 场景1: 动态初始化时序平滑 (推荐)
# 使用动态方法(逐帧对比)进行预热, 这是最稳健的选项。
# =================================================================
echo "================================================================="
echo "Running Scenario 1: Temporal Smoothing with DYNAMIC Initialization"
echo "================================================================="

INIT_METHOD="dynamic"
OUTPUT_DIR="/data/refined/$INIT_METHOD/$CASE"

python run.py \
    --img-path "$IMAGE_PATH" \
    --outdir "$OUTPUT_DIR" \
    --encoder "$ENCODER_TYPE" \
    --load-from "$CHECKPOINT" \
    --max-depth "$MAX_DEPTH" \
    --temporal-smoothing \
    --init-method "$INIT_METHOD" \
    --window-size "$WINDOW_SIZE" \
    --sigma "$SIGMA" \
    --input-size "$INPUT_SIZE" \
    --save-npz \
    --grayscale

echo "Scenario 1 finished. Results are in $OUTPUT_DIR"

# =================================================================
# 场景2: 退火初始化时序平滑
# 预热阶段的权重从低到高平滑过渡, 适合模型初始几帧不稳定的情况。
# 要运行此场景, 请注释掉上面的场景1, 并取消下面的注释。
# =================================================================
echo "================================================================="
echo "Running Scenario 2: Temporal Smoothing with ANNEAL Initialization"
echo "================================================================="

INIT_METHOD="anneal"
OUTPUT_DIR="/data/refined/$INIT_METHOD/$CASE"

python run.py \
    --img-path "$IMAGE_PATH" \
    --outdir "$OUTPUT_DIR" \
    --encoder "$ENCODER_TYPE" \
    --load-from "$CHECKPOINT" \
    --max-depth "$MAX_DEPTH" \
    --temporal-smoothing \
    --init-method "$INIT_METHOD" \
    --window-size "$WINDOW_SIZE" \
    --input-size "$INPUT_SIZE" \
    --save-npz \
    --grayscale

echo "Scenario 2 finished. Results are in $OUTPUT_DIR"

# =================================================================
# 场景3: 固定值初始化时序平滑
# 预热阶段所有帧的权重都为1.0, 这是最简单的预热方式。
# =================================================================
echo "================================================================="
echo "Running Scenario 3: Temporal Smoothing with FIXED Initialization"
echo "================================================================="

INIT_METHOD="fixed"
OUTPUT_DIR="/data/refined/$INIT_METHOD/$CASE"

python run.py \
    --img-path "$IMAGE_PATH" \
    --outdir "$OUTPUT_DIR" \
    --encoder "$ENCODER_TYPE" \
    --load-from "$CHECKPOINT" \
    --max-depth "$MAX_DEPTH" \
    --temporal-smoothing \
    --init-method "$INIT_METHOD" \
    --window-size "$WINDOW_SIZE" \
    --input-size "$INPUT_SIZE" \
    --save-npz \
    --grayscale

echo "Scenario 3 finished. Results are in $OUTPUT_DIR"

# =================================================================
# 场景4: 不使用时序平滑 (作为对比基线)
# 直接逐帧预测并保存结果。
# =================================================================
# echo "================================================================="
# echo "Running Scenario 4: No Temporal Smoothing (Baseline)"
# echo "================================================================="
#
# OUTPUT_DIR="./output_baseline"
#
# python run.py \
#     --img-path "$IMAGE_PATH" \
#     --outdir "$OUTPUT_DIR" \
#     --encoder "$ENCODER_TYPE" \
#     --load-from "$CHECKPOINT" \
#     \
#     --save-npz \
#     --grayscale
#
# echo "Scenario 4 finished. Results are in $OUTPUT_DIR"
