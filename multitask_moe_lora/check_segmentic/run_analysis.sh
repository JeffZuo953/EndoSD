#!/bin/bash

# =====================================================================================
# 语义分割误分类分析运行脚本 (简化版)
#
# 使用方法:
# 1. 修改 CHECKPOINT_PATH 为您要测试的.pth模型文件的路径。
# 2. 修改 DATASET_CONFIG_COMBINED 为您加载混合验证集（同时包含kidney和colon）的配置名。
# 3. 运行此脚本: ./run_analysis.sh
#
# 脚本会自动遍历数据，并同时输出 kidney 和 colon 的误分类分析报告。
# =====================================================================================

# --- 用户配置 ---

# 指定要评估的模型检查点
CHECKPOINT_PATH="/data/ziyi/multitask/save/train_lesion/multitask_vits_original_uwl_20250917_121352/checkpoint_latest.pth"

# 指定加载混合验证集的数据集配置文件名 (在dataset_configs.py中定义)
# 例如: "server_hk_01" 或任何包含 'kidney' 和 'colon' 验证集的配置
DATASET_CONFIG_COMBINED="server_hk_01"

# --- 脚本主体 ---

# 检查checkpoint文件是否存在
if [ ! -f "$CHECKPOINT_PATH" ]; then
    echo "Error: Checkpoint file not found at $CHECKPOINT_PATH"
    exit 1
fi

echo "========================================================================="
echo "Starting Misclassification Analysis for Kidney and Colon datasets"
echo "Checkpoint: $CHECKPOINT_PATH"
echo "Dataset Config: $DATASET_CONFIG_COMBINED"
echo "========================================================================="

# 使用 torchrun 和 -m 标志来启动脚本
# torchrun 创建分布式环境 (解决 LOCAL_RANK 问题)
# -m 确保从项目根目录正确解析导入 (解决 ImportError 问题)
torchrun --standalone --nproc_per_node=1 -m multitask_moe_lora.check_segmentic.analyze_misclassification \
    --resume-from "$CHECKPOINT_PATH" \
    --dataset-config-name "$DATASET_CONFIG_COMBINED" \
    --area-threshold 100 \
    --encoder vits \
    --num-classes 4 \
    --img-size 518 \
    --mode original \
    --save-path "/data/ziyi/multitask/save/eval_lesion"

if [ $? -ne 0 ]; then
    echo "Error during analysis. Aborting."
    exit 1
fi

echo ""
echo "Analysis complete."
