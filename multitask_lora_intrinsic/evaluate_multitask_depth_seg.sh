#!/bin/bash
export PYTHONDONTWRITEBYTECODE=1

# ==============================================================================
# 多任务评估配置脚本 (深度估计 + 语义分割)
# ==============================================================================

# 默认参数
NUM_GPUS=2
CUDA_DEVICES="0,1" # 指定使用的GPU设备
ENCODER="vits"   # 模型编码器: vits, vitb, vitl, vitg, dinov3_vits16, etc.
FEATURES=64      # 特征维度
NUM_CLASSES=4    # 分割类别数
MAX_DEPTH=0.2    # 最大深度值
SEG_INPUT_TYPE="from_depth" # 分割头输入类型: "last", "last_four", "from_depth"

# 评估参数
BATCH_SIZE=40
SEG_BATCH_SIZE=40 # 分割任务的批处理大小
VAL_BATCH_SIZE=100  # 验证集的批处理大小
IMG_SIZE=518

# 数据路径配置
BASE_DATA_PATH="/media/ExtHDD1/jianfu/data"

# 解析命令行参数
CHECKPOINT_ROOT=""
START_EPOCH=0
END_EPOCH=100

# 显示使用方法
usage() {
    echo "Usage: $0 [OPTIONS]"
    echo "Options:"
    echo "  --checkpoint-root PATH    Root directory containing checkpoint files"
    echo " --start-epoch NUM         Start epoch number (default: 0)"
    echo "  --end-epoch NUM           End epoch number (default: 100)"
    echo "  --help                    Display this help message"
    echo ""
    echo "Example:"
    echo "  $0 --checkpoint-root /path/to/checkpoints --start-epoch 0 --end-epoch 50"
    exit 1
}

# 解析命令行参数
while [[ $# -gt 0 ]]; do
    case $1 in
        --checkpoint-root)
            CHECKPOINT_ROOT="$2"
            shift 2
            ;;
        --start-epoch)
            START_EPOCH="$2"
            shift 2
            ;;
        --end-epoch)
            END_EPOCH="$2"
            shift 2
            ;;
        --help)
            usage
            ;;
        *)
            echo "Unknown option: $1"
            usage
            ;;
    esac
done

# 检查必需参数
if [ -z "$CHECKPOINT_ROOT" ]; then
    echo "错误: 必须指定 --checkpoint-root 参数"
    usage
fi

# 检查checkpoint根目录是否存在
if [ ! -d "$CHECKPOINT_ROOT" ]; then
    echo "错误: Checkpoint根目录不存在: $CHECKPOINT_ROOT"
    exit 1
fi

# 保存路径
BASE_SAVE_PATH="${BASE_DATA_PATH}/evaluate_lesion"
SAVE_PATH="${BASE_SAVE_PATH}/multitask_${ENCODER}_$(date +%Y%m%d_%H%M%S)"

# 创建保存目录
mkdir -p "${SAVE_PATH}"
exec > >(tee -a "${SAVE_PATH}/evaluation.log") 2>&1

# ==============================================================================
# 验证配置
# ==============================================================================

echo "=============================================================================="
echo "多任务评估配置 (深度估计 + 语义分割)"
echo "------------------------------------------------------------------------------"
echo "  编码器:              ${ENCODER}"
echo "  特征维度:            ${FEATURES}"
echo "  分割类别数:          ${NUM_CLASSES}"
echo "  最大深度:            ${MAX_DEPTH}"
echo "  分割输入类型:        ${SEG_INPUT_TYPE}"
echo "  批处理大小 (深度): ${BATCH_SIZE}"
echo "  批处理大小 (分割): ${SEG_BATCH_SIZE}"
echo " 批处理大小 (验证): ${VAL_BATCH_SIZE}"
echo "  图像尺寸:            ${IMG_SIZE}"
echo "  GPU设备:             ${CUDA_DEVICES}"
echo "  Checkpoint根目录:    ${CHECKPOINT_ROOT}"
echo "  Epoch区间:           ${START_EPOCH} - ${END_EPOCH}"
echo "  保存路径:            ${SAVE_PATH}"
echo "=============================================================================="
echo ""

# 设置CUDA设备
if [ -n "$CUDA_DEVICES" ]; then
    export CUDA_VISIBLE_DEVICES=${CUDA_DEVICES}
fi

# 设置分布式环境变量（多GPU模式）
export RANK=0
export LOCAL_RANK=0
export WORLD_SIZE=${NUM_GPUS}
export MASTER_ADDR=localhost
export MASTER_PORT=20599

echo "设置分布式环境变量..."
echo "RANK=${RANK}, WORLD_SIZE=${WORLD_SIZE}, MASTER_ADDR=${MASTER_ADDR}:${MASTER_PORT}"
echo ""

# 构建基础评估命令（多GPU模式）
BASE_EVAL_CMD="python -m torch.distributed.launch --nproc_per_node=${NUM_GPUS} --master_port=${MASTER_PORT} multitask/evaluate_multitask_depth_seg.py \
    --encoder ${ENCODER} \
    --features ${FEATURES} \
    --num-classes ${NUM_CLASSES} \
    --max-depth ${MAX_DEPTH} \
    --seg-input-type ${SEG_INPUT_TYPE} \
    --bs ${BATCH_SIZE} \
    --seg-bs ${SEG_BATCH_SIZE} \
    --val-bs ${VAL_BATCH_SIZE} \
    --img-size ${IMG_SIZE} \
    --save-path \"${SAVE_PATH}\" \
    --mixed-precision"

# 复制代码备份
cp -r /media/ExtHDD1/jianfu/depth/DepthAnythingV2/multitask "${SAVE_PATH}/code"
echo "code backed up to ${SAVE_PATH}/code"
echo ""

# 遍历指定范围内的epoch进行评估
echo "开始多任务评估..."
for (( epoch=$START_EPOCH; epoch<=$END_EPOCH; epoch++ )); do
    CHECKPOINT_PATH="${CHECKPOINT_ROOT}/checkpoint_epoch_${epoch}.pth"
    
    # 检查checkpoint文件是否存在
    if [ -f "$CHECKPOINT_PATH" ]; then
        echo "=============================================================================="
        echo "评估 checkpoint: $CHECKPOINT_PATH"
        echo "=============================================================================="
        
        # 构建当前epoch的评估命令
        EVAL_CMD="${BASE_EVAL_CMD} --pretrained-from \"${CHECKPOINT_PATH}\""
        
        echo "评估命令:"
        echo "$EVAL_CMD"
        echo ""
        
        # 执行评估
        eval $EVAL_CMD
        
        echo "完成评估 checkpoint: $CHECKPOINT_PATH"
        echo ""
    else
        echo "跳过不存在的checkpoint: $CHECKPOINT_PATH"
    fi
done

echo "所有评估完成!"