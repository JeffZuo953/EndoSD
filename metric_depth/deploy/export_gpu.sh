#!/bin/bash
# Export script for GPU-optimized ONNX model

# Set script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Model paths - update these according to your setup
MODEL_PATH="/media/ExtHDD1/jianfu/data/train_4_dataset/same_maxdepth_full_20250613_210808/latest.pth"
OUTPUT_PATH="/media/ExtHDD1/jianfu/data/onnx/da2/depth_anything_v2_vits_518.onnx"

# Export parameters
ENCODER="vits"
MAX_DEPTH=0.2
INPUT_SIZE=518
OPSET_VERSION=14  # ONNX opset 14 has better GPU support

echo "=========================================="
echo "Exporting DepthAnythingV2 for GPU (RTX 3090)"
echo "=========================================="
echo "Model path: $MODEL_PATH"
echo "Output path: $OUTPUT_PATH"
echo "Encoder: $ENCODER"
echo "Input size: ${INPUT_SIZE}x${INPUT_SIZE}"
echo "Max depth: $MAX_DEPTH"
echo "ONNX opset: $OPSET_VERSION"
echo "=========================================="

# Check if CUDA is available
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda if torch.cuda.is_available() else \"N/A\"}')"

# Run export with GPU optimization
python export_onnx_gpu.py \
    --model-path "$MODEL_PATH" \
    --encoder $ENCODER \
    --max-depth $MAX_DEPTH \
    --input-size $INPUT_SIZE \
    --output-path "$OUTPUT_PATH" \
    --opset-version $OPSET_VERSION

if [ $? -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "✅ Export completed successfully!"
    echo "ONNX model saved to: $OUTPUT_PATH"
    echo ""
    echo "To test the model performance:"
    echo "python benchmark_onnx.py --onnx-path $OUTPUT_PATH --test-image <path_to_test_image>"
    echo "=========================================="
else
    echo ""
    echo "=========================================="
    echo "❌ Export failed!"
    echo "Check the error messages above for details."
    echo "=========================================="
    exit 1
fi
