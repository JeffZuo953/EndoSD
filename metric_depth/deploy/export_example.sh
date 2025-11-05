#!/bin/bash
# Example script for exporting DepthAnythingV2 to ONNX on Linux

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

echo "Exporting DepthAnythingV2 model to ONNX format..."
echo "Model path: $MODEL_PATH"
echo "Output path: $OUTPUT_PATH"
echo "Encoder: $ENCODER"
echo "Input size: ${INPUT_SIZE}x${INPUT_SIZE}"
echo "Max depth: $MAX_DEPTH"
echo "----------------------------------------"

# Run export
python export_onnx.py \
    --model-path "$MODEL_PATH" \
    --encoder $ENCODER \
    --max-depth $MAX_DEPTH \
    --input-size $INPUT_SIZE \
    --output-path "$OUTPUT_PATH"

if [ $? -eq 0 ]; then
    echo "----------------------------------------"
    echo "Export completed successfully!"
    echo "ONNX model saved to: $OUTPUT_PATH"
else
    echo "----------------------------------------"
    echo "Export failed!"
    exit 1
fi
