#!/bin/bash
# Example script for running DepthAnythingV2 ONNX inference on Linux

# Set script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Default parameters
# Hardcoded parameters
INPUT_SIZE=518
ONNX_MODEL="/media/ExtHDD1/jianfu/data/onnx/da2/depth_anything_v2_vits_${INPUT_SIZE}.onnx"
INPUT_PATH="/media/ExtHDD1/jianfu/data/inhouse/case3/img/0001.png"
OUTPUT_DIR="./onnx_results"
GRAYSCALE=true
PRED_ONLY=false
SAVE_NUMPY=false
NO_RESIZE_BACK=false

# Check if ONNX model exists
if [ ! -f "$ONNX_MODEL" ]; then
    echo "Error: ONNX model not found: $ONNX_MODEL"
    echo "Please run export_example.sh first to create the ONNX model."
    exit 1
fi

echo "Running DepthAnythingV2 ONNX inference with hardcoded parameters..."
echo "ONNX model: $ONNX_MODEL"
echo "Input path: $INPUT_PATH"
echo "Output directory: $OUTPUT_DIR"
echo "Grayscale: $GRAYSCALE"
echo "Pred Only: $PRED_ONLY"
echo "Save Numpy: $SAVE_NUMPY"
echo "No Resize Back: $NO_RESIZE_BACK"
echo "----------------------------------------"

# Build command
CMD="python run_onnx.py --onnx-path \"$ONNX_MODEL\" --img-path \"$INPUT_PATH\" --output-dir \"$OUTPUT_DIR\""

if [ "$GRAYSCALE" = true ]; then
    CMD="$CMD --grayscale"
fi

if [ "$PRED_ONLY" = true ]; then
    CMD="$CMD --pred-only"
fi

if [ "$SAVE_NUMPY" = true ]; then
    CMD="$CMD --save-numpy"
fi

if [ "$NO_RESIZE_BACK" = true ]; then
    CMD="$CMD --no-resize-back"
fi

# Run inference
eval $CMD

if [ $? -eq 0 ]; then
    echo "----------------------------------------"
    echo "Inference completed successfully!"
    echo "Results saved to: $OUTPUT_DIR"
else
    echo "----------------------------------------"
    echo "Inference failed!"
    exit 1
fi
