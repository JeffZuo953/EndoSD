#!/bin/bash
# Batch processing script for DepthAnythingV2 ONNX inference

# Set script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Configuration
ONNX_MODEL="depth_anything_v2_vits_518.onnx"
BASE_OUTPUT_DIR="./batch_results"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# Check if ONNX model exists
if [ ! -f "$ONNX_MODEL" ]; then
    echo "Error: ONNX model not found: $ONNX_MODEL"
    echo "Please run export_example.sh first to create the ONNX model."
    exit 1
fi

echo "Starting batch inference with DepthAnythingV2 ONNX model"
echo "Model: $ONNX_MODEL"
echo "Timestamp: $TIMESTAMP"
echo "========================================"

# Process test images from text file
echo -e "\n[1/3] Processing images from test_mapping.txt..."
if [ -f "/root/c3vd/test_mapping.txt" ]; then
    OUTPUT_DIR="${BASE_OUTPUT_DIR}/test_mapping_${TIMESTAMP}"
    python run_onnx.py \
        --onnx-path "$ONNX_MODEL" \
        --img-path "/root/c3vd/test_mapping.txt" \
        --output-dir "$OUTPUT_DIR" \
        --grayscale
    echo "Results saved to: $OUTPUT_DIR"
else
    echo "Warning: test_mapping.txt not found, skipping..."
fi

# Process single test image
echo -e "\n[2/3] Processing single test image..."
TEST_IMAGE="/data/c3vd/test/color/trans_t4_a/0381_color.png"
if [ -f "$TEST_IMAGE" ]; then
    OUTPUT_DIR="${BASE_OUTPUT_DIR}/single_test_${TIMESTAMP}"
    python run_onnx.py \
        --onnx-path "$ONNX_MODEL" \
        --img-path "$TEST_IMAGE" \
        --output-dir "$OUTPUT_DIR" \
        --pred-only \
        --grayscale \
        --save-numpy
    echo "Results saved to: $OUTPUT_DIR"
else
    echo "Warning: Test image not found, skipping..."
fi

# Process directory of images
echo -e "\n[3/3] Processing directory of images..."
TEST_DIR="/data/c3vd/test/color"
if [ -d "$TEST_DIR" ]; then
    OUTPUT_DIR="${BASE_OUTPUT_DIR}/directory_test_${TIMESTAMP}"
    python run_onnx.py \
        --onnx-path "$ONNX_MODEL" \
        --img-path "$TEST_DIR" \
        --output-dir "$OUTPUT_DIR"
    echo "Results saved to: $OUTPUT_DIR"
else
    echo "Warning: Test directory not found, skipping..."
fi

echo -e "\n========================================"
echo "Batch processing completed!"
echo "All results saved under: $BASE_OUTPUT_DIR"

# Generate summary report
REPORT_FILE="${BASE_OUTPUT_DIR}/inference_report_${TIMESTAMP}.txt"
echo "Generating summary report..."
{
    echo "DepthAnythingV2 ONNX Inference Report"
    echo "Generated: $(date)"
    echo "Model: $ONNX_MODEL"
    echo ""
    echo "Processed directories:"
    find "$BASE_OUTPUT_DIR" -name "*_${TIMESTAMP}" -type d | while read dir; do
        count=$(find "$dir" -name "*.png" | wc -l)
        echo "  - $dir: $count images"
    done
} > "$REPORT_FILE"

echo "Report saved to: $REPORT_FILE"
