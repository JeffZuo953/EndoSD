#!/bin/bash
# Example script for running DepthAnythingV2 PyTorch inference

# Set script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Hardcoded parameters
ENCODER="vits"
MODEL_PATH="/media/ExtHDD1/jianfu/data/train_4_dataset/same_maxdepth_full_20250613_210808/latest.pth"
INPUT_PATH="/media/ExtHDD1/jianfu/data/inhouse/case3/img/0001.png"
OUTPUT_DIR="./pytorch_results"
PRED_ONLY=false
GRAYSCALE=true
SAVE_NUMPY=false
MAX_DEPTH=0.2
DEVICE="cuda"
NO_RESIZE_BACK=false

# Check if PyTorch model exists
if [ ! -f "$MODEL_PATH" ]; then
    echo "Error: PyTorch model not found: $MODEL_PATH"
    echo "Please make sure the model checkpoint exists."
    exit 1
fi

echo "Running DepthAnythingV2 PyTorch inference with hardcoded parameters..."
echo "Model path: $MODEL_PATH"
echo "Encoder: $ENCODER"
echo "Input path: $INPUT_PATH"
echo "Output directory: $OUTPUT_DIR"
echo "Max depth: $MAX_DEPTH"
echo "Device: $DEVICE"
echo "----------------------------------------"

# Build command
CMD="python run_pytorch.py --model-path \"$MODEL_PATH\" --encoder \"$ENCODER\" --img-path \"$INPUT_PATH\" --output-dir \"$OUTPUT_DIR\" --max-depth $MAX_DEPTH --device $DEVICE"

if [ "$PRED_ONLY" = true ]; then
    CMD="$CMD --pred-only"
fi

if [ "$GRAYSCALE" = true ]; then
    CMD="$CMD --grayscale"
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