# DepthAnythingV2 ONNX Deployment

This directory contains scripts for exporting DepthAnythingV2 models to ONNX format and running inference with the exported models.

## Requirements

Before using these scripts, make sure you have the following packages installed:

```bash
pip install torch torchvision onnx onnxruntime opencv-python matplotlib numpy
```

## Files

- `export_onnx.py`: Script to export PyTorch model to ONNX format
- `run_onnx.py`: Script to run inference using the exported ONNX model
- `README.md`: This file

## Exporting Model to ONNX

To export a trained DepthAnythingV2 model to ONNX format:

```bash
# Example for vits encoder with input size 518x518
python export_onnx.py \
    --model-path /path/to/your/checkpoint.pth \
    --encoder vits \
    --max-depth 0.2 \
    --input-size 518 \
    --output-path depth_anything_v2_vits.onnx
```

### Parameters for export_onnx.py:

- `--model-path`: Path to the PyTorch model checkpoint (required)
- `--encoder`: Model encoder type [vits, vitb, vitl, vitg] (default: vits)
- `--max-depth`: Maximum depth value (default: 0.2)
- `--input-size`: Input image size (default: 518)
- `--output-path`: Output ONNX model path (default: depth_anything_v2.onnx)

### Example with your model:

```bash
# Navigate to the deploy directory
cd D:\_demo\python\data\depth\DepthAnythingV2\metric_depth\deploy

# Export the model
python export_onnx.py \
    --model-path /media/ExtHDD1/jianfu/data/train_multitask_depth_seg_dinov2/multitask_vits_20250716_115006/best_model.pth \
    --encoder vits \
    --max-depth 0.2 \
    --input-size 518 \
    --output-path depth_anything_v2_vits_518.onnx
```

## Running Inference with ONNX Model

To run inference on images using the exported ONNX model:

```bash
# Process a single image
python run_onnx.py \
    --onnx-path depth_anything_v2_vits_518.onnx \
    --img-path /path/to/image.png \
    --output-dir ./results

# Process multiple images from a directory
python run_onnx.py \
    --onnx-path depth_anything_v2_vits_518.onnx \
    --img-path /path/to/image/directory \
    --output-dir ./results

# Process images listed in a text file
python run_onnx.py \
    --onnx-path depth_anything_v2_vits_518.onnx \
    --img-path /path/to/image_list.txt \
    --output-dir ./results
```

### Parameters for run_onnx.py:

- `--onnx-path`: Path to the ONNX model (required)
- `--img-path`: Path to input image(s) or directory (required)
- `--output-dir`: Output directory for results (default: ./onnx_output)
- `--save-numpy`: Save raw depth output as numpy array
- `--pred-only`: Only save the depth prediction (without side-by-side comparison)
- `--grayscale`: Use grayscale instead of colormap

### Example with different options:

```bash
# Save only depth predictions in grayscale
python run_onnx.py \
    --onnx-path depth_anything_v2_vits_518.onnx \
    --img-path /data/c3vd/test/color/trans_t4_a/0381_color.png \
    --output-dir ./results \
    --pred-only \
    --grayscale

# Save raw numpy arrays along with visualizations
python run_onnx.py \
    --onnx-path depth_anything_v2_vits_518.onnx \
    --img-path /data/test_images \
    --output-dir ./results \
    --save-numpy
```

## Performance Notes

1. ONNX models typically run faster than PyTorch models, especially on CPU
2. The exported model has fixed input size (518x518 by default)
3. Images are automatically resized and preprocessed to match the model's requirements
4. The output depth maps are resized back to the original image dimensions

## Linux Shell Scripts

For Linux users, we provide shell scripts for easier usage:

### 1. Export Model Script (`export_example.sh`)

First, make the script executable:
```bash
chmod +x export_example.sh
```

Then run it:
```bash
./export_example.sh
```

The script is pre-configured with the model path from your training. You can edit it to change the model path.

### 2. Run Inference Script (`run_example.sh`)

Make it executable:
```bash
chmod +x run_example.sh
```

Run inference with various options:
```bash
# Process a single image
./run_example.sh --input /path/to/image.png

# Process with grayscale output
./run_example.sh --input /path/to/image.png --grayscale

# Save only predictions with numpy arrays
./run_example.sh --input /path/to/images/ --pred-only --save-numpy

# Show help
./run_example.sh --help
```

### 3. Batch Processing Script (`batch_inference.sh`)

For processing multiple test cases:
```bash
chmod +x batch_inference.sh
./batch_inference.sh
```

This script automatically processes:
- Images listed in test_mapping.txt
- Single test images
- Directories of images

And generates a summary report.

## Troubleshooting

1. If you encounter import errors, ensure you're running from the correct directory:
   ```bash
   cd /path/to/DepthAnythingV2/metric_depth/deploy
   ```

2. If the model fails to load, check that the checkpoint contains the correct state dict structure

3. For CUDA-related issues, you can force CPU inference by setting:
   ```python
   providers=['CPUExecutionProvider']
   ```
   in the `ort.InferenceSession()` call

4. On Linux, if you get permission denied errors, make sure the scripts are executable:
   ```bash
   chmod +x *.sh
   ```
