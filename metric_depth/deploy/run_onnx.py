import argparse
import cv2
import glob
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import os
import onnxruntime as ort
from collections import OrderedDict


INPUT_SIZE = 518


def preprocess_image(image, input_size):
    """Preprocess image for ONNX model inference"""
    h_orig, w_orig = image.shape[:2]
    
    # Convert BGR to RGB and normalize
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) / 255.0
    
    # Resize image to input_size x input_size
    # Keep aspect ratio and ensure multiple of 14
    scale = input_size / max(h_orig, w_orig)
    h_new = int(h_orig * scale)
    w_new = int(w_orig * scale)
    
    # Make sure dimensions are multiple of 14
    h_new = int(np.ceil(h_new / 14) * 14)
    w_new = int(np.ceil(w_new / 14) * 14)
    
    # Resize image
    image_resized = cv2.resize(image, (w_new, h_new), interpolation=cv2.INTER_CUBIC)
    
    # Pad to square if necessary
    if h_new != w_new:
        max_dim = max(h_new, w_new)
        pad_h = (max_dim - h_new) // 2
        pad_w = (max_dim - w_new) // 2
        image_resized = np.pad(image_resized, 
                               ((pad_h, max_dim - h_new - pad_h), 
                                (pad_w, max_dim - w_new - pad_w), 
                                (0, 0)), 
                               mode='constant', 
                               constant_values=0)
        h_new, w_new = max_dim, max_dim
    
    # Resize to exact input size
    if h_new != input_size or w_new != input_size:
        image_resized = cv2.resize(image_resized, (input_size, input_size), 
                                  interpolation=cv2.INTER_CUBIC)
    
    # Normalize with ImageNet statistics
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image_normalized = (image_resized - mean) / std
    
    # Transpose to CHW format and add batch dimension
    image_tensor = np.transpose(image_normalized, (2, 0, 1)).astype(np.float32)
    image_tensor = np.expand_dims(image_tensor, axis=0)
    
    return image_tensor, (h_orig, w_orig)


def postprocess_depth(depth, original_size, resize_back=True):
    """Postprocess depth output to original image size"""
    h_orig, w_orig = original_size
    
    # Remove batch dimension if present
    if depth.ndim == 3:
        depth = depth[0]
    
    # Resize to original dimensions if requested
    if resize_back:
        depth = cv2.resize(depth, (w_orig, h_orig), interpolation=cv2.INTER_CUBIC)
    
    return depth


def run_inference(onnx_path, image_path, output_dir, save_numpy=False,
                  pred_only=False, grayscale=False, no_resize_back=False):
    """Run inference on images using ONNX model"""
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load ONNX model
    print(f"Loading ONNX model from {onnx_path}...")
    ort_session = ort.InferenceSession(onnx_path)
    
    # Get input details
    input_name = ort_session.get_inputs()[0].name
    input_shape = ort_session.get_inputs()[0].shape
    input_size = INPUT_SIZE  # Use global variable for size
    
    print(f"Model input: {input_name}, shape: {input_shape}, target size: {input_size}")
    
    # Process images
    if os.path.isfile(image_path):
        if image_path.endswith('.txt'):
            with open(image_path, 'r') as f:
                filenames = f.read().splitlines()
        else:
            filenames = [image_path]
    else:
        filenames = glob.glob(os.path.join(image_path, '**/*'), recursive=True)
        filenames = [f for f in filenames if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
    
    # Set up colormap
    cmap = matplotlib.colormaps.get_cmap('Spectral')
    
    print(f"\nProcessing {len(filenames)} images...")
    
    for k, filename in enumerate(filenames):
        filename = filename.strip().split()[0]
        
        print(f'Progress {k+1}/{len(filenames)}: {filename}')
        
        # Check if file exists
        if not os.path.exists(filename):
            print(f"Warning: File does not exist: {filename}")
            continue
        
        # Read image
        raw_image = cv2.imread(filename)
        if raw_image is None:
            print(f"Warning: Could not read image file: {filename}")
            continue
        
        try:
            # Preprocess image
            image_tensor, original_size = preprocess_image(raw_image, input_size)
            
            # Run inference
            ort_inputs = {input_name: image_tensor}
            ort_outputs = ort_session.run(None, ort_inputs)
            depth = ort_outputs[0]
            
            # Postprocess depth
            depth = postprocess_depth(depth, original_size, resize_back=not no_resize_back)
            
        except Exception as e:
            print(f"Error processing {filename}: {e}")
            continue
        
        # Save raw depth if requested
        if save_numpy:
            output_path = os.path.join(output_dir, 
                                      os.path.splitext(os.path.basename(filename))[0] + '_depth.npy')
            np.save(output_path, depth)
        
        # Normalize depth for visualization
        depth_vis = (depth - depth.min()) / (depth.max() - depth.min()) * 255.0
        depth_vis = depth_vis.astype(np.uint8)
        
        # Apply colormap
        if grayscale:
            depth_vis = np.repeat(depth_vis[..., np.newaxis], 3, axis=-1)
        else:
            depth_vis = (cmap(depth_vis)[:, :, :3] * 255)[:, :, ::-1].astype(np.uint8)
        
        # Save output
        output_path = os.path.join(output_dir, 
                                  os.path.splitext(os.path.basename(filename))[0] + '.png')
        
        if pred_only:
            cv2.imwrite(output_path, depth_vis)
        else:
            # Create side-by-side comparison
            split_region = np.ones((raw_image.shape[0], 50, 3), dtype=np.uint8) * 255
            combined_result = cv2.hconcat([raw_image, split_region, depth_vis])
            cv2.imwrite(output_path, combined_result)
    
    print(f"\nProcessing complete! Results saved to {output_dir}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run DepthAnythingV2 ONNX Model')
    
    parser.add_argument('--onnx-path', type=str, required=True,
                        help='Path to the ONNX model')
    parser.add_argument('--img-path', type=str, required=True,
                        help='Path to input image(s) or directory')
    parser.add_argument('--output-dir', type=str, default='./onnx_output',
                        help='Output directory for results')
    parser.add_argument('--save-numpy', action='store_true',
                        help='Save raw depth output as numpy array')
    parser.add_argument('--pred-only', action='store_true',
                        help='Only save the depth prediction')
    parser.add_argument('--grayscale', action='store_true',
                        help='Use grayscale instead of colormap')
    parser.add_argument('--no-resize-back', action='store_true',
                        help='Do not resize the output to original image size')
    
    args = parser.parse_args()
    
    run_inference(args.onnx_path, args.img_path, args.output_dir,
                  args.save_numpy, args.pred_only, args.grayscale, args.no_resize_back)
