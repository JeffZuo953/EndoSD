import argparse
import cv2
import glob
import matplotlib
import numpy as np
import os
import torch
from collections import OrderedDict
import sys

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from depth_anything_v2.dpt import DepthAnythingV2

INPUT_SIZE = 518


def strip_module_prefix(state_dict):
    """Remove 'module.' prefix from state dict keys"""
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        new_key = k.replace('module.', '') if k.startswith('module.') else k
        new_state_dict[new_key] = v
    return new_state_dict


def run_inference(args):
    """Run inference on images using PyTorch model"""
    
    print("=" * 60)
    print("PyTorch Model Inference")
    print("=" * 60)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Model configurations
    model_configs = {
        'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
        'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
        'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
        'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
    }
    
    # Initialize model
    print(f"Loading {args.encoder} model from {args.model_path}...")
    model = DepthAnythingV2(**{**model_configs[args.encoder], 'max_depth': args.max_depth})
    
    # Load checkpoint
    checkpoint = torch.load(args.model_path, map_location='cpu')
    state_dict = checkpoint.get("model", checkpoint)
    model.load_state_dict(strip_module_prefix(state_dict))
    
    # Move model to device
    if args.device == 'cuda' and torch.cuda.is_available():
        model = model.cuda()
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        model = model.cpu()
        args.device = 'cpu'
        print("Using CPU")
        
    model.eval()
    
    # Process images
    if os.path.isfile(args.img_path):
        if args.img_path.endswith('.txt'):
            with open(args.img_path, 'r') as f:
                filenames = f.read().splitlines()
        else:
            filenames = [args.img_path]
    else:
        filenames = glob.glob(os.path.join(args.img_path, '**/*'), recursive=True)
        filenames = [f for f in filenames if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
    
    # Set up colormap
    cmap = matplotlib.colormaps.get_cmap('Spectral')
    
    print(f"\nProcessing {len(filenames)} images...")
    
    with torch.no_grad():
        for k, filename in enumerate(filenames):
            filename = filename.strip().split()[0]
            
            print(f'Progress {k+1}/{len(filenames)}: {filename}')
            
            if not os.path.exists(filename):
                print(f"Warning: File does not exist: {filename}")
                continue
            
            raw_image = cv2.imread(filename)
            if raw_image is None:
                print(f"Warning: Could not read image file: {filename}")
                continue
            
            try:
                # Preprocess image
                image_tensor, (h_orig, w_orig) = model.image2tensor(raw_image, INPUT_SIZE)
                if args.device == 'cuda':
                    image_tensor = image_tensor.cuda()
                
                # Run inference
                depth = model(image_tensor)
                
                # Postprocess depth
                depth = depth.cpu().numpy().squeeze()
                if not args.no_resize_back:
                    depth = cv2.resize(depth, (w_orig, h_orig), interpolation=cv2.INTER_CUBIC)
                
            except Exception as e:
                print(f"Error processing {filename}: {e}")
                continue
            
            # Save raw depth if requested
            if args.save_numpy:
                output_path_npy = os.path.join(args.output_dir, os.path.splitext(os.path.basename(filename))[0] + '_depth.npy')
                np.save(output_path_npy, depth)

            # Normalize depth for visualization
            depth_vis = (depth - depth.min()) / (depth.max() - depth.min()) * 255.0
            depth_vis = depth_vis.astype(np.uint8)
            
            # Apply colormap
            if args.grayscale:
                depth_vis = np.repeat(depth_vis[..., np.newaxis], 3, axis=-1)
            else:
                depth_vis = (cmap(depth_vis)[:, :, :3] * 255)[:, :, ::-1].astype(np.uint8)
            
            # Save output
            output_path_png = os.path.join(args.output_dir, os.path.splitext(os.path.basename(filename))[0] + '.png')
            
            if args.pred_only:
                cv2.imwrite(output_path_png, depth_vis)
            else:
                split_region = np.ones((raw_image.shape[0], 50, 3), dtype=np.uint8) * 255
                combined_result = cv2.hconcat([raw_image, split_region, depth_vis])
                cv2.imwrite(output_path_png, combined_result)
                
    print(f"\nProcessing complete! Results saved to {args.output_dir}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run DepthAnythingV2 PyTorch Model')
    
    parser.add_argument('--model-path', type=str, required=True,
                        help='Path to the PyTorch model checkpoint')
    parser.add_argument('--encoder', type=str, required=True,
                        choices=['vits', 'vitb', 'vitl', 'vitg'],
                        help='Model encoder type')
    parser.add_argument('--img-path', type=str, required=True,
                        help='Path to input image(s) or directory')
    parser.add_argument('--output-dir', type=str, default='./pytorch_results',
                        help='Output directory for results')
    parser.add_argument('--pred-only', action='store_true',
                        help='Only save the depth prediction')
    parser.add_argument('--grayscale', action='store_true',
                        help='Use grayscale instead of colormap')
    parser.add_argument('--save-numpy', action='store_true',
                        help='Save raw depth output as numpy array')
    parser.add_argument('--max-depth', type=float, default=1.0,
                        help='Maximum depth value for model initialization')
    parser.add_argument('--input-size', type=int, default=INPUT_SIZE,
                        help='Input image size')
    parser.add_argument('--device', type=str, default='cuda',
                        choices=['cuda', 'cpu'],
                        help='Device to run inference on')
    parser.add_argument('--no-resize-back', action='store_true',
                        help='Do not resize the output to original image size')
    
    args = parser.parse_args()
    
    run_inference(args)