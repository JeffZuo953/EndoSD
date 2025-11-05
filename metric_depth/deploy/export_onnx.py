import argparse
import torch
import torch.nn as nn
import onnx
import onnxruntime as ort
import numpy as np
import sys
import os
from collections import OrderedDict

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from depth_anything_v2.dpt import DepthAnythingV2


def strip_module_prefix(state_dict):
    """Remove 'module.' prefix from state dict keys"""
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        new_key = k.replace('module.', '') if k.startswith('module.') else k
        new_state_dict[new_key] = v
    return new_state_dict


def export_to_onnx(model_path, encoder, max_depth, input_size, output_path, use_gpu=True):
    """Export DepthAnythingV2 model to ONNX format"""
    
    # Model configurations
    model_configs = {
        'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
        'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
        'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
        'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
    }
    
    # Set device
    device = 'cuda' if use_gpu and torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    if device == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    # For ONNX export, we need to temporarily disable xformers if it causes issues
    # But first try with it enabled for better performance
    xformers_disabled = False
    
    # Initialize model
    print(f"Initializing {encoder} model...")
    model = DepthAnythingV2(**{**model_configs[encoder], 'max_depth': max_depth})
    
    # Load checkpoint
    print(f"Loading checkpoint from {model_path}...")
    checkpoint = torch.load(model_path, map_location='cpu')
    state_dict = checkpoint.get("model", checkpoint)
    model.load_state_dict(strip_module_prefix(state_dict))
    
    # Move model to device
    model = model.to(device)
    
    # Set model to evaluation mode
    model.eval()
    
    # Create dummy input
    print(f"Creating dummy input with size {input_size}x{input_size}...")
    dummy_input = torch.randn(1, 3, input_size, input_size).to(device)
    
    # Export to ONNX
    print(f"Exporting to ONNX: {output_path}")
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['depth'],
        dynamic_axes={
            'input': {0: 'batch_size'},
            'depth': {0: 'batch_size'}
        }
    )
    
    print("Export completed!")
    
    # Verify the exported model
    print("\nVerifying ONNX model...")
    onnx_model = onnx.load(output_path)
    onnx.checker.check_model(onnx_model)
    print("ONNX model is valid!")
    
    # Test with ONNX Runtime
    print("\nTesting with ONNX Runtime...")
    ort_session = ort.InferenceSession(output_path)
    
    # Run inference
    ort_inputs = {ort_session.get_inputs()[0].name: dummy_input.numpy()}
    ort_outputs = ort_session.run(None, ort_inputs)
    
    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {ort_outputs[0].shape}")
    print(f"Output range: [{ort_outputs[0].min():.4f}, {ort_outputs[0].max():.4f}]")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Export DepthAnythingV2 to ONNX')
    
    parser.add_argument('--model-path', type=str, required=True,
                        help='Path to the PyTorch model checkpoint')
    parser.add_argument('--encoder', type=str, default='vits', 
                        choices=['vits', 'vitb', 'vitl', 'vitg'],
                        help='Model encoder type')
    parser.add_argument('--max-depth', type=float, default=0.2,
                        help='Maximum depth value')
    parser.add_argument('--input-size', type=int, default=518,
                        help='Input image size')
    parser.add_argument('--output-path', type=str, default='depth_anything_v2.onnx',
                        help='Output ONNX model path')
    
    args = parser.parse_args()
    
    export_to_onnx(args.model_path, args.encoder, args.max_depth, 
                   args.input_size, args.output_path)
