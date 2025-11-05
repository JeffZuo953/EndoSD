import argparse
import torch
import torch.nn as nn
import onnx
import onnxruntime as ort
import numpy as np
import sys
import os
import warnings
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


class DepthAnythingV2ONNX(nn.Module):
    """Wrapper class for ONNX export with simplified attention"""
    def __init__(self, model):
        super().__init__()
        self.model = model
        
    def forward(self, x):
        # Temporarily disable xformers for export
        import os
        old_env = os.environ.get('XFORMERS_ENABLE_MEM_EFF_ATTN', None)
        os.environ['XFORMERS_ENABLE_MEM_EFF_ATTN'] = '0'
        
        try:
            output = self.model(x)
        finally:
            # Restore environment
            if old_env is not None:
                os.environ['XFORMERS_ENABLE_MEM_EFF_ATTN'] = old_env
            elif 'XFORMERS_ENABLE_MEM_EFF_ATTN' in os.environ:
                del os.environ['XFORMERS_ENABLE_MEM_EFF_ATTN']
                
        return output


def export_to_onnx(model_path, encoder, max_depth, input_size, output_path, 
                   use_gpu=True, opset_version=14, simplify=False):
    """Export DepthAnythingV2 model to ONNX format with GPU optimization"""
    
    # Model configurations
    model_configs = {
        'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
        'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
        'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
        'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
    }
    
    # Check CUDA availability
    if use_gpu and not torch.cuda.is_available():
        print("WARNING: CUDA not available, falling back to CPU")
        use_gpu = False
    
    device = 'cuda' if use_gpu else 'cpu'
    print(f"Using device: {device}")
    if device == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA Version: {torch.version.cuda}")
    
    # Temporarily set environment to avoid xformers issues during export
    os.environ['XFORMERS_DISABLE_FLASH_ATTN'] = '1'
    os.environ['XFORMERS_ENABLE_MEM_EFF_ATTN'] = '0'
    
    try:
        # Initialize model
        print(f"\nInitializing {encoder} model...")
        model = DepthAnythingV2(**{**model_configs[encoder], 'max_depth': max_depth})
        
        # Load checkpoint
        print(f"Loading checkpoint from {model_path}...")
        checkpoint = torch.load(model_path, map_location='cpu')
        state_dict = checkpoint.get("model", checkpoint)
        model.load_state_dict(strip_module_prefix(state_dict))
        
        # Move model to device
        model = model.to(device)
        model.eval()
        
        # Wrap model for ONNX export
        model_wrapped = DepthAnythingV2ONNX(model)
        
        # Create dummy input
        print(f"Creating dummy input with size {input_size}x{input_size}...")
        dummy_input = torch.randn(1, 3, input_size, input_size).to(device)
        
        # Test forward pass
        print("Testing forward pass...")
        with torch.no_grad():
            test_output = model_wrapped(dummy_input)
        print(f"Test output shape: {test_output.shape}")
        
        # Export to ONNX
        print(f"\nExporting to ONNX: {output_path}")
        print(f"Using opset version: {opset_version}")
        
        # Make sure output directory exists
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=torch.jit.TracerWarning)
            
            torch.onnx.export(
                model_wrapped,
                dummy_input,
                output_path,
                export_params=True,
                opset_version=opset_version,
                do_constant_folding=True,
                input_names=['input'],
                output_names=['depth'],
                dynamic_axes={
                    'input': {0: 'batch_size'},
                    'depth': {0: 'batch_size'}
                },
                verbose=False
            )
        
        print("Export completed!")
        
        # Verify the exported model
        print("\nVerifying ONNX model...")
        onnx_model = onnx.load(output_path)
        onnx.checker.check_model(onnx_model)
        print("ONNX model is valid!")
        
        # Get model size
        model_size = os.path.getsize(output_path) / (1024 * 1024)
        print(f"Model size: {model_size:.2f} MB")
        
        # Test with ONNX Runtime
        print("\nTesting with ONNX Runtime...")
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if use_gpu else ['CPUExecutionProvider']
        
        try:
            ort_session = ort.InferenceSession(output_path, providers=providers)
            print(f"ONNX Runtime providers: {ort_session.get_providers()}")
            
            # Move dummy input to CPU for ONNX Runtime
            dummy_input_cpu = dummy_input.cpu().numpy()
            
            # Run inference
            ort_inputs = {ort_session.get_inputs()[0].name: dummy_input_cpu}
            ort_outputs = ort_session.run(None, ort_inputs)
            
            print(f"Input shape: {dummy_input.shape}")
            print(f"Output shape: {ort_outputs[0].shape}")
            print(f"Output range: [{ort_outputs[0].min():.4f}, {ort_outputs[0].max():.4f}]")
            
            # Compare outputs
            torch_output = test_output.cpu().numpy()
            onnx_output = ort_outputs[0]
            max_diff = np.abs(torch_output - onnx_output).max()
            print(f"Max difference between PyTorch and ONNX: {max_diff:.6f}")
            
        except Exception as e:
            print(f"Warning: ONNX Runtime test failed: {e}")
            print("The model was still exported successfully and can be used.")
        
        # Optionally simplify the model
        if simplify:
            try:
                print("\nSimplifying ONNX model...")
                import onnxsim
                model_simp, check = onnxsim.simplify(onnx_model)
                if check:
                    onnx.save(model_simp, output_path)
                    new_size = os.path.getsize(output_path) / (1024 * 1024)
                    print(f"Simplified model size: {new_size:.2f} MB")
                else:
                    print("Simplification check failed, keeping original model")
            except ImportError:
                print("onnx-simplifier not installed, skipping simplification")
                print("Install with: pip install onnx-simplifier")
        
        print("\n✅ ONNX export successful!")
        print(f"Model saved to: {output_path}")
        
        # Print usage instructions
        print("\n" + "="*60)
        print("To use this model with ONNX Runtime:")
        print("="*60)
        print("import onnxruntime as ort")
        print("import numpy as np")
        print("")
        print("# For GPU inference:")
        print("providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']")
        print(f"session = ort.InferenceSession('{os.path.basename(output_path)}', providers=providers)")
        print("")
        print("# For CPU-only inference:")
        print("# providers = ['CPUExecutionProvider']")
        print("# session = ort.InferenceSession('model.onnx', providers=providers)")
        print("")
        print("# Run inference")
        print("input_name = session.get_inputs()[0].name")
        print("output = session.run(None, {input_name: input_array})")
        print("="*60)
        
    finally:
        # Clean up environment variables
        if 'XFORMERS_DISABLE_FLASH_ATTN' in os.environ:
            del os.environ['XFORMERS_DISABLE_FLASH_ATTN']
        if 'XFORMERS_ENABLE_MEM_EFF_ATTN' in os.environ:
            del os.environ['XFORMERS_ENABLE_MEM_EFF_ATTN']


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Export DepthAnythingV2 to ONNX (GPU Optimized)')
    
    parser.add_argument('--model-path', type=str, required=True,
                        help='Path to the PyTorch model checkpoint')
    parser.add_argument('--encoder', type=str, default='vits', 
                        choices=['vits', 'vitb', 'vitl', 'vitg'],
                        help='Model encoder type')
    parser.add_argument('--max-depth', type=float, default=0.2,
                        help='Maximum depth value')
    parser.add_argument('--input-size', type=int, default=518,
                        help='Input image size')
    parser.add_argument('--output-path', type=str, default='depth_anything_v2_gpu.onnx',
                        help='Output ONNX model path')
    parser.add_argument('--opset-version', type=int, default=14,
                        help='ONNX opset version (14 recommended for GPU)')
    parser.add_argument('--cpu', action='store_true',
                        help='Export for CPU instead of GPU')
    parser.add_argument('--simplify', action='store_true',
                        help='Simplify the ONNX model (requires onnx-simplifier)')
    
    args = parser.parse_args()
    
    export_to_onnx(
        args.model_path, 
        args.encoder, 
        args.max_depth, 
        args.input_size, 
        args.output_path,
        use_gpu=not args.cpu,
        opset_version=args.opset_version,
        simplify=args.simplify
    )
