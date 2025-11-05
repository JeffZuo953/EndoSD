import argparse
import cv2
import time
import torch
import numpy as np
import matplotlib.pyplot as plt
import psutil
import GPUtil
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


def get_gpu_memory_usage():
    """Get current GPU memory usage in MB"""
    if torch.cuda.is_available():
        # Using GPUtil for more accurate memory reporting
        try:
            gpus = GPUtil.getGPUs()
            if gpus:
                return gpus[0].memoryUsed
        except:
            pass
        # Fallback to torch
        return torch.cuda.memory_allocated() / 1024 / 1024
    return 0


def benchmark_pytorch(model_path, encoder, max_depth, input_size, test_image_path, 
                     num_warmup=10, num_iterations=100, device='cuda'):
    """Benchmark PyTorch model performance"""
    
    print("=" * 60)
    print("PyTorch Model Benchmark")
    print("=" * 60)
    
    # Model configurations
    model_configs = {
        'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
        'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
        'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
        'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
    }
    
    # Initialize model
    print(f"Loading {encoder} model from {model_path}...")
    model = DepthAnythingV2(**{**model_configs[encoder], 'max_depth': max_depth})
    
    # Load checkpoint
    checkpoint = torch.load(model_path, map_location='cpu')
    state_dict = checkpoint.get("model", checkpoint)
    model.load_state_dict(strip_module_prefix(state_dict))
    
    # Move model to device
    if device == 'cuda' and torch.cuda.is_available():
        model = model.cuda()
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        model = model.cpu()
        device = 'cpu'
        print("Using CPU")
    
    model.eval()
    
    # Load and preprocess test image
    print(f"\nLoading test image: {test_image_path}")
    raw_image = cv2.imread(test_image_path)
    if raw_image is None:
        raise ValueError(f"Could not load image: {test_image_path}")
    
    print(f"Original image size: {raw_image.shape[1]}x{raw_image.shape[0]}")
    
    # Use model's preprocessing
    with torch.no_grad():
        image_tensor, (h, w) = model.image2tensor(raw_image, input_size)
        if device == 'cpu':
            image_tensor = image_tensor.cpu()
    
    print(f"Input tensor shape: {image_tensor.shape}")
    print(f"Input tensor device: {image_tensor.device}")
    
    # Get initial memory usage
    if device == 'cuda':
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        initial_memory = get_gpu_memory_usage()
    else:
        initial_memory = psutil.Process().memory_info().rss / 1024 / 1024
    
    # Warmup
    print(f"\nWarming up with {num_warmup} iterations...")
    with torch.no_grad():
        for _ in range(num_warmup):
            _ = model(image_tensor)
            if device == 'cuda':
                torch.cuda.synchronize()
    
    # Benchmark
    print(f"Running benchmark with {num_iterations} iterations...")
    
    if device == 'cuda':
        torch.cuda.synchronize()
    
    inference_times = []
    
    with torch.no_grad():
        for i in range(num_iterations):
            if device == 'cuda':
                torch.cuda.synchronize()
            
            start_time = time.perf_counter()
            output = model(image_tensor)
            
            if device == 'cuda':
                torch.cuda.synchronize()
            
            end_time = time.perf_counter()
            inference_times.append((end_time - start_time) * 1000)  # Convert to ms
    
    # Get peak memory usage
    if device == 'cuda':
        peak_memory = get_gpu_memory_usage()
        memory_used = peak_memory - initial_memory
    else:
        peak_memory = psutil.Process().memory_info().rss / 1024 / 1024
        memory_used = peak_memory - initial_memory
    
    # Calculate statistics
    inference_times = np.array(inference_times)
    mean_time = np.mean(inference_times)
    std_time = np.std(inference_times)
    min_time = np.min(inference_times)
    max_time = np.max(inference_times)
    fps = 1000.0 / mean_time  # Convert from ms to FPS
    
    # Print results
    print("\n" + "=" * 60)
    print("BENCHMARK RESULTS - PyTorch")
    print("=" * 60)
    print(f"Model: {encoder}")
    print(f"Device: {device}")
    print(f"Input size: {input_size}x{input_size}")
    print(f"Output shape: {output.shape}")
    print(f"\nTiming Statistics:")
    print(f"  Mean inference time: {mean_time:.2f} ms")
    print(f"  Std deviation: {std_time:.2f} ms")
    print(f"  Min inference time: {min_time:.2f} ms")
    print(f"  Max inference time: {max_time:.2f} ms")
    print(f"  FPS: {fps:.2f}")
    print(f"\nMemory Usage:")
    print(f"  Memory used: {memory_used:.2f} MB")
    print(f"  Peak memory: {peak_memory:.2f} MB")
    print("=" * 60)
    
    return {
        'mean_time': mean_time,
        'std_time': std_time,
        'min_time': min_time,
        'max_time': max_time,
        'fps': fps,
        'memory_used': memory_used,
        'inference_times': inference_times
    }


def plot_performance(results, output_path='pytorch_performance.png'):
    """Create performance visualization"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Inference time distribution
    ax1.hist(results['inference_times'], bins=30, alpha=0.7, color='blue', edgecolor='black')
    ax1.axvline(results['mean_time'], color='red', linestyle='--', 
                label=f'Mean: {results["mean_time"]:.2f} ms')
    ax1.set_xlabel('Inference Time (ms)')
    ax1.set_ylabel('Frequency')
    ax1.set_title('PyTorch Inference Time Distribution')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Performance metrics
    metrics = ['FPS', 'Memory (MB)']
    values = [results['fps'], results['memory_used']]
    colors = ['green', 'orange']
    
    bars = ax2.bar(metrics, values, color=colors, alpha=0.7)
    ax2.set_ylabel('Value')
    ax2.set_title('PyTorch Performance Metrics')
    
    # Add value labels on bars
    for bar, value in zip(bars, values):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{value:.1f}', ha='center', va='bottom')
    
    ax2.grid(True, alpha=0.3, axis='y')
    
    plt.suptitle(f'PyTorch Model Performance Analysis\n'
                 f'Mean: {results["mean_time"]:.2f}±{results["std_time"]:.2f} ms, '
                 f'FPS: {results["fps"]:.2f}')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nPerformance plot saved to: {output_path}")
    
    # Close the plot to free memory
    plt.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Benchmark PyTorch DepthAnythingV2 Model')
    
    parser.add_argument('--model-path', type=str, required=True,
                        help='Path to the PyTorch model checkpoint')
    parser.add_argument('--encoder', type=str, default='vits', 
                        choices=['vits', 'vitb', 'vitl', 'vitg'],
                        help='Model encoder type')
    parser.add_argument('--max-depth', type=float, default=0.2,
                        help='Maximum depth value')
    parser.add_argument('--input-size', type=int, default=518,
                        help='Input image size')
    parser.add_argument('--test-image', type=str, required=True,
                        help='Path to test image')
    parser.add_argument('--num-warmup', type=int, default=10,
                        help='Number of warmup iterations')
    parser.add_argument('--num-iterations', type=int, default=100,
                        help='Number of benchmark iterations')
    parser.add_argument('--device', type=str, default='cuda',
                        choices=['cuda', 'cpu'],
                        help='Device to run benchmark on')
    parser.add_argument('--output-plot', type=str, default='pytorch_performance.png',
                        help='Output path for performance plot')
    
    args = parser.parse_args()
    
    # Run benchmark
    results = benchmark_pytorch(
        args.model_path,
        args.encoder,
        args.max_depth,
        args.input_size,
        args.test_image,
        args.num_warmup,
        args.num_iterations,
        args.device
    )
    
    # Create visualization
    plot_performance(results, args.output_plot)
