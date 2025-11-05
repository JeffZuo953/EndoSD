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


def benchmark_pytorch_pure(model_path, encoder, max_depth, input_size, test_image_path, 
                          num_warmup=10, num_iterations=100, device='cuda'):
    """Benchmark PyTorch model performance with minimal overhead"""
    
    print("=" * 60)
    print("PyTorch Model Pure Inference Benchmark")
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
    
    # Load and preprocess test image ONCE
    print(f"\nLoading test image: {test_image_path}")
    raw_image = cv2.imread(test_image_path)
    if raw_image is None:
        raise ValueError(f"Could not load image: {test_image_path}")
    
    print(f"Original image size: {raw_image.shape[1]}x{raw_image.shape[0]}")
    
    # Preprocess image
    with torch.no_grad():
        image_tensor, (h, w) = model.image2tensor(raw_image, input_size)
        if device == 'cuda':
            image_tensor = image_tensor.cuda()
    
    print(f"Input tensor shape: {image_tensor.shape}")
    print(f"Input tensor device: {image_tensor.device}")
    
    # Warmup
    print(f"\nWarming up with {num_warmup} iterations...")
    with torch.no_grad():
        for _ in range(num_warmup):
            _ = model(image_tensor)
            if device == 'cuda':
                torch.cuda.synchronize()
    
    # Clear cache before benchmark
    if device == 'cuda':
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    
    print(f"Running pure inference benchmark with {num_iterations} iterations...")
    
    # Method 1: Standard timing (with overhead)
    standard_times = []
    
    # Method 2: CUDA Events (GPU only, minimal overhead)
    cuda_event_times = []
    
    # Method 3: Batch timing (amortized overhead)
    batch_time = 0
    
    if device == 'cuda':
        # Method 2: CUDA Events timing
        starter = torch.cuda.Event(enable_timing=True)
        ender = torch.cuda.Event(enable_timing=True)
        
        with torch.no_grad():
            for i in range(num_iterations):
                # Standard timing
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                output = model(image_tensor)
                torch.cuda.synchronize()
                t1 = time.perf_counter()
                standard_times.append((t1 - t0) * 1000)
                
                # CUDA event timing (separate pass to avoid interference)
                starter.record()
                output = model(image_tensor)
                ender.record()
                torch.cuda.synchronize()
                cuda_event_times.append(starter.elapsed_time(ender))
        
        # Method 3: Batch timing
        torch.cuda.synchronize()
        batch_start = time.perf_counter()
        with torch.no_grad():
            for _ in range(num_iterations):
                output = model(image_tensor)
        torch.cuda.synchronize()
        batch_end = time.perf_counter()
        batch_time = ((batch_end - batch_start) * 1000) / num_iterations
        
    else:
        # CPU timing
        with torch.no_grad():
            for i in range(num_iterations):
                t0 = time.perf_counter()
                output = model(image_tensor)
                t1 = time.perf_counter()
                standard_times.append((t1 - t0) * 1000)
        
        # Batch timing for CPU
        batch_start = time.perf_counter()
        with torch.no_grad():
            for _ in range(num_iterations):
                output = model(image_tensor)
        batch_end = time.perf_counter()
        batch_time = ((batch_end - batch_start) * 1000) / num_iterations
    
    # Measure overhead
    print("\nMeasuring timing overhead...")
    overhead_times = []
    dummy_tensor = torch.randn(1, 1, device=device)
    
    for _ in range(100):
        if device == 'cuda':
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            torch.cuda.synchronize()
            t1 = time.perf_counter()
        else:
            t0 = time.perf_counter()
            t1 = time.perf_counter()
        overhead_times.append((t1 - t0) * 1000)
    
    # Memory usage
    if device == 'cuda':
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        
        with torch.no_grad():
            _ = model(image_tensor)
            torch.cuda.synchronize()
        
        memory_used = torch.cuda.max_memory_allocated() / 1024 / 1024
    else:
        memory_used = psutil.Process().memory_info().rss / 1024 / 1024
    
    # Calculate statistics
    standard_times = np.array(standard_times)
    overhead_times = np.array(overhead_times)
    
    # Primary results (CUDA events if available, otherwise standard)
    if device == 'cuda' and cuda_event_times:
        primary_times = np.array(cuda_event_times)
        method_name = "CUDA Events"
    else:
        primary_times = standard_times
        method_name = "perf_counter"
    
    mean_time = np.mean(primary_times)
    std_time = np.std(primary_times)
    min_time = np.min(primary_times)
    max_time = np.max(primary_times)
    median_time = np.median(primary_times)
    p95_time = np.percentile(primary_times, 95)
    p99_time = np.percentile(primary_times, 99)
    fps = 1000.0 / mean_time
    
    mean_overhead = np.mean(overhead_times)
    
    # Print results
    print("\n" + "=" * 60)
    print("PURE INFERENCE BENCHMARK RESULTS - PyTorch")
    print("=" * 60)
    print(f"Model: {encoder}")
    print(f"Device: {device}")
    print(f"Input size: {input_size}x{input_size}")
    print(f"Output shape: {output.shape}")
    print(f"\nTiming Method Comparison:")
    print(f"  Primary method: {method_name}")
    print(f"  Standard timing mean: {np.mean(standard_times):.3f} ms")
    if device == 'cuda' and cuda_event_times:
        print(f"  CUDA Events mean: {np.mean(cuda_event_times):.3f} ms")
    print(f"  Batch timing mean: {batch_time:.3f} ms")
    print(f"\nPrimary Results ({method_name}):")
    print(f"  Mean inference time: {mean_time:.3f} ms")
    print(f"  Std deviation: {std_time:.3f} ms")
    print(f"  Min inference time: {min_time:.3f} ms")
    print(f"  Max inference time: {max_time:.3f} ms")
    print(f"  Median inference time: {median_time:.3f} ms")
    print(f"  95th percentile: {p95_time:.3f} ms")
    print(f"  99th percentile: {p99_time:.3f} ms")
    print(f"  FPS: {fps:.2f}")
    print(f"\nOverhead Analysis:")
    print(f"  Mean measurement overhead: {mean_overhead:.6f} ms")
    print(f"  Estimated actual inference: {mean_time - mean_overhead:.3f} ms")
    print(f"\nMemory Usage:")
    print(f"  Peak memory used: {memory_used:.2f} MB")
    print("=" * 60)
    
    return {
        'mean_time': mean_time,
        'std_time': std_time,
        'min_time': min_time,
        'max_time': max_time,
        'median_time': median_time,
        'p95_time': p95_time,
        'p99_time': p99_time,
        'fps': fps,
        'memory_used': memory_used,
        'inference_times': primary_times,
        'standard_times': standard_times,
        'cuda_event_times': cuda_event_times if device == 'cuda' else None,
        'batch_time': batch_time,
        'overhead_times': overhead_times,
        'method_name': method_name
    }


def plot_performance_comparison(results, output_path='pytorch_pure_performance.png'):
    """Create performance visualization with method comparison"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. Inference time distribution
    ax1.hist(results['inference_times'], bins=50, alpha=0.7, color='blue', 
             edgecolor='black', density=True, label=results['method_name'])
    if results['cuda_event_times'] is not None:
        ax1.hist(results['standard_times'], bins=50, alpha=0.5, color='red', 
                 edgecolor='black', density=True, label='Standard Timing')
    ax1.axvline(results['mean_time'], color='red', linestyle='--', linewidth=2)
    ax1.axvline(results['median_time'], color='green', linestyle='--', linewidth=2)
    ax1.set_xlabel('Inference Time (ms)')
    ax1.set_ylabel('Density')
    ax1.set_title('Inference Time Distribution')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Time series
    ax2.plot(results['inference_times'][:200], alpha=0.7, linewidth=1, label=results['method_name'])
    ax2.axhline(results['mean_time'], color='red', linestyle='--', alpha=0.5)
    ax2.axhline(results['batch_time'], color='green', linestyle='--', alpha=0.5, label='Batch avg')
    ax2.set_xlabel('Iteration')
    ax2.set_ylabel('Inference Time (ms)')
    ax2.set_title('Inference Time Stability')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Method comparison
    if results['cuda_event_times'] is not None:
        methods = ['CUDA\nEvents', 'Standard\nTiming', 'Batch\nTiming']
        means = [np.mean(results['cuda_event_times']), 
                 np.mean(results['standard_times']), 
                 results['batch_time']]
    else:
        methods = ['Standard\nTiming', 'Batch\nTiming']
        means = [np.mean(results['standard_times']), results['batch_time']]
    
    bars = ax3.bar(methods, means, color=['blue', 'red', 'green'][:len(methods)], alpha=0.7)
    ax3.set_ylabel('Mean Time (ms)')
    ax3.set_title('Timing Method Comparison')
    for bar, mean in zip(bars, means):
        ax3.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                f'{mean:.3f}', ha='center', va='bottom')
    ax3.grid(True, alpha=0.3, axis='y')
    
    # 4. Performance metrics
    metrics = ['FPS', 'Memory\n(MB)', 'Mean\n(ms)', 'Std\n(ms)', 'P95\n(ms)']
    values = [results['fps'], results['memory_used'], results['mean_time'], 
              results['std_time'], results['p95_time']]
    colors = ['green', 'orange', 'blue', 'red', 'purple']
    
    bars = ax4.bar(metrics, values, color=colors, alpha=0.7)
    ax4.set_ylabel('Value')
    ax4.set_title('Performance Metrics Summary')
    for bar, value in zip(bars, values):
        ax4.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                f'{value:.1f}', ha='center', va='bottom')
    ax4.grid(True, alpha=0.3, axis='y')
    
    plt.suptitle(f'PyTorch Pure Inference Benchmark ({results["method_name"]})\n'
                 f'Mean: {results["mean_time"]:.3f}±{results["std_time"]:.3f} ms, '
                 f'FPS: {results["fps"]:.2f}')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nPerformance plot saved to: {output_path}")
    plt.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Pure PyTorch DepthAnythingV2 Inference Benchmark')
    
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
    parser.add_argument('--num-warmup', type=int, default=20,
                        help='Number of warmup iterations')
    parser.add_argument('--num-iterations', type=int, default=200,
                        help='Number of benchmark iterations')
    parser.add_argument('--device', type=str, default='cuda',
                        choices=['cuda', 'cpu'],
                        help='Device to run benchmark on')
    parser.add_argument('--output-plot', type=str, default='pytorch_pure_performance.png',
                        help='Output path for performance plot')
    
    args = parser.parse_args()
    
    # Run benchmark
    results = benchmark_pytorch_pure(
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
    plot_performance_comparison(results, args.output_plot)
