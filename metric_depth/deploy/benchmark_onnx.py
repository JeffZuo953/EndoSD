import argparse
import cv2
import time
import numpy as np
import matplotlib.pyplot as plt
import onnxruntime as ort
import psutil
import os


def preprocess_image(image, input_size=518):
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


def benchmark_onnx(onnx_path, test_image_path, num_warmup=10, num_iterations=100, 
                   providers=None):
    """Benchmark ONNX model performance"""
    
    print("=" * 60)
    print("ONNX Model Benchmark")
    print("=" * 60)
    
    # Set providers
    if providers is None:
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
    
    # Load ONNX model
    print(f"Loading ONNX model from {onnx_path}...")
    try:
        ort_session = ort.InferenceSession(onnx_path, providers=providers)
        print(f"Active provider: {ort_session.get_providers()[0]}")
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Falling back to CPU provider only")
        ort_session = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])
        print(f"Active provider: {ort_session.get_providers()[0]}")
    
    # Get model info
    input_name = ort_session.get_inputs()[0].name
    input_shape = ort_session.get_inputs()[0].shape
    output_name = ort_session.get_outputs()[0].name
    
    # Extract input size (assuming square input)
    input_size = input_shape[2] if len(input_shape) >= 3 else 518
    
    print(f"Input: {input_name}, shape: {input_shape}")
    print(f"Output: {output_name}")
    
    # Load and preprocess test image
    print(f"\nLoading test image: {test_image_path}")
    raw_image = cv2.imread(test_image_path)
    if raw_image is None:
        raise ValueError(f"Could not load image: {test_image_path}")
    
    print(f"Original image size: {raw_image.shape[1]}x{raw_image.shape[0]}")
    
    # Preprocess image
    image_tensor, (h_orig, w_orig) = preprocess_image(raw_image, input_size)
    print(f"Input tensor shape: {image_tensor.shape}")
    
    # Get initial memory usage
    initial_memory = psutil.Process().memory_info().rss / 1024 / 1024
    
    # Warmup
    print(f"\nWarming up with {num_warmup} iterations...")
    for _ in range(num_warmup):
        _ = ort_session.run(None, {input_name: image_tensor})
    
    # Benchmark
    print(f"Running benchmark with {num_iterations} iterations...")
    
    inference_times = []
    
    for i in range(num_iterations):
        start_time = time.perf_counter()
        outputs = ort_session.run(None, {input_name: image_tensor})
        end_time = time.perf_counter()
        
        inference_times.append((end_time - start_time) * 1000)  # Convert to ms
    
    # Get output info
    output = outputs[0]
    
    # Get peak memory usage
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
    print("BENCHMARK RESULTS - ONNX")
    print("=" * 60)
    print(f"Provider: {ort_session.get_providers()[0]}")
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
        'inference_times': inference_times,
        'provider': ort_session.get_providers()[0]
    }


def plot_performance(results, output_path='onnx_performance.png'):
    """Create performance visualization"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Inference time distribution
    ax1.hist(results['inference_times'], bins=30, alpha=0.7, color='green', edgecolor='black')
    ax1.axvline(results['mean_time'], color='red', linestyle='--', 
                label=f'Mean: {results["mean_time"]:.2f} ms')
    ax1.set_xlabel('Inference Time (ms)')
    ax1.set_ylabel('Frequency')
    ax1.set_title('ONNX Inference Time Distribution')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Performance metrics
    metrics = ['FPS', 'Memory (MB)']
    values = [results['fps'], results['memory_used']]
    colors = ['blue', 'orange']
    
    bars = ax2.bar(metrics, values, color=colors, alpha=0.7)
    ax2.set_ylabel('Value')
    ax2.set_title('ONNX Performance Metrics')
    
    # Add value labels on bars
    for bar, value in zip(bars, values):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{value:.1f}', ha='center', va='bottom')
    
    ax2.grid(True, alpha=0.3, axis='y')
    
    plt.suptitle(f'ONNX Model Performance Analysis ({results["provider"]})\n'
                 f'Mean: {results["mean_time"]:.2f}±{results["std_time"]:.2f} ms, '
                 f'FPS: {results["fps"]:.2f}')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nPerformance plot saved to: {output_path}")
    
    # Close the plot to free memory
    plt.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Benchmark ONNX DepthAnythingV2 Model')
    
    parser.add_argument('--onnx-path', type=str, required=True,
                        help='Path to the ONNX model')
    parser.add_argument('--test-image', type=str, required=True,
                        help='Path to test image')
    parser.add_argument('--num-warmup', type=int, default=10,
                        help='Number of warmup iterations')
    parser.add_argument('--num-iterations', type=int, default=100,
                        help='Number of benchmark iterations')
    parser.add_argument('--provider', type=str, default='auto',
                        choices=['auto', 'cuda', 'cpu'],
                        help='Execution provider (auto will try CUDA first)')
    parser.add_argument('--output-plot', type=str, default='onnx_performance.png',
                        help='Output path for performance plot')
    
    args = parser.parse_args()
    
    # Set providers based on user choice
    if args.provider == 'cuda':
        providers = ['CUDAExecutionProvider']
    elif args.provider == 'cpu':
        providers = ['CPUExecutionProvider']
    else:
        providers = None  # Use default
    
    # Run benchmark
    results = benchmark_onnx(
        args.onnx_path,
        args.test_image,
        args.num_warmup,
        args.num_iterations,
        providers
    )
    
    # Create visualization
    plot_performance(results, args.output_plot)
