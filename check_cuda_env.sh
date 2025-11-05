#!/bin/bash

echo "=========================================="
echo "CUDA Environment Check Script"
echo "=========================================="
echo ""

# 检查系统信息
echo "=== System Information ==="
echo "Hostname: $(hostname)"
echo "OS: $(cat /etc/os-release | grep PRETTY_NAME | cut -d '"' -f2)"
echo "Kernel: $(uname -r)"
echo ""

# 检查NVIDIA驱动
echo "=== NVIDIA Driver ==="
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
    echo ""
else
    echo "nvidia-smi not found. NVIDIA driver may not be installed."
    echo ""
fi

# 检查CUDA版本
echo "=== CUDA Version ==="
if command -v nvcc &> /dev/null; then
    nvcc --version | grep "release"
    echo "CUDA Path: $(which nvcc)"
else
    echo "nvcc not found. CUDA toolkit may not be installed."
fi

# 检查其他可能的CUDA路径
echo ""
echo "Checking common CUDA paths:"
for cuda_path in /usr/local/cuda* /opt/cuda*; do
    if [ -d "$cuda_path" ]; then
        echo "Found: $cuda_path"
        if [ -f "$cuda_path/version.txt" ]; then
            echo "  Version: $(cat $cuda_path/version.txt)"
        elif [ -f "$cuda_path/version.json" ]; then
            echo "  Version: $(cat $cuda_path/version.json | grep version | head -1)"
        fi
    fi
done
echo ""

# 检查cuDNN
echo "=== cuDNN Version ==="
# 检查常见的cuDNN位置
cudnn_found=false
for lib_path in /usr/local/cuda*/lib64 /usr/lib/x86_64-linux-gnu /usr/local/lib; do
    if [ -f "$lib_path/libcudnn.so" ]; then
        cudnn_found=true
        echo "Found cuDNN at: $lib_path"
        # 尝试获取版本信息
        if [ -f "$lib_path/../include/cudnn_version.h" ]; then
            grep -E "CUDNN_MAJOR|CUDNN_MINOR|CUDNN_PATCHLEVEL" "$lib_path/../include/cudnn_version.h" | head -3
        elif [ -f "$lib_path/../include/cudnn.h" ]; then
            grep -E "CUDNN_MAJOR|CUDNN_MINOR|CUDNN_PATCHLEVEL" "$lib_path/../include/cudnn.h" | head -3
        fi
        # 检查符号链接
        ls -la "$lib_path"/libcudnn.so* 2>/dev/null | grep -E "libcudnn.so.[0-9]+" | head -1
    fi
done

if [ "$cudnn_found" = false ]; then
    echo "cuDNN library not found in common locations."
fi
echo ""

# 检查环境变量
echo "=== Environment Variables ==="
echo "PATH: $PATH" | grep -o "[^:]*cuda[^:]*" | sort -u || echo "No CUDA paths in PATH"
echo ""
echo "LD_LIBRARY_PATH: $LD_LIBRARY_PATH" | grep -o "[^:]*cuda[^:]*" | sort -u || echo "No CUDA paths in LD_LIBRARY_PATH"
echo ""

# 检查Python环境中的CUDA相关包
echo "=== Python CUDA Packages ==="
if command -v python &> /dev/null || command -v python3 &> /dev/null; then
    PYTHON_CMD=$(command -v python3 || command -v python)
    echo "Using Python: $PYTHON_CMD"
    $PYTHON_CMD -c "
import sys
print(f'Python version: {sys.version.split()[0]}')

packages = ['torch', 'tensorflow', 'onnxruntime', 'onnxruntime-gpu']
for pkg in packages:
    try:
        module = __import__(pkg)
        print(f'{pkg}: {module.__version__}', end='')
        if pkg == 'torch':
            import torch
            print(f' (CUDA available: {torch.cuda.is_available()})', end='')
            if torch.cuda.is_available():
                print(f' (CUDA version: {torch.version.cuda})', end='')
        elif pkg == 'tensorflow':
            import tensorflow as tf
            print(f' (GPU devices: {len(tf.config.list_physical_devices(\"GPU\"))})', end='')
        print()
    except ImportError:
        print(f'{pkg}: not installed')
" 2>/dev/null || echo "Failed to check Python packages"
else
    echo "Python not found"
fi
echo ""

# 检查GPU使用情况
echo "=== GPU Utilization ==="
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total --format=csv
else
    echo "nvidia-smi not available"
fi
echo ""

# 检查ONNX Runtime特定要求
echo "=== ONNX Runtime Requirements Check ==="
echo "ONNX Runtime GPU requires:"
echo "- CUDA 12.x (you have: $(nvcc --version 2>/dev/null | grep release | awk '{print $5}' | cut -d',' -f1 || echo 'not found'))"
echo "- cuDNN 9.x (you have: check cuDNN section above)"
echo ""

# 提供建议
echo "=== Recommendations ==="
cuda_version=$(nvcc --version 2>/dev/null | grep release | awk '{print $5}' | cut -d',' -f1 | cut -d'.' -f1)
if [ -z "$cuda_version" ]; then
    echo "1. Install CUDA 12.x from: https://developer.nvidia.com/cuda-downloads"
elif [ "$cuda_version" -lt "12" ]; then
    echo "1. Upgrade to CUDA 12.x from: https://developer.nvidia.com/cuda-downloads"
else
    echo "1. CUDA version appears compatible"
fi

if [ "$cudnn_found" = false ]; then
    echo "2. Install cuDNN 9.x from: https://developer.nvidia.com/cudnn"
else
    echo "2. Verify cuDNN version is 9.x (check version info above)"
fi

echo "3. Ensure CUDA paths are in PATH and LD_LIBRARY_PATH"
echo "4. For ONNX Runtime GPU, install: pip install onnxruntime-gpu"
echo ""
echo "=========================================="
