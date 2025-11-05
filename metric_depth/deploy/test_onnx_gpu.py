import onnxruntime as ort
import os

print("onnxruntime version:", ort.__version__)
print("Available providers:", ort.get_available_providers())
print("LD_LIBRARY_PATH:", os.environ.get('LD_LIBRARY_PATH', 'Not set'))

# 测试创建 session
try:
    providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
    print("\nTrying to create session with providers:", providers)
    # 使用一个简单的 ONNX 模型路径
    session = ort.InferenceSession("/media/ExtHDD1/jianfu/data/onnx/da2/depth_anything_v2_vits_518.onnx", providers=providers)
    print("Success! Active providers:", session.get_providers())
except Exception as e:
    print("Error:", e)
