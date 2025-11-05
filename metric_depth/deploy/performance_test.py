import time
import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
import onnxruntime as ort
from depth_anything_v2.dpt import DepthAnythingV2


def test_pytorch_performance(model_path, encoder, max_depth, input_size, image_path):
    # Initialize PyTorch model
    model_configs = {
        'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
        'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
        'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
        'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
    }

    model = DepthAnythingV2(**{**model_configs[encoder], 'max_depth': max_depth})
    checkpoint = torch.load(model_path, map_location='cpu')
    state_dict = checkpoint.get("model", checkpoint)
    model.load_state_dict(state_dict)
    model.eval()

    # Preprocess image
    raw_image = cv2.imread(image_path)
    image, _ = model.image2tensor(raw_image, input_size)

    # Warm up
    for _ in range(10):
        with torch.no_grad():
            _ = model.forward(image)

    # Measure inference time
    torch.cuda.reset_max_memory_allocated()
    torch.cuda.synchronize()
    start_time = time.time()
    with torch.no_grad():
        for _ in range(100):
            _ = model.forward(image)
    torch.cuda.synchronize()
    end_time = time.time()
    elapsed_time = end_time - start_time
    fps = 100 / elapsed_time
    memory_usage = torch.cuda.max_memory_allocated() / 1024**2

    return fps, memory_usage


def test_onnx_performance(onnx_model_path, input_size, image_path):
    # Create ONNX runtime session
    ort_session = ort.InferenceSession(onnx_model_path)
    input_name = ort_session.get_inputs()[0].name

    # Preprocess image
    raw_image = cv2.imread(image_path)
    image = cv2.cvtColor(raw_image, cv2.COLOR_BGR2RGB) / 255.0
    image_resized = cv2.resize(image, (input_size, input_size), interpolation=cv2.INTER_CUBIC)
    image_normalized = (image_resized - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
    image_tensor = np.transpose(image_normalized, (2, 0, 1)).astype(np.float32)
    image_tensor = np.expand_dims(image_tensor, axis=0)

    # Warm up
    for _ in range(10):
        ort_session.run(None, {input_name: image_tensor})

    # Measure inference time
    start_time = time.time()
    for _ in range(100):
        ort_session.run(None, {input_name: image_tensor})
    end_time = time.time()
    elapsed_time = end_time - start_time
    fps = 100 / elapsed_time

    return fps


def plot_performance(fps_pytorch, memory_pytorch, fps_onnx, output_path):
    fig, ax1 = plt.subplots()

    ax1.set_xlabel('Model')
    ax1.set_ylabel('Frames per Second (FPS)', color='tab:blue')
    ax1.bar(['PyTorch', 'ONNX'], [fps_pytorch, fps_onnx], color='tab:blue', alpha=0.6)
    ax1.tick_params(axis='y', labelcolor='tab:blue')

    ax2 = ax1.twinx()
    ax2.set_ylabel('Memory Usage (MB)', color='tab:red')
    ax2.bar(['PyTorch'], [memory_pytorch], color='tab:red', alpha=0.6)
    ax2.tick_params(axis='y', labelcolor='tab:red')

    fig.tight_layout()
    plt.title('Model Performance')
    plt.savefig(output_path)
    plt.close()


if __name__ == "__main__":
    model_path = "<path_to_model_checkpoint>"
    encoder = "vits"
    max_depth = 0.2
    input_size = 518
    image_path = "<path_to_test_image>"
    onnx_model_path = "depth_anything_v2_vits_518.onnx"
    
    fps_pytorch, memory_pytorch = test_pytorch_performance(model_path, encoder, max_depth, input_size, image_path)
    fps_onnx = test_onnx_performance(onnx_model_path, input_size, image_path)
    
    plot_performance(fps_pytorch, memory_pytorch, fps_onnx, "performance_comparison.png")

    print(f"PyTorch Model - FPS: {fps_pytorch:.2f}, Memory Usage: {memory_pytorch:.2f} MB")
    print(f"ONNX Model - FPS: {fps_onnx:.2f}")

