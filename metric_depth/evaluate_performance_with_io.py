# realtime_inference.py
import argparse
import logging
import time
import pprint
import os

import torch
import torch.nn.functional as F
from torchvision.transforms.functional import normalize

# Make sure opencv-python is installed: pip install opencv-python
import cv2
import numpy as np

from depth_anything_v2.dpt import DepthAnythingV2
from util.utils import init_log

# Example Command:
# python realtime_inference.py --encoder vits --img-size 518 --load-from /path/to/your/checkpoint.pth --camera-index 0

parser = argparse.ArgumentParser(
    description="Depth Anything V2 Real-time Inference from Camera"
)

# --- Model Arguments ---
parser.add_argument(
    "--encoder", default="vitl", choices=["vits", "vitb", "vitl", "vitg"]
)
parser.add_argument(
    "--load-from", type=str, required=True, help="Path to the model checkpoint"
)
parser.add_argument(
    "--max-depth",
    default=1.0,
    type=float,  # Adjust based on expected scene scale if needed
    help="Maximum depth value used for relative normalization in the model (might influence scaling)",
)

# --- Input Arguments ---
parser.add_argument(
    "--img-size",
    default=518,
    type=int,
    help="Size to resize camera frames to for model input",
)
parser.add_argument(
    "--camera-index", default=0, type=int, help="Index of the camera device to use"
)
parser.add_argument(
    "--input-height",
    default=480,
    type=int,
    help="Target height for camera capture (if supported)",
)
parser.add_argument(
    "--input-width",
    default=640,
    type=int,
    help="Target width for camera capture (if supported)",
)


# --- Performance Arguments ---
parser.add_argument(
    "--warmup-frames",
    default=10,
    type=int,
    help="Number of warmup frames before starting measurements",
)
parser.add_argument(
    "--display-output",
    action="store_true",
    default=True,
    help="Display the depth map output in a window",
)
parser.add_argument(
    "--no-display-output",
    dest="display_output",
    action="store_false",
    help="Do not display the depth map output",
)


# --- Preprocessing Constants ---
# Normalization parameters (typically ImageNet)
MEAN = torch.tensor([0.485, 0.456, 0.406]).cuda().view(3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).cuda().view(3, 1, 1)


def preprocess_frame(frame, target_size):
    """Converts OpenCV frame to model input tensor."""
    if frame is None:
        return None
    # 1. BGR to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # 2. To Tensor (HWC -> CHW) and Normalize to [0, 1]
    tensor = torch.from_numpy(rgb_frame).permute(2, 0, 1).float() / 255.0
    # 3. Resize
    # Use align_corners=True consistent with many DPT models if needed, else False might be better
    resized_tensor = F.interpolate(
        tensor.unsqueeze(0),
        size=(target_size, target_size),
        mode="bicubic",
        align_corners=False,
    ).squeeze(0)
    # 4. Normalize
    normalized_tensor = normalize(
        resized_tensor, MEAN.cpu(), STD.cpu()
    )  # Normalize on CPU before moving
    # 5. Add Batch Dimension and Move to GPU
    return normalized_tensor.unsqueeze(0).cuda()


def postprocess_depth(pred_depth, original_hw):
    """Converts raw model output to a displayable depth map."""
    # 1. Resize to original frame size
    # Use align_corners=True matching potential use in model, else False
    resized_pred = F.interpolate(
        pred_depth.unsqueeze(0).unsqueeze(0),
        size=original_hw,
        mode="bicubic",
        align_corners=False,
    ).squeeze()
    # 2. Move to CPU and NumPy
    depth_np = resized_pred.cpu().numpy()
    # 3. Normalize for visualization (e.g., 0-255)
    depth_norm = cv2.normalize(depth_np, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    # 4. Apply colormap
    depth_colored = cv2.applyColorMap(
        depth_norm, cv2.COLORMAP_INFERNO
    )  # Or COLORMAP_JET, etc.
    return depth_colored


def main():
    args = parser.parse_args()

    logger = init_log("global", logging.INFO)
    logger.propagate = 0

    all_args = {**vars(args)}
    logger.info(
        "Real-time Inference Configuration:\n{}\n".format(pprint.pformat(all_args))
    )

    # --- Model Loading ---
    model_configs = {
        "vits": {"encoder": "vits", "features": 64, "out_channels": [48, 96, 192, 384]},
        "vitb": {
            "encoder": "vitb",
            "features": 128,
            "out_channels": [96, 192, 384, 768],
        },
        "vitl": {
            "encoder": "vitl",
            "features": 256,
            "out_channels": [256, 512, 1024, 1024],
        },
        "vitg": {
            "encoder": "vitg",
            "features": 384,
            "out_channels": [1536, 1536, 1536, 1536],
        },
    }
    model = DepthAnythingV2(
        **{**model_configs[args.encoder], "max_depth": args.max_depth}
    )
    try:
        model.load_state_dict(
            torch.load(args.load_from, map_location="cpu"), strict=False
        )
        logger.info(f"Model loaded successfully from {args.load_from}")
    except Exception as e:
        logger.error(f"Error loading model from {args.load_from}: {e}")
        return

    model.to("cuda").eval()
    logger.info("Model moved to CUDA and set to evaluation mode.")

    # --- Camera Initialization ---
    cap = cv2.VideoCapture(args.camera_index)
    if not cap.isOpened():
        logger.error(f"Error: Cannot open camera with index {args.camera_index}")
        return

    # Attempt to set camera resolution (may not be supported by all cameras/drivers)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.input_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.input_height)
    actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    logger.info(
        f"Camera opened. Requested resolution: {args.input_width}x{args.input_height}. Actual resolution: {actual_width}x{actual_height}"
    )
    original_hw = (actual_height, actual_width)
    model_input_size = (args.img_size, args.img_size)

    # --- Warmup Phase ---
    logger.info(f"Starting warmup ({args.warmup_frames} frames)...")
    for _ in range(args.warmup_frames):
        ret, frame = cap.read()
        if not ret:
            logger.warning("Warning: Failed to read frame during warmup.")
            continue
        input_tensor = preprocess_frame(frame, args.img_size)
        if input_tensor is None:
            continue
        with torch.no_grad():
            _ = model(input_tensor)
    torch.cuda.synchronize()
    logger.info("Warmup finished.")

    # --- Real-time Inference Loop ---
    logger.info("Starting real-time inference loop (press 'q' in window to quit)...")
    frame_count = 0
    total_end_to_end_time = 0.0
    total_gpu_time_ms = 0.0

    # CUDA events for GPU timing
    gpu_start_event = torch.cuda.Event(enable_timing=True)
    gpu_end_event = torch.cuda.Event(enable_timing=True)

    start_time = time.perf_counter()  # Overall timer for FPS calculation period

    while True:
        frame_start_time = time.perf_counter()

        # 1. Read Frame
        ret, frame = cap.read()
        if not ret:
            logger.error("Error: Failed to read frame from camera. Exiting.")
            break

        # 2. Preprocess Frame
        input_tensor = preprocess_frame(frame, args.img_size)
        if input_tensor is None:
            logger.warning("Warning: Frame preprocessing failed.")
            continue

        # 3. Inference (Timed with CUDA events)
        gpu_start_event.record()
        with torch.no_grad():
            pred_depth = model(input_tensor)
        gpu_end_event.record()
        torch.cuda.synchronize()  # Wait for GPU ops to finish

        # 4. Postprocess for Display
        display_depth = postprocess_depth(pred_depth.squeeze(), original_hw)

        # --- End-to-End Timing ---
        frame_end_time = time.perf_counter()
        current_e2e_time = frame_end_time - frame_start_time
        total_end_to_end_time += current_e2e_time
        current_gpu_time_ms = gpu_start_event.elapsed_time(gpu_end_event)
        total_gpu_time_ms += current_gpu_time_ms
        frame_count += 1

        # 5. Display Output (using cv2.imshow as placeholder for "xLauncher")
        if args.display_output:
            # Combine original frame and depth map for side-by-side view
            combined_output = np.concatenate((frame, display_depth), axis=1)
            cv2.imshow("Depth Anything V2 - Realtime (Input | Output)", combined_output)

            # Exit condition
            if cv2.waitKey(1) & 0xFF == ord("q"):
                logger.info("Exit key 'q' pressed. Stopping inference.")
                break
        else:
            # Add a small delay if not displaying to prevent tight loop hogging CPU
            # Or implement a different termination condition (e.g., run for N frames)
            if frame_count % 100 == 0:  # Print status periodically if not displaying
                logger.info(f"Processed {frame_count} frames...")
            # time.sleep(0.001)

    # --- End Loop ---
    end_time = time.perf_counter()
    total_duration = end_time - start_time

    # --- Results ---
    logger.info("Inference loop stopped.")

    if frame_count > 0:
        avg_e2e_time_sec = total_end_to_end_time / frame_count
        overall_fps = 1.0 / avg_e2e_time_sec if avg_e2e_time_sec > 0 else 0

        avg_gpu_time_ms = total_gpu_time_ms / frame_count
        gpu_only_fps = 1000.0 / avg_gpu_time_ms if avg_gpu_time_ms > 0 else 0

        logger.info("===================== Real-time Performance =====================")
        logger.info(
            f"Model:                {args.encoder} loaded from {args.load_from}"
        )
        logger.info(f"Camera Resolution:    {actual_width}x{actual_height}")
        logger.info(f"Model Input Size:     {args.img_size}x{args.img_size}")
        logger.info(f"Total Test Duration:  {total_duration:.3f} seconds")
        logger.info(f"Frames Processed:     {frame_count}")
        logger.info(
            f"Avg GPU Time/Frame:   {avg_gpu_time_ms:.3f} ms ({gpu_only_fps:.2f} FPS GPU-only)"
        )
        logger.info(
            f"Avg Total Time/Frame: {avg_e2e_time_sec*1000:.3f} ms ({overall_fps:.2f} FPS Overall)"
        )
        logger.info("=================================================================")
    else:
        logger.warning("No frames were processed.")

    # --- Cleanup ---
    cap.release()
    if args.display_output:
        cv2.destroyAllWindows()
    logger.info("Camera released and windows closed.")


if __name__ == "__main__":
    main()
