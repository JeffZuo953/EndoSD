# evaluate_performance.py
import argparse
import logging
import time
import pprint

import torch
import torch.nn.functional as F
from torchvision.transforms.functional import normalize, resize, to_pil_image

from depth_anything_v2.dpt import DepthAnythingV2
from util.utils import init_log
from PIL import Image
import numpy as np

# python evaluate_performance.py --encoder vits --img-size 518 --load-from /data/depthanything/depth_anything_v2_metric_vkitti_vits.pth --test-duration 30

parser = argparse.ArgumentParser(
    description="Depth Anything V2 Metric Depth Performance Evaluation"
)

parser.add_argument(
    "--encoder", default="vitl", choices=["vits", "vitb", "vitl", "vitg"]
)
parser.add_argument("--img-size", default=518, type=int)
parser.add_argument(
    "--load-from", type=str, required=True, help="Path to the model checkpoint"
)
parser.add_argument(
    "--max-depth", default=20, type=float
)  # Although not directly used in inference speed, keep for model loading
parser.add_argument(
    "--test-duration",
    default=30,
    type=int,
    help="Duration in seconds to run the performance test",
)
parser.add_argument(
    "--warmup-iters",
    default=10,
    type=int,
    help="Number of warmup iterations before timing",
)
# Add an argument for a sample image if needed, otherwise use a dummy tensor
# parser.add_argument("--sample-image", type=str, help="Path to a sample image for testing")


def main():
    args = parser.parse_args()

    logger = init_log("global", logging.INFO)
    logger.propagate = 0

    all_args = {**vars(args)}
    logger.info(
        "Performance Test Configuration:\n{}\n".format(pprint.pformat(all_args))
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
        **{
            **model_configs[args.encoder],
            "max_depth": args.max_depth,
        }  # max_depth might influence internal model scaling, keep it
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
    logger.info(f"Model moved to CUDA and set to evaluation mode.")

    # --- Prepare Sample Input ---
    # Using a dummy tensor for consistent input without I/O bottleneck
    # You could replace this with loading a real image if needed, but do it outside the loop
    img_size = (args.img_size, args.img_size)
    dummy_input = torch.randn(1, 3, img_size[0], img_size[1]).cuda().float()
    logger.info(f"Using dummy input tensor of size {dummy_input.shape}")

    # --- Warmup Phase ---
    logger.info(f"Starting warmup ({args.warmup_iters} iterations)...")
    for _ in range(args.warmup_iters):
        with torch.no_grad():
            _ = model(dummy_input)
            # Optional: include interpolation if it's part of the core pipeline speed measurement
            # pred = F.interpolate(
            #     pred[:, None], img_size, mode="bilinear", align_corners=True
            # )[0, 0]
    torch.cuda.synchronize()  # Wait for warmup to complete
    logger.info("Warmup finished.")

    # --- Performance Measurement ---
    logger.info(f"Starting performance test for {args.test_duration} seconds...")
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    frame_count = 0
    total_time_ms = 0.0
    test_start_time = time.time()

    while time.time() - test_start_time < args.test_duration:
        start_event.record()
        with torch.no_grad():
            # --- Core Inference Step ---
            pred = model(dummy_input)
            # --- Minimal Post-processing (Optional) ---
            # Keep this minimal or exclude if measuring *pure* model speed
            # pred = F.interpolate(
            #      pred[:, None], img_size, mode="bilinear", align_corners=True
            #  )[0, 0]
            # -----------------------------

        end_event.record()
        torch.cuda.synchronize()  # IMPORTANT: Wait for the GPU operation to finish before getting time

        # --- Timing ---
        # Output here is 'asynchronous' in the sense it's not happening
        # We are only timing the inference + sync
        current_time_ms = start_event.elapsed_time(end_event)
        total_time_ms += current_time_ms
        frame_count += 1

        # Optional: Print progress
        # if frame_count % 10 == 0:
        #     elapsed_test_time = time.time() - test_start_time
        #     current_fps = frame_count / elapsed_test_time
        #     print(f"  Processed {frame_count} frames in {elapsed_test_time:.2f}s ({current_fps:.2f} FPS)", end='\r')

    # --- Results ---
    logger.info("Performance test finished.")
    print()  # New line after potential progress indicator

    if frame_count > 0:
        average_time_ms = total_time_ms / frame_count
        average_fps = (
            1000.0 / average_time_ms
        )  # Convert ms per frame to frames per second
        total_duration_sec = total_time_ms / 1000.0

        logger.info("==================== Performance Results ====================")
        logger.info(f"Model:          {args.encoder} loaded from {args.load_from}")
        logger.info(f"Input Size:     {img_size[0]}x{img_size[1]}")
        logger.info(
            f"Test Duration:  ~{args.test_duration} seconds (actual GPU time: {total_duration_sec:.3f} s)"
        )
        logger.info(f"Warmup Iters:   {args.warmup_iters}")
        logger.info(f"Frames Processed: {frame_count}")
        logger.info(f"Average Time/Frame: {average_time_ms:.3f} ms")
        logger.info(f"Average FPS:        {average_fps:.2f}")
        logger.info("=============================================================")
    else:
        logger.warning("No frames were processed during the test duration.")


if __name__ == "__main__":
    main()
