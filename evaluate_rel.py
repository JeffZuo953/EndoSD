import argparse
import logging
import os
import pprint
import cv2  # Import time module

import torch
from torch.utils.data import DataLoader, ConcatDataset  # Keep ConcatDataset import for consistency, although not used in this file
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms.functional
from tqdm import tqdm  # Added tqdm
import numpy as np
from collections import OrderedDict


def strip_module_prefix(state_dict):
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        new_key = k.replace("module.", "")
        new_state_dict[new_key] = v
    return new_state_dict


def normalize_least_squares(pred, depth):
    # 确保张量在同一个设备上并且是浮点类型
    pred = pred.float()
    depth = depth.float()
    a = torch.median(depth) / torch.median(pred)
    b = 0
    normalized_pred = a * pred + b

    return normalized_pred, a, b


from depth_anything_v2.dpt import DepthAnythingV2
from metric_depth.util.metric import eval_depth
from metric_depth.util.utils import init_log
# from metric_depth.dataset.c3vd import C3VD
from metric_depth.dataset.simcol import Simcol
# from dataset.endomapper import Endomapper
from metric_depth.dataset.inhouse import InHouse

parser = argparse.ArgumentParser(
    description="Depth Anything V2 for Metric Depth Estimation")

parser.add_argument("--encoder",
                    default="vitl",
                    choices=["vits", "vitb", "vitl", "vitg"])
parser.add_argument("--save-path", type=str, required=True)
parser.add_argument("--load-from",
                    type=str,
                    required=True,
                    help="Path to the model checkpoint")
parser.add_argument("--bs", default=20, type=int)  # 40, 95, 175
parser.add_argument("--normalize", action="store_true")


def main():
    args = parser.parse_args()

    logger = init_log("global", logging.INFO)
    logger.propagate = 0

    all_args = {**vars(args)}
    logger.info("{}\n".format(pprint.pformat(all_args)))
    writer = SummaryWriter(args.save_path)
    valset = ConcatDataset([
        # C3VD(
        #     "/root/c3vd/test_mapping.txt",
        #     # 'dataset/splits/c3vd/val.txt',
        #     "train",
        #     size=size,
        # ),
        # Simcol("/data/simcol/val_paths.txt", "val", size=size, max_depth=1),
        # Endomapper(
        #     "dataset/splits/endomapper/train.txt",
        #     "train",
        #     size=size,
        # ),
        InHouse("/data/inhouse/val.txt", "val", max_depth=50, size=(960, 540))
    ])
    valloader = DataLoader(
        valset,
        batch_size=args.bs,  # vitl
        # batch_size=95, # vitb
        # batch_size=175, # vits
        pin_memory=True,
        num_workers=16)
    total_samples = len(valset)  # Get total number of samples

    model_configs = {
        "vits": {
            "encoder": "vits",
            "features": 64,
            "out_channels": [48, 96, 192, 384]
        },
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
    model = DepthAnythingV2(**{**model_configs[args.encoder]})

    loaded_state_dict = torch.load(args.load_from, map_location="cpu")
    if isinstance(loaded_state_dict, dict) and "model" in loaded_state_dict:
        model.load_state_dict(strip_module_prefix(loaded_state_dict["model"]))
    else:
        model.load_state_dict(strip_module_prefix(loaded_state_dict))

    # model.load_state_dict(torch.load(args.load_from, map_location="cpu"), strict=False) # Removed duplicate load
    model.to("cuda").eval()

    # Initialize CUDA events for timing
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    results = {
        "d1": torch.tensor([0.0]).cuda(),
        "d2": torch.tensor([0.0]).cuda(),
        "d3": torch.tensor([0.0]).cuda(),
        "abs_rel": torch.tensor([0.0]).cuda(),
        "sq_rel": torch.tensor([0.0]).cuda(),
        "rmse": torch.tensor([0.0]).cuda(),
        "rmse_log": torch.tensor([0.0]).cuda(),
        "log10": torch.tensor([0.0]).cuda(),
        "silog": torch.tensor([0.0]).cuda(),
    }
    total_evaluated_samples = torch.tensor([0.0]).cuda()

    inference_times = []  # Initialize list to store inference times
    ratios = []
    for i, sample in enumerate(tqdm(valloader)):  # Wrapped with tqdm
        img = sample["image"].cuda().float()
        depth_batch = sample["depth"].cuda()
        valid_mask_batch = sample["valid_mask"].cuda()

        with torch.no_grad():
            start_event.record()  # Record start event
            pred_batch = model(img)
            end_event.record()  # Record end event

        # Wait for the events to complete
        torch.cuda.synchronize()

        # Calculate inference time in milliseconds
        inference_time_ms = start_event.elapsed_time(end_event)
        inference_times.append(inference_time_ms)  # Append to list

        # Evaluate each sample in the batch
        batch_size = img.shape[0]
        for b in range(batch_size):
            depth = depth_batch[b]
            valid_mask = valid_mask_batch[b]
            pred_sample = pred_batch[b]
            # Get prediction for single sample (shape: C, H, W)

            # Interpolate single sample prediction while keeping channel dimension
            interpolated_pred = F.interpolate(
                pred_sample.unsqueeze(0).unsqueeze(0),
                # Add batch dim: 1, C, H, W
                depth.shape[-2:],
                # Target size: H_target, W_target
                mode="bilinear",
                align_corners=False)  # Output shape: 1, C, H_target, W_target

            # Select the first channel and remove batch dimension
            pred = interpolated_pred.squeeze(0).squeeze(0)
            # Output shape: H_target, W_target

            # Apply inverse depth for relative evaluation
            pred = 1 / pred
            pred = torch.clamp(pred, min=1e-3, max=1000)

            if valid_mask.sum() < 10:
                continue

            pred_valid = pred[valid_mask]
            depth_valid = depth[valid_mask]

            if args.normalize:
                pred_valid, param_a, param_b = normalize_least_squares(
                    pred_valid, depth_valid)
                ratios.append(param_a)
            cur_results = eval_depth(pred_valid, depth_valid)

            for k in results.keys():
                results[k] += cur_results[k]

            total_evaluated_samples += 1
            origin = pred.cpu().numpy()
            filename = sample["image_path"][0].split("/")[-3:]
            output_dir1 = os.path.join(args.save_path, "output")
            os.makedirs(output_dir1, exist_ok=True)
            output_dir2 = os.path.join(output_dir1, filename[-3])
            os.makedirs(output_dir2, exist_ok=True)
            output_path = os.path.join(output_dir2, filename[-1])
            np.savez_compressed(output_path.replace(".png", ".npz"),
                                depth=origin)
            # Add image saving logic similar to metric_depth/evaluate.py
            # Check if normalization was applied to get a and b
            if args.normalize and i * batch_size + b < 10:
                filename = sample["image_path"][b].split("/")[-2:]
                output_dir1 = os.path.join(args.save_path, "examples")
                os.makedirs(output_dir1, exist_ok=True)
                output_dir2 = os.path.join(output_dir1, filename[-2])
                os.makedirs(output_dir2, exist_ok=True)
                output_path = os.path.join(output_dir2, filename[-1])
                # Use the full normalized prediction for saving
                normalized_pred = param_a * pred + param_b
                pred_16bit = (normalized_pred * 65535).to(torch.uint16)
                cv2.imwrite(output_path, pred_16bit.cpu().numpy())
                pred_16bit = (pred - pred.min()) / (pred.max() -
                                                    pred.min()) * 65535
                cv2.imwrite(output_path.replace(".png", ".max_normalized.png"),
                            pred_16bit.cpu().numpy())

    # Sum of inference times (in ms) and convert to seconds
    eval_duration_sec = sum(inference_times) / 1000.0
    eval_duration_min = eval_duration_sec / 60.0

    ratio_med = 1.0
    ratio_std = 0.0
    if ratios:
        ratios = np.array(ratios)
        ratio_med = np.median(ratios)
        ratio_std = np.std(ratios / ratio_med)
        print(" Scaling ratios | med: {:0.3f} | std: {:0.3f}".format(
            ratio_med, ratio_std))

    # --- Added Performance Metrics Calculation ---
    if total_evaluated_samples.item() > 0:
        avg_time_per_frame_ms = (eval_duration_sec /
                                 total_evaluated_samples.item()) * 1000
        fps = total_evaluated_samples.item() / eval_duration_sec
    else:
        avg_time_per_frame_ms = 0
        fps = 0

    logger.info(
        "=========================================================================================="
    )
    logger.info(
        "{:>8}, {:>8}, {:>8}, {:>8}, {:>8}, {:>8}, {:>8}, {:>8}, {:>8}".format(
            *tuple(results.keys())))
    logger.info(
        "{:8.3f}, {:8.3f}, {:8.3f}, {:8.3f}, {:8.3f}, {:8.3f}, {:8.3f}, {:8.3f}, {:8.3f}"
        .format(*tuple([(v / total_evaluated_samples).item()
                        for v in results.values()])))
    logger.info(
        "=========================================================================================="
    )
    # --- Added Logging ---
    logger.info(f"Total validation samples: {total_samples}")
    logger.info(f"Evaluated samples: {int(total_evaluated_samples.item())}")
    logger.info(
        f"Total evaluation duration: {eval_duration_sec:.2f} seconds ({eval_duration_min:.2f} minutes)"
    )
    logger.info(f"Average time per frame: {avg_time_per_frame_ms:.2f} ms"
                )  # Log to console
    logger.info(f"FPS: {fps:.2f}")  # Log to console
    # --- End Added Logging ---
    print()

    for name, metric in results.items():
        writer.add_scalar(f"eval/{name}",
                          (metric / total_evaluated_samples).item(), 0)
    writer.add_scalar("eval/duration_seconds", eval_duration_sec, 0)
    writer.add_scalar("eval/duration_minutes", eval_duration_min, 0)
    writer.add_scalar("eval/avg_time_per_frame_ms", avg_time_per_frame_ms, 0)
    writer.add_scalar("eval/fps", fps, 0)
    writer.add_scalar("eval/ratio_median", ratio_med, 0)
    writer.add_scalar("eval/ratio_std", ratio_std, 0)


if __name__ == "__main__":
    main()
