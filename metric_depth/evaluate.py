import argparse
import logging
import os
import pprint
import time  # Import time module

import numpy as np
import cv2
import torch
from torch.utils.data import DataLoader, ConcatDataset
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
import matplotlib.cm as cm

def depth_to_colormap(depth_map: np.ndarray) -> np.ndarray:
    """
    将深度图转换为热度图。
    Args:
        depth_map (np.ndarray): 深度图，单通道浮点数数组。
    Returns:
        np.ndarray: 热度图，RGB图像数组 (0-255)。
    """
    # 归一化深度图到0-1范围
    min_val = np.min(depth_map)
    max_val = np.max(depth_map)
    if max_val - min_val > 0:
        normalized_depth = (depth_map - min_val) / (max_val - min_val)
    else:
        normalized_depth = np.zeros_like(depth_map)

    # 使用viridis颜色映射
    colormap = cm.get_cmap('viridis')
    heatmap = colormap(normalized_depth)[:, :, :3]  # 取RGB通道
    heatmap = (heatmap * 255).astype(np.uint8)
    return heatmap

from depth_anything_v2.dpt_features import DepthAnythingV2
from util.metric import eval_depth
from util.utils import init_log
from dataset.c3vd import C3VD
from dataset.simcol import Simcol
# from dataset.endomapper import Endomapper
from dataset.inhouse import InHouse
from tqdm import tqdm  # Added tqdm

from collections import OrderedDict


def strip_module_prefix(state_dict):
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        new_key = k.replace("module.", "")
        new_state_dict[new_key] = v
    return new_state_dict


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
parser.add_argument("--max-depth", default=20, type=float)
parser.add_argument("--normalize", action="store_true")
parser.add_argument("--bs", default=20, type=int)  # 40, 95, 175


def normalize_and_save_image(data: np.ndarray, path: str, is_dino_feature: bool = False):
    """
    将浮点数数组归一化到0-255并保存为灰度PNG图片。
    Args:
        data (np.ndarray): 输入数据，可以是DINO特征或深度图。
        path (str): 保存图片的路径。
        is_dino_feature (bool): 如果是DINO特征，则取第一个通道进行可视化。
    """
    if is_dino_feature and data.ndim == 3: # C, H, W
        # For DINO features, take the first channel for visualization or average
        # Here, we take the first channel as a simple visualization
        data = data[0, :, :] # Take the first channel

    min_val = np.min(data)
    max_val = np.max(data)
    
    if max_val - min_val > 0:
        normalized_data = (data - min_val) / (max_val - min_val) * 255
    else:
        normalized_data = np.zeros_like(data)

    cv2.imwrite(path, normalized_data.astype(np.uint8))

INHOUSE_VAL_SPLIT = "/data/inhouse/val.txt"

def main():
    args = parser.parse_args()

    logger = init_log("global", logging.INFO)
    logger.propagate = 0

    all_args = {**vars(args)}
    logger.info("{}\n".format(pprint.pformat(all_args)))
    writer = SummaryWriter(args.save_path)

    # valset = C3VD('/root/Depth-Anything-V2/metric_depth/dataset/splits/c3vd/val.txt', 'val', size=size, max_depth=args.max_depth)

    valset = ConcatDataset([
        # C3VD(
        #     "/root/c3vd/test_mapping.txt",
        #     # 'dataset/splits/c3vd/val.txt',
        #     "train",
        #     size=size,
        # ),
        # Simcol(
        #     "/data/simcol/val_paths.txt",
        #     "val",
        #     size=size,
        # ),
        # Endomapper(
        #     "dataset/splits/endomapper/train.txt",
        #     "train",
        #     size=size,
        # ),
        InHouse(INHOUSE_VAL_SPLIT, "val")
        # InHouse(
        #     "/data/inhouse/val.txt",
        #     "train",
        #     # size=(1920, 1080),
        #     # size=(960, 540),
        #     size=(400, 400),
        #     max_depth=50.0)
    ])
    valloader = DataLoader(
        valset,
        batch_size=1, # args.bs,
        # batch_size=40, # vitl
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
    model = DepthAnythingV2(
        **{
            **model_configs[args.encoder], "max_depth": args.max_depth
        })

    loaded_state_dict = torch.load(args.load_from, map_location="cpu")
    if isinstance(loaded_state_dict, dict) and "model" in loaded_state_dict:
        model.load_state_dict(strip_module_prefix(loaded_state_dict["model"]))
    else:
        model.load_state_dict(strip_module_prefix(loaded_state_dict))

    model.load_state_dict(torch.load(args.load_from, map_location="cpu"),
                          strict=False)
    model.to("cuda").eval()

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

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    for i, sample in enumerate(tqdm(valloader)):  # Wrapped with tqdm
        if i >= 3:
            break
        img = sample["image"].cuda().float()
        depth_batch = sample["depth"].cuda()
        valid_mask_batch = sample["valid_mask"].cuda()

        with torch.no_grad():
            start_event.record()  # Record start event
            pred_batch, features = model.forward_with_features(img)
            end_event.record()  # Record end event

        # Wait for the events to complete
        torch.cuda.synchronize()

        # Calculate inference time in milliseconds
        inference_time_ms = start_event.elapsed_time(end_event)
        inference_times.append(inference_time_ms)  # Append to list

        # Evaluate each sample in the batch
        batch_size = img.shape[0]
        for b in range(batch_size):
            depth_sample = depth_batch[b]
            valid_mask = valid_mask_batch[b]
            pred_sample = pred_batch[b]
            # Get prediction for single sample (shape: C, H, W)

            # # Save original image
            # original_image_path = sample["image_path"][b]
            # original_image = Image.open(original_image_path).convert("RGB")
            # original_image_output_dir = os.path.join(args.save_path, "original_images")
            # os.makedirs(original_image_output_dir, exist_ok=True)
            # original_image_filename_base, original_image_ext = os.path.splitext(os.path.basename(original_image_path))
            # original_image_size_str = f"_{original_image.height}x{original_image.width}"
            # original_image_output_path = os.path.join(original_image_output_dir, f"{original_image_filename_base}{original_image_size_str}{original_image_ext}")
            # original_image.save(original_image_output_path)

            # # Save DINO features
            # dino_features = features[0][b].cpu().numpy() # Assuming the first feature map is the DINO output
            # dino_output_dir = os.path.join(args.save_path, "dino_features")
            # os.makedirs(dino_output_dir, exist_ok=True)
            # dino_feature_size_str = f"_{dino_features.shape[0]}x{dino_features.shape[1]}x{dino_features.shape[2]}"
            # dino_output_path = os.path.join(dino_output_dir, f"{original_image_filename_base}_dino{dino_feature_size_str}.png")
            # normalize_and_save_image(dino_features, dino_output_path, is_dino_feature=True)

            # # Save model raw output (pred_batch)
            # model_raw_output_dir = os.path.join(args.save_path, "model_raw_output")
            # os.makedirs(model_raw_output_dir, exist_ok=True)
            # pred_sample_np = pred_sample.cpu().numpy()
            # model_raw_output_size_str = f"_{pred_sample_np.shape[0]}x{pred_sample_np.shape[1]}"
            # model_raw_output_path = os.path.join(model_raw_output_dir, f"{original_image_filename_base}_raw_output{model_raw_output_size_str}.png")
            # normalize_and_save_image(pred_sample_np, model_raw_output_path)

            # # Save model raw output as heatmap
            # raw_heatmap_output_dir = os.path.join(args.save_path, "raw_heatmap_output")
            # os.makedirs(raw_heatmap_output_dir, exist_ok=True)
            # raw_heatmap_output_path = os.path.join(raw_heatmap_output_dir, f"{original_image_filename_base}_raw_heatmap{model_raw_output_size_str}.png")
            
            # raw_heatmap_image = depth_to_colormap(pred_sample_np)
            # cv2.imwrite(raw_heatmap_output_path, cv2.cvtColor(raw_heatmap_image, cv2.COLOR_RGB2BGR))

            # Interpolate single sample prediction while keeping channel dimension
            interpolated_pred = F.interpolate(
                pred_sample.unsqueeze(0).unsqueeze(0),
                # Add batch dim: 1, C, H, W
                depth_sample.shape[-2:],  # Target size: H_target, W_target
                mode="bilinear",
                align_corners=False)  # Output shape: 1, C, H_target, W_target

            # Select the first channel and remove batch dimension
            pred = interpolated_pred.squeeze(0).squeeze(0)
            # Output shape: H_target, W_target

            # Save interpolated output
            # interpolated_output_dir = os.path.join(args.save_path, "interpolated_output")
            # os.makedirs(interpolated_output_dir, exist_ok=True)
            # pred_np = pred.cpu().numpy()
            # interpolated_output_size_str = f"_{pred_np.shape[0]}x{pred_np.shape[1]}"
            # interpolated_output_path = os.path.join(interpolated_output_dir, f"{original_image_filename_base}_interpolated{interpolated_output_size_str}.png")
            # normalize_and_save_image(pred_np, interpolated_output_path)

            # # Save interpolated output as heatmap
            # heatmap_output_dir = os.path.join(args.save_path, "heatmap_output")
            # os.makedirs(heatmap_output_dir, exist_ok=True)
            # heatmap_output_path = os.path.join(heatmap_output_dir, f"{original_image_filename_base}_heatmap{interpolated_output_size_str}.png")
            
            # heatmap_image = depth_to_colormap(pred_np)
            # cv2.imwrite(heatmap_output_path, cv2.cvtColor(heatmap_image, cv2.COLOR_RGB2BGR))

            depth_mask = depth_sample[valid_mask]
            pred_mask = pred[valid_mask]

            if valid_mask.sum() < 10:
                continue

            cur_results = eval_depth(pred_mask, depth_mask)

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

            if i * batch_size + b < 3:
                filename = sample["image_path"][b].split("/")[-2:]
                output_dir1 = os.path.join(args.save_path, "examples")
                os.makedirs(output_dir1, exist_ok=True)
                output_dir2 = os.path.join(output_dir1, filename[-2])
                os.makedirs(output_dir2, exist_ok=True)
                output_path = os.path.join(output_dir2, filename[-1])
                pred_16bit = ((pred / args.max_depth) * 65535).to(torch.uint16)
                cv2.imwrite(output_path, pred_16bit.cpu().numpy())

                output_dir1 = os.path.join(args.save_path,
                                           "examples_normalized")
                os.makedirs(output_dir1, exist_ok=True)
                output_dir2 = os.path.join(output_dir1, filename[-2])
                os.makedirs(output_dir2, exist_ok=True)
                output_path = os.path.join(output_dir2, filename[-1])
                pred_16bit = (((pred - pred.min()) /
                               (pred.max() - pred.min())) * 65535).to(
                                   torch.uint16)
                cv2.imwrite(output_path, pred_16bit.cpu().numpy())

                txt_output_path = output_path.replace(".png", ".txt")
                np.savetxt(txt_output_path, pred.cpu().numpy(), fmt='%.8f')
                np.savetxt(txt_output_path.replace(".txt", ".gt.txt"),
                           depth_sample.cpu().numpy(),
                           fmt='%.8f')

    eval_duration_ms = sum(inference_times)
    eval_duration_sec = eval_duration_ms / 1000
    eval_duration_min = eval_duration_sec / 60

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
        f"Total evaluation duration: {eval_duration_sec:.4f} seconds ({eval_duration_min:.4f} minutes)"
    )
    logger.info(f"Average time per frame: {avg_time_per_frame_ms:.4f} ms")
    # Log to console
    logger.info(f"FPS: {fps:.4f}")  # Log to console
    # --- End Added Logging ---
    print()

    for name, metric in results.items():
        writer.add_scalar(f"eval/{name}",
                          (metric / total_evaluated_samples).item(), 0)
    writer.add_scalar("eval/duration_seconds", eval_duration_sec, 0)
    writer.add_scalar("eval/duration_minutes", eval_duration_min, 0)
    writer.add_scalar("eval/avg_time_per_frame_ms", avg_time_per_frame_ms, 0)
    writer.add_scalar("eval/fps", fps, 0)


if __name__ == "__main__":
    main()
