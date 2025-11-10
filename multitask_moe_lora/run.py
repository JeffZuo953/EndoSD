#!/usr/bin/env python3
"""
多任务模型推理脚本 (深度估计 + 语义分割)
"""

import sys
import os
import argparse
import logging
import pprint
import warnings
import numpy as np
import torch
import torch.nn.functional as F
import cv2
import matplotlib.pyplot as plt
from torchvision.transforms import Compose
from torch.utils.data import Dataset, DataLoader

# 添加父目录到 Python 路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from multitask.depth_anything_v2.dpt_multitask import create_multitask_model
from multitask.util.model_io import load_weights
from multitask.util.visualize import save_depth_output, save_seg_output
from multitask.dataset.transform import Resize, NormalizeImage, PrepareForNet
from multitask.util.utils import init_log
import pandas as pd
# 假设GT尺寸的CSV文件路径是固定的
try:
    from size_prediction.dataset.polyp_size_estimation_sfsnet import SIZE_CSV_PATH
except ImportError:
    # 提供一个备用路径，以防万一
    SIZE_CSV_PATH = "size_prediction/dataset/SyntheticDatabase_Size_GT.csv"


def sanitize_path_for_filename(path: str) -> str:
    """Convert an absolute path into a filesystem-friendly identifier."""
    abs_path = os.path.abspath(path)
    sanitized = abs_path.replace(os.sep, '_')
    if os.altsep:
        sanitized = sanitized.replace(os.altsep, '_')
    sanitized = sanitized.replace(':', '_')
    return sanitized or os.path.basename(abs_path)


def _first_in_batch(value):
    """Return the first element from a DataLoader-collated field."""
    if isinstance(value, torch.Tensor):
        if value.ndim == 0:
            return value
        return value[0]
    if isinstance(value, list):
        return _first_in_batch(value[0])
    return value


def _extract_hw(original_shape_entry):
    """Extract (H, W) from a collated original_shape field."""
    value = _first_in_batch(original_shape_entry)
    if isinstance(value, torch.Tensor):
        if value.numel() != 2:
            raise ValueError("original_shape tensor must contain two elements (H, W)")
        value = value.tolist()
    if isinstance(value, (list, tuple)):
        if len(value) < 2:
            raise ValueError("original_shape must provide height and width")
        return int(value[0]), int(value[1])
    if isinstance(value, torch.Size):
        if len(value) < 2:
            raise ValueError("original_shape torch.Size must have two dimensions")
        return int(value[0]), int(value[1])
    raise ValueError(f"Unsupported original_shape format: {type(value)}")


def _extract_tensor_from_batch(batch, key):
    """Fetch a tensor-like field from a batch and ensure it lives on CPU."""
    if key not in batch:
        return None
    value = _first_in_batch(batch[key])
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    return torch.as_tensor(value)


def _extract_valid_mask(batch):
    """Retrieve a boolean valid_mask tensor from the batch if available."""
    mask = _extract_tensor_from_batch(batch, 'valid_mask')
    if mask is None:
        return None
    return mask.bool()


def _save_depth_ground_truth(batch, filename_token, outdir, normalization, colormap, save_img, save_pt, save_npz, valid_mask=None):
    depth_tensor = _extract_tensor_from_batch(batch, 'depth')
    if depth_tensor is None:
        return
    mask = valid_mask if valid_mask is not None else _extract_valid_mask(batch)
    save_depth_output(
        depth_tensor,
        filename_token,
        outdir,
        normalization,
        colormap,
        save_img,
        save_pt,
        save_npz,
        valid_mask=mask,
    )


class InferenceDataset(Dataset):
    """用于推理的数据集，可以处理单个文件、文件列表txt或pt文件"""

    def __init__(self, input_path: str, preprocess: bool, img_size: int, ensure_multiple_of: int = 14, base_data_path: str = None):
        super().__init__()
        self.preprocess = preprocess
        self.original_shapes = {}
        self.base_data_path = base_data_path

        if os.path.isfile(input_path) and input_path.endswith('.txt'):
            with open(input_path, 'r') as f:
                self.filelist = [line.strip() for line in f if line.strip()]
        elif os.path.isfile(input_path):
            self.filelist = [input_path]
        else:
            raise FileNotFoundError(f"Input path not found or is not a valid file: {input_path}")
        
        # GT size lookup
        try:
            df = pd.read_csv(SIZE_CSV_PATH)
            df['key'] = df['VIDEO'] + '/' + df['IMAGE']
            self.size_lookup = df.set_index('key')['GROUND_TRUTH_SIZE'].to_dict()
        except FileNotFoundError:
            warnings.warn(f"GT size CSV not found at {SIZE_CSV_PATH}. GT sizes will be 0.0.")
            self.size_lookup = {}

        self.transform = None
        if self.preprocess:
            net_w, net_h = img_size, img_size
            self.transform = Compose([
                Resize(
                    width=net_w,
                    height=net_h,
                    resize_target=False,
                    keep_aspect_ratio=False,
                    ensure_multiple_of=ensure_multiple_of,
                    image_interpolation_method=cv2.INTER_CUBIC,
                ),
                NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                PrepareForNet(),
            ])

    def __len__(self):
        return len(self.filelist)

    def __getitem__(self, index):
        # filelist中的是相对路径
        rel_path = self.filelist[index]
        original_rel_path = rel_path  # For debugging

        # HOTFIX: Insert 'img' directory for image paths from filelist.
        # This is specific to the SyntheticDatabase structure where images are in an 'img' subfolder.
        if not rel_path.endswith(('.pt', '.pth')):
            # Check if 'img' is already in the path to avoid double insertion
            if '/img/' not in rel_path and '\\img\\' not in rel_path:
                path_parts = os.path.split(rel_path)
                # Ensure it's a path like 'vid40/0029.jpg' (has a directory component)
                if path_parts[0] and path_parts[1]:
                    rel_path = os.path.join(path_parts[0], 'img', path_parts[1])

        # Only join with base_data_path if rel_path is not already absolute
        if self.base_data_path and not os.path.isabs(rel_path):
            fpath = os.path.join(self.base_data_path, rel_path)
        else:
            fpath = rel_path

        if fpath.endswith(('.pt', '.pth')):
            data = torch.load(fpath, map_location='cpu')
            if isinstance(data, dict) and 'image' in data:
                image_tensor = data['image']
            else:
                image_tensor = data
            
            if not isinstance(image_tensor, torch.Tensor):
                image_tensor = torch.as_tensor(image_tensor)

            original_shape_tensor = torch.tensor(list(image_tensor.shape[-2:]), dtype=torch.int64)
            naming_path = data.get('image_path', fpath) if isinstance(data, dict) else fpath

            sample = {
                'image': image_tensor,
                'filename': fpath,
                'original_shape': original_shape_tensor,
                'naming_path': naming_path,
            }
            
            if isinstance(data, dict):
                if 'semseg_mask' in data:
                    semseg_mask = data['semseg_mask']
                    sample['semseg_mask'] = semseg_mask
                if 'seg_valid_mask' in data:
                    sample['seg_valid_mask'] = data['seg_valid_mask']
                if 'semseg_mask' in sample:
                    semseg_mask = sample['semseg_mask']
                    seg_ignore = (semseg_mask == 255)
                    seg_valid = sample.get('seg_valid_mask')
                    if seg_valid is not None:
                        seg_ignore = seg_ignore | (~seg_valid.bool())
                        semseg_mask = semseg_mask.clone()
                        semseg_mask[~seg_valid.bool()] = 255
                        sample['semseg_mask'] = semseg_mask
                    sample['ignore_mask'] = seg_ignore
                if 'valid_mask' in data:
                    sample['valid_mask'] = data['valid_mask']
                if 'depth_valid_mask' in data:
                    sample['depth_valid_mask'] = data['depth_valid_mask']
                if 'depth' in data:
                    sample['depth'] = data['depth']
                if 'max_depth' in data:
                    sample['max_depth'] = torch.tensor(float(data['max_depth']))

            sample['gt_mm'] = torch.tensor(0.0, dtype=torch.float32) # pt文件没有GT
            return sample

        raw_image = cv2.imread(fpath)
        if raw_image is None:
            raise ValueError(f"Could not read image at {fpath}")

        h, w = raw_image.shape[:2]
        self.original_shapes[os.path.basename(fpath)] = (h, w)

        image = cv2.cvtColor(raw_image, cv2.COLOR_BGR2RGB) / 255.0

        if self.transform:
            sample_data = self.transform({"image": image})
            image = sample_data['image']

        # 查找GT尺寸
        # Need to construct lookup key from original path without /img/
        video_id, image_name = os.path.split(rel_path)
        # Remove '/img' from video_id if present (e.g., 'vid40/img' -> 'vid40')
        if video_id.endswith('/img') or video_id.endswith('\\img'):
            video_id = os.path.dirname(video_id)
        lookup_key = f"{video_id}/{image_name}"
        gt_size_mm = float(self.size_lookup.get(lookup_key, 0.0))

        sample = {
            'image': torch.from_numpy(image),
            'filename': fpath,
            'original_shape': torch.tensor([h, w], dtype=torch.int64),
            'naming_path': fpath,
        }
        sample['raw_image'] = raw_image
        sample['gt_mm'] = torch.tensor(gt_size_mm, dtype=torch.float32)
        return sample




def main():
    parser = argparse.ArgumentParser(description="Multi-task Inference Script")

    # Model Args
    parser.add_argument("--encoder", default="vits", choices=["vits", "vitb", "vitl", "vitg", "dinov3_vits16", "dinov3_vits16plus", "dinov3_vitb16", "dinov3_vitl16", "dinov3_vith16plus", "dinov3_vit7b16"])
    parser.add_argument("--features", default=64, type=int)
    parser.add_argument("--num-classes", default=3, type=int)
    parser.add_argument("--max-depth", default=0.2, type=float)
    parser.add_argument("--seg-input-type", default="last_four", choices=["last", "last_four", "from_depth"], help="Input type for segmentation head ('last', 'last_four', 'from_depth')")
    parser.add_argument("--pretrained-from", type=str, default=None, help="Path to pretrained weights")
    parser.add_argument("--dinov3-repo-path", type=str, default="/media/ExtHDD1/jianfu/depth/DepthAnythingV2/dinov3", help="Path to the local dinov3 repository")
    
    # PEFT Args
    parser.add_argument("--mode", type=str, default="original", choices=["original", "lora-only", "legacy-lora"], help="PEFT mode")
    parser.add_argument("--num-experts", type=int, default=8, help="Number of experts for MoE")
    parser.add_argument("--top-k", type=int, default=2, help="Top-k routing for MoE")
    parser.add_argument("--lora-r", type=int, default=4, help="LoRA rank")
    parser.add_argument("--lora-alpha", type=int, default=8, help="LoRA alpha")

    # Data Args
    parser.add_argument("--input-path", type=str, required=True, help="Path to a single image, a .pt file, or a .txt filelist")
    parser.add_argument("--output-path", type=str, required=True, help="Path to save the outputs")
    parser.add_argument("--img-size", default=518, type=int)
    parser.add_argument("--preprocess", action='store_true', help="Apply preprocessing to input images")
    parser.add_argument("--base-data-path", type=str, default=None, help="Base path for data if filelist contains relative paths")

    # Task Args
    parser.add_argument("--task", type=str, default="all", choices=["depth", "seg", "all"], help="Task to perform")

    # Output Args
    parser.add_argument("--save-image", action='store_true', help="Save output as image")
    parser.add_argument("--save-pt", action='store_true', help="Save raw output as .pt file")
    parser.add_argument("--save-npz", action='store_true', help="Save output as .npz file")
    parser.add_argument("--save-gt", action='store_true', help="Also export ground-truth depth maps when available")
    parser.add_argument("--save-gt-only", action='store_true', help="Only export ground-truth depth maps without running inference")
    parser.add_argument("--normalization", type=str, default="min-max", choices=["min-max", "max"], help="Depth normalization for visualization")
    parser.add_argument("--colormap", type=str, default="gray", help="Colormap for depth image")

    args = parser.parse_args()

    if args.save_gt_only:
        args.save_gt = True

    if not args.save_gt_only and not args.pretrained_from:
        parser.error("--pretrained-from is required unless --save-gt-only is set.")

    logger = init_log("multitask_inference", logging.INFO)
    logger.propagate = 0
    logger.info("\n{}\n".format(pprint.pformat(vars(args))))

    model = None
    if not args.save_gt_only:
        # 1. Create Model
        model = create_multitask_model(
            encoder=args.encoder,
            num_classes=args.num_classes,
            features=args.features,
            max_depth=args.max_depth,
            seg_input_type=args.seg_input_type,
            dinov3_repo_path=args.dinov3_repo_path,
            frozen_backbone=True,  # Inference is always with frozen backbone
            mode=args.mode,
            num_experts=args.num_experts,
            top_k=args.top_k,
            lora_r=args.lora_r,
            lora_alpha=args.lora_alpha
        )

        # 2. Load Weights (with key compatibility from training script)
        load_weights(model, args.pretrained_from)

        model.cuda()
        model.eval()

    # 3. Create Dataset and DataLoader
    # 3. Create Dataset and DataLoader
    ensure_multiple_of = 16 if 'dinov3' in args.encoder else 14
    logger.info(f"Using ensure_multiple_of = {ensure_multiple_of} for encoder {args.encoder}")
    
    dataset = InferenceDataset(args.input_path, args.preprocess, args.img_size, ensure_multiple_of=ensure_multiple_of, base_data_path=args.base_data_path)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4)

    depth_output_root = os.path.join(args.output_path, 'depth')
    depth_gt_output_root = os.path.join(args.output_path, 'depth_gt')
    seg_output_root = os.path.join(args.output_path, 'seg')

    if args.save_gt_only:
        logger.info("Exporting ground-truth depth maps without running inference...")
        for batch in loader:
            filename_for_log = str(_first_in_batch(batch['filename']))
            naming_entry = batch.get('naming_path', batch['filename'])
            naming_path = str(_first_in_batch(naming_entry))
            output_token = sanitize_path_for_filename(naming_path)
            logger.info(f"Processing {filename_for_log} -> {output_token} (GT only)")
            _save_depth_ground_truth(
                batch,
                output_token,
                depth_gt_output_root,
                args.normalization,
                args.colormap,
                args.save_image,
                args.save_pt,
                args.save_npz,
            )
        logger.info("Ground-truth export finished.")
        return

    # 4. Inference Loop
    with torch.no_grad():
        for batch in loader:
            input_tensor = batch['image'].cuda()
            filename_for_log = str(_first_in_batch(batch['filename']))
            naming_entry = batch.get('naming_path', batch['filename'])
            naming_path = str(_first_in_batch(naming_entry))
            output_filename = sanitize_path_for_filename(naming_path)
            original_h, original_w = _extract_hw(batch['original_shape'])

            logger.info(f"Processing {filename_for_log} -> {output_filename}")

            valid_mask_cpu = _extract_valid_mask(batch)

            # --- Depth Task ---
            if args.task in ['depth', 'all']:
                outputs = model(input_tensor, task='depth')
                pred_depth = outputs['depth']
                # Resize to original
                pred_depth = F.interpolate(pred_depth.unsqueeze(0), (original_h, original_w), mode='bilinear', align_corners=False).squeeze()
                save_depth_output(
                    pred_depth,
                    output_filename,
                    depth_output_root,
                    args.normalization,
                    args.colormap,
                    args.save_image,
                    args.save_pt,
                    args.save_npz,
                    valid_mask=valid_mask_cpu,
                )

            if args.save_gt:
                _save_depth_ground_truth(
                    batch,
                    output_filename,
                    depth_gt_output_root,
                    args.normalization,
                    args.colormap,
                    args.save_image,
                    args.save_pt,
                    args.save_npz,
                    valid_mask=valid_mask_cpu,
                )

            # --- Segmentation Task ---
            if args.task in ['seg', 'all']:
                outputs = model(input_tensor, task='seg')
                pred_seg = outputs['seg']  # Logits
                # Resize to original
                pred_seg = F.interpolate(pred_seg, (original_h, original_w), mode='bilinear', align_corners=False).squeeze()

                # Get ignore mask from batch if available
                ignore_mask = batch.get('ignore_mask', None)
                if ignore_mask is not None:
                    ignore_mask = ignore_mask.squeeze().cuda()

                save_seg_output(pred_seg, output_filename, seg_output_root, args.save_image, args.save_pt, args.save_npz, ignore_mask=ignore_mask)

    logger.info("Inference finished.")


if __name__ == "__main__":
    main()
