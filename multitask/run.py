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


class InferenceDataset(Dataset):
    """用于推理的数据集，可以处理单个文件、文件列表txt或pt文件"""

    def __init__(self, input_path: str, preprocess: bool, img_size: int, ensure_multiple_of: int = 14, base_data_path: str = None):
        super().__init__()
        self.preprocess = preprocess
        self.original_shapes = {}

        if os.path.isfile(input_path) and input_path.endswith('.txt'):
            with open(input_path, 'r') as f:
                self.filelist = [line.strip() for line in f if line.strip()]
        elif os.path.isfile(input_path):
            self.filelist = [input_path]
        else:
            raise FileNotFoundError(f"Input path not found or is not a valid file: {input_path}")
        
        # If a base data path is provided, prepend it to each file path
        # If a base data path is provided, prepend it to each file path
        if base_data_path:
            new_filelist = []
            for p in self.filelist:
                # HOTFIX for SyntheticDatabase structure
                if not p.endswith(('.pt', '.pth')):
                    path_parts = os.path.split(p)
                    if len(path_parts) == 2 and path_parts[0] and path_parts[1]:
                        p = os.path.join(path_parts[0], 'img', path_parts[1])
                new_filelist.append(os.path.join(base_data_path, p))
            self.filelist = new_filelist

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
        fpath = self.filelist[index]

        if fpath.endswith(('.pt', '.pth')):
            data = torch.load(fpath, map_location='cpu')
            if isinstance(data, dict) and 'image' in data:
                image_tensor = data['image']
            else:
                image_tensor = data
            
            sample = {'image': image_tensor, 'filename': fpath, 'original_shape': image_tensor.shape[-2:]}
            
            if isinstance(data, dict) and 'semseg_mask' in data:
                semseg_mask = data['semseg_mask']
                # Create ignore mask where semseg_mask is 255
                ignore_mask = (semseg_mask == 255)
                sample['ignore_mask'] = ignore_mask
                
            return sample

        raw_image = cv2.imread(fpath)
        if raw_image is None:
            raise ValueError(f"Could not read image at {fpath}")

        h, w = raw_image.shape[:2]
        self.original_shapes[os.path.basename(fpath)] = (h, w)

        image = cv2.cvtColor(raw_image, cv2.COLOR_BGR2RGB) / 255.0

        if self.transform:
            sample = self.transform({"image": image})
            image = sample['image']

        return {'image': torch.from_numpy(image), 'filename': fpath, 'original_shape': (h, w)}




def main():
    parser = argparse.ArgumentParser(description="Multi-task Inference Script")

    # Model Args
    parser.add_argument("--encoder", default="vits", choices=["vits", "vitb", "vitl", "vitg", "dinov3_vits16", "dinov3_vits16plus", "dinov3_vitb16", "dinov3_vitl16", "dinov3_vith16plus", "dinov3_vit7b16"])
    parser.add_argument("--features", default=64, type=int)
    parser.add_argument("--num-classes", default=3, type=int)
    parser.add_argument("--max-depth", default=0.2, type=float)
    parser.add_argument("--seg-input-type", default="last_four", choices=["last", "last_four", "from_depth"], help="Input type for segmentation head ('last', 'last_four', 'from_depth')")
    parser.add_argument("--pretrained-from", type=str, required=True, help="Path to pretrained weights")
    parser.add_argument("--dinov3-repo-path", type=str, default="/media/ExtHDD1/jianfu/depth/DepthAnythingV2/dinov3", help="Path to the local dinov3 repository")

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
    parser.add_argument("--normalization", type=str, default="min-max", choices=["min-max", "max"], help="Depth normalization for visualization")
    parser.add_argument("--colormap", type=str, default="gray", help="Colormap for depth image")

    args = parser.parse_args()

    logger = init_log("multitask_inference", logging.INFO)
    logger.propagate = 0
    logger.info("\n{}\n".format(pprint.pformat(vars(args))))

    # 1. Create Model
    model = create_multitask_model(
        encoder=args.encoder,
        num_classes=args.num_classes,
        features=args.features,
        max_depth=args.max_depth,
        seg_input_type=args.seg_input_type,
        dinov3_repo_path=args.dinov3_repo_path,
        frozen_backbone=True  # Inference is always with frozen backbone
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

    # 4. Inference Loop
    with torch.no_grad():
        for batch in loader:
            input_tensor = batch['image'].cuda()
            filename = batch['filename'][0]
            original_h, original_w = batch['original_shape']

            logger.info(f"Processing {filename}...")
            
            # Generate a new filename by replacing slashes with underscores
            # This prevents overwriting files with the same name from different directories
            output_filename = filename.lstrip('/').replace('/', '_')

            # --- Depth Task ---
            if args.task in ['depth', 'all']:
                outputs = model(input_tensor, task='depth')
                pred_depth = outputs['depth']
                # Resize to original
                pred_depth = F.interpolate(pred_depth.unsqueeze(0), (original_h, original_w), mode='bilinear', align_corners=False).squeeze()
                save_depth_output(pred_depth, output_filename, os.path.join(args.output_path, 'depth'), args.normalization, args.colormap, args.save_image, args.save_pt, args.save_npz)

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

                save_seg_output(pred_seg, output_filename, os.path.join(args.output_path, 'seg'), args.save_image, args.save_pt, args.save_npz, ignore_mask=ignore_mask)

    logger.info("Inference finished.")


if __name__ == "__main__":
    main()
