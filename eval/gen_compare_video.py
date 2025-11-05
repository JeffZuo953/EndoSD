#!/usr/bin/env python3
"""
多任务模型推理脚本 (深度估计 + 语义分割)
根据输入图片、模型权重，生成预测的深度图和分割图。
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
from glob import glob
from tqdm import tqdm

# 添加父目录到 Python 路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from multitask.depth_anything_v2.dpt_multitask import create_multitask_model
from multitask.dataset.inhouse_seg import get_palette
from metric_depth.dataset.transform import Resize, NormalizeImage, PrepareForNet
from multitask.util.utils import init_log, get_new_filename_from_path


class InferenceDataset(Dataset):
    """用于推理的数据集，处理一个目录下的所有图片"""

    def __init__(self, input_path: str, img_size: int):
        super().__init__()
        self.filelist = sorted(glob(os.path.join(input_path, '*')))
        self.img_size = img_size
        
        net_w, net_h = img_size, img_size
        self.transform = Compose([
            Resize(
                width=net_w,
                height=net_h,
                resize_target=False,
                keep_aspect_ratio=False,
                image_interpolation_method=cv2.INTER_CUBIC,
            ),
            NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            PrepareForNet(),
        ])

    def __len__(self):
        return len(self.filelist)

    def __getitem__(self, index):
        fpath = self.filelist[index]
        
        raw_image = cv2.imread(fpath)
        if raw_image is None:
            raise ValueError(f"Could not read image at {fpath}")

        h, w = raw_image.shape[:2]
        image = cv2.cvtColor(raw_image, cv2.COLOR_BGR2RGB) / 255.0

        sample = self.transform({"image": image})
        image = sample['image']

        return {'image': torch.from_numpy(image), 'filename': fpath, 'original_shape': (h, w)}


def save_depth_output(pred, filename, outdir):
    """保存深度预测结果，先除以0.05，再归一化为灰度图"""
    base_filename = get_new_filename_from_path(filename)
    
    img_dir = os.path.join(outdir, 'depth_pred')
    os.makedirs(img_dir, exist_ok=True)
    
    pred_np = pred.squeeze().cpu().numpy()
    
    # 按照要求，先除以 0.05
    pred_np = pred_np / 0.05
    
    # 归一化到 [0, 1] 范围以便于可视化
    normalized = (pred_np - pred_np.min()) / (pred_np.max() - pred_np.min() + 1e-8)
    
    plt.imsave(os.path.join(img_dir, f"{base_filename}.png"), normalized, cmap='gray')


def save_seg_output(pred, filename, outdir):
    """保存分割预测结果"""
    base_filename = get_new_filename_from_path(filename)
    
    img_dir = os.path.join(outdir, 'seg_pred')
    os.makedirs(img_dir, exist_ok=True)
    
    # pred is logits [C, H, W], get class indices
    pred_idx = pred.argmax(dim=0)
    pred_np = pred_idx.cpu().numpy().astype(np.uint8)
    
    palette = get_palette()
    color_seg = np.zeros((pred_np.shape[0], pred_np.shape[1], 3), dtype=np.uint8)
    for label, color in enumerate(palette):
        color_seg[pred_np == label, :] = color
    color_seg = cv2.cvtColor(color_seg, cv2.COLOR_RGB2BGR)
    cv2.imwrite(os.path.join(img_dir, f"{base_filename}.png"), color_seg)


def main():
    parser = argparse.ArgumentParser(description="Generate predictions for depth and segmentation.")
    
    # Model Args
    parser.add_argument("--encoder", default="vits", choices=["vits", "vitb", "vitl", "vitg"])
    parser.add_argument("--features", default=64, type=int)
    parser.add_argument("--num-classes", default=3, type=int)
    parser.add_argument("--max-depth", default=0.2, type=float)
    parser.add_argument("--model-path", type=str, required=True, help="Path to pretrained weights (.pth)")

    # Data Args
    parser.add_argument("--input-dir", type=str, required=True, help="Path to the directory of input images")
    parser.add_argument("--output-dir", type=str, required=True, help="Path to save the outputs")
    parser.add_argument("--img-size", default=518, type=int)

    args = parser.parse_args()

    logger = init_log("multitask_generate_preds", logging.INFO)
    logger.propagate = 0
    logger.info("\n{}\n".format(pprint.pformat(vars(args))))

    # 1. Create Model
    model = create_multitask_model(
        encoder=args.encoder,
        num_classes=args.num_classes,
        features=args.features,
        max_depth=args.max_depth,
        frozen_backbone=True
    )

    # 2. Load Weights
    logger.info(f"Loading pretrained weights from: {args.model_path}")
    checkpoint = torch.load(args.model_path, map_location="cpu")
    
    state_dict = checkpoint.get('model_state_dict', checkpoint.get('model', checkpoint))
    model.load_state_dict(state_dict, strict=False)
    
    model.cuda()
    model.eval()

    # 3. Create Dataset and DataLoader
    dataset = InferenceDataset(args.input_dir, args.img_size)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4)

    # 4. Inference Loop
    with torch.no_grad():
        for batch in tqdm(loader, desc="Processing images"):
            input_tensor = batch['image'].cuda()
            filename = batch['filename'][0]
            original_h, original_w = batch['original_shape']

            # --- Depth Task ---
            depth_outputs = model(input_tensor, task='depth')
            pred_depth = depth_outputs['depth']
            pred_depth = F.interpolate(pred_depth.unsqueeze(0), (original_h, original_w), mode='bilinear', align_corners=False).squeeze()
            save_depth_output(pred_depth, filename, args.output_dir)

            # --- Segmentation Task ---
            seg_outputs = model(input_tensor, task='seg')
            pred_seg = seg_outputs['seg']
            pred_seg = F.interpolate(pred_seg, (original_h, original_w), mode='bilinear', align_corners=False).squeeze()
            save_seg_output(pred_seg, filename, args.output_dir)

    logger.info(f"Inference finished. Predictions saved in {args.output_dir}")


if __name__ == "__main__":
    main()