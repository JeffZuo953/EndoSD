#!/usr/bin/env python3
"""
生成并可视化分割真值（Ground Truth）的脚本
可以处理单个.pt文件，或一个包含.pt文件路径的.txt列表。
"""

import os
import argparse
import logging
import pprint
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

# 添加父目录到 Python 路径
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from multitask.util.palette import get_palette
from multitask.util.utils import init_log, get_new_filename_from_path

class GTDataset(Dataset):
    """用于加载GT .pt文件的数据集"""
    def __init__(self, input_path: str):
        super().__init__()
        if os.path.isfile(input_path) and input_path.endswith('.txt'):
            with open(input_path, 'r') as f:
                self.filelist = [line.strip() for line in f if line.strip()]
        elif os.path.isfile(input_path) and input_path.endswith('.pt'):
            self.filelist = [input_path]
        else:
            raise ValueError(f"输入必须是 .txt 文件列表或单个 .pt 文件: {input_path}")

    def __len__(self):
        return len(self.filelist)

    def __getitem__(self, index):
        fpath = self.filelist[index]
        return {'gt_path': fpath}


def visualize_and_save_seg_gt(gt_path: str, output_dir: str, origin_image_dir: str = None):
    """
    读取单个分割真值 .pt 文件，将其可视化并以新文件名保存。
    如果提供了 origin_image_dir，则同时还原并保存原始RGB图像。
    """
    logger = logging.getLogger("gen_seg_gt")

    try:
        data = torch.load(gt_path, map_location='cpu')
        
        # 提取分割掩码
        if isinstance(data, dict) and 'semseg_mask' in data:
            gt_mask = data['semseg_mask']
        else:
            gt_mask = data # 假设整个文件就是掩码
        
        if isinstance(gt_mask, torch.Tensor):
            gt_mask = gt_mask.cpu().numpy()
        
        if gt_mask is None:
            raise ValueError("无法从 .pt 文件中提取掩码")
            
        # 提取并处理原始图像（如果需要）
        if origin_image_dir:
            image_tensor = data.get('image')
            if image_tensor is not None:
                # --- 逆转 run.py 中的 NormalizeImage 预处理 ---
                mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
                std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
                
                # 反归一化: (tensor * std) + mean
                image_tensor = image_tensor.cpu() * std + mean
                image_tensor = torch.clamp(image_tensor, 0, 1)
                
                # 转换为 NumPy 格式 (H, W, C) 以便 cv2 使用
                rgb_image_np = image_tensor.permute(1, 2, 0).numpy()
                
                # 从 [0, 1] 范围转换为 [0, 255] 的 uint8
                rgb_image_np = (rgb_image_np * 255).astype(np.uint8)
                
                # OpenCV 使用 BGR 格式，因此需要从 RGB 转换
                bgr_image_np = cv2.cvtColor(rgb_image_np, cv2.COLOR_RGB2BGR)
                
                # 保存原始图像
                new_filename = get_new_filename_from_path(gt_path) + '.png'
                output_filepath = os.path.join(origin_image_dir, new_filename)
                os.makedirs(os.path.dirname(output_filepath), exist_ok=True)
                cv2.imwrite(output_filepath, bgr_image_np)
                logger.info(f"已将原始图像从 {gt_path} 保存到: {output_filepath}")
            else:
                logger.warning(f"请求保存原始图像，但在 {gt_path} 中未找到 'image' 键。")

    except Exception as e:
        logger.error(f"无法读取或处理真值文件 {gt_path}: {e}")
        return

    # 获取调色板并进行分割掩码的可视化
    palette = get_palette()
    color_seg = np.zeros((gt_mask.shape[0], gt_mask.shape[1], 3), dtype=np.uint8)
    for label, color in enumerate(palette):
        color_seg[gt_mask == label, :] = color
    color_seg = cv2.cvtColor(color_seg, cv2.COLOR_RGB2BGR)

    # 根据输入路径生成新的文件名并保存
    new_filename = get_new_filename_from_path(gt_path) + '.png'
    output_filepath = os.path.join(output_dir, new_filename)
    
    os.makedirs(os.path.dirname(output_filepath), exist_ok=True)
    cv2.imwrite(output_filepath, color_seg)
    logger.info(f"已将 {gt_path} 的可视化真值保存到: {output_filepath}")


def main():
    parser = argparse.ArgumentParser(description="Generate and Visualize Segmentation Ground Truth")
    parser.add_argument("--input-path", type=str, required=True, help="Path to a .txt filelist or a single .pt ground truth file")
    parser.add_argument("--output-path", type=str, required=True, help="Path to the base directory to save the visualized output images")
    parser.add_argument("--origin-image-dir", type=str, default=None, help="Optional. Path to save the original RGB images.")
    
    args = parser.parse_args()
    
    logger = init_log("gen_seg_gt", logging.INFO)
    logger.propagate = 0
    logger.info("\n{}\n".format(pprint.pformat(vars(args))))
    
    dataset = GTDataset(args.input_path)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4)

    for batch in loader:
        gt_path = batch['gt_path'][0]
        visualize_and_save_seg_gt(gt_path, args.output_path, args.origin_image_dir)
    
    logger.info("处理完成。")


if __name__ == "__main__":
    main()