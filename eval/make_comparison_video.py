#!/usr/bin/env python3
"""
将 GT 和预测的深度图与分割图合成为一个四宫格对比视频。

用法:
python eval/make_comparison_video.py \
    --gt-depth-dir <path_to_gt_depth_pngs> \
    --gt-seg-dir <path_to_gt_seg_pngs> \
    --pred-depth-dir <path_to_pred_depth_pngs> \
    --pred-seg-dir <path_to_pred_seg_pngs> \
    --output-video <output_video_path.mp4>
"""

import cv2
import numpy as np
import os
import argparse
from glob import glob
from tqdm import tqdm

def main():
    parser = argparse.ArgumentParser(description="Create a 2x2 comparison video of depth and segmentation.")
    parser.add_argument('--gt-depth-dir', type=str, required=True, help="Directory for ground truth depth images (PNG).")
    parser.add_argument('--gt-seg-dir', type=str, required=True, help="Directory for ground truth segmentation images (PNG).")
    parser.add_argument('--pred-depth-dir', type=str, required=True, help="Directory for predicted depth images (PNG).")
    parser.add_argument('--pred-seg-dir', type=str, required=True, help="Directory for predicted segmentation images (PNG).")
    parser.add_argument('--output-video', type=str, required=True, help="Path to save the output video file (e.g., comparison.mp4).")
    parser.add_argument('--fps', type=int, default=10, help="Frames per second for the output video.")
    
    args = parser.parse_args()

    # 以 GT 深度图目录为基准，查找所有文件
    gt_depth_files = sorted(glob(os.path.join(args.gt_depth_dir, '*.png')))
    if not gt_depth_files:
        print(f"Error: No PNG files found in {args.gt_depth_dir}")
        return

    # 初始化视频写入器
    first_img = cv2.imread(gt_depth_files[0])
    h, w, _ = first_img.shape
    
    # 定义视频编码和创建 VideoWriter 对象
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') # 或者 'XVID'
    video_writer = cv2.VideoWriter(args.output_video, fourcc, args.fps, (w * 2, h * 2))

    print(f"Found {len(gt_depth_files)} frames. Starting video creation...")

    for gt_depth_path in tqdm(gt_depth_files, desc="Processing frames"):
        basename = os.path.basename(gt_depth_path)
        
        # 构建其他三个文件的路径
        gt_seg_path = os.path.join(args.gt_seg_dir, basename)
        pred_depth_path = os.path.join(args.pred_depth_dir, basename)
        pred_seg_path = os.path.join(args.pred_seg_dir, basename)

        # 检查所有文件是否存在
        if not all(os.path.exists(p) for p in [gt_seg_path, pred_depth_path, pred_seg_path]):
            print(f"Warning: Skipping {basename}, corresponding file not found in all directories.")
            continue

        # 读取所有图像
        gt_depth_img = cv2.imread(gt_depth_path)
        gt_seg_img = cv2.imread(gt_seg_path)
        pred_depth_img = cv2.imread(pred_depth_path)
        pred_seg_img = cv2.imread(pred_seg_path)
        
        # 确保所有图像都是 3 通道 BGR
        if len(gt_depth_img.shape) == 2:
            gt_depth_img = cv2.cvtColor(gt_depth_img, cv2.COLOR_GRAY2BGR)
        if len(pred_depth_img.shape) == 2:
            pred_depth_img = cv2.cvtColor(pred_depth_img, cv2.COLOR_GRAY2BGR)

        # 添加标签
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.8
        font_color = (0, 255, 255)  # BGR for Yellow
        thickness = 2
        
        # 左上角 (GT Depth)
        cv2.putText(gt_depth_img, 'gt', (10, 30), font, font_scale, font_color, thickness, cv2.LINE_AA)
        # 右上角 (Pred Depth)
        cv2.putText(pred_depth_img, 'pred', (10, 30), font, font_scale, font_color, thickness, cv2.LINE_AA)
        # 左下角 (GT Seg)
        cv2.putText(gt_seg_img, 'gt', (10, 30), font, font_scale, font_color, thickness, cv2.LINE_AA)
        # 右下角 (Pred Seg)
        cv2.putText(pred_seg_img, 'pred', (10, 30), font, font_scale, font_color, thickness, cv2.LINE_AA)

        # 拼接图像
        top_row = np.hstack((gt_depth_img, pred_depth_img))
        bottom_row = np.hstack((gt_seg_img, pred_seg_img))
        combined_frame = np.vstack((top_row, bottom_row))
        
        video_writer.write(combined_frame)

    video_writer.release()
    print(f"Successfully created video: {args.output_video}")

if __name__ == '__main__':
    main()