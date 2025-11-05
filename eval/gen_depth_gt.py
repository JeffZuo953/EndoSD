#!/usr/bin/env python3
"""
从 .pt 文件中提取 'depth' 键的内容，除以 max_depth，然后保存为灰度图。

python eval/gen_depth_gt.py --input-pt <您的pt文件路径> --output-dir <输出目录> --max-depth <最大深度值>

"""

import os
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Generate grayscale depth image from a .pt file.")
    parser.add_argument("--input-pt", type=str, required=True, help="Path to the input .pt file.")
    parser.add_argument("--output-dir", type=str, required=True, help="Directory to save the output grayscale image.")
    parser.add_argument("--max-depth", type=float, required=True, help="Maximum depth value to normalize the depth map.")

    args = parser.parse_args()

    # 1. 检查输入文件是否存在
    input_path = Path(args.input_pt)
    if not input_path.is_file():
        print(f"Error: Input file not found at {args.input_pt}")
        return

    # 2. 创建输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        # 3. 加载 .pt 文件
        data = torch.load(args.input_pt, map_location='cpu')

        # 检查 'depth' 键是否存在
        if 'depth' not in data:
            print(f"Error: 'depth' key not found in {args.input_pt}")
            # 尝试将整个 tensor 作为 depth
            if isinstance(data, torch.Tensor):
                depth_tensor = data
                print("Assuming the entire .pt file is the depth tensor.")
            else:
                print(f"Error: Cannot determine depth data. Available keys: {list(data.keys()) if isinstance(data, dict) else 'Not a dict'}")
                return
        else:
            depth_tensor = data['depth']

        # 4. 提取、处理数据
        depth_np = depth_tensor.squeeze().cpu().numpy()

        # 除以 max_depth
        normalized_depth = depth_np / args.max_depth

        # 将值裁剪到 [0, 1] 范围，以便于保存为图像
        clipped_depth = np.clip(normalized_depth, 0, 1)

        # 5. 保存为灰度图
        output_filename = input_path.stem + ".png"
        output_path = output_dir / output_filename

        plt.imsave(output_path, clipped_depth, cmap='gray')

        print(f"Successfully saved depth image to {output_path}")

    except Exception as e:
        print(f"An error occurred: {e}")


if __name__ == "__main__":
    main()

# /media/ExtHDD1/jianfu/data/train_multitask_depth_seg/multitask_vits_20250712_235549/checkpoint_epoch_40.pth
