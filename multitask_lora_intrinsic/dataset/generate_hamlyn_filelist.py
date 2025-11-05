#!/usr/bin/env python3
"""
自动生成 Hamlyn 数据集的文件列表

使用方法:
    python generate_hamlyn_filelist.py --base_dir /path/to/hamlyn --output train.txt

参数:
    --base_dir: Hamlyn 数据集的根目录
    --output: 输出文件列表的路径
    --sequences: 可选，指定要包含的序列名称，例如 "rectified01,rectified02"
"""

import argparse
import os
from pathlib import Path


def generate_hamlyn_filelist(base_dir, output_file, sequences=None):
    """
    自动生成 Hamlyn 数据集的文件列表

    Args:
        base_dir (str): Hamlyn 数据集的根目录
        output_file (str): 输出文件列表的路径
        sequences (list): 要包含的序列名称列表，如果为 None 则包含所有序列
    """
    file_list = []
    base_path = Path(base_dir)

    # 如果没有指定序列，则查找所有 rectified 序列
    if sequences is None:
        seq_dirs = sorted(base_path.glob("rectified*"))
    else:
        seq_dirs = [base_path / seq for seq in sequences]

    for seq_dir in seq_dirs:
        if not seq_dir.is_dir():
            print(f"警告: 序列目录不存在: {seq_dir}")
            continue

        color_dir = seq_dir / "color"
        depth_dir = seq_dir / "depth"

        if not color_dir.exists():
            print(f"警告: color 目录不存在: {color_dir}")
            continue

        if not depth_dir.exists():
            print(f"警告: depth 目录不存在: {depth_dir}")
            continue

        # 获取所有图像文件
        img_files = sorted(color_dir.glob("*.jpg"))

        if len(img_files) == 0:
            print(f"警告: 在 {color_dir} 中没有找到 .jpg 图像文件")
            continue

        print(f"处理序列: {seq_dir.name}")
        print(f"  找到 {len(img_files)} 个图像文件")

        for img_file in img_files:
            frame_id = img_file.stem  # 不带扩展名的文件名

            # 检查对应的深度文件是否存在
            depth_file = depth_dir / f"{frame_id}.png"
            if not depth_file.exists():
                print(f"  警告: 缺少深度文件: {depth_file}")
                continue

            file_list.append(f"{seq_dir} {frame_id}\n")

    # 写入文件
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, 'w') as f:
        f.writelines(file_list)

    print(f"\n生成完成!")
    print(f"  总样本数: {len(file_list)}")
    print(f"  文件列表保存到: {output_file}")

    # 显示前几行作为示例
    if len(file_list) > 0:
        print(f"\n文件列表示例 (前5行):")
        for i, line in enumerate(file_list[:5]):
            print(f"  {line.strip()}")


def main():
    parser = argparse.ArgumentParser(
        description="生成 Hamlyn 数据集的文件列表",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 生成所有序列的文件列表
  python generate_hamlyn_filelist.py --base_dir /data/hamlyn --output /data/hamlyn/train.txt

  # 只生成特定序列的文件列表
  python generate_hamlyn_filelist.py --base_dir /data/hamlyn --output /data/hamlyn/train.txt --sequences rectified01,rectified02
        """
    )

    parser.add_argument(
        "--base_dir",
        type=str,
        required=True,
        help="Hamlyn 数据集的根目录"
    )

    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="输出文件列表的路径"
    )

    parser.add_argument(
        "--sequences",
        type=str,
        default=None,
        help="要包含的序列名称，用逗号分隔，例如 'rectified01,rectified02'"
    )

    args = parser.parse_args()

    # 解析序列列表
    sequences = None
    if args.sequences:
        sequences = [s.strip() for s in args.sequences.split(',')]

    # 生成文件列表
    generate_hamlyn_filelist(args.base_dir, args.output, sequences)


if __name__ == "__main__":
    main()
