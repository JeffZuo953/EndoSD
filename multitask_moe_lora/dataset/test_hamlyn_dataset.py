#!/usr/bin/env python3
"""
测试 Hamlyn 数据集加载是否正常

使用方法:
    python test_hamlyn_dataset.py --base_dir /path/to/hamlyn --filelist /path/to/train.txt
"""

import argparse
import sys
import os

# 添加父目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dataset.hamlyn import HamlynDataset
import torch


def test_dataset_loading(base_dir, filelist_path):
    """
    测试 Hamlyn 数据集加载
    """
    print("=" * 80)
    print("测试 Hamlyn 数据集加载")
    print("=" * 80)

    # 创建数据集实例
    print(f"\n1. 创建数据集实例...")
    print(f"   文件列表: {filelist_path}")
    print(f"   根目录: {base_dir}")

    try:
        dataset = HamlynDataset(
            filelist_path=filelist_path,
            rootpath=base_dir,
            mode="train",
            size=(518, 518),
            max_depth=0.3,
            depth_scale=1000.0,
            cache_intrinsics=True,
        )
        print(f"   ✓ 数据集创建成功!")
        print(f"   数据集大小: {len(dataset)} 个样本")
    except Exception as e:
        print(f"   ✗ 数据集创建失败: {e}")
        return False

    # 测试加载第一个样本
    print(f"\n2. 测试加载第一个样本...")
    try:
        sample = dataset[0]
        print(f"   ✓ 样本加载成功!")

        print(f"\n   样本信息:")
        print(f"   - Image shape: {sample['image'].shape}")
        print(f"   - Image dtype: {sample['image'].dtype}")
        print(f"   - Image range: [{sample['image'].min():.4f}, {sample['image'].max():.4f}]")

        print(f"\n   - Depth shape: {sample['depth'].shape}")
        print(f"   - Depth dtype: {sample['depth'].dtype}")
        print(f"   - Depth range: [{sample['depth'].min():.4f}, {sample['depth'].max():.4f}]")

        print(f"\n   - Valid mask shape: {sample['valid_mask'].shape}")
        print(f"   - Valid mask dtype: {sample['valid_mask'].dtype}")
        print(f"   - Valid pixels: {sample['valid_mask'].sum().item()} / {sample['valid_mask'].numel()}")

        print(f"\n   - Intrinsics shape: {sample['intrinsics'].shape}")
        print(f"   - Intrinsics:\n{sample['intrinsics']}")

        print(f"\n   - Image path: {sample['image_path']}")
        print(f"   - Depth path: {sample['depth_path']}")
        print(f"   - Max depth: {sample['max_depth']}")
        print(f"   - Source type: {sample['source_type']}")

    except Exception as e:
        print(f"   ✗ 样本加载失败: {e}")
        import traceback
        traceback.print_exc()
        return False

    # 测试加载多个样本
    print(f"\n3. 测试加载前 5 个样本...")
    try:
        num_samples = min(5, len(dataset))
        for i in range(num_samples):
            sample = dataset[i]
            valid_ratio = sample['valid_mask'].sum().item() / sample['valid_mask'].numel()
            depth_mean = sample['depth'][sample['valid_mask']].mean().item() if sample['valid_mask'].any() else 0.0

            print(f"   样本 {i}:")
            print(f"     路径: {os.path.basename(sample['image_path'])}")
            print(f"     有效像素比例: {valid_ratio:.2%}")
            print(f"     平均深度: {depth_mean:.4f} m")

        print(f"   ✓ 所有样本加载成功!")

    except Exception as e:
        print(f"   ✗ 加载多个样本时失败: {e}")
        import traceback
        traceback.print_exc()
        return False

    print(f"\n" + "=" * 80)
    print("✓ 所有测试通过!")
    print("=" * 80)
    return True


def main():
    parser = argparse.ArgumentParser(
        description="测试 Hamlyn 数据集加载"
    )

    parser.add_argument(
        "--base_dir",
        type=str,
        required=True,
        help="Hamlyn 数据集的根目录"
    )

    parser.add_argument(
        "--filelist",
        type=str,
        required=True,
        help="文件列表路径 (train.txt)"
    )

    args = parser.parse_args()

    # 检查路径是否存在
    if not os.path.exists(args.base_dir):
        print(f"错误: 数据集根目录不存在: {args.base_dir}")
        sys.exit(1)

    if not os.path.exists(args.filelist):
        print(f"错误: 文件列表不存在: {args.filelist}")
        sys.exit(1)

    # 运行测试
    success = test_dataset_loading(args.base_dir, args.filelist)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
