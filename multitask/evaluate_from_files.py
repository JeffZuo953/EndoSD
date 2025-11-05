#!/usr/bin/env python3
"""
从文件列表评估多任务模型性能的脚本。
"""

import argparse
import torch
import numpy as np
import os
import sys
from tqdm import tqdm

# 添加父目录以便导入
sys.path.insert(0, os.path.dirname(__file__))
from util.metric import eval_depth, SegMetric

def find_gt_file(pred_filename, gt_files):
    """根据预测文件名在真实数据列表中查找对应的文件。"""
    # 预测文件名格式: <原始路径中的斜杠替换为下划线>.pt
    # 我们需要将它转换回可能的文件名来进行匹配
    base_name = os.path.basename(pred_filename).replace('.pt', '')
    
    # 尝试几种可能的匹配模式
    possible_matches = [
        base_name,
        base_name.replace('_', '/')
    ]
    
    for p in possible_matches:
        for gt_file in gt_files:
            if p in gt_file:
                return gt_file
    return None

def main():
    parser = argparse.ArgumentParser(description="从文件列表评估多任务模型")
    
    # 输入文件列表
    parser.add_argument("--pred-list", type=str, required=True, help="预测结果 .pt 文件的列表 (.txt)")
    parser.add_argument("--gt-list", type=str, required=True, help="真实数据 .pt 文件的列表 (.txt)")
    parser.add_argument("--task", type=str, required=True, choices=["depth", "seg"], help="要评估的任务")
    
    # 分割任务相关参数
    parser.add_argument("--num-classes", type=int, default=4, help="分割任务的类别数")

    args = parser.parse_args()

    # 1. 读取文件列表
    with open(args.pred_list, 'r') as f:
        pred_files = [line.strip() for line in f.readlines()]
    with open(args.gt_list, 'r') as f:
        gt_files = [line.strip() for line in f.readlines()]

    if not pred_files:
        print("错误: 预测文件列表为空。")
        return

    # 2. 根据任务进行评估
    if args.task == "depth":
        all_metrics = []
        for pred_file in tqdm(pred_files, desc="评估深度估计"):
            gt_file = find_gt_file(pred_file, gt_files)
            if not gt_file:
                print(f"警告: 找不到与 {pred_file} 匹配的真实数据文件，跳过。")
                continue

            pred_depth = torch.load(pred_file, map_location="cpu").squeeze()
            gt_data = torch.load(gt_file, map_location="cpu")
            gt_depth = gt_data["depth"].squeeze()

            # 创建有效值掩码
            valid_mask = (gt_depth > 0)
            
            if torch.any(valid_mask):
                metrics = eval_depth(pred_depth[valid_mask], gt_depth[valid_mask])
                all_metrics.append(metrics)

        if all_metrics:
            # 计算平均指标
            avg_metrics = {k: np.mean([d[k] for d in all_metrics]) for k in all_metrics[0].keys()}
            print("\n--- 深度估计评估结果 ---")
            for k, v in avg_metrics.items():
                print(f"  {k}: {v:.4f}")
            print("-------------------------\n")
        else:
            print("没有可用于评估的有效深度图。")

    elif args.task == "seg":
        seg_metric = SegMetric(args.num_classes)
        for pred_file in tqdm(pred_files, desc="评估语义分割"):
            gt_file = find_gt_file(pred_file, gt_files)
            if not gt_file:
                print(f"警告: 找不到与 {pred_file} 匹配的真实数据文件，跳过。")
                continue

            pred_seg_logits = torch.load(pred_file, map_location="cpu")
            pred_seg = torch.argmax(pred_seg_logits, dim=0).squeeze()
            
            gt_data = torch.load(gt_file, map_location="cpu")
            gt_seg = gt_data["semseg_mask"].squeeze().long()
            
            seg_metric.update(pred_seg, gt_seg)

        scores = seg_metric.get_scores()
        print("\n--- 语义分割评估结果 ---")
        print(f"  mIoU: {scores.get('miou', 0):.4f}")
        print(f"  mDice: {scores.get('mdice', 0):.4f}")
        print(f"  Overall Acc: {scores.get('acc_overall', 0):.4f}")
        print("-------------------------\n")

if __name__ == "__main__":
    main()