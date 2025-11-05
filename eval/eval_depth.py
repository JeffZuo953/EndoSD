#!/usr/bin/env python3
"""
深度估计评估脚本
- GT: 使用 DepthCacheDataset 加载
- Pred: 使用 PredPtDataset 加载，输入为包含pt文件路径的txt
"""

import sys
import os
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader
import logging
import pprint

# 添加父目录到Python路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from multitask.dataset.cache_utils import DepthCacheDataset
from multitask.util.metric import eval_depth
from multitask.util.utils import init_log
from eval.eval_utils import PredPtDataset

def main():
    parser = argparse.ArgumentParser(description="Depth Estimation Evaluation Script")
    parser.add_argument("--pred-txt", type=str, required=True, help="Path to the txt file containing prediction .pt file paths.")
    parser.add_argument("--gt-cache", type=str, required=True, help="Path to the ground truth depth cache file.")
    parser.add_argument("--min-depth", default=1e-6, type=float, help="Minimum depth value for evaluation.")
    parser.add_argument("--max-depth", default=0.2, type=float, help="Maximum depth value for evaluation.")
    
    args = parser.parse_args()

    logger = init_log("eval_depth", logging.INFO)
    logger.propagate = 0
    logger.info("\n{}\n".format(pprint.pformat(vars(args))))

    # 1. 加载数据集
    try:
        pred_dataset = PredPtDataset(args.pred_txt)
        gt_dataset = DepthCacheDataset(args.gt_cache)
    except FileNotFoundError as e:
        logger.error(e)
        return

    if len(pred_dataset) != len(gt_dataset):
        logger.error(f"Prediction and GT dataset sizes do not match! Pred: {len(pred_dataset)}, GT: {len(gt_dataset)}")
        return

    pred_loader = DataLoader(pred_dataset, batch_size=1, shuffle=False, num_workers=4)
    gt_loader = DataLoader(gt_dataset, batch_size=1, shuffle=False, num_workers=4)

    # 2. 评估
    metric_depth_val = []
    
    for i, (pred_tensor, gt_batch) in enumerate(zip(pred_loader, gt_loader)):
        
        pred = pred_tensor.squeeze() # 假设pt文件中存的是单个tensor
        target_gt = gt_batch["depth"].squeeze() # [1, 1, H, W] -> [H, W]
        
        if pred.shape != target_gt.shape:
            logger.warning(f"Shape mismatch at index {i}. Pred: {pred.shape}, GT: {target_gt.shape}. Skipping.")
            continue

        # 使用与训练时一致的有效掩码逻辑
        valid_mask = (target_gt > 0) & (target_gt >= args.min_depth) & (target_gt <= args.max_depth)
        
        if torch.any(valid_mask):
            metric_depth_val.append(eval_depth(pred[valid_mask], target_gt[valid_mask]))
        else:
            logger.warning(f"No valid pixels found for sample {i}. Skipping.")

    # 3. 计算并打印结果
    if not metric_depth_val:
        logger.error("No samples were evaluated. Exiting.")
        return
        
    keys = metric_depth_val[0].keys()
    avg_metrics = {k: np.mean([d[k] for d in metric_depth_val]) for k in keys}

    logger.info("================== Evaluation Results ==================")
    logger.info(f"Evaluated {len(metric_depth_val)} samples.")
    for k, v in avg_metrics.items():
        logger.info(f"  {k}: {v:.6f}")
    logger.info("========================================================")


if __name__ == "__main__":
    main()