#!/usr/bin/env python3
"""
语义分割评估脚本
- GT: 使用 SegCacheDataset 加载
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

from multitask.dataset.cache_utils import SegCacheDataset
from multitask.util.metric import SegMetric
from multitask.util.utils import init_log
from eval.eval_utils import PredPtDataset

def main():
    parser = argparse.ArgumentParser(description="Segmentation Evaluation Script")
    parser.add_argument("--pred-txt", type=str, required=True, help="Path to the txt file containing prediction .pt file paths.")
    parser.add_argument("--gt-cache", type=str, required=True, help="Path to the ground truth segmentation cache file.")
    parser.add_argument("--num-classes", type=int, required=True, help="Number of segmentation classes.")
    
    args = parser.parse_args()

    logger = init_log("eval_seg", logging.INFO)
    logger.propagate = 0
    logger.info("\n{}\n".format(pprint.pformat(vars(args))))

    # 1. 加载数据集
    try:
        pred_dataset = PredPtDataset(args.pred_txt)
        gt_dataset = SegCacheDataset(args.gt_cache)
    except FileNotFoundError as e:
        logger.error(e)
        return

    if len(pred_dataset) != len(gt_dataset):
        logger.error(f"Prediction and GT dataset sizes do not match! Pred: {len(pred_dataset)}, GT: {len(gt_dataset)}")
        return

    pred_loader = DataLoader(pred_dataset, batch_size=1, shuffle=False, num_workers=4)
    gt_loader = DataLoader(gt_dataset, batch_size=1, shuffle=False, num_workers=4)

    # 2. 评估
    metric_seg = SegMetric(args.num_classes)
    
    for i, (pred_tensor, gt_batch) in enumerate(zip(pred_loader, gt_loader)):
        
        # pred_tensor 应该是 [num_classes, H, W] 的 logits 或 [H, W] 的 class indices
        # 我们假设它是 class indices, 因为这更常见于后处理保存的结果
        pred = pred_tensor.squeeze() # 假设pt文件中存的是单个tensor
        target_gt = gt_batch["semseg_mask"].squeeze() # [1, H, W] -> [H, W]
        
        if pred.shape != target_gt.shape:
            logger.warning(f"Shape mismatch at index {i}. Pred: {pred.shape}, GT: {target_gt.shape}. Skipping.")
            continue
        
        # SegMetric 需要 B, H, W 格式, 所以我们给它加上batch维度
        metric_seg.update(pred.unsqueeze(0), target_gt.unsqueeze(0))

    # 3. 计算并打印结果
    seg_scores = metric_seg.get_scores()
    
    logger.info("================== Evaluation Results ==================")
    logger.info(f"Evaluated {len(pred_dataset)} samples.")
    
    # 打印关键指标
    logger.info(f"  Mean IoU: {seg_scores.get('Mean IoU', 0):.4f}")
    logger.info(f"  Pixel Accuracy: {seg_scores.get('pixel_accuracy', 0):.4f}")
    logger.info(f"  Mean Precision: {seg_scores.get('mean_precision', 0):.4f}")
    logger.info(f"  Mean Recall: {seg_scores.get('mean_recall', 0):.4f}")
    
    # 打印每个类别的IoU
    logger.info("  Per-class IoU:")
    for class_idx in range(args.num_classes):
        class_iou = seg_scores.get(f'Class_{class_idx}_intersection_over_union', 0)
        logger.info(f"    Class {class_idx}: {class_iou:.4f}")
        
    logger.info("========================================================")


if __name__ == "__main__":
    main()