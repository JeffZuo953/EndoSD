#!/usr/bin/env python3
"""
语义分割评估脚本
- GT: 从PNG图片目录加载
- Pred: 从PNG图片目录加载

python eval/eval_seg_png_new.py --gt-dir /media/ExtHDD1/jianfu/data/inference_multitask_seg/gt_vis --pred-dir /media/ExtHDD1/jianfu/data/inference_multitask_seg/checkpoint_epoch_499_20250716_152905/seg/image --output-csv /media/ExtHDD1/jianfu/data/inference_multitask_seg/checkpoint_epoch_499_20250716_152905/seg/evaluation_results.csv
"""

import sys
import os
import argparse
import numpy as np
import torch
from PIL import Image
import logging
import pprint
import glob
from tqdm import tqdm
import csv

# 添加父目录到Python路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from multitask.util.utils import init_log

class SimpleSegMetric:
    """
    一个简单的、基于NumPy的分割指标计算器。
    """
    def __init__(self, num_classes: int):
        self.num_classes = num_classes
        self.confusion_matrix = np.zeros((num_classes, num_classes), dtype=np.int64)

    def update(self, gt: np.ndarray, pred: np.ndarray):
        """
        使用GT和Pred更新混淆矩阵。
        gt和pred应该是已经映射到类别索引的2D NumPy数组。
        """
        mask = (gt >= 0) & (gt < self.num_classes)
        hist = np.bincount(
            self.num_classes * gt[mask].astype(int) + pred[mask].astype(int),
            minlength=self.num_classes**2
        ).reshape(self.num_classes, self.num_classes)
        self.confusion_matrix += hist

    def get_scores(self) -> dict:
        """
        从混淆矩阵计算所有指标。
        """
        # 使用 float32 以匹配 PyTorch 的默认精度
        hist = self.confusion_matrix.astype(np.float32)
        
        # 全局指标
        pixel_accuracy = np.diag(hist).sum() / hist.sum()
        
        # 各类别指标
        iou = np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist))
        dice = 2 * np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0))
        precision = np.diag(hist) / hist.sum(axis=0)
        recall = np.diag(hist) / hist.sum(axis=1)
        
        # 处理NaN值 (当分母为0时)
        iou = np.nan_to_num(iou)
        dice = np.nan_to_num(dice)
        precision = np.nan_to_num(precision)
        recall = np.nan_to_num(recall)
        
        # 平均指标
        mean_iou = np.mean(iou)
        mean_dice = np.mean(dice)
        
        scores = {
            'pixel_accuracy': pixel_accuracy,
            'Mean IoU': mean_iou,
            'Mean Dice': mean_dice,
        }
        
        for i in range(self.num_classes):
            scores[f'Class_{i}_intersection_over_union'] = iou[i]
            scores[f'Class_{i}_dice'] = dice[i]
            scores[f'Class_{i}_precision'] = precision[i]
            scores[f'Class_{i}_recall'] = recall[i]
            
        return scores

def map_pixels_to_classes(img_array: np.ndarray) -> np.ndarray:
    """
    将灰度图像素值映射到类别索引。
    类别0: 黑色 (0)
    类别1: 白色 (255)
    类别2: 灰色 (128)
    """
    class_map = np.full(img_array.shape, -1, dtype=np.int64)
    class_map[img_array == 0] = 0    # Black -> Class 0
    class_map[img_array == 255] = 1  # White -> Class 1
    class_map[img_array == 128] = 2  # Gray -> Class 2

    # 检查是否有未映射的像素值
    if np.any(class_map == -1):
        unique_values = np.unique(img_array)
        unmapped_values = [v for v in unique_values if v not in [0, 128, 255]]
        if unmapped_values:
            logging.warning(f"图像中包含未预期的像素值: {unmapped_values}。这些值将被忽略(映射为类别0)。")
            class_map[class_map == -1] = 0 # 将未映射的值归为背景类

    return class_map

def main():
    parser = argparse.ArgumentParser(description="Segmentation Evaluation Script from PNGs")
    parser.add_argument("--pred-dir", type=str, required=True, help="Path to the directory containing prediction PNG files.")
    parser.add_argument("--gt-dir", type=str, required=True, help="Path to the directory containing ground truth PNG files.")
    parser.add_argument("--num-classes", type=int, default=3, help="Number of segmentation classes. Default: 3")
    parser.add_argument("--output-csv", type=str, help="Path to save the evaluation results in CSV format.")
    
    args = parser.parse_args()

    logger = init_log("eval_seg_png", logging.INFO)
    logger.propagate = 0
    logger.info("\n{}\n".format(pprint.pformat(vars(args))))

    # 1. 查找预测图片
    pred_files = sorted(glob.glob(os.path.join(args.pred_dir, '*.png')))
    if not pred_files:
        logger.error(f"在预测目录中未找到PNG文件: {args.pred_dir}")
        return
    
    logger.info(f"找到 {len(pred_files)} 个预测图像。")

    # 2. 评估
    metric_seg = SimpleSegMetric(args.num_classes)
    
    evaluated_samples = 0
    for pred_path in tqdm(pred_files, desc="正在评估"):
        base_name = os.path.basename(pred_path)
        gt_path = os.path.join(args.gt_dir, base_name)
        
        if not os.path.exists(gt_path):
            logger.warning(f"未找到 {base_name} 对应的真值文件，已跳过。")
            continue
        
        try:
            # 加载 pred 和 gt 图像
            pred_img = Image.open(pred_path).convert('L') # 转换为灰度图
            gt_img = Image.open(gt_path).convert('L')
            
            pred_np = np.array(pred_img)
            gt_np = np.array(gt_img)
            
            if pred_np.shape != gt_np.shape:
                logger.warning(f"图像尺寸不匹配 {base_name}. Pred: {pred_np.shape}, GT: {gt_np.shape}. 已跳过。")
                continue
            
            # 像素值映射到类别索引
            pred_classes = map_pixels_to_classes(pred_np)
            gt_classes = map_pixels_to_classes(gt_np)
            
            # 使用新的Metric类进行更新
            metric_seg.update(gt_classes, pred_classes)
            evaluated_samples += 1

        except Exception as e:
            logger.error(f"处理文件 {base_name} 时出错: {e}")
            continue

    # 3. 计算并打印结果
    seg_scores = metric_seg.get_scores()
    
    logger.info("================== 评估结果 ==================")
    logger.info(f"评估了 {evaluated_samples} 个样本。")
    
    # 打印关键指标
    logger.info(f"  像素准确率 (Pixel Accuracy): {seg_scores.get('pixel_accuracy', 0):.4f}")
    logger.info(f"  平均交并比 (Mean IoU): {seg_scores.get('Mean IoU', 0):.4f}")
    logger.info(f"  平均Dice系数 (Mean Dice): {seg_scores.get('Mean Dice', 0):.4f}")
    
    # 打印每个类别的指标
    logger.info("  各类别指标:")
    headers = ["类别", "IoU", "Dice", "Precision", "Recall"]
    rows = []
    
    class_names = {0: "0 (黑色)", 1: "1 (白色)", 2: "2 (灰色)"}

    for class_idx in range(args.num_classes):
        class_name = class_names.get(class_idx, str(class_idx))
        iou = seg_scores.get(f'Class_{class_idx}_intersection_over_union', 0)
        dice = seg_scores.get(f'Class_{class_idx}_dice', 0)
        precision = seg_scores.get(f'Class_{class_idx}_precision', 0)
        recall = seg_scores.get(f'Class_{class_idx}_recall', 0)
        rows.append([class_name, f"{iou:.4f}", f"{dice:.4f}", f"{precision:.4f}", f"{recall:.4f}"])
    
    # 为了对齐
    col_widths = [max(len(str(item)) for item in col) for col in zip(headers, *rows)]
    header_line = " | ".join(header.ljust(width) for header, width in zip(headers, col_widths))
    logger.info(f"    {header_line}")
    logger.info(f"    {'-' * len(header_line)}")
    for row in rows:
        row_line = " | ".join(item.ljust(width) for item, width in zip(row, col_widths))
        logger.info(f"    {row_line}")
    
    # 4. 保存结果到CSV (如果提供了路径)
    if args.output_csv:
        try:
            with open(args.output_csv, 'w', newline='', encoding='utf-8') as csvfile:
                writer = csv.writer(csvfile)
                
                # 写入总体指标
                writer.writerow(['Metric', 'Value'])
                writer.writerow(['Evaluated Samples', evaluated_samples])
                writer.writerow(['Pixel Accuracy', seg_scores.get('pixel_accuracy', 0)])
                writer.writerow(['Mean IoU', seg_scores.get('Mean IoU', 0)])
                writer.writerow(['Mean Dice', seg_scores.get('Mean Dice', 0)])
                
                writer.writerow([]) # 空行
                
                # 写入每个类别的指标
                writer.writerow(headers)
                
                for class_idx in range(args.num_classes):
                    class_name = class_names.get(class_idx, str(class_idx))
                    iou = seg_scores.get(f'Class_{class_idx}_intersection_over_union', 0)
                    dice = seg_scores.get(f'Class_{class_idx}_dice', 0)
                    precision = seg_scores.get(f'Class_{class_idx}_precision', 0)
                    recall = seg_scores.get(f'Class_{class_idx}_recall', 0)
                    writer.writerow([class_name, iou, dice, precision, recall])
            
            logger.info(f"结果已无损保存到: {args.output_csv}")

        except Exception as e:
            logger.error(f"保存CSV文件失败: {e}")
        
    logger.info("========================================================")


if __name__ == "__main__":
    main()