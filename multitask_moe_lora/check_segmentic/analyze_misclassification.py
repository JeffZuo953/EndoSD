#!/usr/bin/env python3
"""
语义分割误分类分析脚本

该脚本用于评估一个训练好的多任务模型在特定数据集上对其他任务类别的误分类情况。
例如，在 'kidney' 数据集上检测模型是否错误地预测了 'colon' 相关的类别。

主要功能:
1. 加载指定的模型检查点。
2. 在指定的数据集（如 'kidney' 或 'colon'）上进行推理。
3. 统计特定“外来”类别的误预测情况。
4. 计算并输出一系列指标来量化误分类的严重程度。
"""

import os
import argparse
import torch
import numpy as np
from tqdm import tqdm
from typing import Dict, List, Any


from ..util.config import TrainingConfig, args_to_config, create_parser
from ..util.model_setup import setup_complete_model
from ..util.data_utils import setup_dataloaders
from ..util.train_utils import setup_training_environment, cleanup_distributed


class MisclassificationAnalyzer:
    """
    误分类分析器 (针对特定数据集类型)
    """
    def __init__(self, dataset_name: str, target_labels: List[int], area_threshold: int):
        self.dataset_name = dataset_name
        self.target_labels = target_labels
        self.area_threshold = area_threshold
        
        # 初始化统计变量
        self.total_images = 0
        self.images_with_misclassification = 0
        self.images_with_large_area_misclassification = 0
        self.total_pixels = 0
        self.total_misclassified_pixels = 0

    def analyze_sample(self, pred_mask: torch.Tensor):
        """
        分析单个样本的预测掩码
        """
        self.total_images += 1
        image_pixels = pred_mask.numel()
        self.total_pixels += image_pixels
        
        misclassified_pixels_mask = torch.zeros_like(pred_mask, dtype=torch.bool)
        for label in self.target_labels:
            misclassified_pixels_mask |= (pred_mask == label)
        
        num_misclassified = torch.sum(misclassified_pixels_mask).item()
        
        if num_misclassified > 0:
            self.images_with_misclassification += 1
            self.total_misclassified_pixels += num_misclassified
            
            if num_misclassified > self.area_threshold:
                self.images_with_large_area_misclassification += 1

    def report(self) -> Dict[str, Any]:
        """
        生成并打印分析报告
        """
        report_data = {
            "dataset_name": self.dataset_name,
            "total_images": self.total_images,
            "images_with_misclassification": self.images_with_misclassification,
            "image_misclassification_ratio": (self.images_with_misclassification / self.total_images) if self.total_images > 0 else 0,
            
            "total_pixels": self.total_pixels,
            "total_misclassified_pixels": self.total_misclassified_pixels,
            "pixel_misclassification_ratio": (self.total_misclassified_pixels / self.total_pixels) if self.total_pixels > 0 else 0,
            
            "images_with_large_area_misclassification": self.images_with_large_area_misclassification,
            "large_area_image_ratio": (self.images_with_large_area_misclassification / self.total_images) if self.total_images > 0 else 0,
            
            "target_labels": self.target_labels,
            "area_threshold": self.area_threshold,
        }
        
        print(f"\n--- Misclassification Report for '{self.dataset_name.upper()}' Dataset ---")
        print(f"Analyzed {report_data['total_images']} images.")
        print(f"Targeting misclassified labels: {report_data['target_labels']}")
        print("-" * 50)
        print(f"Image Ratio with Errors: {report_data['image_misclassification_ratio']:.4f} ({report_data['images_with_misclassification']} / {report_data['total_images']})")
        print(f"Pixel Ratio of Errors:   {report_data['pixel_misclassification_ratio']:.6f} ({report_data['total_misclassified_pixels']} / {report_data['total_pixels']})")
        print(f"Large Area Image Ratio:  {report_data['large_area_image_ratio']:.4f} ({report_data['images_with_large_area_misclassification']} / {report_data['total_images']}) (Threshold > {self.area_threshold} pixels)")
        print("-" * 50)
        
        return report_data


def main():
    """主执行函数"""
    parser = create_parser()
    # 添加此脚本特定的参数
    parser.add_argument("--area-threshold", type=int, default=100, help="Pixel count threshold for 'large area' misclassification.")
    
    args = parser.parse_args()
    config = args_to_config(args)

    try:
        # 1. 设置分布式环境
        rank, world_size, logger, writer = setup_training_environment(config)
        
        # 2. 加载模型
        if not config.resume_from:
            raise ValueError("--resume-from argument is required to specify the model checkpoint.")
        
        logger.info(f"Loading model from: {config.resume_from}")
        setup_bundle = setup_complete_model(config, logger=logger)
        model = setup_bundle["model"]
        device = torch.device(f"cuda:{rank}")
        model.to(device)
        model.eval()
        
        # 3. 设置数据加载器
        logger.info(f"Loading data using dataset config: '{config.dataset_config_name}'")
        _, _, _, val_seg_loader = setup_dataloaders(config)
        
        # 4. 初始化两个分析器
        analyzers = {
            'kidney': MisclassificationAnalyzer(
                dataset_name='kidney',
                target_labels=[3],  # 在 KİDNEY (结石) 数据上，寻找外来的 POLYP (3) 标签
                area_threshold=args.area_threshold
            ),
            'colon': MisclassificationAnalyzer(
                dataset_name='colon',
                target_labels=[1, 2],  # 在 COLON (息肉) 数据上，寻找外来的 STONE (1) 和 LASER (2) 标签
                area_threshold=args.area_threshold
            )
        }

        # 5. 遍历数据并分发给分析器
        logger.info("Starting analysis...")
        with torch.no_grad():
            for batch in tqdm(val_seg_loader, desc="Analyzing All Datasets"):
                images = batch['image'].to(device)
                source_types = batch.get("source_type", [])
                
                outputs = model(images, task='seg')
                preds = torch.argmax(outputs['seg'], dim=1)
                
                for i in range(preds.shape[0]):
                    sample_type = source_types[i] if i < len(source_types) else None
                    if sample_type in analyzers:
                        analyzers[sample_type].analyze_sample(preds[i])

        # 6. 打印所有报告
        for analyzer in analyzers.values():
            analyzer.report()

    finally:
        # 清理分布式环境
        cleanup_distributed()


if __name__ == "__main__":
    main()
