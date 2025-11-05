#!/usr/bin/env python3
"""
数据处理模块
包含数据整理函数和数据加载器创建逻辑
"""

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from typing import Dict, Any, List
from torch.utils.data import ConcatDataset

from ..dataset.cache_utils import DepthCacheDataset, SegCacheDataset
from .config import TrainingConfig


def collate_fn_multitask(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    多任务数据整理函数
    
    Args:
        batch: 批次数据列表，每个元素包含image、depth（可选）、semseg_mask（可选）等字段
        
    Returns:
        整理后的批次数据字典
    """
    max_h = max(item["image"].shape[-2] for item in batch)
    max_w = max(item["image"].shape[-1] for item in batch)

    # 确保尺寸是14的倍数
    stride = 14
    if max_h % stride != 0:
        max_h = max_h + (stride - max_h % stride)
    if max_w % stride != 0:
        max_w = max_w + (stride - max_w % stride)

    images, depths, masks, max_depths, source_types = [], [], [], [], []

    for item in batch:
        image = item["image"]
        h, w = image.shape[-2:]

        pad_h = max_h - h
        pad_w = max_w - w

        # 填充图像
        padded_image = F.pad(image, (0, pad_w, 0, pad_h), mode="constant", value=0)
        images.append(padded_image)

        # 记录数据源类型
        if "source_type" in item:
            source_types.append(item["source_type"])

        # 填充深度（如果存在）
        if "depth" in item:
            depth = item["depth"]
            if depth.dim() == 2:
                depth = depth.unsqueeze(0)
            if depth.dim() == 3 and depth.shape[0] != 1:
                depth = depth[0:1]

            padded_depth = F.pad(depth, (0, pad_w, 0, pad_h), mode="constant", value=0)
            depths.append(padded_depth)
            max_depths.append(item.get('max_depth', 1.0))

        # 填充分割掩码（如果存在）
        if "semseg_mask" in item:
            mask = item["semseg_mask"]
            if mask.dim() == 3:
                mask = mask.squeeze(0) if mask.shape[0] == 1 else mask[0]
            elif mask.dim() == 4:
                mask = mask.squeeze()

            ignore_idx = 255
            padded_mask = F.pad(mask, (0, pad_w, 0, pad_h), mode="constant", value=ignore_idx)
            masks.append(padded_mask)

    result = {"image": torch.stack(images)}

    if source_types:
        result["source_type"] = source_types

    if depths:
        depth_tensor = torch.stack(depths)
        if depth_tensor.dim() == 3:
            depth_tensor = depth_tensor.unsqueeze(1)
        result["depth"] = depth_tensor
        result["max_depth"] = torch.tensor(max_depths)

    if masks:
        mask_tensor = torch.stack(masks)
        if mask_tensor.dim() == 4:
            mask_tensor = mask_tensor.squeeze(1)
        result["semseg_mask"] = mask_tensor

    return result


def create_datasets(config: TrainingConfig) -> tuple:
    """
    创建训练和验证数据集 - 支持多数据集
    
    Args:
        config: 训练配置
        
    Returns:
        (train_depth_dataset, val_depth_dataset, train_seg_dataset, val_seg_dataset)
    """
    # --- 深度数据集 ---
    train_depth_dataset = ConcatDataset([
        DepthCacheDataset("/media/ssd2t/jianfu/data/inhouse/cache/train_cache.txt", dataset_type="kidney"),
        DepthCacheDataset("/media/ssd2t/jianfu/data/endomapper_sim_sub/cache/train_cache.txt", dataset_type="colon"),
    ])

    val_depth_dataset = ConcatDataset([
        DepthCacheDataset("/media/ssd2t/jianfu/data/inhouse/cache/val_cache.txt", dataset_type="kidney"),
        DepthCacheDataset("/media/ssd2t/jianfu/data/endomapper_sim_sub/cache/val_cache.txt", dataset_type="colon"),
    ])

    # --- 分割训练数据集 ---
    train_seg_dataset = ConcatDataset([
        SegCacheDataset("/media/ssd2t/jianfu/data/seg_inhouse/cache/train_cache.txt", dataset_type="kidney"),
        SegCacheDataset("/media/ssd2t/jianfu/data/polyp/CVC-ClinicDB/cache/train_cache.txt", dataset_type="colon"),
        SegCacheDataset("/media/ssd2t/jianfu/data/polyp/bkai-igh-neopolyp/cache/train_cache.txt", dataset_type="colon"),
        SegCacheDataset("/media/ssd2t/jianfu/data/polyp/CVC-EndoScene/TrainDataset/cache/train_cache.txt", dataset_type="colon"),
        SegCacheDataset("/media/ssd2t/jianfu/data/polyp/kvasir-seg/cache/train_cache.txt", dataset_type="colon"),
    ])

    # --- 分割验证数据集 (合并) ---
    val_seg_dataset = ConcatDataset([
        SegCacheDataset("/media/ssd2t/jianfu/data/seg_inhouse/cache/val_cache.txt", dataset_type="kidney"),
        SegCacheDataset("/media/ssd2t/jianfu/data/polyp/CVC-EndoScene/ValidationDataset/cache/train_cache.txt", dataset_type="colon"),
        SegCacheDataset("/media/ssd2t/jianfu/data/polyp/ETIS-LaribPolypDB/cache/train_cache.txt", dataset_type="colon"),
    ])

    return train_depth_dataset, val_depth_dataset, train_seg_dataset, val_seg_dataset


def create_dataloaders(config: TrainingConfig, train_depth_dataset: ConcatDataset, val_depth_dataset: ConcatDataset, train_seg_dataset: ConcatDataset,
                       val_seg_dataset: ConcatDataset) -> tuple:
    """
    创建数据加载器
    
    Args:
        config: 训练配置
        train_depth_dataset: 训练深度数据集
        val_depth_dataset: 验证深度数据集
        train_seg_dataset: 训练分割数据集
        val_seg_dataset: 验证分割数据集
        
    Returns:
        (train_depth_loader, train_seg_loader, val_depth_loader, val_seg_loader)
    """
    # 为每个任务创建分布式采样器
    train_depth_sampler = torch.utils.data.distributed.DistributedSampler(train_depth_dataset)
    train_seg_sampler = torch.utils.data.distributed.DistributedSampler(train_seg_dataset)

    # 设置分割训练的批次大小（可以比深度训练更小以节省内存）
    seg_batch_size = config.seg_bs if config.seg_bs is not None else config.bs

    # 创建数据加载器, shuffle=False因为sampler会处理随机化
    train_depth_loader = DataLoader(train_depth_dataset,
                                    batch_size=config.bs,
                                    shuffle=False,
                                    num_workers=4,
                                    pin_memory=True,
                                    drop_last=True,
                                    collate_fn=collate_fn_multitask,
                                    sampler=train_depth_sampler)

    train_seg_loader = DataLoader(train_seg_dataset,
                                  batch_size=seg_batch_size,
                                  shuffle=False,
                                  num_workers=4,
                                  pin_memory=True,
                                  drop_last=True,
                                  collate_fn=collate_fn_multitask,
                                  sampler=train_seg_sampler)

    # 验证集通常不需要分布式采样或打乱
    val_depth_loader = DataLoader(val_depth_dataset, batch_size=config.val_bs, shuffle=False, num_workers=4, pin_memory=True, collate_fn=collate_fn_multitask)

    val_seg_loader = DataLoader(val_seg_dataset, batch_size=config.val_bs, shuffle=False, num_workers=4, pin_memory=True, collate_fn=collate_fn_multitask)

    return train_depth_loader, train_seg_loader, val_depth_loader, val_seg_loader


def setup_dataloaders(config: TrainingConfig) -> tuple:
    """
    一站式数据加载器设置函数
    
    Args:
        config: 训练配置
        
    Returns:
        (train_depth_loader, train_seg_loader, val_depth_loader, val_seg_loader)
    """
    # 创建数据集
    train_depth_dataset, val_depth_dataset, train_seg_dataset, val_seg_dataset = create_datasets(config)

    # 创建数据加载器
    train_depth_loader, train_seg_loader, val_depth_loader, val_seg_loader = create_dataloaders(config, train_depth_dataset, val_depth_dataset, train_seg_dataset, val_seg_dataset)

    return train_depth_loader, train_seg_loader, val_depth_loader, val_seg_loader


def log_batch_info(logger, config: TrainingConfig) -> None:
    """记录批次大小信息"""
    seg_batch_size = config.seg_bs if config.seg_bs is not None else config.bs
    logger.info(f"Using batch sizes - Depth: {config.bs}, Segmentation: {seg_batch_size}")


def get_batch_size_info(config: TrainingConfig) -> Dict[str, int]:
    """
    获取批次大小信息
    
    Args:
        config: 训练配置
        
    Returns:
        包含各种批次大小的字典
    """
    seg_batch_size = config.seg_bs if config.seg_bs is not None else config.bs

    return {'depth_batch_size': config.bs, 'seg_batch_size': seg_batch_size, 'val_batch_size': config.val_bs}
