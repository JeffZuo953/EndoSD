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
from ..dataset.endonerf import EndonerfDataset
from ..dataset.hamlyn import HamlynDataset
from ..dataset.kidney3d import Kidney3DDataset
from ..dataset.stereomis import StereoMISDataset
from .config import TrainingConfig


def collate_fn_multitask(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    多任务数据整理函数
    0
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
    intrinsics_list = []

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

        if "intrinsics" in item:
            intrinsics_list.append(item["intrinsics"])

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

    if intrinsics_list:
        intrinsics_tensor = torch.stack([intr.to(torch.float32) for intr in intrinsics_list])
        result["intrinsics"] = intrinsics_tensor

    return result


def _make_collate_fn(stride: int):
    """Pad batch H/W to multiple of `stride` (e.g., 14 for DINOv2, 16 for DINOv3)."""
    def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        max_h = max(item["image"].shape[-2] for item in batch)
        max_w = max(item["image"].shape[-1] for item in batch)

        if max_h % stride != 0:
            max_h = max_h + (stride - max_h % stride)
        if max_w % stride != 0:
            max_w = max_w + (stride - max_w % stride)

        images, depths, masks, max_depths, source_types = [], [], [], [], []
        intrinsics_list = []
        for item in batch:
            image = item["image"]
            h, w = image.shape[-2:]
            pad_h = max_h - h
            pad_w = max_w - w
            images.append(F.pad(image, (0, pad_w, 0, pad_h), mode="constant", value=0))

            if "source_type" in item:
                source_types.append(item["source_type"])

            if "depth" in item:
                depth = item["depth"]
                if depth.dim() == 2:
                    depth = depth.unsqueeze(0)
                if depth.dim() == 3 and depth.shape[0] != 1:
                    depth = depth[0:1]
                depths.append(F.pad(depth, (0, pad_w, 0, pad_h), mode="constant", value=0))
                max_depths.append(item.get('max_depth', 1.0))

            if "semseg_mask" in item:
                mask = item["semseg_mask"]
                if mask.dim() == 3:
                    mask = mask.squeeze(0) if mask.shape[0] == 1 else mask[0]
                elif mask.dim() == 4:
                    mask = mask.squeeze()
                ignore_idx = 255
                masks.append(F.pad(mask, (0, pad_w, 0, pad_h), mode="constant", value=ignore_idx))
            if "intrinsics" in item:
                intrinsics_list.append(item["intrinsics"])

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
        if intrinsics_list:
            intrinsics_tensor = torch.stack([intr.to(torch.float32) for intr in intrinsics_list])
            result["intrinsics"] = intrinsics_tensor
        return result

    return collate_fn


def create_datasets(config: TrainingConfig) -> tuple:
    """
    创建训练和验证数据集 - 支持多数据集
    
    Args:
        config: 训练配置
        
    Returns:
        (train_depth_dataset, val_depth_dataset, train_seg_dataset, val_seg_dataset)
    """
    # ==============================================================================
    #  数据集路径配置
    #  通过切换 ACTIVE_DATASET 变量来选择不同的数据集路径配置
    # ==============================================================================

    # 从配置中获取数据集和转换的名称
    active_dataset = config.dataset_config_name
    active_transform = config.path_transform_name

    DATASET_PATHS = {
        'server_sz': {
            "depth_train_inhouse": "/media/ssd2t/jianfu/data/inhouse/cache/train_cache.txt",
            "depth_train_endomapper": "/media/ssd2t/jianfu/data/endomapper_sim_sub/cache/train_cache.txt",
            "depth_val_inhouse": "/media/ssd2t/jianfu/data/inhouse/cache/val_cache.txt",
            "depth_val_endomapper": "/media/ssd2t/jianfu/data/endomapper_sim_sub/cache/val_cache.txt",
            "seg_train_inhouse": "/media/ssd2t/jianfu/data/seg_inhouse/cache/train_cache.txt",
            "seg_train_cvc_clinicdb": "/media/ssd2t/jianfu/data/polyp/CVC-ClinicDB/cache/train_cache.txt",
            "seg_train_bkai": "/media/ssd2t/jianfu/data/polyp/bkai-igh-neopolyp/cache/train_cache.txt",
            "seg_train_cvc_endoscene": "/media/ssd2t/jianfu/data/polyp/CVC-EndoScene/TrainDataset/cache/train_cache.txt",
            "seg_train_kvasir": "/media/ssd2t/jianfu/data/polyp/kvasir-seg/cache/train_cache.txt",
            "seg_val_inhouse": "/media/ssd2t/jianfu/data/seg_inhouse/cache/val_cache.txt",
            "seg_val_cvc_endoscene": "/media/ssd2t/jianfu/data/polyp/CVC-EndoScene/ValidationDataset/cache/train_cache.txt",
            "seg_val_etis": "/media/ssd2t/jianfu/data/polyp/ETIS-LaribPolypDB/cache/train_cache.txt",
        },
        'server_hk_01': {
            "depth_train_inhouse": "/data/ziyi/multitask/data/inhouse/media/ssd2t/jianfu/data/inhouse/cache/train_cache.txt",
            "depth_train_endomapper": "/data/ziyi/multitask/data/endomapper_sim_sub/cache/train_cache.txt",
            "depth_val_inhouse": "/data/ziyi/multitask/data/inhouse/media/ssd2t/jianfu/data/inhouse/cache/val_cache.txt",
            "depth_val_endomapper": "/data/ziyi/multitask/data/endomapper_sim_sub/cache/val_cache.txt",
            "seg_train_inhouse": "/data/ziyi/multitask/data/seg_inhouse/media/ssd2t/jianfu/data/seg_inhouse/cache/train_cache.txt",
            "seg_train_cvc_clinicdb": "/data/ziyi/multitask/data/clinicDB/cache/train_cache.txt",
            "seg_train_bkai": "/data/ziyi/multitask/data/bkai-igh-neopolyp/cache/train_cache.txt",
            "seg_train_cvc_endoscene": "/data/ziyi/multitask/data/TrainDataset/media/ssd2t/jianfu/data/polyp/CVC-EndoScene/TrainDataset/cache/train_cache.txt",
            "seg_train_kvasir": "/data/ziyi/multitask/data/kvasir-seg/media/ssd2t/jianfu/data/polyp/kvasir-seg/cache/train_cache.txt",
            "seg_val_inhouse": "/data/ziyi/multitask/data/seg_inhouse/media/ssd2t/jianfu/data/seg_inhouse/cache/val_cache.txt",
            "seg_val_cvc_endoscene": "/data/ziyi/multitask/data/ValidationDataset/media/ssd2t/jianfu/data/polyp/CVC-EndoScene/ValidationDataset/cache/train_cache.txt",
            "seg_val_etis": "/data/ziyi/multitask/data/ETIS-LaribPolypDB/media/ssd2t/jianfu/data/polyp/ETIS-LaribPolypDB/cache/train_cache.txt",
        },
        'hamlyn': {
            "depth_train": "/media/ExtHDD1/jianfu/data/Hamlyn/train.txt",
            "depth_val": "/media/ExtHDD1/jianfu/data/Hamlyn/val.txt",
        },
        'endonerf': {
            "depth_train": config.endonerf_filelist, # Assume filelist contains frame_ids
            "depth_val": config.endonerf_filelist,   # Use same for val for now, or specify separate
            "rootpath": config.endonerf_rootpath,
        },
        "stereomis": {
            "train_filelist": "/media/ExtHDD1/jianfu/data/StereoMIS/train_sequences.txt",
            "val_filelist": "/media/ExtHDD1/jianfu/data/StereoMIS/val_sequences.txt",
            "root_path": "/media/ExtHDD1/jianfu/data/StereoMIS",
        },
        "kidney3d": {
            "rootpath": "/media/ExtHDD1/jianfu/data/Kidney3D-CT-depth-seg", # config.kidney3d_rootpath will override this
        }
    }

    paths = DATASET_PATHS[active_dataset]

    # 定义路径转换函数映射
    PATH_TRANSFORM_CONFIGS = {
        'sz_to_hk': {
            "depth_train_inhouse":
                lambda p: p.replace("/media/ExtHDD1/jianfu", "/data/ziyi/multitask/data/inhouse/media/ssd2t/jianfu"),
            "depth_val_inhouse":
                lambda p: p.replace("/media/ExtHDD1/jianfu", "/data/ziyi/multitask/data/inhouse/media/ssd2t/jianfu"),
            "depth_train_endomapper":
                lambda p: p.replace("/media/ssd2t/jianfu/data/endomapper_sim_sub", "/data/ziyi/multitask/data/endomapper_sim_sub"),
            "depth_val_endomapper":
                lambda p: p.replace("/media/ssd2t/jianfu/data/endomapper_sim_sub", "/data/ziyi/multitask/data/endomapper_sim_sub"),
            "seg_train_inhouse":
                lambda p: p.replace("/media/ssd2t/jianfu/data/seg_inhouse", "/data/ziyi/multitask/data/seg_inhouse/media/ssd2t/jianfu/data/seg_inhouse"),
            "seg_val_inhouse":
                lambda p: p.replace("/media/ssd2t/jianfu/data/seg_inhouse", "/data/ziyi/multitask/data/seg_inhouse/media/ssd2t/jianfu/data/seg_inhouse"),
            "seg_train_cvc_clinicdb":
                lambda p: p.replace("/media/ssd2t/jianfu/data/polyp/CVC-ClinicDB", "/data/ziyi/multitask/data/clinicDB"),
            "seg_train_bkai":
                lambda p: p.replace("/media/ssd2t/jianfu/data/polyp/bkai-igh-neopolyp", "/data/ziyi/multitask/data/bkai-igh-neopolyp"),
            "seg_train_cvc_endoscene":
                lambda p: p.replace("/media/ssd2t/jianfu/data/polyp/CVC-EndoScene/TrainDataset",
                                    "/data/ziyi/multitask/data/TrainDataset/media/ssd2t/jianfu/data/polyp/CVC-EndoScene/TrainDataset"),
            "seg_train_kvasir":
                lambda p: p.replace("/media/ssd2t/jianfu/data/polyp/kvasir-seg", "/data/ziyi/multitask/data/kvasir-seg/media/ssd2t/jianfu/data/polyp/kvasir-seg"),
            "seg_val_cvc_endoscene":
                lambda p: p.replace("/media/ssd2t/jianfu/data/polyp/CVC-EndoScene/ValidationDataset",
                                    "/data/ziyi/multitask/data/ValidationDataset/media/ssd2t/jianfu/data/polyp/CVC-EndoScene/ValidationDataset"),
            "seg_val_etis":
                lambda p: p.replace("/media/ssd2t/jianfu/data/polyp/ETIS-LaribPolypDB",
                                    "/data/ziyi/multitask/data/ETIS-LaribPolypDB/media/ssd2t/jianfu/data/polyp/ETIS-LaribPolypDB"),
        }
    }

    transform_map = PATH_TRANSFORM_CONFIGS.get(active_transform, {}) if active_transform else {}

    train_depth_datasets: List[Any] = []
    val_depth_datasets: List[Any] = []
    train_seg_datasets: List[Any] = []
    val_seg_datasets: List[Any] = []

    if active_dataset == 'hamlyn':
        if "depth_train" in paths:
            train_depth_datasets.append(HamlynDataset(paths["depth_train"], mode="train", size=(config.img_size, config.img_size), max_depth=config.max_depth))
        if "depth_val" in paths:
            val_depth_datasets.append(HamlynDataset(paths["depth_val"], mode="val", size=(config.img_size, config.img_size), max_depth=config.max_depth))
    elif active_dataset == 'endonerf':
        if "depth_train" in paths and "rootpath" in paths:
            train_depth_datasets.append(EndonerfDataset(paths["depth_train"], rootpath=paths["rootpath"], mode="train", size=(config.img_size, config.img_size), max_depth=config.max_depth))
        if "depth_val" in paths and "rootpath" in paths:
            val_depth_datasets.append(EndonerfDataset(paths["depth_val"], rootpath=paths["rootpath"], mode="val", size=(config.img_size, config.img_size), max_depth=config.max_depth))
    elif active_dataset == 'stereomis':
        if "train_filelist" in paths and "root_path" in paths:
            train_depth_datasets.append(StereoMISDataset(
                filelist_path=config.stereomis_filelist if config.stereomis_filelist else paths["train_filelist"],
                rootpath=config.stereomis_rootpath if config.stereomis_rootpath else paths["root_path"],
                mode="train",
                size=(config.img_size, config.img_size),
                max_depth=config.max_depth,
            ))
            train_seg_datasets.append(StereoMISDataset(
                filelist_path=config.stereomis_filelist if config.stereomis_filelist else paths["train_filelist"],
                rootpath=config.stereomis_rootpath if config.stereomis_rootpath else paths["root_path"],
                mode="train",
                size=(config.img_size, config.img_size),
                max_depth=config.max_depth,
            ))
        if "val_filelist" in paths and "root_path" in paths:
            val_depth_datasets.append(StereoMISDataset(
                filelist_path=config.stereomis_filelist if config.stereomis_filelist else paths["val_filelist"],
                rootpath=config.stereomis_rootpath if config.stereomis_rootpath else paths["root_path"],
                mode="val",
                size=(config.img_size, config.img_size),
                max_depth=config.max_depth,
            ))
            val_seg_datasets.append(StereoMISDataset(
                filelist_path=config.stereomis_filelist if config.stereomis_filelist else paths["val_filelist"],
                rootpath=config.stereomis_rootpath if config.stereomis_rootpath else paths["root_path"],
                mode="val",
                size=(config.img_size, config.img_size),
                max_depth=config.max_depth,
            ))
    elif active_dataset == 'kidney3d':
        if "rootpath" in paths:
            _rootpath = config.kidney3d_rootpath if config.kidney3d_rootpath else paths["rootpath"]
            train_depth_datasets.append(Kidney3DDataset(
                rootpath=_rootpath,
                mode="train",
                size=(config.img_size, config.img_size),
                max_depth=config.max_depth,
            ))
            val_depth_datasets.append(Kidney3DDataset(
                rootpath=_rootpath,
                mode="val",
                size=(config.img_size, config.img_size),
                max_depth=config.max_depth,
            ))
            # Kidney3D dataset currently does not have separate seg masks, use depth dataset for seg tasks
            train_seg_datasets.append(Kidney3DDataset(
                rootpath=_rootpath,
                mode="train",
                size=(config.img_size, config.img_size),
                max_depth=config.max_depth,
            ))
            val_seg_datasets.append(Kidney3DDataset(
                rootpath=_rootpath,
                mode="val",
                size=(config.img_size, config.img_size),
                max_depth=config.max_depth,
            ))
    elif active_dataset in ['server_sz', 'server_hk_01']:
        # --- 深度数据集 ---
        train_depth_datasets.extend([
            DepthCacheDataset(paths["depth_train_inhouse"], dataset_type="kidney", path_transform=transform_map.get("depth_train_inhouse")),
            DepthCacheDataset(paths["depth_train_endomapper"], dataset_type="colon", path_transform=transform_map.get("depth_train_endomapper")),
        ])

        # --- 深度验证数据集 (合并) ---
        val_depth_datasets.extend([
            DepthCacheDataset(paths["depth_val_inhouse"], dataset_type="kidney", path_transform=transform_map.get("depth_val_inhouse")),
            DepthCacheDataset(paths["depth_val_endomapper"], dataset_type="colon", path_transform=transform_map.get("depth_val_endomapper")),
        ])

        # --- 分割训练数据集 ---
        train_seg_datasets.extend([
            SegCacheDataset(paths["seg_train_inhouse"], dataset_type="kidney", path_transform=transform_map.get("seg_train_inhouse")),
            SegCacheDataset(paths["seg_train_cvc_clinicdb"], dataset_type="colon", path_transform=transform_map.get("seg_train_cvc_clinicdb")),
            SegCacheDataset(paths["seg_train_bkai"], dataset_type="colon", path_transform=transform_map.get("seg_train_bkai")),
            SegCacheDataset(paths["seg_train_cvc_endoscene"], dataset_type="colon", path_transform=transform_map.get("seg_train_cvc_endoscene")),
            SegCacheDataset(paths["seg_train_kvasir"], dataset_type="colon", path_transform=transform_map.get("seg_train_kvasir")),
        ])

        # --- 分割验证数据集 (合并) ---
        val_seg_datasets.extend([
            SegCacheDataset(paths["seg_val_inhouse"], dataset_type="kidney", path_transform=transform_map.get("seg_val_inhouse")),
            SegCacheDataset(paths["seg_val_cvc_endoscene"], dataset_type="colon", path_transform=transform_map.get("seg_val_cvc_endoscene")),
            SegCacheDataset(paths["seg_val_etis"], dataset_type="colon", path_transform=transform_map.get("seg_val_etis")),
        ])
    else:
        raise ValueError(f"Unknown dataset configuration name: {active_dataset}")

    train_depth_dataset = ConcatDataset(train_depth_datasets) if train_depth_datasets else None
    val_depth_dataset = ConcatDataset(val_depth_datasets) if val_depth_datasets else None
    train_seg_dataset = ConcatDataset(train_seg_datasets) if train_seg_datasets else None
    val_seg_dataset = ConcatDataset(val_seg_datasets) if val_seg_datasets else None

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

    # 动态选择 patch stride（dinov2=14，dinov3=16），并构造对应 collate_fn
    encoder = getattr(config, 'encoder', 'vits').lower()
    stride = 16 if 'dinov3' in encoder else 14
    collate = _make_collate_fn(stride)

    # 创建数据加载器, shuffle=False因为sampler会处理随机化
    train_depth_loader = DataLoader(train_depth_dataset,
                                    batch_size=config.bs,
                                    shuffle=False,
                                    num_workers=4,
                                    pin_memory=True,
                                    drop_last=True,
                                    collate_fn=collate,
                                    sampler=train_depth_sampler)

    train_seg_loader = DataLoader(train_seg_dataset,
                                  batch_size=seg_batch_size,
                                  shuffle=False,
                                  num_workers=4,
                                  pin_memory=True,
                                  drop_last=True,
                                  collate_fn=collate,
                                  sampler=train_seg_sampler)

    # 创建验证加载器
    val_depth_sampler = torch.utils.data.distributed.DistributedSampler(val_depth_dataset, shuffle=False)
    val_seg_sampler = torch.utils.data.distributed.DistributedSampler(val_seg_dataset, shuffle=False)
    
    val_depth_loader = DataLoader(val_depth_dataset, batch_size=config.val_bs, shuffle=False, num_workers=4, pin_memory=True, collate_fn=collate, sampler=val_depth_sampler)
    val_seg_loader = DataLoader(val_seg_dataset, batch_size=config.val_bs, shuffle=False, num_workers=4, pin_memory=True, collate_fn=collate, sampler=val_seg_sampler)

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
