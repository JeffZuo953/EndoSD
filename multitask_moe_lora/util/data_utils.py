#!/usr/bin/env python3
"""
数据处理模块
包含数据整理函数和数据加载器创建逻辑
"""

import os
import logging
import math
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset as TorchDataset, Subset
from typing import Dict, Any, List, Optional, Set
from torch.utils.data import ConcatDataset

from ..dataset.cache_utils import DepthCacheDataset, SegCacheDataset
from ..dataset.filelist_seg_depth import FileListSegDepthDataset
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

    images, depths, masks, max_depths, source_types, dataset_names = [], [], [], [], [], []
    valid_masks, depth_valid_masks, seg_valid_masks = [], [], []
    camera_intrinsics, camera_norms, camera_sizes = [], [], []

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
        if "dataset_name" in item:
            dataset_names.append(item["dataset_name"])

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

        if "valid_mask" in item:
            valid = item["valid_mask"].float()
            padded_valid = F.pad(valid, (0, pad_w, 0, pad_h), mode="constant", value=0)
            valid_masks.append(padded_valid.to(torch.bool))

        if "depth_valid_mask" in item:
            depth_valid = item["depth_valid_mask"].float()
            padded_depth_valid = F.pad(depth_valid, (0, pad_w, 0, pad_h), mode="constant", value=0)
            depth_valid_masks.append(padded_depth_valid.to(torch.bool))

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

        if "seg_valid_mask" in item:
            seg_valid = item["seg_valid_mask"].float()
            padded_seg_valid = F.pad(seg_valid, (0, pad_w, 0, pad_h), mode="constant", value=0)
            seg_valid_masks.append(padded_seg_valid.to(torch.bool))

        if "camera_intrinsics" in item:
            camera_mat = item["camera_intrinsics"]
            if not torch.is_tensor(camera_mat):
                camera_mat = torch.as_tensor(camera_mat, dtype=torch.float32)
            camera_intrinsics.append(camera_mat.to(torch.float32))

        if "camera_intrinsics_norm" in item:
            camera_norm = item["camera_intrinsics_norm"]
            if not torch.is_tensor(camera_norm):
                camera_norm = torch.as_tensor(camera_norm, dtype=torch.float32)
            camera_norms.append(camera_norm.to(torch.float32))

        if "camera_size" in item:
            size_tensor = item["camera_size"]
            if not torch.is_tensor(size_tensor):
                size_tensor = torch.as_tensor(size_tensor, dtype=torch.float32)
            camera_sizes.append(size_tensor.to(torch.float32))

    result = {"image": torch.stack(images)}

    if source_types:
        result["source_type"] = source_types
    if dataset_names:
        result["dataset_name"] = dataset_names

    if depths:
        depth_tensor = torch.stack(depths)
        if depth_tensor.dim() == 3:
            depth_tensor = depth_tensor.unsqueeze(1)
        result["depth"] = depth_tensor
        result["max_depth"] = torch.tensor(max_depths)

    if valid_masks:
        result["valid_mask"] = torch.stack(valid_masks)

    if depth_valid_masks:
        result["depth_valid_mask"] = torch.stack(depth_valid_masks)

    if masks:
        mask_tensor = torch.stack(masks)
        if mask_tensor.dim() == 4:
            mask_tensor = mask_tensor.squeeze(1)
        result["semseg_mask"] = mask_tensor

    if seg_valid_masks:
        result["seg_valid_mask"] = torch.stack(seg_valid_masks)

    if camera_intrinsics:
        result["camera_intrinsics"] = torch.stack(camera_intrinsics)
    if camera_norms:
        result["camera_intrinsics_norm"] = torch.stack(camera_norms)
    if camera_sizes:
        result["camera_size"] = torch.stack(camera_sizes)

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

        images, depths, masks, max_depths, source_types, dataset_names = [], [], [], [], [], []
        valid_masks, depth_valid_masks, seg_valid_masks = [], [], []
        camera_intrinsics, camera_norms, camera_sizes = [], [], []
        for item in batch:
            image = item["image"]
            h, w = image.shape[-2:]
            pad_h = max_h - h
            pad_w = max_w - w
            images.append(F.pad(image, (0, pad_w, 0, pad_h), mode="constant", value=0))

            if "source_type" in item:
                source_types.append(item["source_type"])
            if "dataset_name" in item:
                dataset_names.append(item["dataset_name"])

            if "depth" in item:
                depth = item["depth"]
                if depth.dim() == 2:
                    depth = depth.unsqueeze(0)
                if depth.dim() == 3 and depth.shape[0] != 1:
                    depth = depth[0:1]
                depths.append(F.pad(depth, (0, pad_w, 0, pad_h), mode="constant", value=0))
                max_depths.append(item.get('max_depth', 1.0))

            if "valid_mask" in item:
                valid = item["valid_mask"].float()
                valid_masks.append(F.pad(valid, (0, pad_w, 0, pad_h), mode="constant", value=0).to(torch.bool))

            if "depth_valid_mask" in item:
                depth_valid = item["depth_valid_mask"].float()
                depth_valid_masks.append(F.pad(depth_valid, (0, pad_w, 0, pad_h), mode="constant", value=0).to(torch.bool))

            if "semseg_mask" in item:
                mask = item["semseg_mask"]
                if mask.dim() == 3:
                    mask = mask.squeeze(0) if mask.shape[0] == 1 else mask[0]
                elif mask.dim() == 4:
                    mask = mask.squeeze()
                ignore_idx = 255
                masks.append(F.pad(mask, (0, pad_w, 0, pad_h), mode="constant", value=ignore_idx))

            if "seg_valid_mask" in item:
                seg_valid = item["seg_valid_mask"].float()
                seg_valid_masks.append(F.pad(seg_valid, (0, pad_w, 0, pad_h), mode="constant", value=0).to(torch.bool))

            if "camera_intrinsics" in item:
                camera_mat = item["camera_intrinsics"]
                if not torch.is_tensor(camera_mat):
                    camera_mat = torch.as_tensor(camera_mat, dtype=torch.float32)
                camera_intrinsics.append(camera_mat.to(torch.float32))

            if "camera_intrinsics_norm" in item:
                camera_norm = item["camera_intrinsics_norm"]
                if not torch.is_tensor(camera_norm):
                    camera_norm = torch.as_tensor(camera_norm, dtype=torch.float32)
                camera_norms.append(camera_norm.to(torch.float32))

            if "camera_size" in item:
                size_tensor = item["camera_size"]
                if not torch.is_tensor(size_tensor):
                    size_tensor = torch.as_tensor(size_tensor, dtype=torch.float32)
                camera_sizes.append(size_tensor.to(torch.float32))

        result = {"image": torch.stack(images)}
        if source_types:
            result["source_type"] = source_types
        if dataset_names:
            result["dataset_name"] = dataset_names
        if depths:
            depth_tensor = torch.stack(depths)
            if depth_tensor.dim() == 3:
                depth_tensor = depth_tensor.unsqueeze(1)
            result["depth"] = depth_tensor
            result["max_depth"] = torch.tensor(max_depths)
        if valid_masks:
            result["valid_mask"] = torch.stack(valid_masks)
        if depth_valid_masks:
            result["depth_valid_mask"] = torch.stack(depth_valid_masks)
        if masks:
            mask_tensor = torch.stack(masks)
            if mask_tensor.dim() == 4:
                mask_tensor = mask_tensor.squeeze(1)
            result["semseg_mask"] = mask_tensor
        if seg_valid_masks:
            result["seg_valid_mask"] = torch.stack(seg_valid_masks)
        if camera_intrinsics:
            result["camera_intrinsics"] = torch.stack(camera_intrinsics)
        if camera_norms:
            result["camera_intrinsics_norm"] = torch.stack(camera_norms)
        if camera_sizes:
            result["camera_size"] = torch.stack(camera_sizes)
        return result

    return collate_fn


class CyclingStepDistributedSampler(torch.utils.data.distributed.DistributedSampler):
    """
    Distributed sampler that iterates dataset indices with a fixed stride and epoch-dependent offset.

    Example (step=10):
        epoch 0 -> indices [0, 10, 20, ...]
        epoch 1 -> indices [1, 11, 21, ...]
    """

    def __init__(self, dataset: TorchDataset, step: int, num_replicas=None, rank=None, drop_last: bool = False):
        super().__init__(dataset, num_replicas=num_replicas, rank=rank, shuffle=False, drop_last=drop_last)
        self.step = max(int(step), 1)
        self.current_epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.current_epoch = max(int(epoch), 0)

    def _build_indices(self) -> List[int]:
        dataset_len = len(self.dataset)
        if dataset_len == 0:
            return []

        offset_base = min(self.step, dataset_len)
        offset = self.current_epoch % offset_base
        indices = list(range(offset, dataset_len, self.step))
        if not indices:
            indices = [offset % dataset_len]
        return indices

    def __iter__(self):
        indices = self._build_indices()
        if not indices:
            return iter([])

        if self.drop_last:
            usable = (len(indices) // self.num_replicas) * self.num_replicas
            indices = indices[:usable]
        else:
            remainder = len(indices) % self.num_replicas
            if remainder != 0 and indices:
                pad_size = self.num_replicas - remainder
                indices.extend(indices[:pad_size])

        if not indices:
            return iter([])

        return iter(indices[self.rank::self.num_replicas])

    def __len__(self) -> int:
        indices = self._build_indices()
        if not indices:
            return 0
        if self.drop_last:
            return len(indices) // self.num_replicas
        return int(math.ceil(len(indices) / self.num_replicas))


def _limit_dataset(dataset: TorchDataset, max_samples: Optional[int]) -> TorchDataset:
    if max_samples is None:
        return dataset
    dataset_len = len(dataset)
    if dataset_len == 0:
        logging.warning("Dataset %s has no samples; skipping max_samples_per_dataset limit.", getattr(dataset, 'dataset_name', dataset))
        return dataset
    limit = min(dataset_len, max_samples)
    if limit <= 0:
        logging.warning("max_samples_per_dataset <=0 after evaluation for dataset %s; returning original dataset.", getattr(dataset, 'dataset_name', dataset))
        return dataset
    if limit == dataset_len:
        return dataset
    subset = Subset(dataset, list(range(limit)))
    base_name = getattr(dataset, 'dataset_name', getattr(dataset, 'dataset_type', dataset.__class__.__name__))
    setattr(subset, 'dataset_name', f"{base_name}[:{limit}]")
    setattr(subset, 'dataset_type', getattr(dataset, 'dataset_type', 'unknown'))
    return subset


def _create_dataset_or_none(factory, label: str):
    try:
        return factory()
    except FileNotFoundError as exc:
        logging.warning("Skipping dataset %s: %s", label, exc)
        return None


def _sample_dataset_by_step(dataset: Optional[TorchDataset], step: int) -> Optional[TorchDataset]:
    if dataset is None or step is None or step <= 1:
        return dataset
    dataset_len = len(dataset)
    if dataset_len == 0:
        return dataset
    indices = list(range(0, dataset_len, step))
    if not indices:
        return dataset
    subset = Subset(dataset, indices)
    for attr in ('dataset_name', 'dataset_type'):
        if hasattr(dataset, attr):
            setattr(subset, attr, getattr(dataset, attr))
    return subset


def _concat_datasets(datasets: List[TorchDataset], dataset_kind: str) -> TorchDataset:
    if not datasets:
        raise ValueError(f"No datasets configured for {dataset_kind}. Please check dataset config.")
    if len(datasets) == 1:
        return datasets[0]
    return ConcatDataset(datasets)

DATASET_MODALITY_RULES: Dict[str, Dict[str, Dict[str, Set[str]]]] = {
    "fd_depth": {
        "fd": {
            "depth": {
                "SCARED",
                "StereoMIS",
                "dVPN",
                "EndoVis2017",
                "EndoVis2018",
                "C3VDv2",
                "SimCol",
                "Kidney3D",
            },
            "depth_val": {
                "hamlyn",
                "EndoNeRF",
                "C3VD",
                "EndoMapper",
            },
            "seg": None,
            "seg_val": None,
        },
    },
    "ls": {
        "mt": {
            "depth": {"EndoSynth", "EndoVis2017", "EndoVis2018"},
            "depth_val": {"EndoSynth", "EndoVis2017", "EndoVis2018"},
            "seg": {"EndoSynth", "EndoVis2017", "EndoVis2018"},
            "seg_val": {"EndoSynth", "EndoVis2017", "EndoVis2018"},
        },
        "fd": {
            "depth": {"EndoSynth", "StereoMIS", "EndoVis2017", "EndoVis2018", "EndoNeRF", "SCARED"},
            "depth_val": {"EndoSynth", "StereoMIS", "EndoVis2017", "EndoVis2018", "EndoNeRF", "hamlyn", "SCARED"},
            "seg": None,
        },
    },
    "no": {
        "mt": {
            "depth": {
                "Kidney3D",
                "RIRS-SegC",
                "RIRS-SegP",
                "clinicDB",
                "CVC-EndoScene",
                "Kvasir-SEG",
                "bkai-igh-neopolyp",
                "ETIS-LaribPolypDB",
            },
            "depth_val": {"Kidney3D"},
            "seg": {
                "Kidney3D",
                "RIRS-SegC",
                "RIRS-SegP",
                "clinicDB",
                "CVC-EndoScene",
                "Kvasir-SEG",
                "bkai-igh-neopolyp",
                "ETIS-LaribPolypDB",
            },
            "seg_val": {"Kidney3D", "RIRS-SegC", "CVC-EndoScene", "Kvasir-SEG"},
        },
        "fd": {
            "depth": {
                "endomapper_sim",
                "Kidney3D",
                "c3vd",
                "c3vdv2",
                "SimCol",
                "simcol",
            },
            "depth_val": {"Kidney3D"},
            "seg": None,
        },
    },
    "no_ls": {
        "mt": {
            "depth": {
                "endomapper_sim",
                "Kidney3D",
                "StereoMIS",
                "EndoVis2017",
                "EndoVis2018",
                "EndoSynth",
                "EndoNeRF",
            },
            "depth_val": {
                "endomapper_sim",
                "Kidney3D",
                "StereoMIS",
                "EndoVis2017",
                "EndoVis2018",
                "EndoSynth",
                "EndoNeRF",
            },
            "seg": {
                "Kidney3D",
                "RIRS-SegP",
                "RIRS-SegC",
                "clinicDB",
                "CVC-EndoScene",
                "Kvasir-SEG",
                "bkai-igh-neopolyp",
                "EndoVis2017",
                "EndoVis2018",
                "EndoSynth",
                "EndoNeRF",
            },
            "seg_val": {
                "Kidney3D",
                "RIRS-SegC",
                "CVC-EndoScene",
                "Kvasir-SEG",
                "EndoVis2017",
                "EndoVis2018",
                "EndoSynth",
                "EndoNeRF",
            },
        },
    },
}


def _detect_domain_from_config(config_name: str) -> Optional[str]:
    name_lower = config_name.lower()
    if "fd_depth" in name_lower:
        return "fd_depth"
    if "no_ls" in name_lower:
        return "no_ls"
    if "ls" in name_lower and "no" not in name_lower:
        return "ls"
    if "no" in name_lower and "ls" not in name_lower:
        return "no"
    return None


def _get_dataset_name(ds: TorchDataset) -> Optional[str]:
    return getattr(ds, "dataset_name", getattr(ds, "dataset_type", None))


def _apply_modality_filter(datasets: List[TorchDataset],
                           allowed_names: Optional[Set[str]],
                           dataset_kind: str) -> List[TorchDataset]:
    if not datasets or not allowed_names:
        return datasets
    filtered: List[TorchDataset] = []
    for ds in datasets:
        name = _get_dataset_name(ds)
        base_name = str(name).split('[')[0] if name else None
        if base_name in allowed_names:
            filtered.append(ds)
        else:
            logging.debug("Filtering dataset '%s' out of %s due to modality rule.", name, dataset_kind)
    if not filtered:
        raise ValueError(f"No datasets remain for {dataset_kind} after applying modality filter {allowed_names}.")
    return filtered


def _apply_dataset_include(datasets: List[TorchDataset],
                           include_names: Optional[List[str]],
                           dataset_kind: str) -> List[TorchDataset]:
    if not datasets or not include_names:
        return datasets
    include_set = {name.strip() for name in include_names if name.strip()}
    filtered: List[TorchDataset] = []
    for ds in datasets:
        name = _get_dataset_name(ds)
        base_name = str(name).split('[')[0] if name else None
        if base_name in include_set:
            filtered.append(ds)
        else:
            logging.debug("Filtering dataset '%s' out of %s due to include filter %s.", name, dataset_kind, include_set)
    if not filtered:
        raise ValueError(f"No datasets remain for {dataset_kind} after applying dataset include filter {include_set}.")
    return filtered


def _flatten_dataset_parts(dataset: TorchDataset) -> List[Dict[str, Any]]:
    """解构嵌套的ConcatDataset/Subset，返回扁平化的子数据集列表。"""
    parts: List[Dict[str, Any]] = []

    if isinstance(dataset, ConcatDataset):
        for sub in dataset.datasets:
            parts.extend(_flatten_dataset_parts(sub))
        return parts

    if isinstance(dataset, Subset):
        base = dataset.dataset
        base_name = getattr(base, 'dataset_name', getattr(base, 'dataset_type', base.__class__.__name__))
        parts.append({
            'name': getattr(dataset, 'dataset_name', f"{base_name}[:{len(dataset)}]"),
            'dataset_type': getattr(dataset, 'dataset_type', getattr(base, 'dataset_type', 'unknown')),
            'count': len(dataset),
        })
        return parts

    parts.append({
        'name': getattr(dataset, 'dataset_name', getattr(dataset, 'dataset_type', dataset.__class__.__name__)),
        'dataset_type': getattr(dataset, 'dataset_type', 'unknown'),
        'count': len(dataset),
    })
    return parts


def summarize_loader_composition(data_loader: DataLoader) -> List[Dict[str, Any]]:
    """
    汇总数据加载器内部各子数据集的样本数。
    Returns:
        列表项包含 name / dataset_type / count / cumulative / total / index / total_parts
    """
    dataset = data_loader.dataset
    parts = _flatten_dataset_parts(dataset)
    total = sum(part['count'] for part in parts)
    cumulative = 0
    total_parts = len(parts)
    for idx, part in enumerate(parts, start=1):
        cumulative += part['count']
        part['cumulative'] = cumulative
        part['total'] = total
        part['index'] = idx
        part['total_parts'] = total_parts
    return parts


def log_loader_composition(logger, data_loader, phase: str, task: str, rank: int, epoch: Optional[int] = None) -> None:
    """在日志中输出数据加载器的组成情况（仅rank=0）。"""
    if rank != 0:
        return

    summaries = summarize_loader_composition(data_loader)
    if not summaries:
        logger.info(f"[{phase}-{task}] No datasets available.")
        return

    total = summaries[0]['total']
    head_prefix = f"[{phase}-{task}]"
    if epoch is not None:
        head_prefix += f" Epoch {epoch}"
    logger.info(f"{head_prefix} Using {len(summaries)} datasets, total {total} samples.")

    for part in summaries:
        logger.info(
            f"{head_prefix} Dataset {part['index']}/{part['total_parts']}: "
            f"{part['name']} ({part['dataset_type']}) -> {part['count']}/{total} samples, cumulative {part['cumulative']}/{total}"
        )


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

    modality = getattr(config, "dataset_modality", "mt").lower()
    dataset_domain = _detect_domain_from_config(active_dataset)
    modality_rule = DATASET_MODALITY_RULES.get(dataset_domain, {}).get(modality)

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
        'no_ls_v1': {
            "depth_train_caches": [
                {
                    "path": "/data/ziyi/multitask/data/NO/endomapper_sim/cache/train_cache.txt",
                    "dataset_type": "NO",
                    "transform_key": "depth_train_endomapper",
                },
                {
                    "path": "/data/ziyi/multitask/data/NO/Kidney3D-CT-depth-seg/cache_pt/train_cache.txt",
                    "dataset_type": "NO",
                    "name": "Kidney3D",
                },
                {
                    "path": "/data/ziyi/multitask/data/LS/StereoMIS/cache_pt/train_cache.txt",
                    "dataset_type": "LS",
                    "name": "StereoMIS",
                },
                {
                    "path": "/data/ziyi/multitask/data/LS/EndoVis2017/cache_pt/train_cache.txt",
                    "dataset_type": "LS",
                    "name": "EndoVis2017",
                },
                {
                    "path": "/data/ziyi/multitask/data/LS/EndoVis2018/cache_pt/train_cache.txt",
                    "dataset_type": "LS",
                    "name": "EndoVis2018",
                },
                {
                    "path": "/data/ziyi/multitask/data/LS/EndoSynth/cache_pt/train_cache.txt",
                    "dataset_type": "LS",
                    "name": "EndoSynth",
                },
                {
                    "path": "/data/ziyi/multitask/data/LS/EndoNeRF/cache_pt/train_cache.txt",
                    "dataset_type": "LS",
                    "name": "EndoNeRF",
                },
            ],
            "depth_val_caches": [
                {
                    "path": "/data/ziyi/multitask/data/NO/endomapper_sim/cache/val_cache.txt",
                    "dataset_type": "NO",
                    "transform_key": "depth_val_endomapper",
                },
                {
                    "path": "/data/ziyi/multitask/data/NO/Kidney3D-CT-depth-seg/cache_pt/val_cache.txt",
                    "dataset_type": "NO",
                    "name": "Kidney3D",
                },
                {
                    "path": "/data/ziyi/multitask/data/LS/StereoMIS/cache_pt/val_cache.txt",
                    "dataset_type": "LS",
                    "name": "StereoMIS",
                },
                {
                    "path": "/data/ziyi/multitask/data/LS/EndoVis2017/cache_pt/val_cache.txt",
                    "dataset_type": "LS",
                    "name": "EndoVis2017",
                },
                {
                    "path": "/data/ziyi/multitask/data/LS/EndoVis2018/cache_pt/val_cache.txt",
                    "dataset_type": "LS",
                    "name": "EndoVis2018",
                },
                {
                    "path": "/data/ziyi/multitask/data/LS/EndoSynth/cache_pt/val_cache.txt",
                    "dataset_type": "LS",
                    "name": "EndoSynth",
                },
                {
                    "path": "/data/ziyi/multitask/data/LS/EndoNeRF/cache_pt/val_cache.txt",
                    "dataset_type": "LS",
                    "name": "EndoNeRF",
                },
            ],
            "seg_train_caches": [
                {
                    "path": "/data/ziyi/multitask/data/NO/RIRS-SegP/train_cache.txt",
                    "dataset_type": "NO",
                    "transform_key": "seg_train_rirs",
                },
                {
                    "path": "/data/ziyi/multitask/data/NO/bkai-igh-neopolyp/cache/train_cache.txt",
                    "dataset_type": "NO",
                    "transform_key": "seg_train_bkai",
                },
                {
                    "path": "/data/ziyi/multitask/data/NO/clinicDB/cache/train_cache.txt",
                    "dataset_type": "NO",
                    "transform_key": "seg_train_clinicdb",
                },
                {
                    "path": "/data/ziyi/multitask/data/NO/CVC-EndoScene/TrainDataset/cache/train_cache.txt",
                    "dataset_type": "NO",
                    "transform_key": "seg_train_cvc_endoscene",
                },
                {
                    "path": "/data/ziyi/multitask/data/NO/kvasir-SEG-split/cache_pt/train_cache.txt",
                    "dataset_type": "NO",
                    "name": "Kvasir-SEG",
                },
                {
                    "path": "/data/ziyi/multitask/data/NO/Kidney3D-CT-depth-seg/cache_pt/train_cache.txt",
                    "dataset_type": "NO",
                    "name": "Kidney3D",
                },
                {
                    "path": "/data/ziyi/multitask/data/LS/EndoVis2017/cache_pt/train_cache.txt",
                    "dataset_type": "LS",
                    "name": "EndoVis2017",
                },
                {
                    "path": "/data/ziyi/multitask/data/LS/EndoVis2018/cache_pt/train_cache.txt",
                    "dataset_type": "LS",
                    "name": "EndoVis2018",
                },
                {
                    "path": "/data/ziyi/multitask/data/LS/EndoSynth/cache_pt/train_cache.txt",
                    "dataset_type": "LS",
                    "name": "EndoSynth",
                },
                {
                    "path": "/data/ziyi/multitask/data/LS/EndoNeRF/cache_pt/train_cache.txt",
                    "dataset_type": "LS",
                    "name": "EndoNeRF",
                },
            ],
            "seg_val_caches": [
                {
                    "path": "/data/ziyi/multitask/data/NO/CVC-EndoScene/ValidationDataset/cache/train_cache.txt",
                    "dataset_type": "NO",
                    "transform_key": "seg_val_cvc_endoscene",
                },
                {
                    "path": "/data/ziyi/multitask/data/NO/kvasir-SEG-split/cache_pt/val_cache.txt",
                    "dataset_type": "NO",
                    "name": "Kvasir-SEG",
                },
                {
                    "path": "/data/ziyi/multitask/data/NO/Kidney3D-CT-depth-seg/cache_pt/val_cache.txt",
                    "dataset_type": "NO",
                    "name": "Kidney3D",
                },
                {
                    "path": "/data/ziyi/multitask/data/LS/EndoVis2017/cache_pt/val_cache.txt",
                    "dataset_type": "LS",
                    "name": "EndoVis2017",
                },
                {
                    "path": "/data/ziyi/multitask/data/LS/EndoVis2018/cache_pt/val_cache.txt",
                    "dataset_type": "LS",
                    "name": "EndoVis2018",
                },
                {
                    "path": "/data/ziyi/multitask/data/LS/EndoSynth/cache_pt/val_cache.txt",
                    "dataset_type": "LS",
                    "name": "EndoSynth",
                },
                {
                    "path": "/data/ziyi/multitask/data/LS/EndoNeRF/cache_pt/val_cache.txt",
                    "dataset_type": "LS",
                    "name": "EndoNeRF",
                },
            ],
        },
        'no_ls_local': {
            "depth_train_caches": [
                {
                    "path": "/data/ziyi/multitask/data/NO/endomapper_sim/cache/train_cache.txt",
                    "dataset_type": "NO",
                    "transform_key": "depth_train_endomapper",
                },
                {
                    "path": "/data/ziyi/multitask/data/NO/Kidney3D-CT-depth-seg/cache_pt/train_cache.txt",
                    "dataset_type": "NO",
                    "name": "Kidney3D",
                },
                {
                    "path": "/data/ziyi/multitask/data/LS/StereoMIS/cache_pt/train_cache.txt",
                    "dataset_type": "LS",
                    "name": "StereoMIS",
                },
                {
                    "path": "/data/ziyi/multitask/data/LS/EndoVis2017/cache_pt/all_cache.txt",
                    "dataset_type": "LS",
                    "name": "EndoVis2017",
                },
                {
                    "path": "/data/ziyi/multitask/data/LS/EndoVis2018/cache_pt/train_cache.txt",
                    "dataset_type": "LS",
                    "name": "EndoVis2018",
                },
                {
                    "path": "/data/ziyi/multitask/data/LS/EndoSynth/cache_pt/train_cache.txt",
                    "dataset_type": "LS",
                    "name": "EndoSynth",
                },
                {
                    "path": "/data/ziyi/multitask/data/LS/EndoNeRF/cache_pt/train_cache.txt",
                    "dataset_type": "LS",
                    "name": "EndoNeRF",
                },
            ],
            "depth_val_caches": [
                {
                    "path": "/data/ziyi/multitask/data/NO/endomapper_sim/cache/val_cache.txt",
                    "dataset_type": "NO",
                    "transform_key": "depth_val_endomapper",
                },
                {
                    "path": "/data/ziyi/multitask/data/NO/Kidney3D-CT-depth-seg/cache_pt/val_cache.txt",
                    "dataset_type": "NO",
                    "name": "Kidney3D",
                },
                {
                    "path": "/data/ziyi/multitask/data/LS/StereoMIS/cache_pt/val_cache.txt",
                    "dataset_type": "LS",
                    "name": "StereoMIS",
                },
                {
                    "path": "/data/ziyi/multitask/data/LS/EndoVis2017/cache_pt/val_cache.txt",
                    "dataset_type": "LS",
                    "name": "EndoVis2017",
                },
                {
                    "path": "/data/ziyi/multitask/data/LS/EndoVis2018/cache_pt/val_cache.txt",
                    "dataset_type": "LS",
                    "name": "EndoVis2018",
                },
                {
                    "path": "/data/ziyi/multitask/data/LS/EndoSynth/cache_pt/val_cache.txt",
                    "dataset_type": "LS",
                    "name": "EndoSynth",
                },
                {
                    "path": "/data/ziyi/multitask/data/LS/EndoNeRF/cache_pt/val_cache.txt",
                    "dataset_type": "LS",
                    "name": "EndoNeRF",
                },
            ],
            "seg_train_caches": [
                {
                    "path": "/data/ziyi/multitask/data/NO/RIRS-SegP/train_cache.txt",
                    "dataset_type": "NO",
                    "transform_key": "seg_train_rirs",
                },
                {
                    "path": "/data/ziyi/multitask/data/NO/bkai-igh-neopolyp/cache/train_cache.txt",
                    "dataset_type": "NO",
                    "transform_key": "seg_train_bkai",
                },
                {
                    "path": "/data/ziyi/multitask/data/NO/clinicDB/cache/train_cache.txt",
                    "dataset_type": "NO",
                    "transform_key": "seg_train_clinicdb",
                },
                {
                    "path": "/data/ziyi/multitask/data/NO/CVC-EndoScene/TrainDataset/cache/train_cache.txt",
                    "dataset_type": "NO",
                    "transform_key": "seg_train_cvc_endoscene",
                },
                {
                    "path": "/data/ziyi/multitask/data/NO/kvasir-SEG-split/cache_pt/train_cache.txt",
                    "dataset_type": "NO",
                    "name": "Kvasir-SEG",
                },
                {
                    "path": "/data/ziyi/multitask/data/NO/Kidney3D-CT-depth-seg/cache_pt/train_cache.txt",
                    "dataset_type": "NO",
                    "name": "Kidney3D",
                },
                {
                    "path": "/data/ziyi/multitask/data/LS/EndoVis2017/cache_pt/all_cache.txt",
                    "dataset_type": "LS",
                    "name": "EndoVis2017",
                },
                {
                    "path": "/data/ziyi/multitask/data/LS/EndoVis2018/cache_pt/train_cache.txt",
                    "dataset_type": "LS",
                    "name": "EndoVis2018",
                },
                {
                    "path": "/data/ziyi/multitask/data/LS/EndoSynth/cache_pt/train_cache.txt",
                    "dataset_type": "LS",
                    "name": "EndoSynth",
                },
                {
                    "path": "/data/ziyi/multitask/data/LS/EndoNeRF/cache_pt/train_cache.txt",
                    "dataset_type": "LS",
                    "name": "EndoNeRF",
                },
            ],
            "seg_val_caches": [
                {
                    "path": "/data/ziyi/multitask/data/NO/RIRS-SegP/val_cache.txt",
                    "dataset_type": "NO",
                    "transform_key": "seg_val_rirs",
                },
                {
                    "path": "/data/ziyi/multitask/data/NO/CVC-EndoScene/ValidationDataset/cache/train_cache.txt",
                    "dataset_type": "NO",
                    "transform_key": "seg_val_cvc_endoscene",
                },
                {
                    "path": "/data/ziyi/multitask/data/NO/kvasir-SEG-split/cache_pt/val_cache.txt",
                    "dataset_type": "NO",
                    "name": "Kvasir-SEG",
                },
                {
                    "path": "/data/ziyi/multitask/data/NO/Kidney3D-CT-depth-seg/cache_pt/val_cache.txt",
                    "dataset_type": "NO",
                    "name": "Kidney3D",
                },
                {
                    "path": "/data/ziyi/multitask/data/LS/EndoVis2017/cache_pt/val_cache.txt",
                    "dataset_type": "LS",
                    "name": "EndoVis2017",
                },
                {
                    "path": "/data/ziyi/multitask/data/LS/EndoVis2018/cache_pt/val_cache.txt",
                    "dataset_type": "LS",
                    "name": "EndoVis2018",
                },
                {
                    "path": "/data/ziyi/multitask/data/LS/EndoSynth/cache_pt/val_cache.txt",
                    "dataset_type": "LS",
                    "name": "EndoSynth",
                },
                {
                    "path": "/data/ziyi/multitask/data/LS/EndoNeRF/cache_pt/val_cache.txt",
                    "dataset_type": "LS",
                    "name": "EndoNeRF",
                },
            ],
        },
        'no_only_v1': {
            "depth_train_caches": [
                {
                    "path": "/data/ziyi/multitask/data/NO/endomapper_sim/cache/train_cache.txt",
                    "dataset_type": "NO",
                    "transform_key": "depth_train_endomapper",
                },
                {
                    "path": "/data/ziyi/multitask/data/NO/c3vd/cache/train_cache.txt",
                    "dataset_type": "NO",
                },
                {
                    "path": "/data/ziyi/multitask/data/NO/c3vdv2/cache/cache.txt",
                    "dataset_type": "NO",
                },
            ],
            "depth_val_caches": [
                {
                    "path": "/data/ziyi/multitask/data/NO/endomapper_sim/cache/val_cache.txt",
                    "dataset_type": "NO",
                    "transform_key": "depth_val_endomapper",
                },
                {
                    "path": "/data/ziyi/multitask/data/NO/c3vd/cache/test/test_cache.txt",
                    "dataset_type": "NO",
                },
            ],
            "depth_train_filelists": [
                {
                    "path": "/data/ziyi/multitask/data/NO/Kidney3D-CT-depth-seg/filelists/train.txt",
                    "dataset_type": "NO",
                    "max_depth": 0.05,
                    "depth_scale": 1.0,
                    "name": "Kidney3D",
                },
            ],
            "depth_val_filelists": [
                {
                    "path": "/data/ziyi/multitask/data/NO/Kidney3D-CT-depth-seg/filelists/val.txt",
                    "dataset_type": "NO",
                    "max_depth": 0.05,
                    "depth_scale": 1.0,
                    "name": "Kidney3D",
                },
            ],
            "seg_train_caches": [
                {
                    "path": "/data/ziyi/multitask/data/NO/RIRS-SegP/train_cache.txt",
                    "dataset_type": "NO",
                    "transform_key": "seg_train_rirs",
                },
                {
                    "path": "/data/ziyi/multitask/data/NO/RIRS-SegC/train_split_cache.txt",
                    "dataset_type": "NO",
                    "name": "RIRS-SegC",
                },
                {
                    "path": "/data/ziyi/multitask/data/NO/bkai-igh-neopolyp/cache/train_cache.txt",
                    "dataset_type": "NO",
                    "transform_key": "seg_train_bkai",
                },
                {
                    "path": "/data/ziyi/multitask/data/NO/clinicDB/cache/train_cache.txt",
                    "dataset_type": "NO",
                    "transform_key": "seg_train_clinicdb",
                },
                {
                    "path": "/data/ziyi/multitask/data/NO/CVC-EndoScene/TrainDataset/cache/train_cache.txt",
                    "dataset_type": "NO",
                    "transform_key": "seg_train_cvc_endoscene",
                    "name": "CVC-EndoScene",
                },
                {
                    "path": "/data/ziyi/multitask/data/NO/kvasir-seg/cache/train_split_cache.txt",
                    "dataset_type": "NO",
                    "transform_key": "seg_train_kvasir",
                    "name": "Kvasir-SEG",
                },
                {
                    "path": "/data/ziyi/multitask/data/NO/ETIS-LaribPolypDB/train_cache.txt",
                    "dataset_type": "NO",
                },
            ],
            "seg_train_filelists": [
                {
                    "path": "/data/ziyi/multitask/data/NO/Kidney3D-CT-depth-seg/filelists/train.txt",
                    "dataset_type": "NO",
                    "max_depth": 0.05,
                    "depth_scale": 1.0,
                    "name": "Kidney3D",
                },
            ],
            "seg_val_caches": [
                {
                    "path": "/data/ziyi/multitask/data/NO/CVC-EndoScene/ValidationDataset/cache/train_cache.txt",
                    "dataset_type": "NO",
                    "transform_key": "seg_val_cvc_endoscene",
                    "name": "CVC-EndoScene",
                },
                {
                    "path": "/data/ziyi/multitask/data/NO/RIRS-SegC/val_split_cache.txt",
                    "dataset_type": "NO",
                    "name": "RIRS-SegC",
                },
                {
                    "path": "/data/ziyi/multitask/data/NO/kvasir-seg/cache/val_split_cache.txt",
                    "dataset_type": "NO",
                    "transform_key": "seg_val_kvasir",
                    "name": "Kvasir-SEG",
                },
            ],
            "seg_val_filelists": [
                {
                    "path": "/data/ziyi/multitask/data/NO/Kidney3D-CT-depth-seg/filelists/val.txt",
                    "dataset_type": "NO",
                    "max_depth": 0.05,
                    "depth_scale": 1.0,
                    "name": "Kidney3D",
                },
            ],
        },
        'ls_only_v1': {
            "depth_train_caches": [
                {
                    "path": "/data/ziyi/multitask/data/LS/StereoMIS/cache_pt/train_cache.txt",
                    "dataset_type": "LS",
                },
                {
                    "path": "/data/ziyi/multitask/data/LS/EndoSynth/cache_pt/train_cache.txt",
                    "dataset_type": "LS",
                },
                {
                    "path": "/data/ziyi/multitask/data/LS/EndoVis2017/cache_pt/train_cache.txt",
                    "dataset_type": "LS",
                    "name": "EndoVis2017",
                },
                {
                    "path": "/data/ziyi/multitask/data/LS/SCARED/cache/train_cache.txt",
                    "dataset_type": "LS",
                },
            ],
            "depth_val_caches": [
                {
                    "path": "/data/ziyi/multitask/data/LS/StereoMIS/cache_pt/val_cache.txt",
                    "dataset_type": "LS",
                },
                {
                    "path": "/data/ziyi/multitask/data/LS/EndoSynth/cache_pt/val_cache.txt",
                    "dataset_type": "LS",
                },
                {
                    "path": "/data/ziyi/multitask/data/LS/SCARED/cache/test_cache.txt",
                    "dataset_type": "LS",
                },
                {
                    "path": "/data/ziyi/multitask/data/LS/EndoVis2017/cache_pt/val_cache.txt",
                    "dataset_type": "LS",
                    "name": "EndoVis2017",
                },
            ],
            "depth_train_filelists": [
                {
                    "path": "/data/ziyi/multitask/data/LS/EndoVis2018/filelists/train.txt",
                    "dataset_type": "LS",
                    "max_depth": 0.3,
                    "depth_scale": 1.0,
                },
                {
                    "path": "/data/ziyi/multitask/data/LS/EndoNeRF/filelists/train.txt",
                    "dataset_type": "LS",
                    "max_depth": 0.3,
                    "depth_scale": 255.0 / 0.3,
                },
            ],
            "depth_val_filelists": [
                {
                    "path": "/data/ziyi/multitask/data/LS/EndoVis2018/filelists/val.txt",
                    "dataset_type": "LS",
                    "max_depth": 0.3,
                    "depth_scale": 1.0,
                },
                {
                    "path": "/data/ziyi/multitask/data/LS/EndoNeRF/filelists/val.txt",
                    "dataset_type": "LS",
                    "max_depth": 0.3,
                    "depth_scale": 255.0 / 0.3,
                },
            ],
            "seg_train_filelists": [
                {
                    "path": "/data/ziyi/multitask/data/LS/EndoVis2018/filelists/train.txt",
                    "dataset_type": "LS",
                    "max_depth": 0.3,
                    "depth_scale": 1.0,
                },
                {
                    "path": "/data/ziyi/multitask/data/LS/EndoNeRF/filelists/train.txt",
                    "dataset_type": "LS",
                    "max_depth": 0.3,
                    "depth_scale": 255.0 / 0.3,
                },
            ],
            "seg_val_filelists": [
                {
                    "path": "/data/ziyi/multitask/data/LS/EndoVis2018/filelists/val.txt",
                    "dataset_type": "LS",
                    "max_depth": 0.3,
                    "depth_scale": 1.0,
                },
                {
                    "path": "/data/ziyi/multitask/data/LS/EndoNeRF/filelists/val.txt",
                    "dataset_type": "LS",
                    "max_depth": 0.3,
                    "depth_scale": 255.0 / 0.3,
                },
            ],
            "seg_train_caches": [
                {
                    "path": "/data/ziyi/multitask/data/LS/EndoSynth/cache_pt/train_cache.txt",
                    "dataset_type": "LS",
                },
                {
                    "path": "/data/ziyi/multitask/data/LS/EndoVis2017/cache_pt/train_cache.txt",
                    "dataset_type": "LS",
                    "name": "EndoVis2017",
                },
            ],
            "seg_val_caches": [
                {
                    "path": "/data/ziyi/multitask/data/LS/EndoSynth/cache_pt/val_cache.txt",
                    "dataset_type": "LS",
                },
                {
                    "path": "/data/ziyi/multitask/data/LS/EndoVis2017/cache_pt/val_cache.txt",
                    "dataset_type": "LS",
                    "name": "EndoVis2017",
                },
            ],
        },
        'fd_depth_fm_v1': {
            "depth_train_caches": [
                {
                    "path": "/home/ziyi/ssde/data/LS/SCARED/cache/train_all_cache.txt",
                    "dataset_type": "LS",
                    "name": "SCARED",
                },
                {
                    "path": "/data/ziyi/multitask/data/LS/StereoMIS/cache_pt/all_cache.txt",
                    "dataset_type": "LS",
                    "name": "StereoMIS",
                },
                {
                    "path": "/data/ziyi/multitask/data/LS/EndoVis2017/cache_pt/all_cache.txt",
                    "dataset_type": "LS",
                    "name": "EndoVis2017",
                },
                {
                    "path": "/data/ziyi/multitask/data/LS/EndoVis2018/cache_pt/train_cache.txt",
                    "dataset_type": "LS",
                    "name": "EndoVis2018",
                },
                {
                    "path": "/data/ziyi/multitask/data/LS/EndoVis2018/cache_pt/val_cache.txt",
                    "dataset_type": "LS",
                    "name": "EndoVis2018",
                },
                {
                    "path": "/data/ziyi/multitask/data/NO/c3vdv2/cache/cache.txt",
                    "dataset_type": "NO",
                    "name": "C3VDv2",
                },
                {
                    "path": "/data/ziyi/multitask/data/NO/Kidney3D-CT-depth-seg/cache_pt/train_cache.txt",
                    "dataset_type": "NO",
                    "name": "Kidney3D",
                },
                {
                    "path": "/home/ziyi/ssde/data/simcol/cache/train_all_cache.txt",
                    "dataset_type": "NO",
                    "name": "SimCol",
                },
                {
                    "path": "/home/ziyi/ssde/data/dVPN/cache/train_all_cache.txt",
                    "dataset_type": "LS",
                    "name": "dVPN",
                },
            ],
            "depth_train_filelists": [],
            "depth_val_caches": [
                {
                    "path": "/data/ziyi/multitask/data/LS/hamlyn/cache_pt/all_cache.txt",
                    "dataset_type": "LS",
                    "name": "hamlyn",
                },
                {
                    "path": "/data/ziyi/multitask/data/LS/EndoNeRF/cache_pt/all_cache.txt",
                    "dataset_type": "LS",
                    "name": "EndoNeRF",
                },
                {
                    "path": "/data/ziyi/multitask/data/NO/c3vd/cache/all_cache.txt",
                    "dataset_type": "NO",
                    "name": "C3VD",
                },
                {
                    "path": "/data/ziyi/multitask/data/NO/endomapper_sim/cache/all_cache.txt",
                    "dataset_type": "NO",
                    "name": "EndoMapper",
                },
            ],
            "depth_val_filelists": [],
        },
        'ls_only_v1_filelist': {
            "depth_train_filelists": [
                {"path": "/data/ziyi/multitask/data/LS/EndoVis2018/filelists/train.txt", "dataset_type": "LS", "max_depth": 0.3, "depth_scale": 1.0, "name": "EndoVis2018"},
            ],
            "depth_val_filelists": [
                {"path": "/data/ziyi/multitask/data/LS/EndoVis2018/filelists/val.txt", "dataset_type": "LS", "max_depth": 0.3, "depth_scale": 1.0, "name": "EndoVis2018"},
            ],
            "seg_train_filelists": [
                {"path": "/data/ziyi/multitask/data/LS/EndoVis2018/filelists/train.txt", "dataset_type": "LS", "max_depth": 0.3, "depth_scale": 1.0, "name": "EndoVis2018"},
            ],
            "seg_val_filelists": [
                {"path": "/data/ziyi/multitask/data/LS/EndoVis2018/filelists/val.txt", "dataset_type": "LS", "max_depth": 0.3, "depth_scale": 1.0, "name": "EndoVis2018"},
            ],
        },
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
        },
        'no_ls_default': {
            "depth_train_endomapper":
                lambda p: p.replace("/media/ssd2t/jianfu/data/endomapper_sim_sub/cache",
                                    "/data/ziyi/multitask/data/NO/endomapper_sim/cache"),
            "depth_val_endomapper":
                lambda p: p.replace("/media/ssd2t/jianfu/data/endomapper_sim_sub/cache",
                                    "/data/ziyi/multitask/data/NO/endomapper_sim/cache"),
            "seg_train_rirs":
                lambda p: p.replace("/media/ssd2t/jianfu/data/seg_data_v2/cache/img",
                                    "/data/ziyi/multitask/data/NO/RIRS-SegP/img"),
            "seg_val_rirs":
                lambda p: p.replace("/media/ssd2t/jianfu/data/seg_data_v2/cache/img",
                                    "/data/ziyi/multitask/data/NO/RIRS-SegP/img"),
            "seg_train_bkai":
                lambda p: p.replace("/media/ssd2t/jianfu/data/polyp/bkai-igh-neopolyp/cache",
                                    "/data/ziyi/multitask/data/NO/bkai-igh-neopolyp/cache"),
            "seg_train_clinicdb":
                lambda p: p.replace("/media/ssd2t/jianfu/data/polyp/CVC-ClinicDB/cache",
                                    "/data/ziyi/multitask/data/NO/clinicDB/cache"),
            "seg_train_cvc_endoscene":
                lambda p: p.replace("/media/ssd2t/jianfu/data/polyp/CVC-EndoScene/TrainDataset/cache",
                                    "/data/ziyi/multitask/data/NO/CVC-EndoScene/TrainDataset/cache"),
            "seg_val_cvc_endoscene":
                lambda p: p.replace("/media/ssd2t/jianfu/data/polyp/CVC-EndoScene/ValidationDataset/cache",
                                    "/data/ziyi/multitask/data/NO/CVC-EndoScene/ValidationDataset/cache"),
            "seg_train_kvasir":
                lambda p: p.replace("/media/ssd2t/jianfu/data/polyp/kvasir-seg/cache",
                                    "/data/ziyi/multitask/data/NO/kvasir-seg/media/ssd2t/jianfu/data/polyp/kvasir-seg/cache"),
            "seg_val_kvasir":
                lambda p: p.replace("/media/ssd2t/jianfu/data/polyp/kvasir-seg/cache",
                                    "/data/ziyi/multitask/data/NO/kvasir-seg/media/ssd2t/jianfu/data/polyp/kvasir-seg/cache"),
        }
        ,
        'no_only_default': {
            "depth_train_endomapper":
                lambda p: p.replace("/media/ssd2t/jianfu/data/endomapper_sim_sub/cache",
                                    "/data/ziyi/multitask/data/NO/endomapper_sim/cache"),
            "depth_val_endomapper":
                lambda p: p.replace("/media/ssd2t/jianfu/data/endomapper_sim_sub/cache",
                                    "/data/ziyi/multitask/data/NO/endomapper_sim/cache"),
            "seg_train_rirs":
                lambda p: p.replace("/media/ssd2t/jianfu/data/seg_data_v2/cache/img",
                                    "/data/ziyi/multitask/data/NO/RIRS-SegP/img"),
            "seg_val_rirs":
                lambda p: p.replace("/media/ssd2t/jianfu/data/seg_data_v2/cache/img",
                                    "/data/ziyi/multitask/data/NO/RIRS-SegP/img"),
            "seg_train_bkai":
                lambda p: p.replace("/media/ssd2t/jianfu/data/polyp/bkai-igh-neopolyp/cache",
                                    "/data/ziyi/multitask/data/NO/bkai-igh-neopolyp/cache"),
            "seg_train_clinicdb":
                lambda p: p.replace("/media/ssd2t/jianfu/data/polyp/CVC-ClinicDB/cache",
                                    "/data/ziyi/multitask/data/NO/clinicDB/cache"),
            "seg_train_cvc_endoscene":
                lambda p: p.replace("/media/ssd2t/jianfu/data/polyp/CVC-EndoScene/TrainDataset/cache",
                                    "/data/ziyi/multitask/data/NO/CVC-EndoScene/TrainDataset/cache"),
            "seg_val_cvc_endoscene":
                lambda p: p.replace("/media/ssd2t/jianfu/data/polyp/CVC-EndoScene/ValidationDataset/cache",
                                    "/data/ziyi/multitask/data/NO/CVC-EndoScene/ValidationDataset/cache"),
            "seg_train_kvasir":
                lambda p: p.replace("/media/ssd2t/jianfu/data/polyp/kvasir-seg/cache",
                                    "/data/ziyi/multitask/data/NO/kvasir-seg/media/ssd2t/jianfu/data/polyp/kvasir-seg/cache"),
            "seg_val_kvasir":
                lambda p: p.replace("/media/ssd2t/jianfu/data/polyp/kvasir-seg/cache",
                                    "/data/ziyi/multitask/data/NO/kvasir-seg/media/ssd2t/jianfu/data/polyp/kvasir-seg/cache"),
        },
        'ls_default': {
            "depth_train_caches": [
                {
                    "path": "/data/ziyi/multitask/data/LS/StereoMIS/cache_pt/train_cache.txt",
                    "dataset_type": "LS",
                    "name": "StereoMIS",
                },
                {
                    "path": "/data/ziyi/multitask/data/LS/EndoVis2017/cache_pt/train_cache.txt",
                    "dataset_type": "LS",
                    "name": "EndoVis2017",
                },
                {
                    "path": "/data/ziyi/multitask/data/LS/EndoVis2018/cache_pt/train_cache.txt",
                    "dataset_type": "LS",
                    "name": "EndoVis2018",
                },
                {
                    "path": "/data/ziyi/multitask/data/LS/EndoSynth/cache_pt/train_cache.txt",
                    "dataset_type": "LS",
                    "name": "EndoSynth",
                },
                {
                    "path": "/data/ziyi/multitask/data/LS/EndoNeRF/cache_pt/train_cache.txt",
                    "dataset_type": "LS",
                    "name": "EndoNeRF",
                },
            ],
            "depth_val_caches": [
                {
                    "path": "/data/ziyi/multitask/data/LS/StereoMIS/cache_pt/val_cache.txt",
                    "dataset_type": "LS",
                    "name": "StereoMIS",
                },
                {
                    "path": "/data/ziyi/multitask/data/LS/EndoVis2017/cache_pt/val_cache.txt",
                    "dataset_type": "LS",
                    "name": "EndoVis2017",
                },
                {
                    "path": "/data/ziyi/multitask/data/LS/EndoVis2018/cache_pt/val_cache.txt",
                    "dataset_type": "LS",
                    "name": "EndoVis2018",
                },
                {
                    "path": "/data/ziyi/multitask/data/LS/EndoSynth/cache_pt/val_cache.txt",
                    "dataset_type": "LS",
                    "name": "EndoSynth",
                },
                {
                    "path": "/data/ziyi/multitask/data/LS/EndoNeRF/cache_pt/val_cache.txt",
                    "dataset_type": "LS",
                    "name": "EndoNeRF",
                },
            ],
            "seg_train_caches": [
                {
                    "path": "/data/ziyi/multitask/data/LS/EndoVis2017/cache_pt/train_cache.txt",
                    "dataset_type": "LS",
                    "name": "EndoVis2017",
                },
                {
                    "path": "/data/ziyi/multitask/data/LS/EndoVis2018/cache_pt/train_cache.txt",
                    "dataset_type": "LS",
                    "name": "EndoVis2018",
                },
                {
                    "path": "/data/ziyi/multitask/data/LS/EndoSynth/cache_pt/train_cache.txt",
                    "dataset_type": "LS",
                    "name": "EndoSynth",
                },
                {
                    "path": "/data/ziyi/multitask/data/LS/EndoNeRF/cache_pt/train_cache.txt",
                    "dataset_type": "LS",
                    "name": "EndoNeRF",
                },
            ],
            "seg_val_caches": [
                {
                    "path": "/data/ziyi/multitask/data/LS/EndoVis2017/cache_pt/val_cache.txt",
                    "dataset_type": "LS",
                    "name": "EndoVis2017",
                },
                {
                    "path": "/data/ziyi/multitask/data/LS/EndoVis2018/cache_pt/val_cache.txt",
                    "dataset_type": "LS",
                    "name": "EndoVis2018",
                },
                {
                    "path": "/data/ziyi/multitask/data/LS/EndoSynth/cache_pt/val_cache.txt",
                    "dataset_type": "LS",
                    "name": "EndoSynth",
                },
                {
                    "path": "/data/ziyi/multitask/data/LS/EndoNeRF/cache_pt/val_cache.txt",
                    "dataset_type": "LS",
                    "name": "EndoNeRF",
                },
            ],
        }
    }

    transform_map = PATH_TRANSFORM_CONFIGS.get(active_transform, {}) if active_transform else {}

    # segmentation label mode selection
    seg_label_mode = "raw"
    if getattr(config, "num_classes", None) is not None:
        if config.num_classes >= 10:
            seg_label_mode = "10class"
        elif config.num_classes == 3:
            seg_label_mode = "3class"

    if "depth_train_caches" in paths or "depth_train_filelists" in paths:
        train_depth_parts: List[TorchDataset] = []
        for entry in paths.get("depth_train_caches", []):
            transform_fn = transform_map.get(entry.get("transform_key")) if transform_map else None
            ds = _create_dataset_or_none(
                lambda entry=entry, transform_fn=transform_fn: DepthCacheDataset(
                    entry["path"],
                    dataset_type=entry.get("dataset_type", "unknown"),
                    path_transform=transform_fn,
                    dataset_name=entry.get("name") or _infer_dataset_name(entry["path"])
                ),
                entry.get("name") or entry["path"],
            )
            if ds is None:
                continue
            train_depth_parts.append(_limit_dataset(ds, config.max_samples_per_dataset))
        for entry in paths.get("depth_train_filelists", []):
            ds = FileListSegDepthDataset(
                filelist_path=entry["path"],
                mode="train",
                size=(config.img_size, config.img_size),
                default_max_depth=entry.get("max_depth", config.max_depth),
                default_depth_scale=entry.get("depth_scale", 1000.0),
                dataset_type=entry.get("dataset_type", "unknown"),
                dataset_name=entry.get("name") or _infer_dataset_name(entry["path"])
            )
            train_depth_parts.append(_limit_dataset(ds, config.max_samples_per_dataset))
        val_depth_parts: List[TorchDataset] = []
        for entry in paths.get("depth_val_caches", []):
            transform_fn = transform_map.get(entry.get("transform_key")) if transform_map else None
            ds = _create_dataset_or_none(
                lambda entry=entry, transform_fn=transform_fn: DepthCacheDataset(
                    entry["path"],
                    dataset_type=entry.get("dataset_type", "unknown"),
                    path_transform=transform_fn,
                    dataset_name=entry.get("name") or _infer_dataset_name(entry["path"])
                ),
                entry.get("name") or entry["path"],
            )
            if ds is None:
                continue
            val_depth_parts.append(_limit_dataset(ds, config.max_samples_per_dataset))
        for entry in paths.get("depth_val_filelists", []):
            ds = FileListSegDepthDataset(
                filelist_path=entry["path"],
                mode="eval",
                size=(config.img_size, config.img_size),
                default_max_depth=entry.get("max_depth", config.max_depth),
                default_depth_scale=entry.get("depth_scale", 1000.0),
                dataset_type=entry.get("dataset_type", "unknown"),
                dataset_name=entry.get("name") or _infer_dataset_name(entry["path"])
            )
            val_depth_parts.append(_limit_dataset(ds, config.max_samples_per_dataset))

        train_depth_parts = _apply_dataset_include(train_depth_parts, config.train_dataset_include, "depth-train")
        val_depth_parts = _apply_dataset_include(val_depth_parts, config.val_dataset_include, "depth-val")

    if modality_rule:
        allowed_depth_train = modality_rule.get("depth")
        allowed_depth_val = modality_rule.get("depth_val", allowed_depth_train)
        train_depth_parts = _apply_modality_filter(train_depth_parts, allowed_depth_train, "depth-train")
        val_depth_parts = _apply_modality_filter(val_depth_parts, allowed_depth_val, "depth-val")

        train_depth_dataset = _concat_datasets(train_depth_parts, "depth-train")
        val_depth_dataset = _concat_datasets(val_depth_parts, "depth-val")
    else:
        train_depth_parts = []
        ds = _create_dataset_or_none(
            lambda: DepthCacheDataset(
                paths["depth_train_inhouse"],
                dataset_type="kidney",
                path_transform=transform_map.get("depth_train_inhouse"),
                dataset_name=_infer_dataset_name(paths["depth_train_inhouse"]),
            ),
            "depth_train_inhouse",
        )
        if ds is not None:
            train_depth_parts.append(_limit_dataset(ds, config.max_samples_per_dataset))
        ds = _create_dataset_or_none(
            lambda: DepthCacheDataset(
                paths["depth_train_endomapper"],
                dataset_type="colon",
                path_transform=transform_map.get("depth_train_endomapper"),
                dataset_name=_infer_dataset_name(paths["depth_train_endomapper"]),
            ),
            "depth_train_endomapper",
        )
        if ds is not None:
            train_depth_parts.append(_limit_dataset(ds, config.max_samples_per_dataset))
        if modality_rule:
            allowed_depth_train = modality_rule.get("depth")
            allowed_depth_val = modality_rule.get("depth_val", allowed_depth_train)
            train_depth_parts = _apply_modality_filter(train_depth_parts, allowed_depth_train, "depth-train")

        val_depth_parts = []
        ds = _create_dataset_or_none(
            lambda: DepthCacheDataset(
                paths["depth_val_inhouse"],
                dataset_type="kidney",
                path_transform=transform_map.get("depth_val_inhouse"),
                dataset_name=_infer_dataset_name(paths["depth_val_inhouse"]),
            ),
            "depth_val_inhouse",
        )
        if ds is not None:
            val_depth_parts.append(_limit_dataset(ds, config.max_samples_per_dataset))
        ds = _create_dataset_or_none(
            lambda: DepthCacheDataset(
                paths["depth_val_endomapper"],
                dataset_type="colon",
                path_transform=transform_map.get("depth_val_endomapper"),
                dataset_name=_infer_dataset_name(paths["depth_val_endomapper"]),
            ),
            "depth_val_endomapper",
        )
        if ds is not None:
            val_depth_parts.append(_limit_dataset(ds, config.max_samples_per_dataset))
        train_depth_parts = _apply_dataset_include(train_depth_parts, config.train_dataset_include, "depth-train")
        val_depth_parts = _apply_dataset_include(val_depth_parts, config.val_dataset_include, "depth-val")
        if modality_rule:
            val_depth_parts = _apply_modality_filter(val_depth_parts, allowed_depth_val, "depth-val")
        train_depth_dataset = _concat_datasets(train_depth_parts, "depth-train")
        val_depth_dataset = _concat_datasets(val_depth_parts, "depth-val")

    train_seg_dataset = None
    val_seg_dataset = None

    if modality == "fd":
        # 深度专用模式不加载分割分支
        train_seg_dataset = None
        val_seg_dataset = None
    elif "seg_train_caches" in paths or "seg_train_filelists" in paths:
        train_seg_parts: List[TorchDataset] = []
        for entry in paths.get("seg_train_caches", []):
            transform_fn = transform_map.get(entry.get("transform_key")) if transform_map else None
            ds = _create_dataset_or_none(
                lambda entry=entry, transform_fn=transform_fn: SegCacheDataset(
                    entry["path"],
                    dataset_type=entry.get("dataset_type", "unknown"),
                    path_transform=transform_fn,
                    dataset_name=entry.get("name") or _infer_dataset_name(entry["path"]),
                    label_mode=seg_label_mode,
                ),
                entry.get("name") or entry["path"],
            )
            if ds is None:
                continue
            train_seg_parts.append(_limit_dataset(ds, config.max_samples_per_dataset))

        for entry in paths.get("seg_train_filelists", []):
            ds = FileListSegDepthDataset(
                filelist_path=entry["path"],
                mode="train",
                size=(config.img_size, config.img_size),
                default_max_depth=entry.get("max_depth", config.max_depth),
                default_depth_scale=entry.get("depth_scale", 1000.0),
                dataset_type=entry.get("dataset_type", "unknown"),
                dataset_name=entry.get("name") or _infer_dataset_name(entry["path"])
            )
            train_seg_parts.append(_limit_dataset(ds, config.max_samples_per_dataset))
        train_seg_parts = _apply_dataset_include(train_seg_parts, config.train_dataset_include, "seg-train")

        val_seg_parts: List[TorchDataset] = []
        for entry in paths.get("seg_val_caches", []):
            transform_fn = transform_map.get(entry.get("transform_key")) if transform_map else None
            ds = _create_dataset_or_none(
                lambda entry=entry, transform_fn=transform_fn: SegCacheDataset(
                    entry["path"],
                    dataset_type=entry.get("dataset_type", "unknown"),
                    path_transform=transform_fn,
                    dataset_name=entry.get("name") or _infer_dataset_name(entry["path"]),
                    label_mode=seg_label_mode,
                ),
                entry.get("name") or entry["path"],
            )
            if ds is None:
                continue
            val_seg_parts.append(_limit_dataset(ds, config.max_samples_per_dataset))

        for entry in paths.get("seg_val_filelists", []):
            ds = FileListSegDepthDataset(
                filelist_path=entry["path"],
                mode="eval",
                size=(config.img_size, config.img_size),
                default_max_depth=entry.get("max_depth", config.max_depth),
                default_depth_scale=entry.get("depth_scale", 1000.0),
                dataset_type=entry.get("dataset_type", "unknown"),
                dataset_name=entry.get("name") or _infer_dataset_name(entry["path"])
            )
            val_seg_parts.append(_limit_dataset(ds, config.max_samples_per_dataset))
        if modality_rule:
            allowed_seg_train = modality_rule.get("seg")
            allowed_seg_val = modality_rule.get("seg_val", allowed_seg_train)
            train_seg_parts = _apply_modality_filter(train_seg_parts, allowed_seg_train, "seg-train")
            val_seg_parts = _apply_modality_filter(val_seg_parts, allowed_seg_val, "seg-val")
        val_seg_parts = _apply_dataset_include(val_seg_parts, config.val_dataset_include, "seg-val")
        train_seg_dataset = _concat_datasets(train_seg_parts, "seg-train")
        val_seg_dataset = _concat_datasets(val_seg_parts, "seg-val")
    else:
        train_seg_parts = []
        for key, dtype in [
            ("seg_train_inhouse", "kidney"),
            ("seg_train_cvc_clinicdb", "colon"),
            ("seg_train_bkai", "colon"),
            ("seg_train_cvc_endoscene", "colon"),
            ("seg_train_kvasir", "colon"),
        ]:
            ds = _create_dataset_or_none(
                lambda key=key, dtype=dtype: SegCacheDataset(
                    paths[key],
                    dataset_type=dtype,
                    path_transform=transform_map.get(key),
                    dataset_name=_infer_dataset_name(paths[key]),
                    label_mode=seg_label_mode,
                ),
                key,
            )
            if ds is not None:
                train_seg_parts.append(_limit_dataset(ds, config.max_samples_per_dataset))
        if modality_rule:
            allowed_seg_train = modality_rule.get("seg")
            allowed_seg_val = modality_rule.get("seg_val", allowed_seg_train)
            train_seg_parts = _apply_modality_filter(train_seg_parts, allowed_seg_train, "seg-train")

        val_seg_parts = []
        for key, dtype in [
            ("seg_val_inhouse", "kidney"),
            ("seg_val_cvc_endoscene", "colon"),
            ("seg_val_etis", "colon"),
        ]:
            ds = _create_dataset_or_none(
                lambda key=key, dtype=dtype: SegCacheDataset(
                    paths[key],
                    dataset_type=dtype,
                    path_transform=transform_map.get(key),
                    dataset_name=_infer_dataset_name(paths[key]),
                    label_mode=seg_label_mode,
                ),
                key,
            )
            if ds is not None:
                val_seg_parts.append(_limit_dataset(ds, config.max_samples_per_dataset))
        if modality_rule:
            val_seg_parts = _apply_modality_filter(val_seg_parts, allowed_seg_val, "seg-val")
        train_seg_parts = _apply_dataset_include(train_seg_parts, config.train_dataset_include, "seg-train")
        val_seg_parts = _apply_dataset_include(val_seg_parts, config.val_dataset_include, "seg-val")
        train_seg_dataset = _concat_datasets(train_seg_parts, "seg-train")
        val_seg_dataset = _concat_datasets(val_seg_parts, "seg-val")

    return train_depth_dataset, val_depth_dataset, train_seg_dataset, val_seg_dataset


def create_dataloaders(config: TrainingConfig,
                       train_depth_dataset: TorchDataset,
                       val_depth_dataset: TorchDataset,
                       train_seg_dataset: TorchDataset,
                       val_seg_dataset: TorchDataset) -> tuple:
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
    # 在验证阶段按步长抽样
    val_step = max(getattr(config, "val_sample_step", 1), 1)
    val_depth_dataset = _sample_dataset_by_step(val_depth_dataset, val_step)
    val_seg_dataset = _sample_dataset_by_step(val_seg_dataset, val_step)

    # 为每个任务创建分布式采样器
    if train_depth_dataset is not None:
        if getattr(config, "train_sample_step", 1) > 1:
            train_depth_sampler = CyclingStepDistributedSampler(train_depth_dataset, step=config.train_sample_step)
        else:
            train_depth_sampler = torch.utils.data.distributed.DistributedSampler(train_depth_dataset)
    else:
        train_depth_sampler = None

    if train_seg_dataset is not None:
        if getattr(config, "train_sample_step", 1) > 1:
            train_seg_sampler = CyclingStepDistributedSampler(train_seg_dataset, step=config.train_sample_step)
        else:
            train_seg_sampler = torch.utils.data.distributed.DistributedSampler(train_seg_dataset)
    else:
        train_seg_sampler = None

    # 设置分割训练的批次大小（可以比深度训练更小以节省内存）
    seg_batch_size = config.seg_bs if config.seg_bs is not None else config.bs

    # 动态选择 patch stride（dinov2=14，dinov3=16），并构造对应 collate_fn
    encoder = getattr(config, 'encoder', 'vits').lower()
    stride = 16 if 'dinov3' in encoder else 14
    collate = _make_collate_fn(stride)

    # 创建数据加载器, shuffle=False因为sampler会处理随机化
    worker_count = 12  # increase dataloader parallelism to fully utilize CPU

    train_depth_loader = DataLoader(
        train_depth_dataset,
        batch_size=config.bs,
        shuffle=False,
        num_workers=worker_count,
        pin_memory=True,
        drop_last=True,
        collate_fn=collate,
        sampler=train_depth_sampler,
    )

    if train_seg_dataset is not None:
        train_seg_loader = DataLoader(
            train_seg_dataset,
            batch_size=seg_batch_size,
            shuffle=False,
            num_workers=worker_count,
            pin_memory=True,
            drop_last=True,
            collate_fn=collate,
            sampler=train_seg_sampler,
        )
    else:
        train_seg_loader = None

    val_depth_sampler = torch.utils.data.distributed.DistributedSampler(val_depth_dataset, shuffle=False) if val_depth_dataset is not None else None
    val_depth_loader = DataLoader(
        val_depth_dataset,
        batch_size=config.val_bs,
        shuffle=False,
        num_workers=worker_count,
        pin_memory=True,
        drop_last=False,
        collate_fn=collate,
        sampler=val_depth_sampler,
    )

    if val_seg_dataset is not None:
        val_seg_sampler = torch.utils.data.distributed.DistributedSampler(val_seg_dataset, shuffle=False)
        val_seg_loader = DataLoader(
            val_seg_dataset,
            batch_size=config.val_bs,
            shuffle=False,
            num_workers=worker_count,
            pin_memory=True,
            drop_last=False,
            collate_fn=collate,
            sampler=val_seg_sampler,
        )
    else:
        val_seg_loader = None

    return train_depth_loader, train_seg_loader, val_depth_loader, val_seg_loader


def setup_dataloaders(config: TrainingConfig) -> tuple:
    """
    一站式数据加载器设置函数。
    """
    train_depth_dataset, val_depth_dataset, train_seg_dataset, val_seg_dataset = create_datasets(config)
    return create_dataloaders(config, train_depth_dataset, val_depth_dataset, train_seg_dataset, val_seg_dataset)


def log_batch_info(logger, config: TrainingConfig) -> None:
    """记录批次大小信息，seg 分支在 fd 模式下可能为 None。"""
    seg_batch_size = config.seg_bs if config.seg_bs is not None else config.bs
    logger.info(f"Using batch sizes - Depth: {config.bs}, Segmentation: {seg_batch_size}")


def get_batch_size_info(config: TrainingConfig) -> Dict[str, int]:
    """
    返回一个简单的批次大小信息字典。
    """
    seg_batch_size = config.seg_bs if config.seg_bs is not None else config.bs
    return {
        'depth_batch_size': config.bs,
        'seg_batch_size': seg_batch_size,
        'val_batch_size': config.val_bs,
    }
