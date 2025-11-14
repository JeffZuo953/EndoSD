#!/usr/bin/env python3
"""
LoRA 多任务推理脚本

功能：
  * 从多任务（深度 + 分割）LoRA 检查点加载模型
  * 按数据集批量跑推理，可选择导出深度 npz、分割 png/npz
  * 复用训练管线的数据配置，便于在不同 dataset_config 上快速评估
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
PARENT_ROOT = REPO_ROOT.parent
if str(PARENT_ROOT) not in sys.path:
    sys.path.insert(0, str(PARENT_ROOT))

from multitask_moe_lora.util.config import TrainingConfig
from multitask_moe_lora.util.data_utils import (
    _make_collate_fn,
    create_datasets,
)
from multitask_moe_lora.util.model_setup import (
    create_and_setup_model,
    load_weights_from_checkpoint,
)
from multitask_moe_lora.util.backbone_compat import ensure_backbone_extra_token_support


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="LoRA 多任务推理脚本（深度+分割）")
    parser.add_argument("--checkpoint", required=True, help="LoRA 训练得到的 checkpoint 路径 (.pth)")
    parser.add_argument("--mode", default="endounid",
                        choices=["original", "lora-only", "legacy-lora", "mtlora", "mtlga", "mtoat", "endounid"],
                        help="模型模式（默认：endounid）")
    parser.add_argument("--encoder", default="vitb", help="骨干网络（默认：vitb）")
    parser.add_argument("--features", type=int, default=64, help="Head 特征维度（默认：64）")
    parser.add_argument("--num-classes", type=int, default=10, help="分割类别数（默认：10）")
    parser.add_argument("--max-depth", type=float, default=0.3, help="深度上限（默认：0.3m）")
    parser.add_argument("--img-size", type=int, default=518, help="输入尺寸（默认：518）")
    parser.add_argument("--dataset-config-name", default="ls_bundle", help="util.data_utils 中定义的数据配置名")
    parser.add_argument("--path-transform-name", default="none", help="路径转换配置（默认：none）")
    parser.add_argument("--dataset-modality", default="mt", choices=["mt", "fd"], help="数据模式（默认：mt）")
    parser.add_argument("--dataset-include", default="", help="逗号分隔，只保留指定数据集（可留空）")
    parser.add_argument("--local-cache-dir", default=None, help="可选的本地缓存目录")
    parser.add_argument("--batch-size", type=int, default=8, help="推理 batch size（默认：8）")
    parser.add_argument("--num-workers", type=int, default=8, help="DataLoader worker 数（默认：8）")
    parser.add_argument("--tasks", default="depth,seg", help="选择任务：depth、seg 或 both，用逗号分隔")
    parser.add_argument("--output-root", default="runs/infer_lora", help="输出根目录（默认：runs/infer_lora）")
    parser.add_argument("--save-depth", action="store_true", help="导出深度 npz")
    parser.add_argument("--save-seg", action="store_true", help="导出分割 PNG/npz")
    parser.add_argument("--seg-format", choices=["png", "npz", "both"], default="png", help="分割结果保存格式")
    parser.add_argument("--semantic-token-count", type=int, default=0, help="语义 token 数（mode=mtoat/endounid 时生效）")
    parser.add_argument("--use-semantic-tokens", action="store_true", help="强制启用语义 token")
    parser.add_argument("--cuda-devices", default=None, help="设置 CUDA_VISIBLE_DEVICES（例如 0,1）")
    parser.add_argument("--gpu-ids", default=None, help="可选的 DataParallel 设备列表（默认使用所有可见 GPU）")
    parser.add_argument("--use-amp", action="store_true", help="使用混合精度推理")
    parser.add_argument("--log-dir", default="runs/infer_lora/logs", help="日志目录（默认：runs/infer_lora/logs）")
    parser.add_argument("--skip-existing", action="store_true", help="若输出文件存在则跳过（默认覆盖）")
    parser.add_argument("--clamp-depth", action="store_true", help="对深度输出执行 [0, max_depth] 截断")

    parser.add_argument("--seg-head-type", default="linear", help="分割 head 类型（默认：linear）")
    parser.add_argument("--seg-input-type", default="from_depth", help="分割输入类型")
    parser.add_argument("--disable-seg-head", action="store_true", help="完全跳过分割 head")

    return parser.parse_args()


def _setup_logger(log_dir: Path) -> logging.Logger:
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "infer.log"
    formatter = logging.Formatter("[%(asctime)s][%(levelname)5s] %(message)s", "%Y-%m-%d %H:%M:%S")
    logger = logging.getLogger("lora_infer")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    stream = logging.StreamHandler()
    stream.setFormatter(formatter)
    logger.addHandler(stream)
    file_handler = logging.FileHandler(log_path, mode="w")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    return logger


def _parse_name_list(raw_value: Optional[str]) -> Optional[List[str]]:
    if not raw_value:
        return None
    entries = [item.strip() for item in raw_value.split(",") if item.strip()]
    return entries or None


def _build_collate_with_meta(stride: int):
    base_collate = _make_collate_fn(stride)

    def collate(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        result = base_collate(batch)
        meta_list: List[Dict[str, Any]] = []
        for item in batch:
            meta_list.append(
                {
                    "height": int(item["image"].shape[-2]),
                    "width": int(item["image"].shape[-1]),
                    "dataset_name": item.get("dataset_name"),
                    "source_type": item.get("source_type"),
                    "original_image_path": item.get("original_image_path"),
                    "cache_path": item.get("image_path"),
                    "depth_path": item.get("depth_path"),
                    "clip_id": item.get("clip_id"),
                }
            )
        result["meta"] = meta_list
        return result

    return collate


def _derive_relative_path(meta: Dict[str, Any], dataset_tokens: Sequence[str]) -> str:
    dataset_name = str(meta.get("dataset_name") or meta.get("source_type") or "dataset")
    dataset_name = dataset_name.strip() or "dataset"
    canonical = dataset_name
    needles = [dataset_name.lower()]
    needles.extend(token.lower() for token in dataset_tokens if token)

    candidates = [
        meta.get("original_image_path"),
        meta.get("depth_path"),
        meta.get("cache_path"),
    ]
    rel_candidate: Optional[str] = None
    for candidate in candidates:
        if not candidate:
            continue
        normalized = os.path.abspath(os.path.expanduser(str(candidate)))
        lowered = normalized.lower()
        match_offset = -1
        for needle in needles:
            idx = lowered.find(needle)
            if idx != -1:
                match_offset = idx + len(needle)
                break
        if match_offset != -1:
            suffix = normalized[match_offset:].lstrip("/\\")
            rel_candidate = os.path.join(canonical, suffix)
            break
        rel_candidate = os.path.join(canonical, os.path.basename(normalized))
        break

    rel_candidate = rel_candidate or os.path.join(canonical, f"sample_{meta.get('cache_path', 'unknown')}")
    base, _ = os.path.splitext(rel_candidate)
    safe = base.replace(":", "_")
    safe = safe.replace("\\", "/").lstrip("/")
    return safe


def _prepare_config(args: argparse.Namespace, logger: logging.Logger) -> TrainingConfig:
    config = TrainingConfig()
    config.encoder = args.encoder
    config.features = args.features
    config.num_classes = args.num_classes
    config.max_depth = float(args.max_depth)
    config.img_size = args.img_size
    config.mode = args.mode
    config.seg_head_type = args.seg_head_type
    config.seg_input_type = args.seg_input_type
    config.disable_seg_head = args.disable_seg_head
    config.dataset_config_name = args.dataset_config_name
    config.path_transform_name = None if args.path_transform_name.lower() == "none" else args.path_transform_name
    config.dataset_modality = args.dataset_modality
    include_list = _parse_name_list(args.dataset_include)
    config.val_dataset_include = include_list
    config.train_dataset_include = include_list
    config.local_cache_dir = args.local_cache_dir
    config.val_sample_step = 1
    config.val_min_samples_per_dataset = 0
    config.save_path = str(Path(args.output_root).resolve())
    config.resume_from = args.checkpoint
    config.resume_full_state = False
    config.mixed_precision = args.use_amp
    config.semantic_token_count = args.semantic_token_count
    config.use_semantic_tokens = args.use_semantic_tokens or (args.semantic_token_count > 0)
    config.bs = args.batch_size
    config.val_bs = args.batch_size
    config.seg_bs = args.batch_size
    logger.info("Config prepared: dataset=%s, include=%s", config.dataset_config_name, include_list)
    return config


def _build_loader(dataset, batch_size: int, num_workers: int, encoder: str) -> Optional[DataLoader]:
    if dataset is None:
        return None
    stride = 16 if "dinov3" in encoder.lower() else 14
    collate = _build_collate_with_meta(stride)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
        collate_fn=collate,
    )


def _prepare_model(config: TrainingConfig, logger: logging.Logger, device_ids: Sequence[int]) -> torch.nn.Module:
    torch.cuda.set_device(device_ids[0])
    ensure_backbone_extra_token_support(logger)
    model = create_and_setup_model(config, logger)
    if len(device_ids) > 1:
        logger.info("Using DataParallel on devices: %s", device_ids)
        model = torch.nn.DataParallel(model, device_ids=device_ids)
    load_weights_from_checkpoint(
        model=model,
        optimizer_depth=None,
        optimizer_seg=None,
        optimizer_camera=None,
        scheduler_depth=None,
        scheduler_seg=None,
        scheduler_camera=None,
        config=config,
        logger=logger,
    )
    model.eval()
    return model


def _save_npz(array: np.ndarray, out_path: Path, key: str, skip: bool) -> bool:
    if skip and out_path.exists():
        return False
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, **{key: array})
    return True


def _save_seg_png(seg_map: np.ndarray, out_path: Path, skip: bool) -> bool:
    if skip and out_path.exists():
        return False
    out_path.parent.mkdir(parents=True, exist_ok=True)
    import cv2

    cv2.imwrite(str(out_path), seg_map.astype(np.uint16))
    return True


def _run_depth_inference(model: torch.nn.Module,
                         loader: DataLoader,
                         output_root: Path,
                         dataset_tokens: Sequence[str],
                         args: argparse.Namespace,
                         logger: logging.Logger) -> None:
    if loader is None:
        logger.info("No depth dataset available，跳过深度推理。")
        return
    out_dir = output_root / "depth"
    saved = 0
    with torch.no_grad():
        for batch in loader:
            images = batch["image"].cuda(non_blocking=True)
            with autocast(enabled=args.use_amp):
                outputs = model(images, task='depth')
                pred = outputs['depth']
            for idx in range(pred.size(0)):
                depth_tensor = pred[idx, 0]
                if args.clamp_depth:
                    depth_tensor = depth_tensor.clamp_(0, args.max_depth)
                depth = depth_tensor.detach().cpu().numpy().astype(np.float16)
                rel_base = _derive_relative_path(batch["meta"][idx], dataset_tokens)
                out_path = out_dir / f"{rel_base}.npz"
                if _save_npz(depth, out_path, "depth", args.skip_existing):
                    saved += 1
    logger.info("Depth inference完成，保存 %d 个文件 (root=%s)", saved, out_dir)


def _run_seg_inference(model: torch.nn.Module,
                       loader: DataLoader,
                       output_root: Path,
                       dataset_tokens: Sequence[str],
                       args: argparse.Namespace,
                       logger: logging.Logger) -> None:
    if loader is None or args.disable_seg_head:
        logger.info("No segmentation dataset or seg_head disabled，跳过分割推理。")
        return
    save_png = args.save_seg and args.seg_format in ("png", "both")
    save_npz = args.save_seg and args.seg_format in ("npz", "both")
    out_dir_png = output_root / "seg_png"
    out_dir_npz = output_root / "seg_npz"
    saved = 0
    with torch.no_grad():
        for batch in loader:
            images = batch["image"].cuda(non_blocking=True)
            with autocast(enabled=args.use_amp):
                outputs = model(images, task='seg')
                logits = outputs['seg']
            pred = logits.argmax(dim=1)
            for idx in range(pred.size(0)):
                seg_map = pred[idx].detach().cpu().numpy().astype(np.uint16)
                rel_base = _derive_relative_path(batch["meta"][idx], dataset_tokens)
                if save_png:
                    if _save_seg_png(seg_map, (out_dir_png / f"{rel_base}.png"), args.skip_existing):
                        saved += 1
                if save_npz:
                    if _save_npz(seg_map.astype(np.uint8), (out_dir_npz / f"{rel_base}.npz"), "mask", args.skip_existing):
                        saved += 1
    if args.save_seg:
        logger.info("Seg inference完成，保存 %d 个文件 (root=%s)", saved, output_root)
    else:
        logger.info("Seg inference完成（未保存结果，仅执行 forward）。")


def main():
    args = _parse_args()
    if args.cuda_devices:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_devices
    if not torch.cuda.is_available():
        raise RuntimeError("此脚本需要 GPU 环境。")
    logger = _setup_logger(Path(args.log_dir))
    tasks = {token.strip().lower() for token in args.tasks.split(",")}
    args.save_depth = args.save_depth or ("depth" in tasks)
    args.save_seg = args.save_seg or ("seg" in tasks)

    config = _prepare_config(args, logger)
    train_depth_dataset, val_depth_dataset, train_seg_dataset, val_seg_dataset = create_datasets(config)

    depth_loader = _build_loader(val_depth_dataset, args.batch_size, args.num_workers, args.encoder)
    seg_loader = _build_loader(val_seg_dataset, args.batch_size, args.num_workers, args.encoder)

    dataset_tokens = _parse_name_list(args.dataset_include) or []
    log_root = Path(args.output_root).resolve()
    log_root.mkdir(parents=True, exist_ok=True)

    if args.gpu_ids:
        device_ids = [int(x.strip()) for x in args.gpu_ids.split(",") if x.strip()]
    else:
        visible = os.environ.get("CUDA_VISIBLE_DEVICES")
        if visible:
            device_ids = [int(x) for x in visible.split(",")]
        else:
            device_ids = list(range(torch.cuda.device_count()))
    if not device_ids:
        device_ids = [0]

    model = _prepare_model(config, logger, device_ids)

    if "depth" in tasks:
        _run_depth_inference(model, depth_loader, log_root, dataset_tokens, args, logger)
    if "seg" in tasks:
        _run_seg_inference(model, seg_loader, log_root, dataset_tokens, args, logger)

    logger.info("LoRA 多任务推理完成。输出目录: %s", log_root)


if __name__ == "__main__":
    main()
