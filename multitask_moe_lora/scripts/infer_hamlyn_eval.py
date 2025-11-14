#!/usr/bin/env python3
"""
Hamlyn 数据集专用的 LoRA 深度推理 + 评估脚本。

特性：
  * 仅使用 filelist 中包含 image01（左目）的样本；
  * 运行 LoRA 模型进行深度预测，同时把预测保存为 .npy；
  * 计算与训练日志一致的深度指标（absrel、rmse、δ1/2/3、mae_cm、acc_2cm 等）；
  * 支持多 GPU（DataParallel）、AMP 以及语义 token。
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import torch
from torch.cuda.amp import autocast
from torch.utils.data import DataLoader

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
PARENT_ROOT = REPO_ROOT.parent
if str(PARENT_ROOT) not in sys.path:
    sys.path.insert(0, str(PARENT_ROOT))

from multitask_moe_lora.dataset.hamlyn import HamlynDataset  # noqa: E402
from multitask_moe_lora.util.config import TrainingConfig  # noqa: E402
from multitask_moe_lora.util.data_utils import _make_collate_fn  # noqa: E402
from multitask_moe_lora.util.metric import eval_depth  # noqa: E402
from multitask_moe_lora.util.model_setup import (  # noqa: E402
    create_and_setup_model,
    load_weights_from_checkpoint,
)
from multitask_moe_lora.util.backbone_compat import ensure_backbone_extra_token_support  # noqa: E402


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Hamlyn 左目深度推理 + 评估")
    parser.add_argument("--checkpoint",
                        default="~/checkpoint_best_absrel_combined.pth",
                        help="模型 checkpoint (.pth)，默认使用 FM best_absrel 权重")
    parser.add_argument("--filelist", default="~/ssde/000/abdo/hamlyn_data/filelists/train.txt",
                        help="Hamlyn filelist（默认：train.txt）")
    parser.add_argument("--root", default="~/ssde/000/abdo/hamlyn_data",
                        help="Hamlyn 数据根目录（包含 color/depth 等子目录）")
    parser.add_argument("--output-root", default="/data/ziyi/result/hamlyn_all",
                        help="推理结果与指标输出目录")
    parser.add_argument("--left-token", default="image01",
                        help="仅保留包含该关键字的行（用于选择左目 image01）")
    parser.add_argument("--batch-size", type=int, default=4, help="推理 batch size")
    parser.add_argument("--num-workers", type=int, default=8, help="DataLoader workers 数量")
    parser.add_argument("--img-size", type=int, default=518, help="模型输入尺寸（wxh）")
    parser.add_argument("--max-depth", type=float, default=0.3, help="深度上限（m）")
    parser.add_argument("--min-depth", type=float, default=1e-3, help="深度下限（m）")
    parser.add_argument("--encoder", default="vitb",
                        choices=["vits", "vitb", "vitl", "vitg",
                                 "dinov3_vits16", "dinov3_vits16plus", "dinov3_vitb16",
                                 "dinov3_vitl16", "dinov3_vitl16plus", "dinov3_vith16plus", "dinov3_vit7b16"])
    parser.add_argument("--features", type=int, default=64, help="head feature 维度")
    parser.add_argument("--mode", default="original",
                        choices=["original", "lora-only", "legacy-lora", "mtlora", "mtlga", "mtoat", "endounid"],
                        help="LoRA / 适配模式")
    parser.add_argument("--lora-r", type=int, default=4, help="LoRA rank")
    parser.add_argument("--lora-alpha", type=int, default=8, help="LoRA alpha")
    parser.add_argument("--semantic-token-count", type=int, default=0, help="语义 token 数量")
    parser.add_argument("--use-semantic-tokens", action="store_true", help="强制启用语义 token")
    parser.add_argument("--cuda-devices", default="0,1",
                        help="CUDA_VISIBLE_DEVICES（默认：0,1）")
    parser.add_argument("--gpu-ids", default=None,
                        help="DataParallel 使用的 GPU 列表（默认：所有可见 GPU）")
    parser.add_argument("--use-amp", action="store_true", help="启用混合精度推理")
    parser.add_argument("--local-cache-dir", default=None, help="Hamlyn 数据本地缓存目录（可选）")
    parser.add_argument("--metrics-json", default=None, help="可选：指定额外的指标输出 JSON 文件名")
    parser.add_argument("--skip-existing", action="store_true", help="若输出 npy 已存在则跳过写入")
    return parser.parse_args()


def _setup_logger(output_root: Path) -> logging.Logger:
    output_root.mkdir(parents=True, exist_ok=True)
    log_path = output_root / "hamlyn_infer.log"
    logger = logging.getLogger("hamlyn_infer")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter("[%(asctime)s][%(levelname)5s] %(message)s", "%Y-%m-%d %H:%M:%S")
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(sh)
    fh = logging.FileHandler(log_path, mode="w")
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    return logger


def _filter_filelist(filelist_path: str, keyword: str, work_dir: Path, logger: logging.Logger) -> Path:
    src = Path(os.path.expanduser(filelist_path))
    if not src.is_file():
        raise FileNotFoundError(f"filelist 不存在: {src}")
    with src.open("r") as f:
        lines = [line.strip() for line in f if line.strip()]
    filtered = [line for line in lines if keyword in line]
    if not filtered:
        raise ValueError(f"filelist {src} 中未找到包含 '{keyword}' 的行，无法锁定左目样本。")
    dst = work_dir / f"hamlyn_{keyword}_filelist.txt"
    with dst.open("w") as f:
        f.write("\n".join(filtered))
    logger.info("筛选 filelist：%s -> %s，保留 %d / %d 行（关键字：%s）",
                src, dst, len(filtered), len(lines), keyword)
    return dst


def _build_collate_with_meta(stride: int):
    base_collate = _make_collate_fn(stride)

    def collate(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        packed = base_collate(batch)
        meta = []
        for item in batch:
            meta.append({
                "image_path": item.get("image_path"),
                "depth_path": item.get("depth_path"),
                "sequence": item.get("sequence"),
                "sequence_path": item.get("sequence_path"),
            })
        packed["meta"] = meta
        return packed

    return collate


def _prepare_config(args: argparse.Namespace, save_path: Path) -> TrainingConfig:
    config = TrainingConfig()
    config.encoder = args.encoder
    config.features = args.features
    config.num_classes = 1
    config.min_depth = args.min_depth
    config.max_depth = args.max_depth
    config.seg_head_type = "none"
    config.seg_input_type = "from_depth"
    config.disable_seg_head = True
    config.mode = args.mode
    config.lora_r = args.lora_r
    config.lora_alpha = args.lora_alpha
    config.semantic_token_count = max(0, args.semantic_token_count)
    config.use_semantic_tokens = bool(args.use_semantic_tokens or config.mode in {"mtoat", "endounid"})
    config.camera_head_mode = "none"
    config.img_size = args.img_size
    config.bs = args.batch_size
    config.val_bs = args.batch_size
    config.save_path = str(save_path)
    config.resume_from = os.path.expanduser(args.checkpoint)
    config.resume_full_state = False
    config.mixed_precision = args.use_amp
    return config


def _resolve_device_ids(args: argparse.Namespace) -> List[int]:
    if args.cuda_devices:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_devices
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if args.gpu_ids:
        ids = [int(x.strip()) for x in args.gpu_ids.split(",") if x.strip()]
    elif visible:
        ids = [int(x.strip()) for x in visible.split(",") if x.strip()]
    else:
        ids = list(range(torch.cuda.device_count()))
    if not ids:
        raise RuntimeError("未检测到可用 GPU。")
    return ids


def _prepare_model(config: TrainingConfig, logger: logging.Logger, device_ids: Sequence[int]) -> torch.nn.Module:
    torch.cuda.set_device(device_ids[0])
    ensure_backbone_extra_token_support(logger)
    model = create_and_setup_model(config, logger)
    if len(device_ids) > 1:
        logger.info("使用 DataParallel，设备：%s", ",".join(map(str, device_ids)))
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


def _prepare_semantic_kwargs(batch: Dict[str, Any], config: TrainingConfig) -> Dict[str, torch.Tensor]:
    if not getattr(config, "use_semantic_tokens", False):
        return {}
    token_ids = batch.get("token_ids") or batch.get("semantic_token_ids")
    if token_ids is None:
        return {}
    if not torch.is_tensor(token_ids):

        if isinstance(token_ids, (list, tuple)) and token_ids:
            if isinstance(token_ids[0], (list, tuple)):
                max_len = max((len(entry) for entry in token_ids), default=0)
                max_len = max(max_len, 1)
                padded = torch.full((len(token_ids), max_len), -1, dtype=torch.long)
                for idx, entry in enumerate(token_ids):
                    if not entry:
                        continue
                    vals = torch.tensor(entry, dtype=torch.long)
                    length = min(len(entry), max_len)
                    padded[idx, :length] = vals[:length]
                token_ids = padded
            else:
                token_ids = torch.as_tensor(token_ids, dtype=torch.long)
        else:
            return {}
    return {"semantic_token_ids": token_ids.cuda(non_blocking=True)}


def _build_output_stem(meta: Dict[str, Any], fallback_idx: int) -> str:
    seq = meta.get("sequence") or "hamlyn"
    depth_path = meta.get("depth_path") or meta.get("image_path")
    if depth_path:
        stem = Path(depth_path).stem
    else:
        stem = f"frame_{fallback_idx:06d}"
    return f"{seq}_{stem}"


def _run_inference_and_eval(model: torch.nn.Module,
                            loader: DataLoader,
                            config: TrainingConfig,
                            output_root: Path,
                            args: argparse.Namespace,
                            logger: logging.Logger) -> Dict[str, float]:
    preds_dir = output_root / "pred_depth"
    preds_dir.mkdir(parents=True, exist_ok=True)
    stats_sum = defaultdict(float)
    agg = {
        "samples": 0,
        "valid_pixels": 0,
        "abs_sum": 0.0,
        "hits_2cm": 0.0,
    }
    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            images = batch["image"].cuda(non_blocking=True)
            sem_kwargs = _prepare_semantic_kwargs(batch, config)
            with autocast(enabled=args.use_amp):
                outputs = model(images, task='depth', **sem_kwargs)
                pred = outputs['depth']
            depth_gt = batch["depth"].to(dtype=torch.float32)
            valid_mask = batch.get("valid_mask")
            if valid_mask is not None:
                valid_mask = valid_mask.to(dtype=torch.bool)
                if valid_mask.dim() == 3:
                    valid_mask = valid_mask.unsqueeze(1)
                elif valid_mask.dim() == 2:
                    valid_mask = valid_mask.unsqueeze(0).unsqueeze(0)
            meta_list = batch.get("meta") or [{}] * pred.size(0)

            for i in range(pred.size(0)):
                pred_map = pred[i].detach().cpu().numpy().astype(np.float32, copy=False)
                gt_map = depth_gt[i].detach().cpu().numpy().astype(np.float32, copy=False)
                if pred_map.ndim > 2:
                    pred_map = np.squeeze(pred_map, axis=0)
                if gt_map.ndim > 2:
                    gt_map = np.squeeze(gt_map, axis=0)
                if batch_idx == 0 and i == 0:
                    logger.info("[debug] pred sample shape=%s, gt sample shape=%s", pred_map.shape, gt_map.shape)
                if pred_map.ndim == 1 and gt_map.ndim == 2 and pred_map.size == gt_map.size:
                    pred_map = pred_map.reshape(gt_map.shape)
                elif pred_map.ndim == 2 and gt_map.ndim == 1 and pred_map.size == gt_map.size:
                    gt_map = gt_map.reshape(pred_map.shape)
                if args.max_depth > 0:
                    np.clip(pred_map, 0.0, args.max_depth, out=pred_map)
                mask = np.isfinite(gt_map) & (gt_map > config.min_depth)
                if valid_mask is not None:
                    vm = valid_mask[i]
                    while vm.dim() > 2:
                        vm = vm[0]
                    vm = vm.detach().cpu().numpy().astype(bool, copy=False)
                    if vm.shape == mask.shape:
                        mask &= vm
                    else:
                        logger.warning(
                            "Valid mask shape %s mismatches depth map %s; ignoring dataset mask for sample %d.",
                            vm.shape,
                            mask.shape,
                            i,
                        )
                mask_flat = mask.astype(bool, copy=False)
                if mask_flat.ndim > 1:
                    mask_flat = mask_flat.reshape(-1)
                pred_vec = pred_map.reshape(-1)
                gt_vec = gt_map.reshape(-1)
                if mask_flat.size != pred_vec.size:
                    logger.warning(
                        "Mask size %d mismatches depth size %d; falling back to depth-only mask for sample %d.",
                        mask_flat.size,
                        pred_vec.size,
                        i,
                    )
                    mask_flat = np.isfinite(gt_vec) & (gt_vec > config.min_depth)
                pred_valid = pred_vec[mask_flat]
                gt_valid = gt_vec[mask_flat]
                if pred_valid.size == 0:
                    continue

                metrics = eval_depth(pred_valid, gt_valid)
                for k, v in metrics.items():
                    stats_sum[k] += v
                agg["samples"] += 1
                agg["valid_pixels"] += pred_valid.size
                abs_diff = np.abs(pred_valid - gt_valid)
                agg["abs_sum"] += abs_diff.sum()
                agg["hits_2cm"] += np.count_nonzero(abs_diff <= 0.02)

                stem = _build_output_stem(meta_list[i], fallback_idx=agg["samples"])
                out_path = preds_dir / f"{stem}.npy"
                if args.skip_existing and out_path.exists():
                    continue
                np.save(out_path, pred_map.astype(np.float32))

            logger.info("批次 %d 完成 (%d 样本)，累计样本 %d。",
                        batch_idx, pred.size(0), agg["samples"])

    if agg["samples"] == 0:
        raise RuntimeError("没有有效样本完成推理，无法计算指标。")

    results = {k: (v / agg["samples"]) for k, v in stats_sum.items()}
    if agg["valid_pixels"] > 0:
        results["mae_cm"] = (agg["abs_sum"] / agg["valid_pixels"]) * 100.0
        results["acc_2cm"] = agg["hits_2cm"] / agg["valid_pixels"]
    results["frames"] = agg["samples"]
    results["valid_pixels"] = int(agg["valid_pixels"])
    return results


def _save_metrics(metrics: Dict[str, float], output_root: Path, custom_path: Optional[str], logger: logging.Logger) -> None:
    default_path = output_root / "hamlyn_metrics.json"
    with default_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    logger.info("指标已写入 %s", default_path)
    if custom_path:
        custom = Path(custom_path)
        custom.parent.mkdir(parents=True, exist_ok=True)
        with custom.open("w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2, ensure_ascii=False)
        logger.info("指标同时写入 %s", custom)


def main() -> None:
    args = _parse_args()
    output_root = Path(os.path.expanduser(args.output_root)).resolve()
    logger = _setup_logger(output_root)
    filtered_filelist = _filter_filelist(args.filelist, args.left_token, output_root, logger)

    stride = 16 if "dinov3" in args.encoder.lower() else 14
    collate = _build_collate_with_meta(stride)
    dataset = HamlynDataset(
        filelist_path=str(filtered_filelist),
        rootpath=os.path.expanduser(args.root),
        mode="eval",
        size=(args.img_size, args.img_size),
        max_depth=args.max_depth,
        min_depth=args.min_depth,
        local_cache_dir=args.local_cache_dir,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
        collate_fn=collate,
    )
    logger.info("Hamlyn 数据集加载完成，共 %d 张左目图像。", len(dataset))

    config = _prepare_config(args, output_root)
    device_ids = _resolve_device_ids(args)
    model = _prepare_model(config, logger, device_ids)

    metrics = _run_inference_and_eval(model, loader, config, output_root, args, logger)
    for k, v in metrics.items():
        logger.info("  %s: %.6f", k, v)
    _save_metrics(metrics, output_root, args.metrics_json, logger)
    logger.info("Hamlyn 推理与评估完成，输出目录：%s", output_root)


if __name__ == "__main__":
    main()
