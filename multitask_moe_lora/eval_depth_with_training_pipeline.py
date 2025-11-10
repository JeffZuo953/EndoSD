#!/usr/bin/env python3
"""
基于训练流水线的深度/分割评估脚本

相较于旧版 eval_hamlyn_depth.py，本脚本完全复用
train_multitask_depth_seg.py / util 模块中的
数据集构建、模型加载以及验证逻辑，
确保评估结果与训练日志一致。
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parent
PARENT_ROOT = REPO_ROOT.parent
if str(PARENT_ROOT) not in sys.path:
    sys.path.insert(0, str(PARENT_ROOT))

from multitask_moe_lora.util.config import (  # noqa: E402
    TrainingConfig,
    args_to_config,
    create_parser,
    validate_config,
)
from multitask_moe_lora.util.data_utils import (  # noqa: E402
    _make_collate_fn,
    _sample_dataset_by_step,
    create_datasets,
    summarize_loader_composition,
)
from multitask_moe_lora.util.model_setup import (  # noqa: E402
    create_and_setup_model,
    load_weights_from_checkpoint,
)
from multitask_moe_lora.util.validation import validate_and_visualize  # noqa: E402


def _build_eval_parser() -> argparse.ArgumentParser:
    parser = create_parser()
    parser.add_argument(
        "--checkpoint",
        required=True,
        help="待评估的 checkpoint (.pth)，会通过训练同款 remap/resize 流程加载",
    )
    parser.add_argument(
        "--metrics-json",
        type=str,
        default=None,
        help="可选：保存最终指标到 JSON 文件",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=8,
        help="DataLoader worker 数量（默认 8，可根据机器调整）",
    )
    parser.add_argument(
        "--eval-epoch-tag",
        type=int,
        default=0,
        help="用于日志展示的 epoch 编号（不会影响计算，只影响打印）",
    )
    parser.add_argument(
        "--depth-only",
        action="store_true",
        help="只评估深度，跳过分割/相机头验证",
    )
    return parser


def _prepare_config(args: argparse.Namespace) -> TrainingConfig:
    config = args_to_config(args)
    errors = validate_config(config)
    if errors:
        msg = "配置非法:\n" + "\n".join(f"  - {err}" for err in errors)
        raise ValueError(msg)

    if config.path_transform_name and config.path_transform_name.lower() == "none":
        config.path_transform_name = None

    config.resume_from = args.checkpoint
    config.resume_full_state = False
    return config


def _setup_logger(save_path: Path) -> logging.Logger:
    save_path.mkdir(parents=True, exist_ok=True)
    log_file = save_path / "eval.log"
    logger = logging.getLogger("eval")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    fmt = logging.Formatter(
        fmt="[%(asctime)s][%(levelname)5s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(sh)

    fh = logging.FileHandler(log_file, mode="w")
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    return logger


def _build_eval_loaders(
    config: TrainingConfig,
    num_workers: int,
    logger: logging.Logger,
    skip_seg: bool,
) -> Tuple[DataLoader, Optional[DataLoader]]:
    _, val_depth_dataset, _, val_seg_dataset = create_datasets(config)
    if val_depth_dataset is None:
        raise ValueError("无法创建深度验证数据集，请检查 dataset 配置。")

    val_step = config.val_sample_step if config.val_sample_step == -1 else max(config.val_sample_step, 1)
    val_min = max(int(getattr(config, "val_min_samples_per_dataset", 0) or 0), 0)

    val_depth_dataset = _sample_dataset_by_step(val_depth_dataset, val_step, val_min)
    if val_seg_dataset is not None:
        val_seg_dataset = _sample_dataset_by_step(val_seg_dataset, val_step, val_min)

    stride = 16 if "dinov3" in config.encoder.lower() else 14
    collate_fn = _make_collate_fn(stride)

    depth_loader = DataLoader(
        val_depth_dataset,
        batch_size=config.val_bs,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
        collate_fn=collate_fn,
    )

    seg_loader: Optional[DataLoader]
    if skip_seg or val_seg_dataset is None:
        seg_loader = None
    else:
        seg_loader = DataLoader(
            val_seg_dataset,
            batch_size=config.val_bs,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=False,
            collate_fn=collate_fn,
        )

    summaries = summarize_loader_composition(depth_loader)
    if summaries:
        logger.info("[Depth] 共 %d 个子数据集，总计 %d 条样本。", len(summaries), summaries[0]["total"])
        for entry in summaries:
            logger.info(
                "    %s (%s) -> %d / %d",
                entry["name"],
                entry.get("dataset_type", "unknown"),
                entry["count"],
                entry["total"],
            )

    if seg_loader is not None:
        seg_summaries = summarize_loader_composition(seg_loader)
        if seg_summaries:
            logger.info("[Seg] 共 %d 个子数据集，总计 %d 条样本。", len(seg_summaries), seg_summaries[0]["total"])

    return depth_loader, seg_loader


def _save_metrics(payload: Dict[str, Any], output_path: Path, logger: logging.Logger) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    logger.info("指标已写入 %s", output_path)


def main() -> None:
    parser = _build_eval_parser()
    args = parser.parse_args()

    config = _prepare_config(args)
    logger = _setup_logger(Path(config.save_path))
    logger.info("=== Eval config ===")
    logger.info("Dataset config: %s (modality=%s)", config.dataset_config_name, config.dataset_modality)
    logger.info("Val include: %s", config.val_dataset_include)
    logger.info("Checkpoint: %s", config.resume_from)

    if not torch.cuda.is_available():
        raise RuntimeError("当前环境无可用 GPU，训练版验证依赖 CUDA。请在 GPU 机器上运行。")
    torch.cuda.set_device(0)

    depth_loader, seg_loader = _build_eval_loaders(config, args.num_workers, logger, args.depth_only)

    model = create_and_setup_model(config, logger)
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

    epoch_tag = args.eval_epoch_tag
    logger.info("=== 开始深度评估 Epoch %s ===", epoch_tag)
    depth_metrics = validate_and_visualize(model, depth_loader, "depth", epoch_tag, config, writer=None, logger=logger)

    seg_metrics: Optional[Dict[str, Any]] = None
    if seg_loader is not None:
        logger.info("=== 开始分割评估 Epoch %s ===", epoch_tag)
        seg_metrics = validate_and_visualize(model, seg_loader, "seg", epoch_tag, config, writer=None, logger=logger)
    else:
        logger.info("Segmentation loader 为空或被跳过，直接结束。")

    if args.metrics_json:
        metrics_payload = {
            "depth": depth_metrics,
            "segmentation": seg_metrics,
        }
        _save_metrics(metrics_payload, Path(args.metrics_json), logger)


if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True
    main()
