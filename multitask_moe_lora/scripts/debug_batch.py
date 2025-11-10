#!/usr/bin/env python3
"""
Debug script to replay a saved bad batch and run forward/backward with anomaly detection.
Usage example:
    PYTHONPATH=/data/ziyi/multitask/code/DepthAnythingV2 \
      python scripts/debug_batch.py \
      --batch-path /path/to/bad_batch.pt \
      --checkpoint /path/to/checkpoint.pth \
      --device cuda:0
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from multitask_moe_lora.util.config import TrainingConfig
from multitask_moe_lora.util.model_setup import create_and_setup_model, load_weights_from_checkpoint
from multitask_moe_lora.util.loss import SiLogLoss


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Replay a saved bad batch to pinpoint NaNs.")
    parser.add_argument("--batch-path", required=True, type=Path, help="Path to bad_batch_*.pt")
    parser.add_argument("--checkpoint", required=True, type=Path, help="Checkpoint (.pth) compatible with the batch.")
    parser.add_argument("--device", default="cuda:0", help="Device to run on (e.g., cuda:0 or cpu).")
    parser.add_argument("--encoder", default="vits", help="Backbone encoder name.")
    parser.add_argument("--features", type=int, default=64, help="Depth head feature size.")
    parser.add_argument("--max-depth", type=float, default=0.3, help="Max depth used during training.")
    parser.add_argument("--min-depth", type=float, default=1e-6, help="Min depth used during training.")
    parser.add_argument("--mode", default="original", help="Training mode (original, lora-only, etc.).")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.batch_path.exists():
        raise FileNotFoundError(f"Batch file not found: {args.batch_path}")
    if not args.checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")

    device = torch.device(args.device)
    torch.autograd.set_detect_anomaly(True)

    batch = torch.load(args.batch_path, map_location="cpu")
    config = TrainingConfig(
        encoder=args.encoder,
        features=args.features,
        num_classes=1,
        min_depth=args.min_depth,
        max_depth=args.max_depth,
        dataset_config_name="fd_depth_fm_v1",
        dataset_modality="fd",
        mode=args.mode,
        save_path=".",
    )
    config.resume_from = str(args.checkpoint)
    config.resume_full_state = False

    dummy_logger = type("L", (object,), {"info": print, "debug": print, "warning": print, "error": print})
    model = create_and_setup_model(config, logger=dummy_logger)  # loads to cuda by default
    load_weights_from_checkpoint(model, None, None, None, None, None, None, config, logger=dummy_logger)
    model.to(device)
    model.train()

    images = batch["image"].to(device)
    depth_gt = batch["depth"].to(device)
    mask = batch["valid_mask"].to(device)

    loss_fn = SiLogLoss().to(device)

    outputs = model(images, task="depth")
    pred_depth = outputs["depth"].unsqueeze(1)
    loss = loss_fn(pred_depth, depth_gt, mask)
    print(f"Forward loss: {loss.item():.6f}")
    loss.backward()
    print("Backward completed without NaN.")


if __name__ == "__main__":
    main()
