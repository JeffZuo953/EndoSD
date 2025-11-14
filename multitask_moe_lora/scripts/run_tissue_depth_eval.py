#!/usr/bin/env python3
"""
Depth-only inference + evaluation for specified dVRK tissue sequences.
"""

from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in os.sys.path:
    os.sys.path.insert(0, str(REPO_ROOT))

from scripts.run_downstream_jobs import (  # type: ignore
    MultitaskInferencer,
    _depth_to_colormap,
    _ensure_dir,
    _compute_depth_metrics,
)


DEFAULT_SEQUENCES = [
    "~/downstreamjobs/dVRK_demo/episode_Forcep_LND_demo/episode_Forcep_LND_tissue_collision",
    "~/downstreamjobs/dVRK_demo/episode_Forcep_LND_demo/episode_Forcep_LND_tissue_collision2",
    "~/downstreamjobs/dVRK_demo/episold_LND_demo/episode_LND_Tissue",
    "~/downstreamjobs/dVRK_demo/episode_Forcep_demo/episode_Forcep_Tissue",
]


def _frame_to_depth(img_path: Path) -> Path:
    fname = img_path.name.replace("frame_", "depth_").replace(".png", ".npy")
    return img_path.parent.parent / "depth_imgs" / fname


def _relative_path(path: Path, base: Path) -> Path:
    try:
        return path.relative_to(base)
    except ValueError:
        return Path(path.name)


def process_sequences(
    inferencer: MultitaskInferencer,
    sequences: List[Path],
    output_root: Path,
    base_root: Path,
) -> Dict[str, Dict[str, float]]:
    records = []
    summary_totals: Dict[str, float] = defaultdict(float)
    summary_count = 0
    per_sequence_totals: Dict[str, Dict[str, float]] = defaultdict(lambda: defaultdict(float))
    per_sequence_counts: Dict[str, int] = defaultdict(int)

    depth_dir = output_root / "depth_npy"
    depth_vis_dir = output_root / "depth_vis"
    metrics_jsonl = output_root / "depth_metrics.jsonl"
    summary_json = output_root / "depth_metrics_summary.json"

    for seq in sequences:
        left_dir = seq / "left_imgs"
        if not left_dir.exists():
            continue
        for img_path in sorted(left_dir.glob("*.png")):
            img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
            if img is None:
                continue
            depth_map, _, _ = inferencer.infer(img)
            rel_path = _relative_path(img_path, base_root)

            depth_save = depth_dir / rel_path.with_suffix(".npy")
            depth_vis_save = depth_vis_dir / rel_path.with_suffix(".png")
            _ensure_dir(depth_save)
            np.save(depth_save, depth_map.astype(np.float32))
            _ensure_dir(depth_vis_save)
            cv2.imwrite(str(depth_vis_save), _depth_to_colormap(depth_map, inferencer.max_depth))

            gt_path = _frame_to_depth(img_path)
            if not gt_path.exists():
                continue
            gt_depth = np.load(gt_path)
            metrics = _compute_depth_metrics(depth_map, gt_depth)
            if not metrics:
                continue
            metrics["sequence"] = str(seq.relative_to(base_root))
            metrics["relative_path"] = str(rel_path)
            records.append(metrics)
            summary_count += 1
            for key, value in metrics.items():
                if key in {"sequence", "relative_path"}:
                    continue
                summary_totals[key] += float(value)
                per_sequence_totals[metrics["sequence"]][key] += float(value)
            per_sequence_counts[metrics["sequence"]] += 1

    _ensure_dir(metrics_jsonl)
    with open(metrics_jsonl, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")

    summary = {}
    if summary_count > 0:
        summary = {k: v / summary_count for k, v in summary_totals.items()}
        summary["samples"] = summary_count
    per_seq_summary = []
    for seq, totals in per_sequence_totals.items():
        count = per_sequence_counts[seq]
        seq_summary = {k: v / count for k, v in totals.items()}
        seq_summary["sequence"] = seq
        seq_summary["samples"] = count
        per_seq_summary.append(seq_summary)

    payload = {
        "overall": summary,
        "per_sequence": per_seq_summary,
    }
    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Depth-only inference for dVRK tissue sequences.")
    parser.add_argument("--checkpoint", required=True, help="Checkpoint path (.pth)")
    parser.add_argument("--output-root", default="~/downstreamjobs/save/tissue", help="Output directory root")
    parser.add_argument("--base-root", default="~/downstreamjobs/dVRK_demo", help="Base path for relative outputs")
    parser.add_argument("--sequences", nargs="+", default=DEFAULT_SEQUENCES, help="List of sequence directories (left_imgs/depth_imgs inside)")
    parser.add_argument("--img-size", type=int, default=518)
    parser.add_argument("--max-depth", type=float, default=0.3)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    sequences = [Path(os.path.expanduser(p)).resolve() for p in args.sequences]
    base_root = Path(os.path.expanduser(args.base_root)).resolve()
    output_root = Path(os.path.expanduser(args.output_root)).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    inferencer = MultitaskInferencer(
        checkpoint=Path(os.path.expanduser(args.checkpoint)),
        device=args.device if os.environ.get("CUDA_VISIBLE_DEVICES") else "cuda",
        encoder="vitb",
        features=64,
        num_classes=10,
        max_depth=args.max_depth,
        img_size=args.img_size,
        seg_head_type="sf",
        seg_input_type="from_depth",
        mode="endounid",
        use_semantic_tokens=True,
        semantic_token_count=10,
    )

    summary = process_sequences(inferencer, sequences, output_root, base_root)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
