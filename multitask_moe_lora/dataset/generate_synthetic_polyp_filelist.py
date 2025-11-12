#!/usr/bin/env python3
"""
Generate train/val file lists for SyntheticDatabase polyp dataset.

Each line in the output file references the relative image path (including `vidXX/img/xxxx.jpg`).
Depth and mask counterparts will be resolved automatically by replacing `/img/` with `/z/` and `/mask/`.

默认生成一个 all.txt（无需再划分 train/val）。
"""

import argparse
import os
from pathlib import Path
from typing import Iterable, List


def _collect_entries(root: Path, videos: Iterable[str]) -> List[str]:
    entries: List[str] = []
    for video in videos:
        img_dir = root / video / "img"
        if not img_dir.exists():
            print(f"[WARN] Missing directory: {img_dir}")
            continue
        for img_file in sorted(img_dir.iterdir()):
            if not img_file.is_file():
                continue
            if img_file.suffix.lower() not in {".jpg", ".png", ".jpeg"}:
                continue
            rel_path = img_file.relative_to(root).as_posix()
            entries.append(rel_path)
    return entries


def _write_list(entries: List[str], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for item in entries:
            f.write(item + "\n")
    print(f"Wrote {len(entries)} entries to {path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate all-in-one filelist for SyntheticDatabase polyp dataset.")
    parser.add_argument("--root", type=Path, required=True, help="Root path of SyntheticDatabase_testingset_PolypSize")
    parser.add_argument("--output-dir", type=Path, default=None, help="Directory to save file list (default: <root>/filelists)")
    parser.add_argument("--videos", type=str, nargs="*", default=None, help="Video folders to include (default: autodetect all vidXX)")
    parser.add_argument("--output-name", type=str, default="all.txt", help="File name for generated list")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    output_dir = args.output_dir or (root / "filelists")

    videos = args.videos
    if not videos:
        videos = sorted([p.name for p in root.iterdir() if p.is_dir() and p.name.startswith("vid")])
    entries = _collect_entries(root, videos)
    if not entries:
        raise RuntimeError("No entries collected. Please verify --videos and dataset root.")
    _write_list(entries, output_dir / args.output_name)


if __name__ == "__main__":
    main()
