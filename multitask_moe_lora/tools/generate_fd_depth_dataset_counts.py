#!/usr/bin/env python3
"""
统计 Foundation Depth (FM) 训练/评估所需数据集的样本数量，并生成 Markdown 表格。

输出文件: FM/dataset_counts_fd_depth.md
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple


WorkspaceRoot = Path(__file__).resolve().parent.parent
OutputPath = WorkspaceRoot / "FM" / "dataset_counts_fd_depth.md"


@dataclass
class SplitCount:
    listed: int
    existing: int
    note: str = ""

    @property
    def status(self) -> str:
        if self.listed == 0:
            return "无数据" if self.existing == 0 else "异常"
        if self.listed == self.existing:
            return "正常"
        if self.existing == 0:
            return "缺失"
        return "部分缺失"


def _normalize_path(base: Path, candidate: str) -> Path:
    candidate_path = Path(candidate.strip())
    if not candidate_path.is_absolute():
        candidate_from_parent = (base / candidate_path).resolve()
        if candidate_from_parent.exists():
            return candidate_from_parent
        candidate_from_root = (WorkspaceRoot / candidate_path).resolve()
        if candidate_from_root.exists():
            return candidate_from_root
        candidate_path = candidate_from_parent
    return candidate_path


def _count_cache_entries(cache_file: Path) -> SplitCount:
    if not cache_file.exists():
        return SplitCount(listed=0, existing=0, note="缓存文件缺失")

    listed = 0
    existing = 0
    missing_examples: List[str] = []

    with cache_file.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            listed += 1
            sample_path = _normalize_path(cache_file.parent, line)
            if sample_path.exists():
                existing += 1
            elif len(missing_examples) < 3:
                missing_examples.append(sample_path.as_posix())

    note = ""
    if listed != existing and missing_examples:
        note = f"缺失示例: {missing_examples[0]}"
    return SplitCount(listed=listed, existing=existing, note=note)


def _parse_filelist_line(line: str) -> Optional[Tuple[Path, Path]]:
    parts = line.split()
    if len(parts) < 2:
        return None
    image_path = Path(parts[0])
    depth_path = Path(parts[1])
    return image_path, depth_path


def _count_filelist_entries(filelist: Path) -> SplitCount:
    if not filelist.exists():
        return SplitCount(listed=0, existing=0, note="文件列表缺失")

    listed = 0
    existing = 0
    missing_examples: List[str] = []

    with filelist.open("r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            parsed = _parse_filelist_line(line)
            if parsed is None:
                continue
            listed += 1
            img_path, depth_path = parsed
            if not img_path.is_absolute():
                img_path = (filelist.parent / img_path).resolve()
            if not depth_path.is_absolute():
                depth_path = (filelist.parent / depth_path).resolve()
            if img_path.exists() and depth_path.exists():
                existing += 1
            elif len(missing_examples) < 3:
                missing_examples.append(img_path.as_posix())

    note = ""
    if listed != existing and missing_examples:
        note = f"缺失示例: {missing_examples[0]}"
    return SplitCount(listed=listed, existing=existing, note=note)


def _count_hamlyn_samples(root: Path) -> SplitCount:
    if not root.exists():
        return SplitCount(listed=0, existing=0, note="Hamlyn 数据目录缺失")

    listed = 0
    existing = 0
    missing_examples: List[str] = []

    image_root = root / "image"
    depth_root = root / "depth"

    def _register_pair(img_path: Path, depth_paths: Sequence[Path]) -> None:
        nonlocal listed, existing, missing_examples
        listed += 1
        if any(path.exists() for path in depth_paths):
            existing += 1
        elif len(missing_examples) < 3:
            missing_examples.append(img_path.as_posix())

    if image_root.exists() and depth_root.exists():
        for seq_dir in sorted(image_root.glob("rectified*")):
            if not seq_dir.is_dir():
                continue
            depth_dir = depth_root / seq_dir.name
            for img_path in seq_dir.glob("*.jpg"):
                stem = img_path.stem
                depth_candidates = [
                    depth_dir / f"{stem}.npy",
                    depth_dir / f"{stem}.png",
                    depth_dir / f"{stem}.tiff",
                ]
                _register_pair(img_path, depth_candidates)
    else:
        for seq_dir in sorted(root.glob("rectified*")):
            if not seq_dir.is_dir():
                continue

            image_dirs = sorted(
                d for d in seq_dir.iterdir()
                if d.is_dir() and d.name.lower().startswith(("image", "color"))
            )
            depth_dirs = sorted(
                d for d in seq_dir.iterdir()
                if d.is_dir() and d.name.lower().startswith("depth")
            )

            if not image_dirs or not depth_dirs:
                continue

            depth_dir_map = {}
            for d_dir in depth_dirs:
                suffix = "".join(ch for ch in d_dir.name if ch.isdigit())
                depth_dir_map[suffix or "default"] = d_dir

            for img_dir in image_dirs:
                suffix = "".join(ch for ch in img_dir.name if ch.isdigit())
                depth_dir = depth_dir_map.get(suffix) or depth_dir_map.get("default")
                if depth_dir is None:
                    continue
                for img_path in img_dir.glob("*.jpg"):
                    stem = img_path.stem
                    depth_candidates = [
                        depth_dir / f"{stem}.png",
                        depth_dir / f"{stem}.npy",
                        depth_dir / f"{stem}.tiff",
                    ]
                    _register_pair(img_path, depth_candidates)

    note = ""
    if listed != existing and missing_examples:
        note = f"缺失示例: {missing_examples[0]}"
    return SplitCount(listed=listed, existing=existing, note=note)


def _count_directory_pairs(root: Path, image_glob: str, depth_suffixes: Iterable[str]) -> SplitCount:
    if not root.exists():
        return SplitCount(listed=0, existing=0, note="目录不存在")

    listed = 0
    existing = 0
    missing_examples: List[str] = []

    for img_path in root.glob(image_glob):
        if not img_path.is_file():
            continue
        listed += 1
        depth_found = False
        for suffix in depth_suffixes:
            depth_path = img_path.with_suffix(suffix)
            if depth_path.exists():
                depth_found = True
                break
        if depth_found:
            existing += 1
        elif len(missing_examples) < 3:
            missing_examples.append(img_path.as_posix())

    note = ""
    if listed != existing and missing_examples:
        note = f"缺失示例: {missing_examples[0]}"
    return SplitCount(listed=listed, existing=existing, note=note)


DatasetSpec = Dict[str, Dict[str, Dict[str, SplitCount]]]


def collect_counts() -> DatasetSpec:
    base = {
        "SCARED": {
            "domain": "LS",
            "train": _count_cache_entries(Path("/home/ziyi/ssde/data/LS/SCARED/cache/train_all_cache.txt")),
        },
        "dVPN": {
            "domain": "LS",
            "train": _count_cache_entries(Path("/home/ziyi/ssde/data/dVPN/cache/train_all_cache.txt")),
        },
        "StereoMIS": {
            "domain": "LS",
            "train": _count_cache_entries(Path("/data/ziyi/multitask/data/LS/StereoMIS/cache/cache_pt/all_cache.txt")),
        },
        "EndoVis2017": {
            "domain": "LS",
            "train": _count_cache_entries(Path("/data/ziyi/multitask/data/LS/EndoVis2017/Endovis2017_seg_depth/cache/train_cache.txt")),
        },
        "EndoVis2018-ISINet": {
            "domain": "LS",
            "train": _count_filelist_entries(Path("/data/ziyi/multitask/data/LS/EndoVis2018/filelists/all.txt")),
        },
        "C3VDv2": {
            "domain": "NO",
            "train": _count_cache_entries(Path("/data/ziyi/multitask/data/NO/c3vdv2/cache/cache.txt")),
        },
        "SimCol": {
            "domain": "NO",
            "train": _count_cache_entries(Path("/home/ziyi/ssde/data/simcol/cache/train_all_cache.txt")),
        },
        "Kidney3D": {
            "domain": "NO",
            "train": _count_cache_entries(Path("/data/ziyi/multitask/data/NO/Kidney3D-CT-depth-seg/cache_pt/train_cache.txt")),
        },
        "Hamlyn Dataset": {
            "domain": "LS",
            "val": _count_cache_entries(Path("/data/ziyi/multitask/data/LS/hamlyn/cache_pt/all_cache.txt")),
        },
        "EndoNeRF": {
            "domain": "LS",
            "val": _count_cache_entries(Path("/data/ziyi/multitask/data/LS/EndoNeRF/cache_pt/all_cache.txt")),
        },
        "C3VD": {
            "domain": "NO",
            "val": _count_cache_entries(Path("/data/ziyi/multitask/data/NO/c3vd/cache/all_cache.txt")),
        },
        "EndoMapper": {
            "domain": "NO",
            "val": _count_cache_entries(Path("/data/ziyi/multitask/data/NO/endomapper_sim/cache/all_cache.txt")),
        },
    }
    return base


def build_markdown_table(spec: DatasetSpec) -> str:
    header = [
        "| 数据集 | 域 | 划分 | 列表条目数 | 实际存在 | 状态 | 备注 |",
        "| --- | --- | --- | ---: | ---: | --- | --- |",
    ]
    rows: List[str] = []

    for dataset_name in sorted(spec.keys()):
        domain = spec[dataset_name]["domain"]
        for split in ("train", "val"):
            if split not in spec[dataset_name]:
                continue
            counts = spec[dataset_name][split]
            rows.append(
                f"| {dataset_name} | {domain} | {split} | {counts.listed} | {counts.existing} | {counts.status} | {counts.note} |"
            )
    return "\n".join(header + rows) + "\n"


def main() -> None:
    spec = collect_counts()
    markdown = "# Foundation Depth 数据集样本统计\n\n" + build_markdown_table(spec)
    OutputPath.parent.mkdir(parents=True, exist_ok=True)
    OutputPath.write_text(markdown, encoding="utf-8")
    print(f"统计结果已写入 {OutputPath}")


if __name__ == "__main__":
    main()
