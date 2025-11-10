#!/usr/bin/env python3
"""
自动生成 Hamlyn 数据集的文件列表

使用方法:
    python generate_hamlyn_filelist.py --base_dir /path/to/hamlyn --output train.txt

参数:
    --base_dir: Hamlyn 数据集的根目录
    --output: 输出文件列表的路径
    --sequences: 可选，指定要包含的序列名称，例如 "rectified01,rectified02"
"""

import argparse
from collections import OrderedDict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

EVAL_PRESET_KEY = "eval"
PRESET_FILTERS: Dict[str, Dict[str, Sequence[str]]] = {
    EVAL_PRESET_KEY: {
        "sequences": ("rectified01", "rectified04", "rectified19", "rectified20"),
        "image_subdirs": ("image01",),
        "depth_subdirs": ("depth01",),
    }
}


def generate_hamlyn_filelist(
    base_dir,
    output_file,
    sequences: Optional[Sequence[str]] = None,
    image_subdirs: Optional[Sequence[str]] = None,
    depth_subdirs: Optional[Sequence[str]] = None,
):
    """
    自动生成 Hamlyn 数据集的文件列表

    Args:
        base_dir (str): Hamlyn 数据集的根目录
        output_file (str): 输出文件列表的路径
        sequences (list): 要包含的序列名称列表，如果为 None 则包含所有序列
        image_subdirs (list): 仅包含指定的图像子目录 (例如 image01)
        depth_subdirs (list): 仅包含指定的深度子目录 (例如 depth01)
    """
    file_list = []
    base_path = Path(base_dir).expanduser()
    try:
        base_path = base_path.resolve()
    except OSError:
        base_path = base_path.absolute()

    # 如果没有指定序列，则查找所有 rectified 序列
    if sequences is None:
        seq_dirs = sorted(base_path.glob("rectified*"))
    else:
        seq_dirs = [base_path / seq for seq in sequences]

    allowed_image_dirs = _normalize_filter(image_subdirs)
    allowed_depth_dirs = _normalize_filter(depth_subdirs)

    for seq_dir in seq_dirs:
        if not seq_dir.is_dir():
            print(f"警告: 序列目录不存在: {seq_dir}")
            continue

        pairs = _discover_image_depth_pairs(
            seq_dir,
            allowed_image_dirs=allowed_image_dirs,
            allowed_depth_dirs=allowed_depth_dirs,
        )
        if not pairs:
            print(f"警告: 在 {seq_dir} 中未找到匹配的图像/深度目录")
            continue

        seq_total = 0

        for token, image_dir, depth_dir in pairs:
            img_files = _gather_files(image_dir, exts=("*.jpg", "*.jpeg", "*.png"))

            if len(img_files) == 0:
                print(f"警告: 在 {image_dir} 中没有找到图像文件 (*.jpg/*.png)")
                continue

            print(f"处理序列: {seq_dir.name} / {image_dir.name}")
            print(f"  找到 {len(img_files)} 个图像文件")

            for img_file in img_files:
                frame_id = img_file.stem  # 不带扩展名的文件名

                # 检查对应的深度文件是否存在
                depth_file = _resolve_matching_file(depth_dir, frame_id)
                if depth_file is None:
                    print(f"  警告: 缺少深度文件: {depth_dir}/{frame_id}.*")
                    continue

                frame_token = f"{token}/{frame_id}" if token else frame_id
                file_list.append(f"{seq_dir.as_posix()} {frame_token}\n")
                seq_total += 1

        if seq_total == 0:
            print(f"  提示: 跳过 {seq_dir.name}，未找到有效的图像-深度配对。")
        else:
            print(f"  序列 {seq_dir.name} 累计样本数: {seq_total}")

    # 写入文件
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, 'w') as f:
        f.writelines(file_list)

    print(f"\n生成完成!")
    print(f"  总样本数: {len(file_list)}")
    print(f"  文件列表保存到: {output_file}")

    # 显示前几行作为示例
    if len(file_list) > 0:
        print(f"\n文件列表示例 (前5行):")
        for i, line in enumerate(file_list[:5]):
            print(f"  {line.strip()}")


def main():
    parser = argparse.ArgumentParser(
        description="生成 Hamlyn 数据集的文件列表",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 生成所有序列的文件列表
  python generate_hamlyn_filelist.py --base_dir /data/hamlyn --output /data/hamlyn/train.txt

  # 只生成特定序列的文件列表
  python generate_hamlyn_filelist.py --base_dir /data/hamlyn --output /data/hamlyn/train.txt --sequences rectified01,rectified02
        """
    )

    parser.add_argument(
        "--base_dir",
        type=str,
        required=True,
        help="Hamlyn 数据集的根目录"
    )

    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="输出文件列表的路径"
    )

    parser.add_argument(
        "--sequences",
        type=str,
        default=None,
        help="要包含的序列名称，用逗号分隔，例如 'rectified01,rectified02'"
    )
    parser.add_argument(
        "--image-subdirs",
        type=str,
        default=None,
        help="仅使用指定图像子目录 (逗号分隔)，例如 'image01,image02'",
    )
    parser.add_argument(
        "--depth-subdirs",
        type=str,
        default=None,
        help="仅使用指定深度子目录 (逗号分隔)，例如 'depth01'",
    )
    parser.add_argument(
        "--preset",
        type=str,
        choices=sorted(PRESET_FILTERS.keys()),
        default=None,
        help="应用预设过滤配置，例如 'eval' 仅使用 P1/P2_0/P2_6/P2_7 的 image01 + depth01。",
    )

    args = parser.parse_args()

    # 解析序列列表
    sequences = None
    if args.sequences:
        sequences = [s.strip() for s in args.sequences.split(',')]

    image_subdirs = _parse_list_argument(args.image_subdirs)
    depth_subdirs = _parse_list_argument(args.depth_subdirs)

    if args.preset:
        preset = PRESET_FILTERS.get(args.preset)
        if preset is not None:
            if "sequences" in preset:
                sequences = list(preset["sequences"])
            if "image_subdirs" in preset:
                image_subdirs = list(preset["image_subdirs"])
            if "depth_subdirs" in preset:
                depth_subdirs = list(preset["depth_subdirs"])
            print(
                f"应用预设 '{args.preset}': "
                f"sequences={sequences}, image_subdirs={image_subdirs}, depth_subdirs={depth_subdirs}"
            )

    # 生成文件列表
    generate_hamlyn_filelist(
        args.base_dir,
        args.output,
        sequences=sequences,
        image_subdirs=image_subdirs,
        depth_subdirs=depth_subdirs,
    )


def _parse_list_argument(value: Optional[str]) -> Optional[List[str]]:
    if not value:
        return None
    result = [item.strip() for item in value.split(",")]
    return [item for item in result if item]


def _normalize_filter(values: Optional[Sequence[str]]) -> Optional[set[str]]:
    if not values:
        return None
    normalized = {val.strip().lower() for val in values if val and val.strip()}
    return normalized or None


def _discover_image_depth_pairs(
    seq_dir: Path,
    allowed_image_dirs: Optional[set[str]] = None,
    allowed_depth_dirs: Optional[set[str]] = None,
) -> List[Tuple[str, Path, Path]]:
    image_dirs = _find_candidate_directories(seq_dir, ("color", "image"))
    if not image_dirs:
        return []

    if allowed_image_dirs is not None:
        filtered_image_dirs = OrderedDict(
            (name, path) for name, path in image_dirs.items() if name.lower() in allowed_image_dirs
        )
        if not filtered_image_dirs:
            print(f"  提示: {seq_dir.name} 中没有匹配的图像子目录 {sorted(allowed_image_dirs)}")
            return []
        image_dirs = filtered_image_dirs

    depth_dirs = _find_candidate_directories(seq_dir, ("depth",))
    if not depth_dirs:
        print(f"警告: 在 {seq_dir} 中未找到深度目录 (depth*)")
        return []

    if allowed_depth_dirs is not None:
        filtered_depth_dirs = OrderedDict(
            (name, path) for name, path in depth_dirs.items() if name.lower() in allowed_depth_dirs
        )
        if not filtered_depth_dirs:
            print(f"  提示: {seq_dir.name} 中没有匹配的深度子目录 {sorted(allowed_depth_dirs)}")
            return []
        depth_dirs = filtered_depth_dirs

    depth_lookup = {name.lower(): path for name, path in depth_dirs.items()}
    default_depth = depth_lookup.get("depth") or next(iter(depth_lookup.values()), None)

    pairs: List[Tuple[str, Path, Path]] = []
    multiple = len(image_dirs) > 1

    for name, image_path in image_dirs.items():
        lower_name = name.lower()
        token = ""
        if multiple or not lower_name.startswith("color"):
            token = name

        depth_path = _match_depth_directory(lower_name, depth_lookup, default_depth)
        if depth_path is None:
            print(f"  警告: 序列 {seq_dir.name} 的图像目录 {name} 找不到对应的深度目录，跳过该子目录。")
            continue
        pairs.append((token, image_path, depth_path))

    return pairs


def _find_candidate_directories(base_path: Path, prefixes: Tuple[str, ...]) -> OrderedDict[str, Path]:
    result: OrderedDict[str, Path] = OrderedDict()
    lowercase_seen: set[str] = set()

    for child in sorted(base_path.iterdir(), key=lambda p: p.name.lower()):
        if not child.is_dir():
            continue
        lower = child.name.lower()
        if any(lower == pref or lower.startswith(pref) for pref in prefixes):
            if lower not in lowercase_seen:
                result[child.name] = child
                lowercase_seen.add(lower)

    for pref in prefixes:
        candidate = base_path / pref
        if candidate.is_dir():
            lower = pref.lower()
            if lower not in lowercase_seen:
                result[pref] = candidate
                lowercase_seen.add(lower)

    return result


def _match_depth_directory(image_key: str, depth_lookup: Dict[str, Path], default_depth: Optional[Path]) -> Optional[Path]:
    candidates: List[str] = []

    if image_key in depth_lookup:
        candidates.append(image_key)

    if image_key.startswith("image"):
        suffix = image_key[len("image") :]
        suffix_clean = suffix.lstrip("_-")
        for candidate in (suffix, suffix_clean, f"depth{suffix}", f"depth{suffix_clean}"):
            if candidate and candidate in depth_lookup:
                candidates.append(candidate)
        digits = "".join(ch for ch in suffix if ch.isdigit())
    else:
        digits = "".join(ch for ch in image_key if ch.isdigit())

    if digits:
        for variant in {
            digits,
            digits.zfill(2),
            digits.lstrip("0") or "0",
            f"depth{digits}",
            f"depth{digits.zfill(2)}",
        }:
            if variant and variant in depth_lookup:
                candidates.append(variant)

    for candidate in candidates:
        if candidate in depth_lookup:
            return depth_lookup[candidate]

    return default_depth


def _gather_files(directory: Path, exts: Iterable[str]) -> list[Path]:
    files: list[Path] = []
    for pattern in exts:
        files.extend(directory.glob(pattern))
    files = [f for f in files if f.is_file()]
    return sorted(dict.fromkeys(files))  # preserve order, remove duplicates


def _resolve_matching_file(directory: Path, frame_id: str) -> Optional[Path]:
    exact_png = directory / f"{frame_id}.png"
    if exact_png.exists():
        return exact_png
    matches = list(directory.glob(f"{frame_id}.*"))
    matches = [m for m in matches if m.is_file()]
    if not matches:
        return None
    return sorted(matches)[0]


if __name__ == "__main__":
    main()
