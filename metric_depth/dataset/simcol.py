import argparse
from dataclasses import dataclass
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

_IMAGE_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_IMAGE_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
_DEPTH_SCALE = 65535.0


@dataclass(frozen=True)
class SimcolSample:
    image_path: Path
    depth_path: Path
    seg_path: Optional[Path] = None


class Simcol(Dataset):

    def __init__(
        self,
        filelist_path: str,
        mode: str,
        size: Optional[Tuple[int, int]] = None,
        max_depth: float = 0.3,
        root_dir: Optional[str] = None,
        max_image_size: int = 518,
    ):
        self.mode = mode
        self.max_depth = max_depth
        self.max_valid_depth = 0.3
        self.max_image_size = max_image_size
        if size is not None:
            self.max_image_size = max(size)

        default_root = os.environ.get("SIMCOL_ROOT", "~/ssde/data/simcol")
        self.root_dir = Path(root_dir or default_root).expanduser().resolve()

        filelist_path = Path(filelist_path).expanduser()
        if not filelist_path.is_absolute():
            filelist_path = (self.root_dir / filelist_path).resolve()
        if not filelist_path.exists():
            raise FileNotFoundError(f"File list not found: {filelist_path}")

        self.samples: List[SimcolSample] = self._load_samples(filelist_path)

    def _resolve_path(self, path_str: str) -> Path:
        path = Path(path_str).expanduser()
        if not path.is_absolute():
            path = self.root_dir / path
        return path.resolve(strict=False)

    def _load_samples(self, filelist_path: Path) -> List[SimcolSample]:
        samples: List[SimcolSample] = []
        with filelist_path.open("r") as f:
            entries = [line.strip() for line in f if line.strip() and not line.startswith("#")]

        for entry in entries:
            tokens = entry.split()
            primary_path = self._resolve_path(tokens[0])

            if len(tokens) == 1 and primary_path.is_dir():
                samples.extend(self._gather_from_directory(primary_path))
                continue

            if len(tokens) >= 2:
                image_path = self._resolve_path(tokens[0])
                depth_path = self._resolve_path(tokens[1])
                seg_path = self._resolve_path(tokens[2]) if len(tokens) >= 3 else None
                if not image_path.exists() or not depth_path.exists():
                    raise FileNotFoundError(f"Missing data for entry: {entry}")
                samples.append(SimcolSample(image_path=image_path,
                                            depth_path=depth_path,
                                            seg_path=seg_path if seg_path and seg_path.exists() else None))
                continue

            raise ValueError(f"Unrecognized entry in file list: {entry}")

        if not samples:
            raise RuntimeError(f"No samples found using file list: {filelist_path}")

        return samples

    def _gather_from_directory(self, directory: Path) -> List[SimcolSample]:
        if not directory.exists():
            raise FileNotFoundError(f"Data directory does not exist: {directory}")

        samples: List[SimcolSample] = []
        color_files = sorted(directory.glob("FrameBuffer_*.png"))
        for color_path in color_files:
            frame_id = color_path.stem.split("_")[-1]
            depth_path = directory / f"Depth_{frame_id}.png"
            if not depth_path.exists():
                continue
            seg_path = self._locate_segmentation_file(directory, frame_id)
            samples.append(SimcolSample(image_path=color_path,
                                        depth_path=depth_path,
                                        seg_path=seg_path))

        if not samples:
            raise RuntimeError(f"No valid samples found in directory: {directory}")

        return samples

    @staticmethod
    def _locate_segmentation_file(directory: Path, frame_id: str) -> Optional[Path]:
        candidates = (
            f"Segmentation_{frame_id}.png",
            f"Segment_{frame_id}.png",
            f"Mask_{frame_id}.png",
            f"Semantic_{frame_id}.png",
            f"Label_{frame_id}.png",
        )
        for name in candidates:
            candidate = directory / name
            if candidate.exists():
                return candidate
        return None

    def __getitem__(self, item: int) -> Dict[str, torch.Tensor]:
        record = self.samples[item]
        image_bgr = cv2.imread(str(record.image_path), cv2.IMREAD_COLOR)
        if image_bgr is None:
            raise FileNotFoundError(f"Unable to read image: {record.image_path}")

        black_mask_raw = np.all(image_bgr == 0, axis=2)
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

        depth_raw = cv2.imread(str(record.depth_path), cv2.IMREAD_UNCHANGED)
        if depth_raw is None:
            raise FileNotFoundError(f"Unable to read depth map: {record.depth_path}")
        if depth_raw.ndim == 3:
            depth_raw = depth_raw[..., 0]

        depth = depth_raw.astype(np.float32) / _DEPTH_SCALE * self.max_depth

        height, width = image_rgb.shape[:2]
        longest_side = max(height, width)
        scale = 1.0
        if longest_side > self.max_image_size:
            scale = self.max_image_size / float(longest_side)
        new_width = max(1, int(round(width * scale)))
        new_height = max(1, int(round(height * scale)))

        if scale != 1.0:
            image_rgb = cv2.resize(image_rgb, (new_width, new_height), interpolation=cv2.INTER_AREA)
            depth = cv2.resize(depth, (new_width, new_height), interpolation=cv2.INTER_NEAREST)
            black_mask = cv2.resize(black_mask_raw.astype(np.uint8),
                                    (new_width, new_height),
                                    interpolation=cv2.INTER_NEAREST).astype(bool)
        else:
            black_mask = black_mask_raw

        valid_mask = (depth > 0.0) & (depth <= self.max_valid_depth) & (~black_mask)
        depth = np.where(valid_mask, depth, 0.0)

        image_norm = (image_rgb - _IMAGE_MEAN) / _IMAGE_STD
        image_tensor = torch.from_numpy(np.transpose(image_norm, (2, 0, 1)).astype(np.float32))
        depth_tensor = torch.from_numpy(depth.astype(np.float32))
        valid_mask_tensor = torch.from_numpy(valid_mask.astype(np.bool_))

        if record.seg_path and record.seg_path.exists():
            seg = cv2.imread(str(record.seg_path), cv2.IMREAD_UNCHANGED)
            if seg is None:
                seg_mask = valid_mask.astype(np.uint8)
            else:
                if seg.ndim == 3:
                    seg = cv2.cvtColor(seg, cv2.COLOR_BGR2GRAY)
                if scale != 1.0:
                    seg = cv2.resize(seg, (new_width, new_height), interpolation=cv2.INTER_NEAREST)
                seg_mask = seg.astype(np.uint8)
        else:
            seg_mask = valid_mask.astype(np.uint8)

        seg_mask = np.where(valid_mask, seg_mask, 0).astype(np.uint8)
        seg_tensor = torch.from_numpy(seg_mask)

        return {
            "image": image_tensor,
            "depth": depth_tensor,
            "valid_mask": valid_mask_tensor,
            "semseg_mask": seg_tensor,
            "image_path": str(record.image_path),
            "depth_path": str(record.depth_path),
            "max_depth": self.max_depth,
        }

    def __len__(self) -> int:
        return len(self.samples)


def _prepare_payload(sample: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    payload: Dict[str, torch.Tensor] = {
        "image": sample["image"].to(torch.float16).contiguous(),
        "depth": sample["depth"].to(torch.float32).contiguous(),
        "valid_mask": sample["valid_mask"].to(torch.bool).contiguous(),
        "semseg_mask": sample["semseg_mask"].to(torch.uint8).contiguous(),
    }
    payload["image_path"] = sample["image_path"]
    payload["depth_path"] = sample["depth_path"]
    payload["max_depth"] = float(sample["max_depth"])
    return payload


def _generate_cache_for_split(
    dataset: Simcol,
    split_name: str,
    cache_root: Path,
    data_root: Path,
) -> Path:
    cache_root.mkdir(parents=True, exist_ok=True)
    split_root = cache_root / split_name
    split_root.mkdir(parents=True, exist_ok=True)
    filelist_path = cache_root / f"{split_name}_cache.txt"

    cache_paths: List[str] = []
    iterator = tqdm(range(len(dataset)), desc=f"Generating {split_name} cache", unit="sample")
    for idx in iterator:
        sample = dataset[idx]
        rel_path = Path(sample["image_path"]).resolve().relative_to(data_root.resolve())
        cache_path = split_root / rel_path.with_suffix(".pt")
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(_prepare_payload(sample), cache_path)
        cache_paths.append(str(cache_path))

    with filelist_path.open("w") as f:
        for path in cache_paths:
            f.write(path + "\n")

    return filelist_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate SimCol dataset cache files.")
    parser.add_argument("--root", type=str, default="~/ssde/data/simcol", help="Dataset root directory.")
    parser.add_argument("--train-split", type=str, default="misc/train_file.txt", help="Train split file relative to root.")
    parser.add_argument("--test-split", type=str, default="misc/test_file.txt", help="Test split file relative to root.")
    parser.add_argument("--output", type=str, default=None, help="Cache output directory (defaults to <root>/cache).")
    parser.add_argument("--max-depth", type=float, default=0.3, help="Maximum depth value in meters.")
    parser.add_argument("--max-image-size", type=int, default=518, help="Upper bound for the longer image side.")
    return parser.parse_args()


def main():
    args = parse_args()
    root_dir = Path(args.root).expanduser().resolve()
    if not root_dir.exists():
        raise FileNotFoundError(f"Dataset root does not exist: {root_dir}")

    cache_root = Path(args.output).expanduser().resolve() if args.output else (root_dir / "cache")
    train_split_path = (root_dir / args.train_split).resolve()
    test_split_path = (root_dir / args.test_split).resolve()

    train_dataset = Simcol(
        filelist_path=str(train_split_path),
        mode="train",
        max_depth=args.max_depth,
        root_dir=str(root_dir),
        max_image_size=args.max_image_size,
    )
    train_cache_list = _generate_cache_for_split(train_dataset, "train", cache_root, root_dir)
    print(f"Train cache list saved to {train_cache_list}")

    if test_split_path.exists():
        test_dataset = Simcol(
            filelist_path=str(test_split_path),
            mode="test",
            max_depth=args.max_depth,
            root_dir=str(root_dir),
            max_image_size=args.max_image_size,
        )
        test_cache_list = _generate_cache_for_split(test_dataset, "test", cache_root, root_dir)
        print(f"Test cache list saved to {test_cache_list}")
    else:
        print(f"Test split file not found, skipping: {test_split_path}")


if __name__ == "__main__":
    main()
