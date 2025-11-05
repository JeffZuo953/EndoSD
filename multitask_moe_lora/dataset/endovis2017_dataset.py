import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision.transforms import Compose

from .transform import NormalizeImage, PrepareForNet, Resize
from .utils import compute_valid_mask, remap_labels, map_ls_semseg_to_10_classes

ENDOVIS2017_REMAP = {
    0: 0,
    1: 1,
    2: 2,
    3: 3,
    4: 4,
    5: 5,
    6: 6,
    7: 7,
    255: 255,
}


_CLASS_ID_LOOKUP: Dict[str, int] = {
    "bipolar_forceps": 1,
    "maryland_bipolar_forceps": 1,
    "prograsp_forceps": 2,
    "large_needle_driver": 3,
    "vessel_sealer": 4,
    "grasping_retractor": 5,
    "monopolar_curved_scissors": 6,
    "other": 7,
}


def _resolve_instrument_id(dirname: str) -> Optional[int]:
    """
    Map a ground truth directory name to a semantic class id.
    """
    name = dirname.replace("_labels", "")
    name = name.replace(" ", "_").lower()

    # Remove left/right prefixes
    if name.startswith("left_") or name.startswith("right_"):
        name = name.split("_", 1)[1]

    # Remove trailing camera markers
    if name.endswith("_left") or name.endswith("_right"):
        name = name.rsplit("_", 1)[0]

    return _CLASS_ID_LOOKUP.get(name)


@dataclass
class _EndoVis2017Sample:
    image_path: Path
    depth_path: Path
    mask_dirs: List[Path]
    frame_token: str
    sequence_name: str


class EndoVis2017Dataset(Dataset):
    """
    Dataset loader for the EndoVis2017 instrument segmentation + depth data.

    Args:
        root_dir: Path containing ``image`` and ``depth`` folders.
        split: One of ``train``, ``val`` or ``all``.
        size: Target (width, height) for resizing.
        max_depth: Maximum depth (in meters) used to clip depth maps and
                   compute valid masks.
    """

    SPLIT_MAP: Dict[str, Set[str]] = {
        "train": {
            "instrument_1_4_training",
            "instrument_5_8_training",
            "instrument_dataset_1",
        },
        "val": {
            "instrument_1_4_testing",
            "instrument_5_8_testing",
            "instrument_9_10_testing",
            "instrument_2017_test",
        },
    }

    def __init__(
        self,
        root_dir: str | os.PathLike[str],
        split: str = "train",
        size: Tuple[int, int] = (518, 518),
        max_depth: float = 0.3,
        min_depth: float = 1e-3,
    ) -> None:
        self.root_dir = Path(root_dir)
        self.split = split
        self.size = size
        self.max_depth = max_depth
        self.min_depth = min_depth
        self._unknown_labels: Set[str] = set()

        image_root = self.root_dir / "image"
        depth_root = self.root_dir / "depth"
        if not image_root.exists():
            raise FileNotFoundError(f"EndoVis2017 image root not found: {image_root}")
        if not depth_root.exists():
            raise FileNotFoundError(f"EndoVis2017 depth root not found: {depth_root}")

        self.samples: List[_EndoVis2017Sample] = self._gather_samples(image_root, depth_root)
        self._intrinsics_map: Dict[Path, torch.Tensor] = self._load_intrinsics_map(image_root)

        net_w, net_h = size
        self.transform = Compose(
            [
                Resize(
                    width=net_w,
                    height=net_h,
                    resize_target=True,
                    keep_aspect_ratio=True,
                    ensure_multiple_of=1,
                    resize_method="upper_bound",
                    image_interpolation_method=cv2.INTER_CUBIC,
                    downscale_only=True,
                ),
                NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                PrepareForNet(),
            ]
        )

    def _allowed_sequences(self, image_root: Path) -> Set[str]:
        if self.split == "all":
            return {p.name for p in image_root.iterdir() if p.is_dir()}
        if self.split not in self.SPLIT_MAP:
            raise ValueError(f"Unknown split '{self.split}'. Supported: train, val, all.")
        return self.SPLIT_MAP[self.split]

    def _gather_samples(self, image_root: Path, depth_root: Path) -> List[_EndoVis2017Sample]:
        allowed = self._allowed_sequences(image_root)
        samples: List[_EndoVis2017Sample] = []

        for left_dir in image_root.glob("**/left_frames"):
            try:
                sequence_token = left_dir.relative_to(image_root).parts[0]
            except ValueError:
                continue

            if allowed and sequence_token not in allowed:
                continue

            depth_dir = depth_root / left_dir.relative_to(image_root)
            if not depth_dir.exists():
                continue

            ground_truth_dir = left_dir.parent / "ground_truth"
            mask_dirs = [d for d in ground_truth_dir.glob("*") if d.is_dir()] if ground_truth_dir.exists() else []

            for image_path in sorted(left_dir.glob("frame*.png")):
                frame_token = image_path.stem  # e.g. frame000123
                depth_path = depth_dir / f"{frame_token}_depth.npy"
                if not depth_path.exists():
                    continue

                samples.append(
                    _EndoVis2017Sample(
                        image_path=image_path,
                        depth_path=depth_path,
                        mask_dirs=mask_dirs,
                        frame_token=frame_token,
                        sequence_name=sequence_token,
                    )
                )

        if not samples:
            raise RuntimeError(f"No samples found for EndoVis2017 in split '{self.split}'.")
        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def _build_mask(self, sample: _EndoVis2017Sample, height: int, width: int) -> np.ndarray:
        mask = np.zeros((height, width), dtype=np.uint8)

        for mask_dir in sample.mask_dirs:
            class_id = _resolve_instrument_id(mask_dir.name)
            if class_id is None:
                if mask_dir.name not in self._unknown_labels:
                    self._unknown_labels.add(mask_dir.name)
                    print(f"[EndoVis2017] Warning: unknown label directory '{mask_dir.name}' skipped.")
                continue

            mask_path = mask_dir / f"{sample.frame_token}.png"
            if not mask_path.exists():
                continue

            mask_img = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            if mask_img is None:
                continue

            if mask_img.shape != (height, width):
                mask_img = cv2.resize(mask_img, (width, height), interpolation=cv2.INTER_NEAREST)

            mask[mask_img > 0] = class_id

        return remap_labels(mask, ENDOVIS2017_REMAP)

    def _crop_black_borders(
        self,
        image: np.ndarray,
        depth: np.ndarray,
        mask: np.ndarray,
        brightness_threshold: float = 8.0 / 255.0,
        min_valid_fraction: float = 0.02,
        margin: int = 2,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        height, width = image.shape[:2]
        brightness = image.mean(axis=2)
        valid_pixels = brightness > brightness_threshold

        row_valid = np.where(valid_pixels.mean(axis=1) > min_valid_fraction)[0]
        col_valid = np.where(valid_pixels.mean(axis=0) > min_valid_fraction)[0]

        if row_valid.size < 4 or col_valid.size < 4:
            return image, depth, mask

        top = max(int(row_valid[0]) - margin, 0)
        bottom = min(int(row_valid[-1]) + 1 + margin, height)
        left = max(int(col_valid[0]) - margin, 0)
        right = min(int(col_valid[-1]) + 1 + margin, width)

        cropped_h = bottom - top
        cropped_w = right - left
        if cropped_h < height * 0.5 or cropped_w < width * 0.5:
            return image, depth, mask

        image_c = image[top:bottom, left:right]
        depth_c = depth[top:bottom, left:right]
        mask_c = mask[top:bottom, left:right] if mask is not None else mask

        return image_c, depth_c, mask_c

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor | str | float]:
        sample = self.samples[idx]

        image = cv2.imread(str(sample.image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f"Unable to read image: {sample.image_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

        depth = np.load(sample.depth_path).astype(np.float32)
        depth = np.where(np.isfinite(depth), depth, 0.0)
        depth = np.clip(depth, 0.0, self.max_depth)

        mask = self._build_mask(sample, depth.shape[0], depth.shape[1])
        mask = map_ls_semseg_to_10_classes(mask, "EndoVis2017")

        image, depth, mask = self._crop_black_borders(image, depth, mask)
        depth[(depth < self.min_depth) | (depth > self.max_depth)] = 0.0

        transformed = self.transform(
            {
                "image": image,
                "depth": depth,
                "semseg_mask": mask,
            }
        )

        image_tensor = torch.from_numpy(transformed["image"])
        depth_tensor = torch.from_numpy(transformed["depth"])
        mask_tensor = torch.from_numpy(transformed["semseg_mask"]).long()
        valid_mask = compute_valid_mask(
            image_tensor,
            depth_tensor,
            min_depth=self.min_depth,
            max_depth=self.max_depth,
            dataset_name="EndoVis2017",
        )
        intrinsics_tensor = self._lookup_intrinsics(sample.image_path)

        result = {
            "image": image_tensor,
            "depth": depth_tensor,
            "semseg_mask": mask_tensor,
            "valid_mask": valid_mask,
            "image_path": str(sample.image_path),
            "depth_path": str(sample.depth_path),
            "max_depth": self.max_depth,
            "source_type": "LS",
            "sequence": sample.sequence_name,
            "frame_token": sample.frame_token,
        }
        if intrinsics_tensor is not None:
            result["intrinsics"] = intrinsics_tensor
        return result

    def _lookup_intrinsics(self, image_path: Path) -> Optional[torch.Tensor]:
        left_dir = image_path.parent.resolve()
        if left_dir in self._intrinsics_map:
            return self._intrinsics_map[left_dir]
        parent = left_dir.parent.resolve()
        return self._intrinsics_map.get(parent)

    def _load_intrinsics_map(self, image_root: Path) -> Dict[Path, torch.Tensor]:
        intrinsics_map: Dict[Path, torch.Tensor] = {}
        for calib_path in image_root.glob("**/camera_calibration.txt"):
            left_dir = calib_path.parent / "left_frames"
            if not left_dir.exists():
                continue
            intrinsics = self._parse_camera_calibration(calib_path)
            if intrinsics is None:
                continue
            intrinsics_map[left_dir.resolve()] = intrinsics
        return intrinsics_map

    @staticmethod
    def _parse_camera_calibration(calib_path: Path) -> Optional[torch.Tensor]:
        fx = fy = cx = cy = None
        try:
            with open(calib_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("//"):
                        continue
                    if line.startswith("Camera-0-F:"):
                        parts = line.split(":", 1)[1].split("//")[0].strip().split()
                        if len(parts) >= 2:
                            fx, fy = map(float, parts[:2])
                    elif line.startswith("Camera-0-C:"):
                        parts = line.split(":", 1)[1].split("//")[0].strip().split()
                        if len(parts) >= 2:
                            cx, cy = map(float, parts[:2])
                    if fx is not None and fy is not None and cx is not None and cy is not None:
                        break
        except OSError:
            return None

        if None in (fx, fy, cx, cy):
            return None
        matrix = torch.tensor(
            [
                [fx, 0.0, cx],
                [0.0, fy, cy],
                [0.0, 0.0, 1.0],
            ],
            dtype=torch.float32,
        )
        return matrix


def _parse_args():
    import argparse

    parser = argparse.ArgumentParser(description="Generate cache for EndoVis2017 dataset.")
    parser.add_argument("--root", type=Path, required=True, help="Path to EndoVis2017 root directory")
    parser.add_argument("--split", type=str, default="train", choices=["train", "val", "all"], help="Dataset split to process")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory to store generated cache .pt files and filelist")
    parser.add_argument("--filelist-name", type=str, default=None, help="Name of the generated cache list file")
    parser.add_argument("--cache-root", type=Path, default=None, help="Cache root path to mirror original structure")
    parser.add_argument("--max-depth", type=float, default=0.3, help="Maximum depth clipping value")
    parser.add_argument("--limit", type=int, default=None, help="Optional limit of samples for quick testing")
    return parser.parse_args()


def _main():
    from dataset.cache_utils import generate_dataset_cache

    args = _parse_args()
    dataset = EndoVis2017Dataset(root_dir=str(args.root), split=args.split, max_depth=args.max_depth)

    if args.limit is not None:
        dataset.samples = dataset.samples[: args.limit]

    filelist_name = args.filelist_name
    if filelist_name is None:
        filelist_name = f"{args.split}_cache.txt"

    cache_root = args.cache_root if args.cache_root is not None else args.output_dir

    generate_dataset_cache(
        dataset=dataset,
        output_dir=str(args.output_dir),
        filelist_name=filelist_name,
        origin_prefix=str(args.root),
        cache_root_path=str(cache_root),
    )


if __name__ == "__main__":
    _main()
