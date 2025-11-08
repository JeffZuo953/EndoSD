import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision.transforms import Compose

from .camera_utils import get_camera_info
from .transform import NormalizeImage, PrepareForNet, Resize
from .utils import compute_valid_mask, remap_labels, map_ls_semseg_to_10_classes

RECTIFIED_DATA_FOLDER = "Endovis2017_seg_depth"
RECTIFIED_DEPTH_SCALE = 1000.0

logger = logging.getLogger(__name__)

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
    frame_token: str
    sequence_name: str
    mask_dirs: List[Path] = field(default_factory=list)
    mask_path: Optional[Path] = None
    camera_path: Optional[Path] = None


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
        self.split = split.lower()
        self.size = size
        self.max_depth = max_depth
        self.min_depth = min_depth
        self._unknown_labels: Set[str] = set()

        self._rectified_root = self._detect_rectified_root(self.root_dir)
        self._rectified_mode = self._rectified_root is not None
        self._sequence_intrinsics: Dict[str, Optional[torch.Tensor]] = {}
        self._missing_intrinsics: Set[str] = set()

        if self._rectified_mode:
            rectified_root = self._rectified_root
            if rectified_root is None:
                raise RuntimeError("Rectified dataset root not resolved.")
            self.samples = self._gather_rectified_samples(rectified_root)
        else:
            image_root = self.root_dir / "image"
            depth_root = self.root_dir / "depth"
            if not image_root.exists():
                raise FileNotFoundError(f"EndoVis2017 image root not found: {image_root}")
            if not depth_root.exists():
                raise FileNotFoundError(f"EndoVis2017 depth root not found: {depth_root}")
            self.samples = self._gather_samples(image_root, depth_root)

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
                        frame_token=frame_token,
                        sequence_name=sequence_token,
                        mask_dirs=mask_dirs,
                    )
                )

        if not samples:
            raise RuntimeError(f"No samples found for EndoVis2017 in split '{self.split}'.")
        return samples

    def _detect_rectified_root(self, root: Path) -> Optional[Path]:
        root = root.expanduser()
        image_root = root / "image"
        depth_root = root / "depth"
        if image_root.exists() and depth_root.exists():
            return None
        direct_rectified = root / "output_rectified"
        if direct_rectified.exists():
            return root
        candidate = root / RECTIFIED_DATA_FOLDER
        if candidate.exists():
            return candidate
        train_dir = root / "train"
        if train_dir.exists():
            for seq_dir in train_dir.iterdir():
                if (seq_dir / "output_rectified").exists():
                    return root
        return None

    def _rectified_split_dirs(self, rectified_root: Path) -> List[Tuple[str, Path]]:
        split_map: List[Tuple[str, Path]] = []
        if self.split in {"train", "all"}:
            split_map.append(("train", rectified_root / "train"))
        if self.split in {"val", "all"}:
            eval_dir = rectified_root / "test"
            if not eval_dir.exists():
                eval_dir = rectified_root / "val"
            split_map.append(("val", eval_dir))
        if self.split not in {"train", "val", "all"}:
            raise ValueError(f"Unknown split '{self.split}'. Supported: train, val, all.")
        return split_map

    def _gather_rectified_samples(self, rectified_root: Path) -> List[_EndoVis2017Sample]:
        samples: List[_EndoVis2017Sample] = []
        sequence_dirs = self._rectified_split_dirs(rectified_root)
        missing_components = 0

        for split_name, split_dir in sequence_dirs:
            if not split_dir.exists():
                continue
            for seq_dir in sorted(split_dir.iterdir()):
                if not seq_dir.is_dir():
                    continue
                rectified_dir = seq_dir / "output_rectified"
                left_dir = rectified_dir / "left"
                depth_dir = rectified_dir / "depth"
                mask_dir = rectified_dir / "left_mask_reid"
                camera_dir = rectified_dir / "camera"
                if not left_dir.exists() or not depth_dir.exists() or not mask_dir.exists():
                    missing_components += 1
                    continue
                for image_path in sorted(left_dir.glob("frame*.png")):
                    frame_token = image_path.stem
                    depth_path = depth_dir / f"{frame_token}_depth.npy"
                    mask_path = mask_dir / f"{frame_token}.png"
                    camera_path = camera_dir / f"{frame_token}.txt"
                    if not depth_path.exists() or not mask_path.exists():
                        continue
                    samples.append(
                        _EndoVis2017Sample(
                            image_path=image_path,
                            depth_path=depth_path,
                            frame_token=frame_token,
                            sequence_name=seq_dir.name,
                            mask_path=mask_path,
                            camera_path=camera_path if camera_path.exists() else None,
                        )
                    )

        if not samples:
            raise RuntimeError(
                f"No rectified samples found for split '{self.split}'. Checked {len(sequence_dirs)} directories, "
                f"{missing_components} sequences had missing components."
            )
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

        if self._rectified_mode:
            image_bgr = cv2.imread(str(sample.image_path), cv2.IMREAD_COLOR)
            if image_bgr is None:
                raise FileNotFoundError(f"Unable to read image: {sample.image_path}")
            image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

            depth = np.load(sample.depth_path).astype(np.float32)
            depth = np.where(np.isfinite(depth), depth, 0.0)
            depth = depth / RECTIFIED_DEPTH_SCALE
            depth = np.clip(depth, 0.0, self.max_depth)

            if sample.mask_path is None:
                raise FileNotFoundError(f"Segmentation mask missing for {sample.frame_token}")
            mask_img = cv2.imread(str(sample.mask_path), cv2.IMREAD_GRAYSCALE)
            if mask_img is None:
                raise FileNotFoundError(f"Unable to read mask: {sample.mask_path}")

            transformed = self.transform(
                {
                    "image": image,
                    "depth": depth,
                    "semseg_mask": mask_img,
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
            intrinsics_tensor = self._get_metadata_intrinsics(sample)

            result = {
                "image": image_tensor,
                "depth": depth_tensor,
                "semseg_mask": mask_tensor,
                "valid_mask": valid_mask,
                "image_path": str(sample.image_path),
                "depth_path": str(sample.depth_path),
                "mask_path": str(sample.mask_path),
                "max_depth": self.max_depth,
                "source_type": "LS",
                "sequence": sample.sequence_name,
                "frame_token": sample.frame_token,
            }
            if intrinsics_tensor is not None:
                result["intrinsics"] = intrinsics_tensor
            return result

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
        intrinsics_tensor = self._get_metadata_intrinsics(sample)

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

    def _get_metadata_intrinsics(self, sample: _EndoVis2017Sample) -> Optional[torch.Tensor]:
        seq_key = sample.sequence_name.lower()
        if seq_key in self._sequence_intrinsics:
            return self._sequence_intrinsics[seq_key]

        camera_info = get_camera_info("endovis2017", str(sample.image_path))
        if camera_info is None:
            if seq_key not in self._missing_intrinsics:
                logger.error(
                    "[EndoVis2017] Missing camera metadata for sequence '%s'; skipping intrinsics.",
                    sample.sequence_name,
                )
                self._missing_intrinsics.add(seq_key)
            self._sequence_intrinsics[seq_key] = None
            return None

        tensor = camera_info.intrinsics.clone().to(torch.float32)
        self._sequence_intrinsics[seq_key] = tensor
        return tensor

    # Metadata-driven intrinsics only; no ad-hoc parsing helpers retained.


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
