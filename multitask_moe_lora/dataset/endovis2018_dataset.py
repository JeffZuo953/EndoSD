import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision.transforms import Compose

from .camera_utils import CameraInfo, get_camera_info
from .transform import NormalizeImage, PrepareForNet, Resize
from .utils import compute_valid_mask

logger = logging.getLogger(__name__)


@dataclass
class _EndoVis2018Sample:
    image_path: Path
    mask_path: Path
    depth_path: Path
    sequence_id: str
    frame_token: str


class EndoVis2018Dataset(Dataset):
    """
    Dataset loader for EndoVis2018 instrument segmentation + depth data.

    Args:
        root_dir: Path containing the official dataset structure. Must include
                  ``EndoVis2018_ISINet_tool/EndoVis_2018_ISINet_tool`` for the split lists,
                  the rectified frame directory
                  ``endovis2018_seg_depth/output_rectified`` (with ``left`` and
                  ``left_mask`` subfolders), and the generated depth maps under
                  ``refictied_depth/left``.
        split: One of ``train``, ``val`` or ``all``.
        size: Target (width, height) for resizing.
        max_depth: Maximum depth (in meters) used for clipping and valid mask.
    """

    def __init__(
        self,
        root_dir: str | Path,
        split: str = "train",
        size: Tuple[int, int] = (518, 518),
        max_depth: float = 0.3,
    ) -> None:
        self.root_dir = Path(root_dir)
        self.split = split
        self.size = size
        self.max_depth = max_depth

        self.image_root = self.root_dir / "EndoVis2018_ISINet_tool" / "EndoVis_2018_ISINet_tool"
        default_rectified = self.root_dir / "endovis2018_seg_depth" / "output_rectified"
        alt_rectified = self.root_dir / "output_rectified"
        if default_rectified.exists():
            self.rectified_root = default_rectified
        elif alt_rectified.exists():
            self.rectified_root = alt_rectified
        else:
            raise FileNotFoundError(
                f"Rectified output directory not found under {default_rectified} or {alt_rectified}"
            )
        self.rectified_image_dir = self.rectified_root / "left"

        mask_reid_dir = self.rectified_root / "left_mask_reid"
        mask_legacy_dir = self.rectified_root / "left_mask"
        if mask_reid_dir.exists():
            self.rectified_mask_dir = mask_reid_dir
            self.rectified_mask_alt_dir = mask_legacy_dir
        elif mask_legacy_dir.exists():
            self.rectified_mask_dir = mask_legacy_dir
            self.rectified_mask_alt_dir = mask_reid_dir
        else:
            self.rectified_mask_dir = mask_reid_dir
            self.rectified_mask_alt_dir = mask_legacy_dir
        self.depth_root = self.root_dir / "refictied_depth" / "left"

        if not self.image_root.exists():
            raise FileNotFoundError(f"EndoVis2018 image root not found: {self.image_root}")
        if not self.depth_root.exists():
            raise FileNotFoundError(f"EndoVis2018 depth root not found: {self.depth_root}")
        if not self.rectified_image_dir.exists():
            raise FileNotFoundError(f"Rectified image directory not found: {self.rectified_image_dir}")
        if not self.rectified_mask_dir.exists():
            if self.rectified_mask_alt_dir.exists():
                self.rectified_mask_dir = self.rectified_mask_alt_dir
            else:
                raise FileNotFoundError(f"Rectified mask directory not found: {self.rectified_mask_dir}")

        self.depth_map: Dict[str, Path] = self._build_depth_lookup()
        self.samples: List[_EndoVis2018Sample] = self._gather_samples()
        self._camera_cache: Dict[str, Optional[CameraInfo]] = {}
        self._missing_camera_sequences: Set[str] = set()

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

    def _build_depth_lookup(self) -> Dict[str, Path]:
        depth_map: Dict[str, Path] = {}
        for depth_path in self.depth_root.glob("*.npy"):
            stem = depth_path.stem
            key = stem[:-6] if stem.endswith("_depth") else stem
            depth_map[key] = depth_path
        if not depth_map:
            raise RuntimeError("No rectified depth files found for EndoVis2018.")
        return depth_map

    def _target_splits(self) -> List[str]:
        if self.split == "all":
            return [d.name for d in self.image_root.iterdir() if d.is_dir()]
        if self.split not in ("train", "val"):
            raise ValueError(f"Unknown split '{self.split}'. Supported: train, val, all.")
        return [self.split]

    def _gather_samples(self) -> List[_EndoVis2018Sample]:
        splits = self._target_splits()
        samples: List[_EndoVis2018Sample] = []

        for split_name in splits:
            split_dir = self.image_root / split_name
            if not split_dir.exists():
                continue

            image_dir = split_dir / "images"
            ann_dir = split_dir / "annotations"
            if not image_dir.exists() or not ann_dir.exists():
                continue

            for image_path in sorted(image_dir.glob("*.png")):
                parts = image_path.stem.split("_")
                if len(parts) < 3:
                    continue
                seq_id = parts[1]
                frame_token = parts[-1].replace("frame", "")
                frame_key = image_path.stem

                rectified_image = self.rectified_image_dir / f"{frame_key}.png"
                rectified_mask = self.rectified_mask_dir / f"{frame_key}.png"
                if not rectified_mask.exists() and self.rectified_mask_alt_dir.exists():
                    alt_mask = self.rectified_mask_alt_dir / f"{frame_key}.png"
                    if alt_mask.exists():
                        rectified_mask = alt_mask
                depth_path = self.depth_map.get(frame_key)

                if depth_path is None or not rectified_image.exists() or not rectified_mask.exists():
                    continue

                samples.append(
                    _EndoVis2018Sample(
                        image_path=rectified_image,
                        mask_path=rectified_mask,
                        depth_path=depth_path,
                        sequence_id=seq_id,
                        frame_token=frame_token,
                    )
                )

        if not samples:
            raise RuntimeError(f"No samples found for EndoVis2018 split '{self.split}'.")
        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor | str | float]:
        sample = self.samples[idx]

        image = cv2.imread(str(sample.image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f"Unable to read image: {sample.image_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

        mask_img = cv2.imread(str(sample.mask_path), cv2.IMREAD_UNCHANGED)
        if mask_img is None:
            raise FileNotFoundError(f"Unable to read mask: {sample.mask_path}")
        if mask_img.ndim == 3:
            mask_img = cv2.cvtColor(mask_img, cv2.COLOR_BGR2GRAY)
        mask = mask_img.astype(np.uint8)

        depth = np.load(sample.depth_path).astype(np.float32)
        depth = np.clip(depth, 0.0, self.max_depth)

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
            max_depth=self.max_depth,
            dataset_name="EndoVis2018",
        )

        result = {
            "image": image_tensor,
            "depth": depth_tensor,
            "semseg_mask": mask_tensor,
            "valid_mask": valid_mask,
            "image_path": str(sample.image_path),
            "depth_path": str(sample.depth_path),
            "max_depth": self.max_depth,
            "source_type": "LS",
            "sequence": sample.sequence_id,
            "frame_token": sample.frame_token,
            "dataset_name": "EndoVis2018",
        }

        camera_info = self._camera_cache.get(sample.sequence_id)
        if camera_info is None:
            camera_info = get_camera_info("endovis2018", str(sample.image_path))
            self._camera_cache[sample.sequence_id] = camera_info

        if camera_info is None:
            seq_key = sample.sequence_id.lower()
            if seq_key not in self._missing_camera_sequences:
                logger.error("[EndoVis2018] Missing camera metadata for sequence '%s'; skipping intrinsics.", sample.sequence_id)
                self._missing_camera_sequences.add(seq_key)
        else:
            intrinsics_tensor = camera_info.intrinsics.clone().to(torch.float32)
            intrinsics_norm = camera_info.intrinsics_norm.clone().to(torch.float32)
            camera_size = torch.tensor([float(camera_info.width), float(camera_info.height)], dtype=torch.float32)

            result["intrinsics"] = intrinsics_tensor
            result["camera_intrinsics"] = intrinsics_tensor
            result["camera_intrinsics_norm"] = intrinsics_norm
            result["camera_size"] = camera_size.clone()
            result["camera_size_original"] = camera_size.clone()
            result["camera_original_image_size"] = camera_size.clone()
            result["camera_image_size"] = torch.tensor(
                [float(image_tensor.shape[-1]), float(image_tensor.shape[-2])],
                dtype=torch.float32,
            )
        return result


def _parse_args():
    import argparse

    parser = argparse.ArgumentParser(description="Generate cache for EndoVis2018 dataset.")
    parser.add_argument("--root", type=Path, required=True, help="Path to EndoVis2018 root directory")
    parser.add_argument("--split", type=str, default="train", choices=["train", "val"], help="Dataset split to process")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory to store generated cache files")
    parser.add_argument("--filelist-name", type=str, default=None, help="Name of generated cache list file")
    parser.add_argument("--cache-root", type=Path, default=None, help="Cache root path mirroring original structure")
    parser.add_argument("--max-depth", type=float, default=0.3, help="Maximum depth clipping value")
    parser.add_argument("--limit", type=int, default=None, help="Optional limit of samples for quick testing")
    return parser.parse_args()


def _main():
    from dataset.cache_utils import generate_dataset_cache

    args = _parse_args()
    dataset = EndoVis2018Dataset(root_dir=str(args.root), split=args.split, max_depth=args.max_depth)

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
