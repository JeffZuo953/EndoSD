import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision.transforms import Compose

from .transform import NormalizeImage, PrepareForNet, Resize
from .utils import compute_valid_mask, remap_labels, map_ls_semseg_to_10_classes

ENDOVIS2018_REMAP = {
    0: 0,
    1: 1,
    2: 2,
    3: 3,
    4: 6,
    5: 7,
    6: 8,
    7: 9,
    255: 255,
}


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
                  ``EndoVis2018_ISINet_tool/EndoVis_2018_ISINet_tool`` and the
                  ``depth`` directory with ``EndoVis2018_Scene_seg`` releases.
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
        self.depth_root = self.root_dir / "depth" / "EndoVis2018_Scene_seg"

        if not self.image_root.exists():
            raise FileNotFoundError(f"EndoVis2018 image root not found: {self.image_root}")
        if not self.depth_root.exists():
            raise FileNotFoundError(f"EndoVis2018 depth root not found: {self.depth_root}")

        self.depth_map: Dict[str, Path] = self._build_depth_lookup()
        self.intrinsics_map: Dict[str, torch.Tensor] = self._load_intrinsics_map()
        self.label_map: Dict[int, int] = self._load_label_mapping()
        self.samples: List[_EndoVis2018Sample] = self._gather_samples()

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
        for seq_dir in self.depth_root.glob("**/seq_*"):
            if not seq_dir.is_dir():
                continue
            seq_id = seq_dir.name.split("_")[-1]
            left_frames = seq_dir / "left_frames"
            if left_frames.exists():
                depth_map[seq_id] = left_frames
        if not depth_map:
            raise RuntimeError("No depth sequences found for EndoVis2018.")
        return depth_map

    def _load_label_mapping(self) -> Dict[int, int]:
        labels_json = self.image_root / "labels.json"
        if not labels_json.exists():
            return {}

        with open(labels_json, "r") as f:
            entries = json.load(f)

        mapping: Dict[int, int] = {}
        for entry in entries:
            class_id = entry.get("classid")
            if class_id is not None:
                mapping[int(class_id)] = int(class_id)
        return mapping

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
                mask_path = ann_dir / image_path.name
                if not mask_path.exists():
                    continue

                parts = image_path.stem.split("_")
                if len(parts) < 3:
                    continue
                seq_id = parts[1]
                frame_token = parts[-1].replace("frame", "")

                depth_dir = self.depth_map.get(seq_id)
                if depth_dir is None:
                    continue

                depth_path = depth_dir / f"frame{frame_token}_depth.npy"
                if not depth_path.exists():
                    continue

                samples.append(
                    _EndoVis2018Sample(
                        image_path=image_path,
                        mask_path=mask_path,
                        depth_path=depth_path,
                        sequence_id=seq_id,
                        frame_token=frame_token,
                    )
                )

        if not samples:
            raise RuntimeError(f"No samples found for EndoVis2018 split '{self.split}'.")
        return samples

    def _load_intrinsics_map(self) -> Dict[str, torch.Tensor]:
        intrinsics_map: Dict[str, torch.Tensor] = {}
        for calib_path in self.root_dir.glob("**/camera_calibration.txt"):
            seq_dir = calib_path.parent
            if not seq_dir.name.startswith("seq_"):
                continue
            seq_id = seq_dir.name.split("_")[-1]
            if seq_id in intrinsics_map:
                continue
            intrinsics = self._parse_camera_calibration(calib_path)
            if intrinsics is not None:
                intrinsics_map[seq_id] = intrinsics
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

        return torch.tensor(
            [
                [fx, 0.0, cx],
                [0.0, fy, cy],
                [0.0, 0.0, 1.0],
            ],
            dtype=torch.float32,
        )

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
        mask = remap_labels(mask_img.astype(np.uint8), ENDOVIS2018_REMAP)
        mask = map_ls_semseg_to_10_classes(mask, "EndoVis2018")

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
        }
        intrinsics_tensor = self.intrinsics_map.get(sample.sequence_id)
        if intrinsics_tensor is not None:
            result["intrinsics"] = intrinsics_tensor
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
