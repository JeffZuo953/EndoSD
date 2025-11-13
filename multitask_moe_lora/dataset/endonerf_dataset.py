import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision.transforms import Compose

from .camera_utils import CameraInfo, normalize_intrinsics
from .metadata.endonerf import EndoNeRFCamera
from .transform import NormalizeImage, PrepareForNet, Resize
from .utils import compute_valid_mask


@dataclass
class _EndoNeRFSample:
    image_path: Path
    depth_path: Path
    seg_path: Path
    depth_valid_path: Path
    sequence: str


logger = logging.getLogger(__name__)


class EndoNeRFDataset(Dataset):
    """Dataset loader for the EndoNeRF synthetic scenes with depth + segmentation.

    Expected directory layout::

        root/endonerf/<sequence>/
            images/*.png               # RGB frames
            depth/*.png                # Depth maps (uint8, scaled by max_depth)
            masks/*.png                # Binary masks: black=valid, white=invalid

        root/EndoNeRF_seg/<sequence>/mask_label_1_6/*.png  # Segmentation labels (grayscale)

    Each cached sample exports both depth and segmentation along with
    separate validity masks:
        * depth_valid_mask : True where depth is valid according to masks/*.png
        * seg_valid_mask   : True except for the bottom 10px band (set to False)
        * valid_mask       : Backwards-compatible depth mask (depth_valid_mask ∧ compute_valid_mask)
    """

    CAMERA_WIDTH: int = EndoNeRFCamera.WIDTH
    CAMERA_HEIGHT: int = EndoNeRFCamera.HEIGHT
    CAMERA_INFO: Optional[CameraInfo] = EndoNeRFCamera.get_camera_info()

    def __init__(
        self,
        root_dir: str | Path,
        size: Tuple[int, int] = (518, 518),
        max_depth: float = 0.3,
        filelist_path: str | Path | None = None,
    ) -> None:
        dataset_root = Path(root_dir).expanduser().resolve()
        if (dataset_root / "endonerf").exists():
            dataset_root = dataset_root / "endonerf"
        self.root_dir = dataset_root
        self.seg_root = dataset_root.parent / "EndoNeRF_seg"
        if not self.seg_root.exists():
            logger.warning("Segmentation root not found at %s; falling back to per-sequence directories", self.seg_root)

        self.size = size
        self.max_depth = max_depth
        self.samples: List[_EndoNeRFSample] = self._gather_samples(filelist_path)
        self._camera_warning_emitted = False

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

    def _resolve_segmentation_path(self, sequence_dir: Path) -> Path:
        sequence = sequence_dir.name
        candidates = [
            self.seg_root / sequence / "mask_label_1_6",
            self.seg_root / sequence,
            sequence_dir / "gt_masks",
            sequence_dir / "mask_label_1_6",
        ]
        for candidate in candidates:
            if candidate.exists():
                return candidate
        raise FileNotFoundError(
            f"Segmentation directory not found for sequence '{sequence}'. "
            f"Checked: {', '.join(str(c) for c in candidates)}"
        )

    def _gather_samples(self, filelist_path: str | Path | None) -> List[_EndoNeRFSample]:
        allowed_images: set[str] | None = None
        if filelist_path is not None:
            allowed_images = self._load_allowed_images(Path(filelist_path))

        samples: List[_EndoNeRFSample] = []
        for sequence_dir in sorted(self.root_dir.iterdir()):
            if not sequence_dir.is_dir():
                continue

            image_dir = sequence_dir / "images"
            depth_dir = sequence_dir / "depth"
            seg_dir = self._resolve_segmentation_path(sequence_dir)
            depth_valid_dir = sequence_dir / "masks"

            if not (image_dir.exists() and depth_dir.exists() and depth_valid_dir.exists()):
                continue

            image_files = sorted(image_dir.glob("*.png"))
            depth_files = sorted(depth_dir.glob("*.png"))
            if not image_files or not depth_files or len(image_files) != len(depth_files):
                raise RuntimeError(f"Mismatched image/depth counts in {sequence_dir}")

            for img_path, depth_path in zip(image_files, depth_files):
                img_resolved = img_path.resolve()
                depth_path = depth_path.resolve()
                if allowed_images is not None and str(img_resolved) not in allowed_images:
                    continue
                seg_candidates = self._build_seg_candidates(seg_dir, img_resolved)
                seg_path = self._first_existing(seg_candidates)
                if seg_path is None:
                    raise FileNotFoundError(
                        f"Segmentation mask missing for {img_path.name}. Tried: {', '.join(str(c) for c in seg_candidates)}"
                    )

                depth_mask_candidates = self._build_depth_mask_candidates(depth_valid_dir, img_resolved)
                depth_valid_path = self._first_existing(depth_mask_candidates)
                if depth_valid_path is None:
                    raise FileNotFoundError(
                        f"Depth valid mask missing for {img_path.name}. Tried: {', '.join(str(c) for c in depth_mask_candidates)}"
                    )

                samples.append(
                    _EndoNeRFSample(
                        image_path=img_resolved,
                        depth_path=depth_path,
                        seg_path=seg_path,
                        depth_valid_path=depth_valid_path,
                        sequence=sequence_dir.name,
                    )
                )

        if not samples:
            raise RuntimeError("No samples found for EndoNeRF dataset.")
        return samples

    @staticmethod
    def get_camera_info(_: Optional[str] = None) -> Optional[CameraInfo]:
        """Return fixed camera intrinsics for EndoNeRF."""
        return EndoNeRFCamera.get_camera_info()

    @staticmethod
    def _load_allowed_images(filelist_path: Path) -> set[str]:
        allowed: set[str] = set()
        with filelist_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                if not parts:
                    continue
                allowed.add(str(Path(parts[0]).resolve()))
        return allowed

    @staticmethod
    def _first_existing(candidates: List[Path]) -> Path | None:
        for candidate in candidates:
            if candidate.exists():
                return candidate
        return None

    @staticmethod
    def _build_seg_candidates(seg_dir: Path, image_path: Path) -> List[Path]:
        stem = image_path.stem
        candidates: List[str] = [image_path.name]

        if stem:
            candidates.append(f"{stem}.png")

        if ".color" in stem:
            base = stem.replace(".color", "")
            candidates.extend(
                [
                    f"{base}.png",
                    f"{base}.color.png",
                ]
            )
        else:
            base = stem

        if base.startswith("frame-"):
            stripped = base.replace("frame-", "")
            if stripped:
                candidates.extend(
                    [
                        f"{stripped}.png",
                        f"frame-{stripped}.png",
                        f"frame-{stripped}.color.png",
                    ]
                )

        # Add ".mask" variants commonly used by raw LS data
        mask_augmented: List[str] = []
        for name in candidates:
            mask_augmented.append(name)
            if name.endswith(".png"):
                base = name[: -len(".png")]
                mask_augmented.append(f"{base}.mask.png")

        # Deduplicate while preserving order
        seen = set()
        unique = []
        for name in mask_augmented:
            if name not in seen:
                unique.append(name)
                seen.add(name)
        return [seg_dir / name for name in unique]

    @staticmethod
    def _build_depth_mask_candidates(mask_dir: Path, image_path: Path) -> List[Path]:
        stem = image_path.stem
        candidates: List[str] = []

        if stem:
            candidates.append(f"{stem}.mask.png")

        base = stem
        if ".color" in stem:
            base = stem.replace(".color", "")
            candidates.append(f"{base}.mask.png")

        if not base.startswith("frame-"):
            candidates.append(f"frame-{base}.mask.png")
        else:
            stripped = base.replace("frame-", "")
            candidates.append(f"{base}.mask.png")
            if stripped:
                candidates.append(f"frame-{stripped}.mask.png")

        if stripped := base.replace("frame-", ""):
            candidates.append(f"{stripped}.mask.png")

        # Deduplicate preserving order
        seen = set()
        unique = []
        for name in candidates:
            if name not in seen:
                unique.append(name)
                seen.add(name)
        return [mask_dir / name for name in unique]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor | str | float]:
        sample = self.samples[idx]

        image = cv2.imread(str(sample.image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f"Unable to read image: {sample.image_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

        depth_raw = cv2.imread(str(sample.depth_path), cv2.IMREAD_UNCHANGED)
        if depth_raw is None:
            raise FileNotFoundError(f"Unable to read depth: {sample.depth_path}")
        depth = depth_raw.astype(np.float32) / 255.0 * self.max_depth

        seg_img = cv2.imread(str(sample.seg_path), cv2.IMREAD_GRAYSCALE)
        if seg_img is None:
            raise FileNotFoundError(f"Unable to read segmentation mask: {sample.seg_path}")
        seg_mask = seg_img.astype(np.uint8)

        depth_valid_img = cv2.imread(str(sample.depth_valid_path), cv2.IMREAD_GRAYSCALE)
        if depth_valid_img is None:
            raise FileNotFoundError(f"Unable to read depth valid mask: {sample.depth_valid_path}")
        depth_valid = (depth_valid_img < 250).astype(np.uint8)

        transformed = self.transform(
            {
                "image": image,
                "depth": depth,
                "semseg_mask": seg_mask,
                "depth_valid_mask": depth_valid,
            }
        )

        image_tensor = torch.from_numpy(transformed["image"])
        depth_tensor = torch.from_numpy(transformed["depth"])
        seg_mask_tensor = torch.from_numpy(transformed["semseg_mask"]).long()
        depth_valid_tensor = torch.from_numpy(transformed["depth_valid_mask"]).to(torch.bool)

        # Create segmentation valid mask (bottom 10px invalid) after resizing
        seg_valid_tensor = torch.ones_like(seg_mask_tensor, dtype=torch.bool)
        invalid_rows = min(10, seg_valid_tensor.shape[0])
        if invalid_rows > 0:
            seg_valid_tensor[-invalid_rows:, :] = False
            seg_mask_tensor = seg_mask_tensor.clone()
            seg_mask_tensor[-invalid_rows:, :] = 255  # ignore label for invalid area

        # Depth valid mask drives legacy valid_mask for backward compatibility
        valid_mask = depth_valid_tensor.clone()
        valid_mask &= compute_valid_mask(
            image_tensor,
            depth_tensor,
            max_depth=self.max_depth,
            dataset_name="EndoNeRF",
        )

        result: Dict[str, torch.Tensor | str | float] = {
            "image": image_tensor,
            "depth": depth_tensor,
            "semseg_mask": seg_mask_tensor,
            "depth_valid_mask": depth_valid_tensor,
            "seg_valid_mask": seg_valid_tensor,
            "valid_mask": valid_mask,
            "image_path": str(sample.image_path),
            "depth_path": str(sample.depth_path),
            "max_depth": self.max_depth,
            "source_type": "LS",
            "sequence": sample.sequence,
        }

        camera_info = self.get_camera_info()
        if camera_info is not None:
            intrinsics = camera_info.intrinsics.clone().to(torch.float32)
            result["camera_intrinsics"] = intrinsics
            result["camera_intrinsics_norm"] = normalize_intrinsics(intrinsics, camera_info.width, camera_info.height)
            result["camera_size"] = torch.tensor([camera_info.width, camera_info.height], dtype=torch.float32)
            result["camera_size_original"] = torch.tensor([camera_info.width, camera_info.height], dtype=torch.float32)
            result["camera_image_size"] = torch.tensor([float(image_tensor.shape[-1]), float(image_tensor.shape[-2])], dtype=torch.float32)
            result["camera_original_image_size"] = torch.tensor([float(image.shape[1]), float(image.shape[0])], dtype=torch.float32)
        elif not self._camera_warning_emitted:
            logger.error("EndoNeRF metadata missing camera intrinsics; skipping camera fields.")
            self._camera_warning_emitted = True

        return result




def _parse_args():
    import argparse

    parser = argparse.ArgumentParser(description="Generate cache for EndoNeRF dataset.")
    parser.add_argument("--root", type=Path, required=True, help="Path to EndoNeRF root directory")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory to store generated cache files")
    parser.add_argument("--filelist", type=Path, default=None, help="Optional existing filelist to subset samples")
    parser.add_argument("--filelist-name", type=str, default=None, help="Name of generated cache list file")
    parser.add_argument("--cache-root", type=Path, default=None, help="Cache root path mirroring original structure")
    parser.add_argument("--max-depth", type=float, default=0.3, help="Maximum depth clipping value")
    parser.add_argument("--val-ratio", type=float, default=0.1, help="Optional validation split ratio when generating two lists")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for shuffling")
    parser.add_argument("--limit", type=int, default=None, help="Optional limit of samples for quick testing")
    return parser.parse_args()


def _main():
    from dataset.cache_utils import generate_dataset_cache

    args = _parse_args()
    dataset = EndoNeRFDataset(root_dir=str(args.root), max_depth=args.max_depth)

    if args.filelist is not None:
        with open(args.filelist, "r", encoding="utf-8") as f:
            target_paths = {line.strip().split()[0] for line in f if line.strip()}
        dataset.samples = [s for s in dataset.samples if str(s.image_path) in target_paths]


    if args.limit is not None:
        dataset.samples = dataset.samples[: args.limit]

    filelist_name = args.filelist_name if args.filelist_name else "cache.txt"
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
