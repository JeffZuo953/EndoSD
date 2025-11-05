import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision.transforms import Compose

from .transform import NormalizeImage, PrepareForNet, Resize
from .utils import compute_valid_mask


@dataclass
class SequenceAssets:
    base_path: Path
    image_dir: Path
    depth_dir: Path
    sequence_name: str
    intrinsics_path: Optional[Path]


class HamlynDataset(Dataset):
    """
    Dataset loader for the Hamlyn Endoscopic Depth and Motion dataset.

    Expected directory layout under ``rootpath``::

        color/        # RGB JPG frames
        depth/        # 16-bit depth PNG files (in millimeters)
        intrinsics.txt # Camera intrinsics (3x3 matrix)

    The file list should contain frame identifiers with base path and frame ID.
    Format: <base_path> <frame_id>
    Example: /path/to/rectified01 frame000000

    For each identifier, the dataset will look for:
        color/<frame_id>.jpg          (e.g., frame000000.jpg)
        depth/<frame_id>.png          (e.g., frame000000.png)

    Depth values in PNG files are stored as 16-bit integers in millimeters
    and converted to meters by dividing by depth_scale (default: 1000.0).

    Camera intrinsics from intrinsics.txt are cached into a single .pt file
    to avoid repeatedly parsing the file at runtime.
    """

    def __init__(
        self,
        filelist_path: str,
        rootpath: str,
        mode: str = "train",
        size: Tuple[int, int] = (518, 518),
        max_depth: float = 0.3,
        intrinsics_cache_path: Optional[str] = None,
        cache_intrinsics: bool = True,
        image_ext: str = ".jpg",
        depth_ext: str = ".png",
        depth_scale: float = 1000.0,
    ) -> None:
        self.rootpath: str = os.path.abspath(os.path.expanduser(rootpath))
        self.mode: str = mode
        self.size: Tuple[int, int] = size
        self.max_depth: float = max_depth
        self.cache_intrinsics: bool = cache_intrinsics
        self.image_ext: str = image_ext
        self.depth_ext: str = depth_ext
        self.depth_scale: float = depth_scale
        self._sequence_cache: Dict[str, SequenceAssets] = {}
        self._intrinsics_cache: Dict[str, torch.Tensor] = {}
        self._intrinsics_cache_dirty: bool = False

        with open(filelist_path, "r") as f:
            self.filelist: List[str] = [line.strip() for line in f if line.strip()]

        if intrinsics_cache_path is None:
            cache_dir = self.rootpath if os.path.exists(self.rootpath) else os.path.dirname(filelist_path)
            intrinsics_cache_path = os.path.join(cache_dir, "intrinsics_cache.pt")
        self.intrinsics_cache_path: Path = Path(intrinsics_cache_path)

        if self.cache_intrinsics and self.intrinsics_cache_path.exists():
            cached = torch.load(self.intrinsics_cache_path)
            if isinstance(cached, dict):
                self._intrinsics_cache = {
                    str(key): torch.as_tensor(val, dtype=torch.float32)
                    for key, val in cached.items()
                }

        net_w, net_h = self.size
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

    def __len__(self) -> int:
        return len(self.filelist)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor | str]:
        frame_token: str = self.filelist[index]

        # Parse the path to extract base_path and frame_id
        # Format: /path/to/rectified01 frame000000
        parts = frame_token.rsplit(' ', 1)
        if len(parts) == 2:
            # Full path format with space separator
            base_path, frame_id = parts
            assets = self._get_sequence_assets(base_path)
        else:
            # Fallback to old behavior if no space found
            frame_id: str = frame_token
            assets = self._get_sequence_assets(self.rootpath)
            base_path = assets.base_path.as_posix()

        image_path = self._resolve_frame_path(assets.image_dir, frame_id, self.image_ext, description="image")
        depth_path = self._resolve_frame_path(assets.depth_dir, frame_id, self.depth_ext, description="depth")

        raw_image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if raw_image is None:
            raise FileNotFoundError(f"Unable to read image: {image_path}")
        image_rgb = cv2.cvtColor(raw_image, cv2.COLOR_BGR2RGB)
        image = image_rgb.astype(np.float32) / 255.0

        if depth_path.lower().endswith('.npy'):
            depth = np.load(depth_path).astype(np.float32)
        else:
            depth_png = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
            if depth_png is None:
                raise FileNotFoundError(f"Unable to read depth map: {depth_path}")
            if depth_png.dtype != np.uint16 and depth_png.dtype != np.uint32:
                depth_png = depth_png.astype(np.uint16)
            depth = depth_png.astype(np.float32) / self.depth_scale

        depth = np.clip(depth, 0.0, self.max_depth)

        sample_dict = {
            "image": image,
            "depth": depth,
        }
        sample = self.transform(sample_dict)
        image_tensor = torch.from_numpy(sample["image"])
        depth_tensor = torch.from_numpy(sample["depth"])
        valid_mask_tensor = compute_valid_mask(
            image_tensor,
            depth_tensor,
            max_depth=self.max_depth,
            dataset_name="hamlyn",
        )

        intrinsics_tensor = self._get_intrinsics_tensor(assets)

        result: Dict[str, torch.Tensor | str] = {
            "image": image_tensor,
            "depth": depth_tensor,
            "valid_mask": valid_mask_tensor,
            "image_path": image_path,
            "max_depth": self.max_depth,
            "sequence": assets.sequence_name,
            "sequence_path": assets.base_path.as_posix(),
        }

        if intrinsics_tensor is not None:
            result["intrinsics"] = intrinsics_tensor
            result["intrinsics_key"] = assets.sequence_name

        result["depth_path"] = depth_path
        result["source_type"] = "hamlyn"

        return result

    def _get_sequence_assets(self, base_path_str: str) -> SequenceAssets:
        if base_path_str in self._sequence_cache:
            return self._sequence_cache[base_path_str]

        base_path = Path(os.path.expanduser(base_path_str))
        if not base_path.exists():
            try:
                resolved = base_path.resolve(strict=True)
            except FileNotFoundError as exc:
                raise FileNotFoundError(f"Hamlyn sequence path does not exist: {base_path}") from exc
            else:
                base_path = resolved

        image_dir = self._resolve_subdirectory(
            base_path,
            preferred=("color", "image"),
            description="image directory"
        )
        depth_dir = self._resolve_subdirectory(
            base_path,
            preferred=("depth",),
            description="depth directory"
        )
        sequence_name = base_path.name
        intrinsics_path = self._resolve_intrinsics_path(base_path, sequence_name)

        assets = SequenceAssets(
            base_path=base_path,
            image_dir=image_dir,
            depth_dir=depth_dir,
            sequence_name=sequence_name,
            intrinsics_path=intrinsics_path,
        )
        self._sequence_cache[base_path_str] = assets
        return assets

    def _resolve_subdirectory(
        self,
        base_path: Path,
        preferred: Tuple[str, ...],
        description: str,
    ) -> Path:
        for name in preferred:
            candidate = base_path / name
            if candidate.is_dir():
                return candidate

        candidates = [
            child for child in base_path.iterdir()
            if child.is_dir() and any(child.name.lower().startswith(name.lower()) for name in preferred)
        ]
        if candidates:
            candidates.sort()
            return candidates[0]

        raise FileNotFoundError(f"Unable to locate {description} under {base_path}")

    def _resolve_frame_path(
        self,
        directory: Path,
        frame_id: str,
        ext: str,
        description: str,
    ) -> str:
        candidate = directory / f"{frame_id}{ext}"
        if candidate.exists():
            return candidate.as_posix()

        matches = list(directory.glob(f"{frame_id}.*"))
        if matches:
            matches.sort()
            return matches[0].as_posix()

        raise FileNotFoundError(f"Hamlyn {description} file not found for frame {frame_id} in {directory}")

    def _resolve_intrinsics_path(self, base_path: Path, sequence_name: str) -> Optional[Path]:
        candidates: List[Path] = [
            base_path / "intrinsics.txt",
            base_path / "camera_intrinsics.txt",
        ]

        seq_digits = "".join(ch for ch in sequence_name if ch.isdigit())
        if seq_digits:
            calibration_root = Path(self.rootpath) / "calibration"
            candidates.append(calibration_root / seq_digits / "intrinsics.txt")
            candidates.append(calibration_root / seq_digits.lstrip("0") / "intrinsics.txt")

        for candidate in candidates:
            if candidate is None:
                continue
            candidate = candidate.expanduser().resolve()
            if candidate.exists():
                return candidate
        return None

    def _get_intrinsics_tensor(self, assets: SequenceAssets) -> Optional[torch.Tensor]:
        seq_key = assets.sequence_name
        if not self.cache_intrinsics:
            return self._read_intrinsics_file(assets.intrinsics_path)

        if seq_key in self._intrinsics_cache:
            return self._intrinsics_cache[seq_key]

        intrinsics_tensor = self._read_intrinsics_file(assets.intrinsics_path)
        if intrinsics_tensor is None:
            return None

        self._intrinsics_cache[seq_key] = intrinsics_tensor
        self._intrinsics_cache_dirty = True
        self._persist_intrinsics_cache()
        return intrinsics_tensor

    def _read_intrinsics_file(self, path: Optional[Path]) -> Optional[torch.Tensor]:
        if path is None or not path.exists():
            return None

        rows: List[List[float]] = []
        try:
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("#"):
                        continue
                    values = [float(x) for x in line.replace(",", " ").split()]
                    if values:
                        rows.append(values)
        except (OSError, ValueError):
            return None

        if len(rows) < 3:
            return None

        matrix = []
        for row in rows[:3]:
            if len(row) < 3:
                return None
            matrix.append(row[:3])

        return torch.tensor(matrix, dtype=torch.float32)

    def _persist_intrinsics_cache(self) -> None:
        if not self.cache_intrinsics:
            return
        if not self._intrinsics_cache_dirty:
            return
        if not self._intrinsics_cache:
            return

        self.intrinsics_cache_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self._intrinsics_cache, self.intrinsics_cache_path)
        self._intrinsics_cache_dirty = False
