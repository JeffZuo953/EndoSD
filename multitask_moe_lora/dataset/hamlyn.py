import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision.transforms import Compose

from .camera_utils import get_camera_info
from .transform import NormalizeImage, PrepareForNet, Resize
from .utils import compute_valid_mask
from ..util.local_cache import LocalCacheManager

logger = logging.getLogger(__name__)


@dataclass
class SequenceAssets:
    base_path: Path
    sequence_name: str
    image_dirs: Dict[str, Path]
    depth_dirs: Dict[str, Path]
    depth_by_image: Dict[str, Path]


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
        min_depth: float = 1e-3,
        intrinsics_cache_path: Optional[str] = None,
        cache_intrinsics: bool = True,
        image_ext: str = ".jpg",
        depth_ext: str = ".png",
        depth_scale: float = 1000.0,
        local_cache_dir: Optional[str] = None,
    ) -> None:
        self.rootpath: str = os.path.abspath(os.path.expanduser(rootpath))
        self.mode: str = mode
        self.size: Tuple[int, int] = size
        self.max_depth: float = max_depth
        self.min_depth: float = max(0.0, float(min_depth))
        self.cache_intrinsics: bool = cache_intrinsics
        self.image_ext: str = image_ext
        self.depth_ext: str = depth_ext
        self.depth_scale: float = depth_scale
        self._sequence_cache: Dict[str, SequenceAssets] = {}
        self._intrinsics_cache: Dict[str, torch.Tensor] = {}
        self._intrinsics_cache_dirty: bool = False
        self._missing_intrinsics: set[str] = set()
        self._cache = LocalCacheManager(local_cache_dir, namespace="native/Hamlyn")

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

        image_path, depth_path = self._resolve_frame_paths(assets, frame_id)
        if self._cache.enabled:
            image_path = self._ensure_local_asset("image", image_path)
            depth_path = self._ensure_local_asset("depth", depth_path)

        raw_image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if raw_image is None:
            raise FileNotFoundError(f"Unable to read image: {image_path}")
        image_rgb = cv2.cvtColor(raw_image, cv2.COLOR_BGR2RGB)
        image = image_rgb.astype(np.float32) / 255.0

        depth = self._load_depth_array(depth_path)
        depth = depth.astype(np.float32, copy=False)
        invalid_depth = (~np.isfinite(depth)) | (depth < self.min_depth) | (depth > self.max_depth)
        if invalid_depth.any():
            depth = depth.copy()
            depth[invalid_depth] = 0.0

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
            min_depth=self.min_depth,
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
            "min_depth": self.min_depth,
            "sequence": assets.sequence_name,
            "sequence_path": assets.base_path.as_posix(),
        }

        if intrinsics_tensor is not None:
            result["intrinsics"] = intrinsics_tensor
            result["intrinsics_key"] = assets.sequence_name

        result["depth_path"] = depth_path
        # Preserve domain label so validation buckets (LS/NO) stay aligned with config
        result["source_type"] = getattr(self, "dataset_type", "hamlyn")
        result["dataset_name"] = "hamlyn"

        return result

    def _ensure_local_asset(self, kind: str, path: str) -> str:
        if not self._cache.enabled:
            return path
        key = f"{kind}|{path}"
        return self._cache.ensure_copy(key, path)

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

        image_dirs = self._discover_directory_mapping(base_path, preferred=("color", "image"))
        depth_dirs = self._discover_directory_mapping(base_path, preferred=("depth",))
        depth_by_image = self._build_depth_mapping(image_dirs, depth_dirs)
        sequence_name = base_path.name

        assets = SequenceAssets(
            base_path=base_path,
            sequence_name=sequence_name,
            image_dirs=image_dirs,
            depth_dirs=depth_dirs,
            depth_by_image=depth_by_image,
        )
        self._sequence_cache[base_path_str] = assets
        return assets

    def _discover_directory_mapping(
        self,
        base_path: Path,
        preferred: Tuple[str, ...],
    ) -> Dict[str, Path]:
        def _add(mapping: Dict[str, Path], key: str, path: Path) -> None:
            key_norm = key.strip().lower()
            if key_norm not in mapping:
                mapping[key_norm] = path

        mapping: Dict[str, Path] = {}

        for name in preferred:
            candidate = base_path / name
            if candidate.is_dir():
                _add(mapping, "", candidate)
                _add(mapping, name, candidate)

        for child in sorted(base_path.iterdir(), key=lambda p: p.name.lower()):
            if not child.is_dir():
                continue
            lower_name = child.name.lower()
            for pref in preferred:
                pref_lower = pref.lower()
                if lower_name == pref_lower or lower_name.startswith(pref_lower):
                    _add(mapping, lower_name, child)
                    break

        if not mapping:
            raise FileNotFoundError(f"Unable to locate directories {preferred} under {base_path}")

        if "" not in mapping:
            first_path = next(iter(mapping.values()))
            _add(mapping, "", first_path)

        return mapping

    def _build_depth_mapping(
        self,
        image_dirs: Dict[str, Path],
        depth_dirs: Dict[str, Path],
    ) -> Dict[str, Path]:
        if not depth_dirs:
            raise FileNotFoundError("Hamlyn depth directories are missing.")

        default_depth = depth_dirs.get("")
        if default_depth is None:
            default_depth = next(iter(depth_dirs.values()))

        mapping: Dict[str, Path] = {"": default_depth}

        for key, _ in image_dirs.items():
            if not key:
                continue

            candidates: List[Path] = []
            lower_key = key.lower()

            if lower_key in depth_dirs:
                candidates.append(depth_dirs[lower_key])

            suffix = ""
            if lower_key.startswith("image"):
                suffix = lower_key[len("image") :]
            if suffix:
                suffix_clean = suffix.lstrip("_")
                candidate_keys = [
                    suffix,
                    suffix_clean,
                    f"depth{suffix}",
                    f"depth{suffix_clean}",
                ]
                digits = "".join(ch for ch in suffix if ch.isdigit())
                if digits:
                    candidate_keys.append(digits)
                    candidate_keys.append(f"depth{digits}")
            else:
                digits = "".join(ch for ch in lower_key if ch.isdigit())
                candidate_keys = [f"depth{digits}", digits] if digits else []

            for candidate_key in candidate_keys:
                if candidate_key and candidate_key.lower() in depth_dirs:
                    candidates.append(depth_dirs[candidate_key.lower()])

            selected = next((path for path in candidates if path is not None), default_depth)
            mapping[lower_key] = selected

        return mapping

    def _split_frame_token(self, frame_id: str) -> Tuple[Optional[str], str]:
        token = frame_id.replace("\\", "/").strip().strip("/")
        if "/" in token:
            parent, leaf = token.rsplit("/", 1)
            parent = parent.strip("/")
            hint = parent.lower() if parent else None
            return hint, leaf
        return None, token

    @staticmethod
    def _unique_paths(paths: List[Path]) -> List[Path]:
        unique: List[Path] = []
        seen: set[str] = set()
        for path in paths:
            key = str(path)
            if key not in seen:
                seen.add(key)
                unique.append(path)
        return unique

    def _gather_image_directories(self, assets: SequenceAssets, subdir_hint: Optional[str]) -> List[Path]:
        candidates: List[Path] = []
        if subdir_hint:
            path = assets.image_dirs.get(subdir_hint)
            if path:
                candidates.append(path)
        default_dir = assets.image_dirs.get("")
        if default_dir:
            candidates.append(default_dir)
        for key in sorted(assets.image_dirs.keys()):
            path = assets.image_dirs[key]
            if path not in candidates:
                candidates.append(path)
        return self._unique_paths(candidates)

    def _gather_depth_directories(self, assets: SequenceAssets, subdir_hint: Optional[str]) -> List[Path]:
        candidates: List[Path] = []
        if subdir_hint:
            mapped = assets.depth_by_image.get(subdir_hint)
            if mapped:
                candidates.append(mapped)
            direct = assets.depth_dirs.get(subdir_hint)
            if direct:
                candidates.append(direct)
        default_dir = assets.depth_by_image.get("") or assets.depth_dirs.get("")
        if default_dir:
            candidates.append(default_dir)
        for key in sorted(assets.depth_dirs.keys()):
            path = assets.depth_dirs[key]
            if path not in candidates:
                candidates.append(path)
        return self._unique_paths(candidates)

    def _resolve_media_path(
        self,
        directories: List[Path],
        frame_name: str,
        ext: str,
        description: str,
        frame_token: str,
        sequence_name: str,
    ) -> str:
        for directory in directories:
            candidate = directory / f"{frame_name}{ext}"
            if candidate.exists():
                return candidate.as_posix()
            matches = sorted(directory.glob(f"{frame_name}.*"))
            if matches:
                return matches[0].as_posix()
        raise FileNotFoundError(
            f"Hamlyn {description} file not found for frame '{frame_token}' in sequence '{sequence_name}'."
        )

    def _resolve_frame_paths(self, assets: SequenceAssets, frame_id: str) -> Tuple[str, str]:
        subdir_hint, frame_name = self._split_frame_token(frame_id)
        image_dirs = self._gather_image_directories(assets, subdir_hint)
        depth_dirs = self._gather_depth_directories(assets, subdir_hint)
        image_path = self._resolve_media_path(
            image_dirs,
            frame_name=frame_name,
            ext=self.image_ext,
            description="image",
            frame_token=frame_id,
            sequence_name=assets.sequence_name,
        )
        depth_path = self._resolve_media_path(
            depth_dirs,
            frame_name=frame_name,
            ext=self.depth_ext,
            description="depth",
            frame_token=frame_id,
            sequence_name=assets.sequence_name,
        )
        return image_path, depth_path

    def _load_depth_array(self, depth_path: str) -> np.ndarray:
        if depth_path.lower().endswith(".npy"):
            array = np.load(depth_path)
            if array.dtype != np.float32:
                array = array.astype(np.float32)
            return array

        depth_png = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
        if depth_png is None:
            raise FileNotFoundError(f"Unable to read depth map: {depth_path}")
        if depth_png.dtype in (np.float32, np.float64):
            depth = depth_png.astype(np.float32, copy=False)
        else:
            depth = depth_png.astype(np.float32, copy=False) / self.depth_scale
        return depth

    def _get_intrinsics_tensor(self, assets: SequenceAssets) -> Optional[torch.Tensor]:
        seq_key = assets.sequence_name.lower()
        if not self.cache_intrinsics:
            camera_info = get_camera_info("hamlyn", str(assets.base_path))
            if camera_info is None:
                if seq_key not in self._missing_intrinsics:
                    logger.error("[Hamlyn] Missing camera metadata for sequence '%s'; skipping intrinsics.", assets.sequence_name)
                    self._missing_intrinsics.add(seq_key)
                return None
            return camera_info.intrinsics.clone().to(torch.float32)

        if seq_key in self._intrinsics_cache:
            return self._intrinsics_cache[seq_key]

        camera_info = get_camera_info("hamlyn", str(assets.base_path))
        if camera_info is None:
            if seq_key not in self._missing_intrinsics:
                logger.error("[Hamlyn] Missing camera metadata for sequence '%s'; skipping intrinsics.", assets.sequence_name)
                self._missing_intrinsics.add(seq_key)
            return None

        intrinsics_tensor = camera_info.intrinsics.clone().to(torch.float32)
        self._intrinsics_cache[seq_key] = intrinsics_tensor
        self._intrinsics_cache_dirty = True
        self._persist_intrinsics_cache()
        return intrinsics_tensor

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
