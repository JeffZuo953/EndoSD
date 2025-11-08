import json
import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision.transforms import Compose

from .camera_utils import CameraInfo, make_camera_info, normalize_intrinsics
from .metadata.scared import SCAREDMetadata
from .transform import NormalizeImage, PrepareForNet, Resize
from .utils import compute_valid_mask


class SCardDataset(Dataset):
    """
    Dataset loader for the SCARED keyframe data.

    Expected directory layout under ``rootpath``::

        depthmap_rectified/   # 16-bit depth PNG files (in millimeters)
        left_rectified/       # RGB PNG frames (left camera)
        poses/                # Per-frame JSON files with calibration data

    The file list should contain frame identifiers with base path and frame ID.
    Format: <base_path> <frame_id>
    Example: /path/to/data frame_data000000

    For each identifier, the dataset will look for:
        left_rectified/<numeric_id>.png          (e.g., 000000.png)
        depthmap_rectified/<numeric_id>.png      (e.g., 000000.png)
        poses/<frame_id>.json                    (e.g., frame_data000000.json)

    Depth values in PNG files are stored as 16-bit integers and converted to
    meters by dividing by depth_scale (default: 1000.0 for millimeters).

    Left-camera intrinsics from the JSON files (``camera-calibration.KL``) are
    cached into a single ``.pt`` file to avoid repeatedly parsing JSON at
    runtime.
    """

    CAMERA_WIDTH: int = 1350
    CAMERA_HEIGHT: int = 1080
    CAMERA_INTRINSICS_ROOT: Path = Path(os.path.expanduser("~/ssde/data/LS/SCARED/intrisics")).resolve()

    # Static registries populated on-demand the first time intrinsics are needed.
    CAMERA_INFO_CACHE: Dict[str, CameraInfo] = {}
    CAMERA_INFO_BY_DATASET: Dict[str, Dict[str, CameraInfo]] = {}

    def __init__(
        self,
        filelist_path: str,
        rootpath: str,
        mode: str = "train",
        size: Tuple[int, int] = (518, 518),
        max_depth: float = 0.3,
        intrinsics_cache_path: Optional[str] = None,
        cache_intrinsics: bool = True,
        image_ext: str = ".png",
        depth_ext: str = ".png",
        pose_ext: str = ".json",
        depth_scale: float = 256000.0,
    ) -> None:
        self.rootpath: str = rootpath
        self.mode: str = mode
        self.size: Tuple[int, int] = size
        self.max_depth: float = max_depth
        self.cache_intrinsics: bool = cache_intrinsics
        self.image_ext: str = image_ext
        self.depth_ext: str = depth_ext
        self.pose_ext: str = pose_ext
        self.depth_scale: float = depth_scale

        with open(filelist_path, "r") as f:
            self.filelist: List[str] = [line.strip() for line in f if line.strip()]

        # Note: image_dir, depth_dir, pose_dir are not used when filelist contains full paths
        self.image_dir: str = os.path.join(self.rootpath, "left_rectified")
        self.depth_dir: str = os.path.join(self.rootpath, "depthmap_rectified")
        self.pose_dir: str = os.path.join(self.rootpath, "poses")

        if intrinsics_cache_path is None:
            # Use rootpath for cache to ensure the directory exists
            cache_dir = self.rootpath if os.path.exists(self.rootpath) else os.path.dirname(filelist_path)
            intrinsics_cache_path = os.path.join(cache_dir, "left_intrinsics.pt")
        self.intrinsics_cache_path: str = intrinsics_cache_path

        self.intrinsics_map: Dict[str, torch.Tensor] = (
            self._load_or_create_intrinsics() if self.cache_intrinsics else {}
        )

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
        # Format: /path/to/data frame_data000000
        parts = frame_token.rsplit(' ', 1)
        if len(parts) == 2:
            # Full path format with space separator
            base_path, frame_id_raw = parts
            frame_id_normalized = self._normalize_key(frame_id_raw)

            # Extract numeric part for image/depth files (e.g., "frame_data000000" -> "000000")
            # Keep full name for pose files (e.g., "frame_data000000.json")
            numeric_id = self._extract_numeric_id(frame_id_normalized)

            image_path: str = os.path.join(base_path, "left_rectified", numeric_id + self.image_ext)
            depth_path: str = os.path.join(base_path, "depthmap_rectified", numeric_id + self.depth_ext)
            pose_path: str = os.path.join(base_path, "poses", frame_id_normalized + self.pose_ext)
        else:
            # Fallback to old behavior if no space found
            frame_id_normalized: str = self._normalize_key(frame_token)
            image_path: str = os.path.join(self.image_dir, frame_id_normalized + self.image_ext)
            depth_path: str = os.path.join(self.depth_dir, frame_id_normalized + self.depth_ext)
            pose_path: str = os.path.join(self.pose_dir, frame_id_normalized + self.pose_ext)

        raw_image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if raw_image is None:
            raise FileNotFoundError(f"Unable to read image: {image_path}")
        image = cv2.cvtColor(raw_image, cv2.COLOR_BGR2RGB) / 255.0

        depth_png = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
        if depth_png is None:
            raise FileNotFoundError(f"Unable to read depth map: {depth_path}")

        # Convert 16-bit PNG depth to meters by dividing by depth_scale
        # SCARED dataset stores depth in millimeters (or scaled units)
        depth = depth_png.astype(np.float32) / self.depth_scale

        # Clip to valid depth range
        depth = np.clip(depth, 0.0, self.max_depth)

        sample = self.transform({"image": image, "depth": depth})
        image_tensor = torch.from_numpy(sample["image"])
        depth_tensor = torch.from_numpy(sample["depth"])

        intrinsics_tensor: Optional[torch.Tensor] = None
        if self.cache_intrinsics:
            intrinsics_tensor = self.intrinsics_map.get(frame_id_normalized)
            if intrinsics_tensor is None:
                intrinsics_tensor = self._read_intrinsics_from_pose(pose_path)
                if intrinsics_tensor is not None:
                    self.intrinsics_map[frame_id_normalized] = intrinsics_tensor
                    self._persist_intrinsics_cache()
        else:
            intrinsics_tensor = self._read_intrinsics_from_pose(pose_path)

        if intrinsics_tensor is None:
            raise FileNotFoundError(f"Unable to load intrinsics for frame {frame_id_normalized} from {pose_path}")

        valid_mask = compute_valid_mask(
            image_tensor,
            depth_tensor,
            max_depth=self.max_depth,
            dataset_name="SCARED",
        )

        width_px = raw_image.shape[1]
        height_px = raw_image.shape[0]

        result: Dict[str, torch.Tensor | str] = {
            "image": image_tensor,
            "depth": depth_tensor,
            "valid_mask": valid_mask,
            "image_path": image_path,
            "max_depth": self.max_depth,
        }

        result["intrinsics"] = intrinsics_tensor
        result["camera_intrinsics"] = intrinsics_tensor
        result["camera_intrinsics_norm"] = normalize_intrinsics(intrinsics_tensor, width_px, height_px)
        result["camera_size"] = torch.tensor([width_px, height_px], dtype=torch.float32)
        result["camera_size_original"] = torch.tensor([self.CAMERA_WIDTH, self.CAMERA_HEIGHT], dtype=torch.float32)
        result["camera_image_size"] = torch.tensor([float(image_tensor.shape[-1]), float(image_tensor.shape[-2])], dtype=torch.float32)
        result["camera_original_image_size"] = torch.tensor([float(width_px), float(height_px)], dtype=torch.float32)
        result["depth_path"] = depth_path
        result["pose_path"] = pose_path
        result["source_type"] = "scard"

        return result

    # ----------------------------------------------------------------------
    # Camera intrinsics helpers
    # ----------------------------------------------------------------------
    @classmethod
    def _load_camera_info(cls) -> None:
        if cls.CAMERA_INFO_CACHE:
            return
        if not cls.CAMERA_INTRINSICS_ROOT.exists():
            return

        for dataset_dir in sorted(cls.CAMERA_INTRINSICS_ROOT.glob("dataset_*")):
            dataset_name = dataset_dir.name.lower()
            for key_dir in sorted(dataset_dir.glob("keyfram_*")):
                key_id = key_dir.name.split("_")[-1]
                yaml_path = key_dir / "endoscope_calibration.yaml"
                if not yaml_path.exists():
                    continue
                fs = cv2.FileStorage(str(yaml_path), cv2.FILE_STORAGE_READ)
                if not fs.isOpened():
                    continue
                matrix = fs.getNode("M1").mat()
                fs.release()
                if matrix is None or matrix.size == 0:
                    continue
                fx = float(matrix[0, 0])
                fy = float(matrix[1, 1])
                cx = float(matrix[0, 2])
                cy = float(matrix[1, 2])
                cache_key = f"{dataset_name}/keyframe_{int(key_id):02d}"
                info = make_camera_info(
                    width=cls.CAMERA_WIDTH,
                    height=cls.CAMERA_HEIGHT,
                    fx=fx,
                    fy=fy,
                    cx=cx,
                    cy=cy,
                )
                cls.CAMERA_INFO_CACHE[cache_key] = info
                dataset_bucket = cls.CAMERA_INFO_BY_DATASET.setdefault(dataset_name, {})
                dataset_bucket[cache_key] = info

    @classmethod
    def _lookup_camera_info(cls, sample_path: Optional[str]) -> Optional[CameraInfo]:
        if not sample_path:
            return None
        info_metadata = SCAREDMetadata.get_camera_info(sample_path)
        if info_metadata is not None:
            return info_metadata
        normalized_path = sample_path.replace("\\", "/").lower()
        dataset_match = re.search(r"dataset_(\d+)", normalized_path)
        keyframe_match = re.search(r"keyfram?e?_(\d+)", normalized_path)
        if not dataset_match or not keyframe_match:
            return None
        dataset_idx = int(dataset_match.group(1))
        keyframe_idx = int(keyframe_match.group(1))
        key = f"dataset_{dataset_idx}/keyframe_{keyframe_idx:02d}"
        info = cls.CAMERA_INFO_CACHE.get(key)
        if info is not None:
            return info
        # fallback to dataset-level (if available)
        prefix = f"dataset_{dataset_idx}/"
        for cache_key, entry in cls.CAMERA_INFO_CACHE.items():
            if cache_key.startswith(prefix):
                return entry
        return None

    @staticmethod
    def get_camera_info(sample_path: Optional[str] = None) -> Optional[CameraInfo]:
        """Return CameraInfo for the given sample path."""
        return SCardDataset._lookup_camera_info(sample_path)

    def _normalize_key(self, value: str) -> str:
        base = os.path.basename(value)
        return os.path.splitext(base)[0]

    def _extract_numeric_id(self, value: str) -> str:
        """
        Extract the numeric part from frame ID.
        E.g., "frame_data000000" -> "000000"
        If no pattern matches, return the original value.
        """
        # Match patterns like "frame_data000000", "frame_000000", etc.
        # and extract the trailing digits
        match = re.search(r'(\d{6,})$', value)
        if match:
            return match.group(1)
        return value

    def _read_intrinsics_from_pose(self, pose_path: str) -> Optional[torch.Tensor]:
        if not os.path.exists(pose_path):
            return None

        try:
            with open(pose_path, "r") as f:
                pose_data = json.load(f)
        except json.JSONDecodeError:
            return None

        calibration = pose_data.get("camera-calibration", {})
        kl_matrix = calibration.get("KL")
        if kl_matrix is None:
            return None

        return torch.as_tensor(kl_matrix, dtype=torch.float32)

    def _load_or_create_intrinsics(self) -> Dict[str, torch.Tensor]:
        if os.path.exists(self.intrinsics_cache_path):
            cached = torch.load(self.intrinsics_cache_path)
            if isinstance(cached, dict):
                return {self._normalize_key(k): torch.as_tensor(v, dtype=torch.float32) for k, v in cached.items()}

        intrinsics: Dict[str, torch.Tensor] = {}
        for token in self.filelist:
            # Parse the path to extract base_path and frame_id
            parts = token.rsplit(' ', 1)
            if len(parts) == 2:
                base_path, frame_id = parts
                frame_id = self._normalize_key(frame_id)
                pose_path = os.path.join(base_path, "poses", frame_id + self.pose_ext)
            else:
                frame_id = self._normalize_key(token)
                pose_path = os.path.join(self.pose_dir, frame_id + self.pose_ext)

            intr = self._read_intrinsics_from_pose(pose_path)
            if intr is not None:
                intrinsics[frame_id] = intr

        if intrinsics:
            # Ensure the cache directory exists before saving
            os.makedirs(os.path.dirname(self.intrinsics_cache_path), exist_ok=True)
            torch.save(intrinsics, self.intrinsics_cache_path)

        return intrinsics

    def _persist_intrinsics_cache(self) -> None:
        if not self.cache_intrinsics:
            return
        if not self.intrinsics_map:
            return
        # Ensure the cache directory exists before saving
        os.makedirs(os.path.dirname(self.intrinsics_cache_path), exist_ok=True)
        torch.save(self.intrinsics_map, self.intrinsics_cache_path)
