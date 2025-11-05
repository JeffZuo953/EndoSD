import os
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision.transforms import Compose

from .transform import NormalizeImage, PrepareForNet, Resize


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
        self.rootpath: str = rootpath
        self.mode: str = mode
        self.size: Tuple[int, int] = size
        self.max_depth: float = max_depth
        self.cache_intrinsics: bool = cache_intrinsics
        self.image_ext: str = image_ext
        self.depth_ext: str = depth_ext
        self.depth_scale: float = depth_scale

        with open(filelist_path, "r") as f:
            self.filelist: List[str] = [line.strip() for line in f if line.strip()]

        self.image_dir: str = os.path.join(self.rootpath, "color")
        self.depth_dir: str = os.path.join(self.rootpath, "depth")

        if intrinsics_cache_path is None:
            cache_dir = self.rootpath if os.path.exists(self.rootpath) else os.path.dirname(filelist_path)
            intrinsics_cache_path = os.path.join(cache_dir, "intrinsics_cache.pt")
        self.intrinsics_cache_path: str = intrinsics_cache_path

        self.intrinsics_tensor: Optional[torch.Tensor] = (
            self._load_or_create_intrinsics() if self.cache_intrinsics else None
        )

        net_w, net_h = self.size
        self.transform = Compose(
            [
                Resize(
                    width=net_w,
                    height=net_h,
                    resize_target=True,
                    keep_aspect_ratio=False,
                    ensure_multiple_of=14,
                    resize_method="lower_bound",
                    image_interpolation_method=cv2.INTER_CUBIC,
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
            image_path: str = os.path.join(base_path, "color", frame_id + self.image_ext)
            depth_path: str = os.path.join(base_path, "depth", frame_id + self.depth_ext)
        else:
            # Fallback to old behavior if no space found
            frame_id: str = frame_token
            image_path: str = os.path.join(self.image_dir, frame_id + self.image_ext)
            depth_path: str = os.path.join(self.depth_dir, frame_id + self.depth_ext)

        raw_image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if raw_image is None:
            raise FileNotFoundError(f"Unable to read image: {image_path}")
        image = cv2.cvtColor(raw_image, cv2.COLOR_BGR2RGB) / 255.0

        depth_png = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
        if depth_png is None:
            raise FileNotFoundError(f"Unable to read depth map: {depth_path}")

        # Convert 16-bit PNG depth to meters by dividing by depth_scale
        # Hamlyn dataset stores depth in millimeters
        depth = depth_png.astype(np.float32) / self.depth_scale

        # Clip to valid depth range (1-300mm -> 0.001-0.3m)
        depth = np.clip(depth, 0.0, self.max_depth)

        original_h, original_w = image.shape[:2]

        sample = self.transform({"image": image, "depth": depth})
        image_tensor = torch.from_numpy(sample["image"])
        depth_tensor = torch.from_numpy(sample["depth"])

        intrinsics_tensor: Optional[torch.Tensor] = None
        if self.cache_intrinsics:
            intrinsics_tensor = self.intrinsics_tensor
        else:
            intrinsics_tensor = self._read_intrinsics_from_file()

        if intrinsics_tensor is None:
            raise FileNotFoundError(f"Unable to load intrinsics from {self.rootpath}")

        intrinsics_tensor = intrinsics_tensor.clone().to(torch.float32)
        resized_h, resized_w = image_tensor.shape[-2:]
        scale_x = resized_w / float(original_w)
        scale_y = resized_h / float(original_h)
        intrinsics_tensor[0, 0] *= scale_x
        intrinsics_tensor[1, 1] *= scale_y
        intrinsics_tensor[0, 2] *= scale_x
        intrinsics_tensor[1, 2] *= scale_y

        result: Dict[str, torch.Tensor | str] = {
            "image": image_tensor,
            "depth": depth_tensor,
            "valid_mask": depth_tensor > 0,
            "image_path": image_path,
            "max_depth": self.max_depth,
        }

        result["intrinsics"] = intrinsics_tensor
        result["depth_path"] = depth_path
        result["source_type"] = "hamlyn"

        return result

    def _read_intrinsics_from_file(self) -> Optional[torch.Tensor]:
        """
        Read camera intrinsics from intrinsics.txt file.
        The file should contain a 3x3 intrinsic matrix.
        """
        # Try different possible intrinsics file locations
        intrinsics_paths = [
            os.path.join(self.rootpath, "intrinsics.txt"),
            os.path.join(os.path.dirname(self.rootpath), "intrinsics.txt"),
        ]

        for intrinsics_path in intrinsics_paths:
            if not os.path.exists(intrinsics_path):
                continue

            try:
                # Read the intrinsics file
                intrinsics_matrix = []
                with open(intrinsics_path, "r") as f:
                    for line in f:
                        line = line.strip()
                        if not line or line.startswith('#'):
                            continue
                        # Parse space or comma separated values
                        values = [float(x) for x in line.replace(',', ' ').split()]
                        intrinsics_matrix.append(values)

                if len(intrinsics_matrix) == 3 and all(len(row) == 3 for row in intrinsics_matrix):
                    return torch.as_tensor(intrinsics_matrix, dtype=torch.float32)
            except (ValueError, IOError):
                continue

        return None

    def _load_or_create_intrinsics(self) -> Optional[torch.Tensor]:
        """
        Load cached intrinsics or create new cache.
        Unlike SCARD, Hamlyn uses a single intrinsics.txt file for all frames.
        """
        if os.path.exists(self.intrinsics_cache_path):
            cached = torch.load(self.intrinsics_cache_path)
            if isinstance(cached, torch.Tensor):
                return cached

        intrinsics_tensor = self._read_intrinsics_from_file()

        if intrinsics_tensor is not None:
            # Ensure the cache directory exists before saving
            os.makedirs(os.path.dirname(self.intrinsics_cache_path), exist_ok=True)
            torch.save(intrinsics_tensor, self.intrinsics_cache_path)

        return intrinsics_tensor
