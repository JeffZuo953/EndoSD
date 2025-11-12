import os
from dataclasses import dataclass
from typing import List, Optional, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision.transforms import Compose

from .camera_utils import get_camera_info, build_intrinsics_matrix
from .transform import NormalizeImage, PrepareForNet, Resize
from .utils import compute_valid_mask


@dataclass
class SyntheticPolypSample:
    rel_image_path: str

    @property
    def rel_depth_path(self) -> str:
        return self.rel_image_path.replace("/img/", "/z/")

    @property
    def rel_mask_path(self) -> str:
        return self.rel_image_path.replace("/img/", "/mask/")


class SyntheticPolypDataset(Dataset):
    """
    Synthetic polyp dataset that provides RGB, depth (z) and segmentation masks.

    Directory layout:
        root/
            vidXX/
                img/*.jpg
                z/*.jpg  (16-bit depth stored as grayscale JPEG)
                mask/*.jpg

    Masks: 255 -> class 4 (polyp), everything else -> class 0 (background).
    Depth: loaded as uint16/uint8, converted to float32 and scaled by ``1 / depth_scale``.
    """

    def __init__(
        self,
        filelist_path: str,
        root_path: str,
        mode: str = "train",
        size: Tuple[int, int] = (518, 518),
        max_depth: float = 0.25,
        depth_scale: float = 1000.0,
        dataset_name: str = "SyntheticPolyp",
        source_type: str = "NO",
        camera_key: Optional[str] = "synthetic_polyp",
    ) -> None:
        if not os.path.exists(filelist_path):
            raise FileNotFoundError(f"File list not found: {filelist_path}")

        with open(filelist_path, "r", encoding="utf-8") as f:
            rel_paths = [line.strip() for line in f if line.strip() and not line.startswith("#")]

        if not rel_paths:
            raise ValueError(f"No entries found in file list: {filelist_path}")

        self.root_path = os.path.abspath(root_path)
        self.filelist_path = filelist_path
        self.mode = mode
        self.size = size
        self.max_depth = max_depth
        self.depth_scale = depth_scale
        self.dataset_name = dataset_name
        self.source_type = source_type

        self.samples: List[SyntheticPolypSample] = [
            SyntheticPolypSample(rel_image_path=path) for path in rel_paths
        ]

        net_w, net_h = size
        self.transform = Compose([
            Resize(
                width=net_w,
                height=net_h,
                resize_target=True,
                keep_aspect_ratio=True,
                ensure_multiple_of=1,
                resize_method="upper_bound",
                image_interpolation_method=cv2.INTER_CUBIC,
                downscale_only=False,
            ),
            NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            PrepareForNet(),
        ])

        self.intrinsics = self._load_intrinsics(camera_key)

    def _load_intrinsics(self, camera_key: Optional[str]) -> torch.Tensor:
        if camera_key:
            info = get_camera_info(camera_key, None)
            if info is not None:
                return info.intrinsics.clone().to(torch.float32)
        # Fallback in case registry is missing – construct from provided specs
        fallback = build_intrinsics_matrix(448.13, 448.13, 320.0, 270.0)
        return fallback.to(torch.float32)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict:
        sample = self.samples[index]
        image_path = os.path.join(self.root_path, sample.rel_image_path)
        depth_path = os.path.join(self.root_path, sample.rel_depth_path)
        mask_path = os.path.join(self.root_path, sample.rel_mask_path)

        image = self._read_image(image_path)
        depth = self._read_depth(depth_path)
        mask = self._read_mask(mask_path)

        transformed = self.transform({
            "image": image,
            "depth": depth,
            "semseg_mask": mask,
        })

        image_tensor = torch.from_numpy(transformed["image"])
        depth_tensor = torch.from_numpy(transformed["depth"])
        mask_tensor = torch.from_numpy(transformed["semseg_mask"]).long()

        valid_mask = compute_valid_mask(
            image_tensor,
            depth_tensor,
            min_depth=0.0,
            max_depth=self.max_depth,
            dataset_name=self.dataset_name,
        )

        return {
            "image": image_tensor,
            "depth": depth_tensor,
            "semseg_mask": mask_tensor,
            "valid_mask": valid_mask,
            "image_path": image_path,
            "depth_path": depth_path,
            "mask_path": mask_path,
            "max_depth": self.max_depth,
            "source_type": self.source_type,
            "dataset_name": self.dataset_name,
            "intrinsics": self.intrinsics.clone(),
        }

    @staticmethod
    def _read_image(path: str) -> np.ndarray:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Image not found: {path}")
        image = cv2.imread(path, cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f"Unable to read image: {path}")
        return (cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)) / 255.0

    def _read_depth(self, path: str) -> np.ndarray:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Depth map not found: {path}")
        depth_raw = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if depth_raw is None:
            raise FileNotFoundError(f"Unable to read depth: {path}")
        depth_raw = depth_raw.astype(np.float32)

        if depth_raw.dtype == np.uint16 or depth_raw.max() > 255:
            max_val = 65535.0
        else:
            max_val = 255.0

        depth = (depth_raw / max(max_val, 1e-6)) * 0.25
        depth = np.clip(depth, 0.0, self.max_depth)
        return depth

    @staticmethod
    def _read_mask(path: str) -> np.ndarray:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Mask not found: {path}")
        mask_raw = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if mask_raw is None:
            raise FileNotFoundError(f"Unable to read mask: {path}")
        mask = np.zeros_like(mask_raw, dtype=np.uint8)
        mask[mask_raw == 255] = 4
        return mask
