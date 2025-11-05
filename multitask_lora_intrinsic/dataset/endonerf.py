import os
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision.transforms import Compose

from .transform import NormalizeImage, PrepareForNet, Resize


class EndonerfDataset(Dataset):
    """
    Dataset loader for the Endonerf dataset.

    Expected directory layout under ``rootpath``::

        color/        # RGB JPG/PNG frames
        depth/        # 16-bit depth PNG files (in millimeters)

    The file list should contain frame identifiers with base path and frame ID.
    Format: <frame_id>
    Example: frame000000

    For each identifier, the dataset will look for:
        color/<frame_id>.jpg (or .png)
        depth/<frame_id>.png

    Depth values in PNG files are stored as 16-bit integers in millimeters
    and converted to meters by dividing by depth_scale (default: 1000.0).
    Camera intrinsics are typically handled by NeRF frameworks and are not
    loaded by this dataset class.
    """

    def __init__(
        self,
        filelist_path: str,
        rootpath: str,
        mode: str = "train",
        size: Tuple[int, int] = (518, 518),
        max_depth: float = 0.3,
        image_ext: str = ".jpg",
        depth_ext: str = ".png",
        depth_scale: float = 1000.0,
    ) -> None:
        self.rootpath: str = rootpath
        self.mode: str = mode
        self.size: Tuple[int, int] = size
        self.max_depth: float = max_depth
        self.image_ext: str = image_ext
        self.depth_ext: str = depth_ext
        self.depth_scale: float = depth_scale

        with open(filelist_path, "r") as f:
            self.filelist: List[str] = [line.strip() for line in f if line.strip()]

        # Assuming 'color' and 'depth' subdirectories directly under rootpath
        self.image_dir: str = os.path.join(self.rootpath, "color")
        self.depth_dir: str = os.path.join(self.rootpath, "depth")

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
        frame_id: str = self.filelist[index]

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
        depth = depth_png.astype(np.float32) / self.depth_scale

        # Clip to valid depth range
        depth = np.clip(depth, 0.0, self.max_depth)

        # original_h, original_w = image.shape[:2] # Not directly used after transform now

        sample = self.transform({"image": image, "depth": depth})
        image_tensor = torch.from_numpy(sample["image"])
        depth_tensor = torch.from_numpy(sample["depth"])

        result: Dict[str, torch.Tensor | str] = {
            "image": image_tensor,
            "depth": depth_tensor,
            "valid_mask": depth_tensor > 0,
            "image_path": image_path,
            "max_depth": self.max_depth,
            "source_type": "endonerf",
        }
        
        # Intrinsics are not loaded by this dataset class as per Endonerf typical setup
        # If needed, they would be loaded separately or inferred by the NeRF model.

        return result
