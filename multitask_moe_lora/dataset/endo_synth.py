import cv2
import torch
from torch.utils.data import Dataset
from torchvision.transforms import Compose
import os
import numpy as np
from typing import Dict, Any, Tuple, List, Optional

from .transform import Resize, NormalizeImage, PrepareForNet
from .utils import compute_valid_mask, remap_labels, map_ls_semseg_to_10_classes

ENDOSYNTH_REMAP = {
    0: 0,
    1: 2,
    2: 1,
    3: 5,
    4: 3,
    5: 6,
    6: 1,
    255: 255,
}


class EndoSynth(Dataset):
    """
    Dataset loader for EndoSynth NPZ files.

    Expected NPZ file format:
        - 'rgb': (H, W, 3) uint8 array, RGB image
        - 'depth': (H, W) float16/float32 array, depth map
        - 'seg': (H, W) uint8 array, segmentation mask

    Filelist format: each line contains the path to an NPZ file
    Example:
        /data/ziyi/multitask/data/EndoSynth/samples/000100.npz
        /data/ziyi/multitask/data/EndoSynth/samples/000101.npz
    """

    def __init__(self, filelist_path: str, mode: str, size: Tuple[int, int] = (518, 518), max_depth: float = 0.3):
        """
        Initialize EndoSynth dataset.

        Args:
            filelist_path: Path to file containing list of NPZ files (one per line)
            mode: 'train' or 'eval'
            size: Target size (width, height) for resizing
            max_depth: Maximum depth value for clipping
        """
        self.mode: str = mode
        self.size: Tuple[int, int] = size
        self.max_depth: float = max_depth

        with open(filelist_path, "r") as f:
            self.filelist: List[str] = f.read().splitlines()

        net_w, net_h = size
        self.transform = Compose([
            Resize(
                width=net_w,
                height=net_h,
                resize_target=True,
                keep_aspect_ratio=True,
                resize_method="upper_bound",
                ensure_multiple_of=1,
                image_interpolation_method=cv2.INTER_CUBIC,
                downscale_only=True,
            ),
            NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            PrepareForNet(),
        ])

    def read_npz_data(self, npz_path: str) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """
        Read RGB image, depth map, and segmentation mask from NPZ file.

        Args:
            npz_path: Path to NPZ file

        Returns:
            Tuple of (rgb, depth, seg) arrays, or None if reading fails
        """
        if not os.path.exists(npz_path):
            print(f"Error: File not found at {npz_path}")
            return None

        try:
            data = np.load(npz_path)

            # Check if required keys exist
            required_keys = ['rgb', 'depth', 'seg']
            for key in required_keys:
                if key not in data:
                    print(f"Error: NPZ file {npz_path} does not contain '{key}' key.")
                    return None

            rgb = data['rgb']  # (H, W, 3) uint8
            depth = data['depth']  # (H, W) float16/float32
            seg = data['seg']  # (H, W) uint8

            # Convert depth to float32 if it's float16
            if depth.dtype == np.float16:
                depth = depth.astype(np.float32)

            return rgb, depth, seg

        except Exception as e:
            print(f"An error occurred while reading NPZ file {npz_path}: {e}")
            return None

    def __getitem__(self, item: int) -> Optional[Dict[str, torch.Tensor | str]]:
        """
        Get a single sample from the dataset.

        Args:
            item: Index of the sample

        Returns:
            Dictionary containing:
                - image: (3, H, W) normalized RGB image tensor
                - depth: (H, W) depth map tensor
                - semseg_mask: (H, W) segmentation mask tensor (long)
                - valid_mask: (H, W) boolean mask for valid depth values
                - image_path: path to the NPZ file
                - depth_path: path to the NPZ file
                - mask_path: path to the NPZ file
                - max_depth: maximum depth value
        """
        npz_path: str = self.filelist[item].strip()

        # Read data from NPZ
        result = self.read_npz_data(npz_path)
        if result is None:
            print(f"Skipping item {item} due to NPZ reading failure.")
            return None

        rgb, depth, seg = result
        seg = remap_labels(seg.astype(np.uint8), ENDOSYNTH_REMAP)
        seg = map_ls_semseg_to_10_classes(seg, "EndoSynth")

        # Normalize RGB to [0, 1]
        image: np.ndarray = rgb.astype(np.float32) / 255.0

        # Clip depth values
        depth = np.clip(depth, 0, self.max_depth)

        # Apply transformations
        sample: Dict[str, np.ndarray] = self.transform({
            "image": image,
            "depth": depth,
            "semseg_mask": seg
        })

        # Convert to tensors
        sample["image"] = torch.from_numpy(sample["image"])
        sample["depth"] = torch.from_numpy(sample["depth"])
        sample["semseg_mask"] = torch.from_numpy(sample["semseg_mask"]).long()

        # Create valid mask (depth > 0 is valid)
        sample["valid_mask"] = compute_valid_mask(
            sample["image"],
            sample["depth"],
            max_depth=self.max_depth,
            dataset_name="EndoSynth",
        )

        # Store paths (all pointing to the same NPZ file)
        sample["image_path"] = npz_path
        sample["depth_path"] = npz_path
        sample["mask_path"] = npz_path
        sample["max_depth"] = self.max_depth
        sample["source_type"] = "LS"

        return sample

    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.filelist)
