import cv2
import torch
from torch.utils.data import Dataset
from torchvision.transforms import Compose
import os
import OpenEXR
import Imath
import numpy as np
from typing import Dict, Any, Tuple, List, Optional

from .transform import Resize, NormalizeImage, PrepareForNet


class InHouse(Dataset):

    def __init__(self, filelist_path: str, mode: str, size: Tuple[int, int] = (960, 540), max_depth: float = 0.05):
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
                resize_target=True if mode == "train" else False,
                # keep_aspect_ratio=True,
                keep_aspect_ratio=False,
                resize_method="lower_bound",
                ensure_multiple_of=14,
                image_interpolation_method=cv2.INTER_CUBIC,
            ),
            NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            PrepareForNet(),
        ])

    def read_exr_depth(self, exr_path: str) -> Optional[np.ndarray]:
        """
        Reads the 'V' channel from an EXR file and returns it as a numpy array.
        Based on the logic in inhouse.py.
        """
        if not os.path.exists(exr_path):
            print(f"Error: File not found at {exr_path}")
            return None

        try:
            file = OpenEXR.InputFile(exr_path)
            header = file.header()
            dw = header['dataWindow']
            size = (dw.max.x - dw.min.x + 1, dw.max.y - dw.min.y + 1)

            channel_names = header['channels'].keys()
            if 'V' not in channel_names:
                print(f"Error: EXR file {exr_path} does not contain a 'V' channel.")
                return None

            data = file.channel('V', Imath.PixelType(Imath.PixelType.FLOAT))
            img_data = np.frombuffer(data, dtype=np.float32).reshape(size[1], size[0])

            return img_data

        except Exception as e:
            print(f"An error occurred while reading EXR file {exr_path}: {e}")
            return None

    def __getitem__(self, item: int) -> Optional[Dict[str, torch.Tensor | str]]:
        # Each entry should provide: "<image_path> <depth_path> <mask_path>"
        parts: List[str] = self.filelist[item].split()
        if len(parts) < 3:
            raise ValueError(f"Expected each line to provide image, depth, and mask paths; got {len(parts)} items: {self.filelist[item]}")

        img_path, depth_path, mask_path = parts[:3]

        raw_image: Optional[np.ndarray] = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if raw_image is None:
            raise FileNotFoundError(f"Unable to read image at {img_path}; please verify the path.")
        image: np.ndarray = cv2.cvtColor(raw_image, cv2.COLOR_BGR2RGB) / 255.0

        # Read depth from EXR file
        depth: Optional[np.ndarray] = self.read_exr_depth(depth_path)

        if depth is None:
            print(f"Skipping item {item} due to EXR reading failure.")
            return None

        depth = np.clip(depth, 0, self.max_depth)

        mask: Optional[np.ndarray] = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise FileNotFoundError(f"Unable to read segmentation mask at {mask_path}; please verify the path.")

        sample: Dict[str, np.ndarray] = self.transform({"image": image, "depth": depth, "semseg_mask": mask})

        sample["image"] = torch.from_numpy(sample["image"])
        sample["depth"] = torch.from_numpy(sample["depth"])
        sample["semseg_mask"] = torch.from_numpy(sample["semseg_mask"]).long()

        # Create a valid mask. Assuming depth > 0 is valid. Adjust if your data has a different convention for invalid depth.
        sample["valid_mask"] = sample["depth"] > 0
        sample["image_path"] = img_path
        sample["depth_path"] = depth_path
        sample["mask_path"] = mask_path
        sample["max_depth"] = self.max_depth

        return sample

    def __len__(self) -> int:
        return len(self.filelist)
