import os
import glob
import logging
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from typing import Dict, List, Tuple

from ..dataset.transform import Resize, NormalizeImage, PrepareForNet
from torchvision.transforms import Compose

class Kidney3DDataset(Dataset):
    def __init__(
        self,
        rootpath: str,
        mode: str = "train",
        size: Tuple[int, int] = (518, 518),
        max_depth: float = 0.2, # Max depth in meters for Kidney3D
        image_ext: str = ".png",
        depth_ext: str = ".png",
        depth_scale: float = 1000.0, # Convert mm to meters
    ) -> None:
        super().__init__()
        self.rootpath = rootpath
        self.mode = mode
        self.size = size
        self.max_depth = max_depth
        self.depth_scale = depth_scale
        self.logger = logging.getLogger(self.__class__.__name__)

        self.transform = Compose([
            Resize(
                width=self.size[0],
                height=self.size[1],
                resize_target=False,
                keep_aspect_ratio=False,
                ensure_multiple_of=14, # Consistent with DepthAnythingV2's DPT model
                image_interpolation_method=cv2.INTER_CUBIC,
            ),
            NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            PrepareForNet(),
        ])

        self.samples: List[Dict[str, str]] = self._load_samples()
        self.logger.info(f"Loaded {len(self.samples)} {mode} samples from Kidney3D dataset.")

    def _load_samples(self) -> List[Dict[str, str]]:
        samples = []
        
        # Kidney3D dataset structure example:
        # rootpath/
        #   train/
        #     images/
        #       patient_001/
        #         0000.png
        #         ...
        #     depth/
        #       patient_001/
        #         0000.png
        #         ...
        #   val/
        #     images/
        #     depth/
        #   test/
        #     images/
        #     depth/
        
        image_dir = os.path.join(self.rootpath, self.mode, "images")
        depth_dir = os.path.join(self.rootpath, self.mode, "depth")

        if not os.path.isdir(image_dir):
            self.logger.error(f"Image directory not found: {image_dir}")
            return []
        if not os.path.isdir(depth_dir):
            self.logger.error(f"Depth directory not found: {depth_dir}")
            return []
        
        image_paths = sorted(glob.glob(os.path.join(image_dir, "**", f"*{self.image_ext}"), recursive=True))
        
        for img_path in image_paths:
            # Reconstruct depth path based on image path
            relative_path = os.path.relpath(img_path, image_dir)
            depth_path = os.path.join(depth_dir, relative_path)
            
            if not os.path.exists(depth_path):
                self.logger.warning(f"Matching depth file not found for {img_path}, skipping.")
                continue
            
            samples.append({
                "image_path": img_path,
                "depth_path": depth_path,
            })
        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor | str]:
        sample_paths = self.samples[index]
        image_path = sample_paths["image_path"]
        depth_path = sample_paths["depth_path"]

        # Load image
        raw_image = cv2.imread(image_path)
        if raw_image is None:
            self.logger.error(f"Failed to load image: {image_path}")
            raise FileNotFoundError(f"Failed to load image: {image_path}")
        
        # Convert to RGB and normalize to [0, 1]
        image = cv2.cvtColor(raw_image, cv2.COLOR_BGR2RGB) / 255.0

        # Load depth
        # Depth maps are 16-bit PNGs, values in mm
        raw_depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
        if raw_depth is None:
            self.logger.error(f"Failed to load depth map: {depth_path}")
            raise FileNotFoundError(f"Failed to load depth map: {depth_path}")
        
        # Convert to float32 and scale to meters
        depth = raw_depth.astype(np.float32) / self.depth_scale

        # Apply transformations
        transformed_data = self.transform({"image": image, "depth": depth})
        image_tensor = transformed_data["image"]
        depth_tensor = transformed_data["depth"]

        # Ensure depth values are within max_depth, clip if necessary
        depth_tensor = torch.clamp(depth_tensor, 0, self.max_depth)

        result: Dict[str, torch.Tensor | str] = {
            "image": image_tensor,
            "depth": depth_tensor,
            "valid_mask": depth_tensor > 0, # Pixels with depth > 0 are valid
            "image_path": image_path,
            "max_depth": self.max_depth,
            "source_type": "kidney3d",
        }
        return result
