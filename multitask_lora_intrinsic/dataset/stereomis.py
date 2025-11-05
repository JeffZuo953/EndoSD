import os
from typing import Dict, List, Optional, Tuple
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision.transforms import Compose
from .transform import NormalizeImage, PrepareForNet, Resize

class StereoMISDataset(Dataset):
    """
    Dataset loader for the StereoMIS dataset.
    Assumes images are in {rootpath}/color/ and depth maps in {rootpath}/depth/.
    Depth maps are expected to be 16-bit PNGs, representing depth in millimeters.
    Camera intrinsics are loaded from a 'calib.txt' file in each sequence directory.
    """
    def __init__(
        self,
        filelist_path: str,
        rootpath: str,
        mode: str = "train",
        size: Tuple[int, int] = (518, 518),
        max_depth: float = 1.0, # StereoMIS depth can be larger, adjust as needed
        image_ext: str = ".png",
        depth_ext: str = ".png",
        depth_scale: float = 1000.0, # Converts mm to meters
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

        # Each line in filelist is expected to be 'sequence_name/frame_id'
        self.sequences: Dict[str, List[str]] = {}
        for entry in self.filelist:
            sequence_name = entry.split('/')[0]
            if sequence_name not in self.sequences:
                self.sequences[sequence_name] = []
            self.sequences[sequence_name].append(entry)
        
        self.all_data_paths: List[Dict[str, str]] = []
        for sequence_name, frames in self.sequences.items():
            for frame_id_full in frames:
                # Assuming frame_id_full is like "seq01/00001"
                # Need to extract just the numeric part for image/depth file naming
                frame_id = os.path.basename(frame_id_full) 
                
                self.all_data_paths.append({
                    "image_path": os.path.join(self.rootpath, frame_id_full + self.image_ext),
                    "depth_path": os.path.join(self.rootpath, frame_id_full + self.depth_ext),
                    "sequence_dir": os.path.join(self.rootpath, sequence_name),
                    "frame_id": frame_id,
                })

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
        return len(self.all_data_paths)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor | str]:
        data_info = self.all_data_paths[index]
        image_path = data_info["image_path"]
        depth_path = data_info["depth_path"]
        sequence_dir = data_info["sequence_dir"]
        frame_id = data_info["frame_id"]

        # Load image
        raw_image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if raw_image is None:
            raise FileNotFoundError(f"Unable to read image: {image_path}")
        image = cv2.cvtColor(raw_image, cv2.COLOR_BGR2RGB) / 255.0

        # Load depth
        depth_png = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
        if depth_png is None:
            raise FileNotFoundError(f"Unable to read depth map: {depth_path}")

        depth = depth_png.astype(np.float32) / self.depth_scale
        depth = np.clip(depth, 0.0, self.max_depth)

        # Load intrinsics from calib.txt
        intrinsics_path = os.path.join(sequence_dir, "calib.txt")
        if not os.path.exists(intrinsics_path):
            raise FileNotFoundError(f"Intrinsics file not found: {intrinsics_path}")
        
        intrinsics = self._read_intrinsics(intrinsics_path)
        
        sample = self.transform({"image": image, "depth": depth})
        image_tensor = torch.from_numpy(sample["image"])
        depth_tensor = torch.from_numpy(sample["depth"])

        result: Dict[str, torch.Tensor | str] = {
            "image": image_tensor,
            "depth": depth_tensor,
            "valid_mask": depth_tensor > 0,
            "image_path": image_path,
            "max_depth": self.max_depth,
            "intrinsics": torch.from_numpy(intrinsics).to(torch.float32),
            "source_type": "stereomis",
        }
        return result

    def _read_intrinsics(self, calib_path: str) -> np.ndarray:
        """Reads camera intrinsics from a calib.txt file."""
        # Assuming calib.txt contains a single line with space-separated values:
        # fx 0 cx 0 fy cy 0 0 1 (flattened 3x3 matrix)
        with open(calib_path, 'r') as f:
            line = f.readline().strip()
        
        parts = line.split()
        if len(parts) == 9: # Standard 3x3 flattened matrix
            intrinsics = np.array([float(p) for p in parts]).reshape(3, 3)
        elif len(parts) == 4: # K11 K22 K13 K23 format (fx fy cx cy)
            fx, fy, cx, cy = [float(p) for p in parts]
            intrinsics = np.array([
                [fx, 0, cx],
                [0, fy, cy],
                [0, 0, 1]
            ], dtype=np.float32)
        else:
            raise ValueError(f"Unsupported intrinsics format in {calib_path}: {line}")
        
        return intrinsics
