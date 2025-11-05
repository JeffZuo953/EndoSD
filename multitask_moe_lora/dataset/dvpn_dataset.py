import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision.transforms import Compose

from .camera_utils import CameraInfo, make_camera_info, register_camera_provider
from .transform import NormalizeImage, PrepareForNet, Resize
from .utils import compute_valid_mask


@dataclass(frozen=True)
class _DVPNSample:
    image_path: Path
    depth_path: Path
    sequence: str


class DVPNDataset(Dataset):
    """Loader for the dVPN (daVinci) depth dataset consisting of PNG RGB frames and depth NPY files."""

    def __init__(
        self,
        root_dir: str | Path,
        split: str = "train",
        size: Tuple[int, int] = (518, 518),
        max_depth: float = 0.3,
        min_depth: float = 1e-4,
        camera_index: int = 0,
    ) -> None:
        self.root_dir = Path(root_dir).expanduser()
        self.split = split.lower()
        self.size = size
        self.max_depth = max_depth
        self.min_depth = min_depth
        self.camera_index = camera_index

        if self.split not in {"train", "test"}:
            raise ValueError(f"Unsupported split '{split}'. Expected 'train' or 'test'.")

        self.image_root = self.root_dir / self.split / f"image_{self.camera_index}"
        self.depth_root = self.root_dir / "depth" / self.split / f"image_{self.camera_index}"

        if not self.image_root.is_dir():
            raise FileNotFoundError(f"Image directory not found: {self.image_root}")
        if not self.depth_root.is_dir():
            raise FileNotFoundError(f"Depth directory not found: {self.depth_root}")

        self.samples: List[_DVPNSample] = self._gather_samples()
        if not self.samples:
            raise RuntimeError(f"No RGB/depth pairs found under {self.image_root}.")

        net_w, net_h = size
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

    def _gather_samples(self) -> List[_DVPNSample]:
        samples: List[_DVPNSample] = []
        for image_dir in sorted(self.image_root.parent.glob("image_*")):
            if image_dir.name != f"image_{self.camera_index}":
                continue
            depth_dir = self.depth_root.parent / image_dir.name
            if not depth_dir.is_dir():
                continue
            sequence = image_dir.name
            for image_path in sorted(image_dir.glob("*.png")):
                frame_id = image_path.stem
                depth_path = depth_dir / f"{frame_id}_depth.npy"
                if not depth_path.exists():
                    continue
                samples.append(
                    _DVPNSample(
                        image_path=image_path,
                        depth_path=depth_path,
                        sequence=sequence,
                    )
                )
        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor | str]:
        sample = self.samples[index]

        image_bgr = cv2.imread(str(sample.image_path), cv2.IMREAD_COLOR)
        if image_bgr is None:
            raise FileNotFoundError(f"Unable to read image: {sample.image_path}")
        image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

        depth_np = np.load(sample.depth_path).astype(np.float32)
        depth_np = np.clip(depth_np, 0.0, self.max_depth)

        transformed = self.transform(
            {
                "image": image,
                "depth": depth_np,
            }
        )

        image_tensor = torch.from_numpy(transformed["image"])
        depth_tensor = torch.from_numpy(transformed["depth"])

        valid_mask = compute_valid_mask(
            image_tensor,
            depth_tensor,
            min_depth=self.min_depth,
            max_depth=self.max_depth,
            dataset_name="dVPN",
        )

        result: Dict[str, torch.Tensor | str] = {
            "image": image_tensor,
            "depth": depth_tensor,
            "valid_mask": valid_mask,
            "image_path": str(sample.image_path),
            "depth_path": str(sample.depth_path),
            "max_depth": self.max_depth,
            "source_type": "LS",
            "dataset_name": "dVPN",
        }
        camera_info = self.get_camera_info()
        if camera_info is not None:
            result["camera_intrinsics"] = camera_info.intrinsics.clone()
            result["camera_intrinsics_norm"] = camera_info.intrinsics_norm.clone()
            result["camera_size"] = torch.tensor([camera_info.width, camera_info.height], dtype=torch.float32)
        return result

    @staticmethod
    def get_camera_info(_: Optional[str] = None) -> Optional[CameraInfo]:
        """Return fixed intrinsics for dVPN dataset."""
        return make_camera_info(
            width=384,
            height=192,
            fx=373.47833252,
            fy=373.47833252,
            cx=182.91804504,
            cy=113.72999573,
        )


register_camera_provider("dvpn", DVPNDataset.get_camera_info)
