import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision.transforms import Compose

from .transform import NormalizeImage, PrepareForNet, Resize
from .utils import compute_valid_mask
from ..util.local_cache import LocalCacheManager


@dataclass
class SampleSpec:
    image_path: str
    depth_path: str
    mask_path: str
    max_depth: float
    depth_scale: float
    min_depth: float
    meta: Dict[str, str]


class FileListSegDepthDataset(Dataset):
    """
    Generic RGB-Depth-Segmentation dataset that is driven by a plain-text file list.

    Each line in the file list must contain at least three whitespace separated fields:
        <image_path> <depth_path> <mask_path> [extra tokens...]

    Extra tokens are interpreted as key=value pairs and can be used to override defaults:
        max_depth=0.1      # Per-sample depth clipping value
        depth_scale=1000   # Divides depth map read from integer formats (e.g., 16-bit PNG)

    Example line:
        /data/.../image/Image0001.png /data/.../depth/Image0001.exr /data/.../mask/Image0001.png max_depth=0.05

    The dataset handles several depth formats automatically based on the file extension:
        .exr   -> OpenEXR V channel (float32)
        .png   -> 16-bit PNG assumed, divided by depth_scale (default 1000.0)
        .npy   -> numpy array stored on disk
        .npz   -> expects an array stored under key 'depth' unless depth_key=... is provided

    Masks are loaded as grayscale images (uint8) unless mask_path ends with .npy/.npz.
    """

    def __init__(
        self,
        filelist_path: str,
        mode: str = "train",
        size: Tuple[int, int] = (518, 518),
        default_max_depth: float = 0.1,
        default_depth_scale: float = 1000.0,
        default_min_depth: float = 0.0,
        dataset_type: str = "unknown",
        dataset_name: Optional[str] = None,
        local_cache_dir: Optional[str] = None,
    ) -> None:
        self.mode: str = mode
        self.size: Tuple[int, int] = size
        self.default_max_depth: float = default_max_depth
        self.default_depth_scale: float = default_depth_scale
        self.default_min_depth: float = default_min_depth
        self.dataset_type: str = dataset_type

        if not os.path.exists(filelist_path):
            raise FileNotFoundError(f"File list not found: {filelist_path}")

        with open(filelist_path, "r", encoding="utf-8") as f:
            raw_lines = [line.strip() for line in f if line.strip() and not line.strip().startswith("#")]

        if not raw_lines:
            raise ValueError(f"No valid entries found in file list: {filelist_path}")

        self.samples: List[SampleSpec] = [self._parse_line(line) for line in raw_lines]
        self.dataset_name: str = dataset_name or self._infer_dataset_name(filelist_path)
        self._local_cache = LocalCacheManager(local_cache_dir, namespace=f"filelist/{self.dataset_name}")
        self._cache_prefix = f"v1|{self.dataset_name}|{self.mode}|{self.size[0]}x{self.size[1]}|{self.default_max_depth}|{self.default_depth_scale}|{self.default_min_depth}|{self.dataset_type}"

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
                downscale_only=True,
            ),
            NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            PrepareForNet(),
        ])

    def _parse_line(self, line: str) -> SampleSpec:
        parts = line.split()
        if len(parts) < 3:
            raise ValueError(f"Expected at least 3 entries per line, got {len(parts)}: {line}")

        image_path, depth_path, mask_path = parts[:3]
        meta_tokens = parts[3:]

        meta: Dict[str, str] = {}
        max_depth = self.default_max_depth
        depth_scale = self.default_depth_scale
        min_depth = self.default_min_depth

        for token in meta_tokens:
            if "=" not in token:
                # Allow single float token to override max_depth for backward compatibility
                try:
                    max_depth = float(token)
                    continue
                except ValueError:
                    raise ValueError(f"Unable to parse token '{token}' in line: {line}")
            key, value = token.split("=", 1)
            key = key.strip().lower()
            value = value.strip()
            meta[key] = value

        if "max_depth" in meta:
            max_depth = float(meta["max_depth"])
        if "depth_scale" in meta:
            depth_scale = float(meta["depth_scale"])
        if "min_depth" in meta:
            min_depth = float(meta["min_depth"])

        return SampleSpec(
            image_path=image_path,
            depth_path=depth_path,
            mask_path=mask_path,
            max_depth=max_depth,
            depth_scale=depth_scale,
            min_depth=min_depth,
            meta=meta,
        )

    def _make_cache_key(self, sample: SampleSpec) -> str:
        return "|".join([
            self._cache_prefix,
            sample.image_path,
            sample.depth_path,
            sample.mask_path,
            f"max={sample.max_depth}",
            f"scale={sample.depth_scale}",
            f"min={sample.min_depth}",
        ])

    def _prepare_cache_payload(self, data: Dict[str, Any]) -> Dict[str, Any]:
        payload: Dict[str, Any] = {}
        for key, value in data.items():
            if torch.is_tensor(value):
                payload[key] = value.detach().cpu()
            else:
                payload[key] = value
        return payload

    def _postprocess_cached_sample(self, data: Dict[str, Any]) -> Dict[str, Any]:
        if "image" in data and torch.is_tensor(data["image"]):
            data["image"] = data["image"].to(torch.float32)
        if "depth" in data and torch.is_tensor(data["depth"]):
            data["depth"] = data["depth"].to(torch.float32)
        if "semseg_mask" in data and torch.is_tensor(data["semseg_mask"]):
            data["semseg_mask"] = data["semseg_mask"].to(torch.long)
        if "valid_mask" in data and torch.is_tensor(data["valid_mask"]):
            data["valid_mask"] = data["valid_mask"].to(torch.bool)
        data.setdefault("dataset_name", self.dataset_name)
        data.setdefault("source_type", data.get("source_type", self.dataset_type))
        return data

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor | str]:
        sample = self.samples[index]

        cache_key: Optional[str] = None
        if self._local_cache.enabled:
            cache_key = self._make_cache_key(sample)
            cached = self._local_cache.load_obj(cache_key)
            if cached is not None:
                return self._postprocess_cached_sample(cached)

        image = self._read_image(sample.image_path)
        depth = self._read_depth(sample.depth_path, sample)
        mask = self._read_mask(sample.mask_path, sample)

        depth = np.clip(depth, 0.0, sample.max_depth)

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
            min_depth=sample.min_depth,
            max_depth=sample.max_depth,
            dataset_name=self.dataset_name,
        )

        result: Dict[str, torch.Tensor | str] = {
            "image": image_tensor,
            "depth": depth_tensor,
            "semseg_mask": mask_tensor,
            "valid_mask": valid_mask,
            "image_path": sample.image_path,
            "depth_path": sample.depth_path,
            "mask_path": sample.mask_path,
            "max_depth": sample.max_depth,
            "source_type": sample.meta.get("source_type", sample.meta.get("dataset_type", self.dataset_type)),
            "dataset_name": self.dataset_name,
        }

        if cache_key and self._local_cache.enabled:
            self._local_cache.save_obj(cache_key, self._prepare_cache_payload(result))
        return result

    def _read_image(self, path: str) -> np.ndarray:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Image not found: {path}")

        image = cv2.imread(path, cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f"Unable to read image: {path}")
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

    def _read_depth(self, path: str, sample: SampleSpec) -> np.ndarray:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Depth file not found: {path}")

        ext = os.path.splitext(path)[1].lower()
        if ext == ".exr":
            return self._read_exr_depth(path)
        if ext in {".png", ".tiff", ".tif"}:
            depth_png = cv2.imread(path, cv2.IMREAD_UNCHANGED)
            if depth_png is None:
                raise FileNotFoundError(f"Unable to read depth map: {path}")
            depth = depth_png.astype(np.float32) / sample.depth_scale
            return depth
        if ext == ".npy":
            arr = np.load(path)
            return arr.astype(np.float32)
        if ext == ".npz":
            depth_key = sample.meta.get("depth_key", "depth")
            with np.load(path) as data:
                if depth_key not in data:
                    raise KeyError(f"Depth key '{depth_key}' not found in npz file: {path}")
                return data[depth_key].astype(np.float32)

        raise ValueError(f"Unsupported depth file format: {path}")

    @staticmethod
    def _read_exr_depth(path: str) -> np.ndarray:
        try:
            import OpenEXR  # type: ignore
            import Imath  # type: ignore
        except ImportError as exc:
            raise ImportError(
                "OpenEXR and Imath are required to read EXR depth files. "
                "Please install them via `pip install openexr Imath`."
            ) from exc

        file = OpenEXR.InputFile(path)
        header = file.header()
        data_window = header["dataWindow"]
        width = data_window.max.x - data_window.min.x + 1
        height = data_window.max.y - data_window.min.y + 1

        channel_name = "V" if "V" in header["channels"] else "R"
        raw = file.channel(channel_name, Imath.PixelType(Imath.PixelType.FLOAT))
        depth = np.frombuffer(raw, dtype=np.float32).reshape((height, width))
        return depth

    def _read_mask(self, path: str, sample: SampleSpec) -> np.ndarray:
        if path.lower() in {"none", "null", "ignore"}:
            # Provide an empty mask when segmentation is not available
            raise FileNotFoundError("Mask path is required but not provided (received placeholder value).")

        if not os.path.exists(path):
            raise FileNotFoundError(f"Mask file not found: {path}")

        ext = os.path.splitext(path)[1].lower()
        if ext in {".npy"}:
            return np.load(path).astype(np.uint8)
        if ext == ".npz":
            mask_key = sample.meta.get("mask_key", "mask")
            with np.load(path) as data:
                if mask_key not in data:
                    raise KeyError(f"Mask key '{mask_key}' not found in npz file: {path}")
                return data[mask_key].astype(np.uint8)

        mask = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if mask is None:
            raise FileNotFoundError(f"Unable to read mask: {path}")

        if mask.ndim == 3:
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        return mask.astype(np.uint8)
    @staticmethod
    def _infer_dataset_name(path: str) -> str:
        norm_path = os.path.normpath(path)
        ignore_tokens = {"cache", "cache_pt", "filelists", "filelist", "txt"}
        directory = os.path.dirname(norm_path)
        while directory:
            base = os.path.basename(directory)
            if base and base.lower() not in ignore_tokens:
                return base
            parent = os.path.dirname(directory)
            if parent == directory:
                break
            directory = parent
        return os.path.splitext(os.path.basename(norm_path))[0] or "dataset"
