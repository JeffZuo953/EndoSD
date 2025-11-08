from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module
from typing import Callable, Dict, Optional

import torch


@dataclass
class CameraInfo:
    """Container for camera intrinsics and image size."""

    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float
    intrinsics: torch.Tensor
    intrinsics_norm: torch.Tensor


def build_intrinsics_matrix(fx: float, fy: float, cx: float, cy: float) -> torch.Tensor:
    """
    Build a 3x3 intrinsic matrix from focal lengths and principal point.
    """
    matrix = torch.zeros((3, 3), dtype=torch.float32)
    matrix[0, 0] = float(fx)
    matrix[1, 1] = float(fy)
    matrix[0, 2] = float(cx)
    matrix[1, 2] = float(cy)
    matrix[2, 2] = 1.0
    return matrix


def normalize_intrinsics(intrinsics: torch.Tensor, width: float, height: float) -> torch.Tensor:
    """
    Convert intrinsics to normalized focal lengths and principal point coordinates.

    Args:
        intrinsics: 3x3 intrinsic matrix.
        width: image width in pixels.
        height: image height in pixels.

    Returns:
        Tensor [4] = [fx/width, fy/height, cx/width, cy/height]
    """
    if not torch.is_tensor(intrinsics):
        intrinsics = torch.as_tensor(intrinsics, dtype=torch.float32)

    width = max(float(width), 1e-6)
    height = max(float(height), 1e-6)
    device = intrinsics.device
    norm = torch.tensor(
        [
            intrinsics[0, 0] / width,
            intrinsics[1, 1] / height,
            intrinsics[0, 2] / width,
            intrinsics[1, 2] / height,
        ],
        dtype=torch.float32,
        device=device,
    )
    return norm


def make_camera_info(width: int, height: int, fx: float, fy: float, cx: float, cy: float) -> CameraInfo:
    """Helper that constructs CameraInfo with normalized intrinsics."""
    intrinsics = build_intrinsics_matrix(fx, fy, cx, cy)
    intrinsics_norm = normalize_intrinsics(intrinsics, width, height)
    return CameraInfo(
        width=int(width),
        height=int(height),
        fx=float(fx),
        fy=float(fy),
        cx=float(cx),
        cy=float(cy),
        intrinsics=intrinsics,
        intrinsics_norm=intrinsics_norm,
    )


CameraProvider = Callable[[Optional[str]], Optional[CameraInfo]]
_CAMERA_INFO_REGISTRY: Dict[str, CameraProvider] = {}
_AUTO_IMPORT_MODULES: Dict[str, str] = {
    "scared": "dataset.metadata.scared",
    "scard": "dataset.metadata.scared",
    "stereomis": "dataset.metadata.stereomis",
    "endovis2017": "dataset.metadata.endovis2017",
    "endovis2018": "dataset.metadata.endovis2018",
    "endonerf": "dataset.metadata.endonerf",
    "dvpn": "dataset.metadata.dvpn",
    "simcol": "dataset.metadata.simcol",
    "c3vd": "dataset.metadata.c3vd",
    "c3vdv2": "dataset.metadata.c3vd",
    "kidney3d": "dataset.metadata.kidney3d",
    "endomapper": "dataset.metadata.endomapper",
    "hamlyn": "dataset.metadata.hamlyn",
}


def register_camera_provider(name: str, provider: CameraProvider) -> None:
    """Register a camera metadata provider for a dataset."""
    if not name:
        return
    _CAMERA_INFO_REGISTRY[name.lower()] = provider


def get_camera_info(dataset_name: str, sample_path: Optional[str] = None) -> Optional[CameraInfo]:
    """Retrieve camera information for the given dataset/sample."""
    if not dataset_name:
        return None
    key = dataset_name.lower()
    provider = _CAMERA_INFO_REGISTRY.get(key)
    if provider is None:
        module_name = _AUTO_IMPORT_MODULES.get(key)
        if module_name:
            try:
                import_module(module_name)
            except ModuleNotFoundError:
                pass
            provider = _CAMERA_INFO_REGISTRY.get(key)
    if provider is None:
        return None
    return provider(sample_path)


__all__ = [
    "CameraInfo",
    "CameraProvider",
    "build_intrinsics_matrix",
    "normalize_intrinsics",
    "make_camera_info",
    "register_camera_provider",
    "get_camera_info",
]
