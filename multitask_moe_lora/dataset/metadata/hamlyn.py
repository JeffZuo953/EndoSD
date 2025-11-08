from __future__ import annotations

import csv
import re
from typing import Dict, Optional

from ..camera_utils import CameraInfo, make_camera_info, register_camera_provider


HAMLYN_INTRINSICS_RAW = [
    ("rectified01", 640, 480, 383.1901395, 383.1901395, 155.9659519195557, 124.3335933685303),
    ("rectified04", 360, 288, 579.05693, 579.05693, 139.9316005706787, 159.0189905166626),
    ("rectified05", 360, 288, 579.05693, 579.05693, 139.9316005706787, 159.0189905166626),
    ("rectified06", 640, 480, 765.8236885, 765.8236885, 276.4727783203125, 253.6752815246582),
    ("rectified08", 640, 480, 765.8236885, 765.8236885, 276.4727783203125, 253.6752815246582),
    ("rectified09", 640, 480, 765.8236885, 765.8236885, 276.4727783203125, 253.6752815246582),
    ("rectified11", 360, 288, 426.532013, 426.532013, 175.2081146240234, 153.1618118286133),
    ("rectified12", 360, 288, 426.532013, 426.532013, 175.2081146240234, 153.1618118286133),
    ("rectified14", 720, 288, 417.9036255, 417.9036255, 373.208288192749, 158.1358108520508),
    ("rectified15", 720, 288, 417.9036255, 417.9036255, 373.208288192749, 158.1358108520508),
    ("rectified16", 720, 288, 417.9036255, 417.9036255, 373.208288192749, 158.1358108520508),
    ("rectified17", 720, 288, 417.9036255, 417.9036255, 373.208288192749, 158.1358108520508),
    ("rectified18", 720, 288, 417.9036255, 417.9036255, 373.208288192749, 158.1358108520508),
    ("rectified19", 720, 288, 417.9036255, 417.9036255, 373.208288192749, 158.1358108520508),
    ("rectified20", 720, 288, 417.9036255, 417.9036255, 373.208288192749, 158.1358108520508),
    ("rectified21", 720, 288, 417.9036255, 417.9036255, 373.208288192749, 158.1358108520508),
    ("rectified22", 720, 288, 417.9036255, 417.9036255, 373.208288192749, 158.1358108520508),
    ("rectified23", 720, 288, 417.9036255, 417.9036255, 373.208288192749, 158.1358108520508),
    ("rectified24", 720, 288, 417.9036255, 417.9036255, 373.208288192749, 158.1358108520508),
    ("rectified25", 720, 288, 455.510315, 455.510315, 238.6288242340088, 124.0845642089844),
    ("rectified26", 720, 288, 455.510315, 455.510315, 238.6288242340088, 124.0845642089844),
    ("rectified27", 720, 288, 455.510315, 455.510315, 238.6288242340088, 124.0845642089844),
]


class HamlynMetadata:
    """Camera intrinsics registry for the Hamlyn dataset."""

    _CAMERA_TABLE: Dict[str, CameraInfo] = {}
    _DEFAULT_CAMERA: Optional[CameraInfo] = None

    @classmethod
    def _ensure_loaded(cls) -> None:
        if cls._CAMERA_TABLE:
            return

        table: Dict[str, CameraInfo] = {}
        for seq_raw, width, height, fx, fy, cx, cy in HAMLYN_INTRINSICS_RAW:
            camera = make_camera_info(width, height, fx, fy, cx, cy)
            normalized = seq_raw.lower()
            candidates = {normalized}
            if normalized.startswith("rectified"):
                suffix = normalized[len("rectified") :].lstrip("_-")
                if suffix:
                    candidates.add(f"rectified{suffix}")
                    candidates.add(suffix)
            digits = "".join(ch for ch in normalized if ch.isdigit())
            if digits:
                candidates.add(digits)
                candidates.add(digits.zfill(2))
                candidates.add(digits.lstrip("0") or "0")
                candidates.add(f"rectified{digits}")
                candidates.add(f"rectified{digits.zfill(2)}")
            for key in filter(None, candidates):
                table[key] = camera

        cls._CAMERA_TABLE = table
        cls._DEFAULT_CAMERA = next(iter(table.values()), None)

    @classmethod
    def get_camera_info(cls, sample_path: Optional[str] = None) -> Optional[CameraInfo]:
        cls._ensure_loaded()
        if not cls._CAMERA_TABLE:
            return None

        if sample_path:
            lowered = sample_path.replace("\\", "/").lower()
            match = re.search(r"rectified[_-]?(\d+)", lowered)
            if match:
                digits = match.group(1)
                variants = [
                    f"rectified{digits}",
                    f"rectified{digits.zfill(2)}",
                    digits,
                    digits.zfill(2),
                    digits.lstrip("0") or "0",
                ]
                for variant in variants:
                    if variant in cls._CAMERA_TABLE:
                        return cls._CAMERA_TABLE[variant]
            for key in cls._CAMERA_TABLE:
                if key and key in lowered:
                    return cls._CAMERA_TABLE[key]

        return cls._DEFAULT_CAMERA


register_camera_provider("hamlyn", HamlynMetadata.get_camera_info)
