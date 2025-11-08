from __future__ import annotations

import re
from typing import Dict, Optional

from ..camera_utils import CameraInfo, make_camera_info, register_camera_provider

ENDOVIS2017_INTRINSICS = [
    ("instrument_dataset_10", 1280, 1024, 1175.1991, 1175.1991, 633.05146, 514.826309),
    ("instrument_dataset_9", 1280, 1024, 1175.1991, 1175.1991, 633.05146, 514.826309),
    ("instrument_dataset_1", 1280, 1024, 1165.64936, 1165.64936, 640.071579, 501.278927),
    ("instrument_dataset_2", 1280, 1024, 1165.64936, 1165.64936, 640.071579, 501.278927),
    ("instrument_dataset_3", 1280, 1024, 1175.1991, 1175.1991, 633.05146, 514.826309),
    ("instrument_dataset_4", 1280, 1024, 1157.99682, 1157.99682, 640.461174, 497.860695),
    ("instrument_dataset_5", 1280, 1024, 1175.1991, 1175.1991, 633.05146, 514.826309),
    ("instrument_dataset_6", 1280, 1024, 1168.09587, 1168.09587, 650.05854, 508.606598),
    ("instrument_dataset_7", 1280, 1024, 1145.14854, 1145.14854, 668.845131, 508.133137),
    ("instrument_dataset_8", 1280, 1024, 1175.1991, 1175.1991, 633.05146, 514.826309),
]


class EndoVis2017Metadata:
    """
    Camera intrinsics registry for the EndoVis2017 dataset.
    """

    _CAMERA_TABLE: Dict[str, CameraInfo] = {}
    _DEFAULT_CAMERA: Optional[CameraInfo] = None

    @classmethod
    def _ensure_loaded(cls) -> None:
        if cls._CAMERA_TABLE:
            return

        table: Dict[str, CameraInfo] = {}
        for seq, width, height, fx, fy, cx, cy in ENDOVIS2017_INTRINSICS:
            camera = make_camera_info(width, height, fx, fy, cx, cy)
            seq_lower = seq.lower()
            table[seq_lower] = camera
            digits = "".join(ch for ch in seq_lower if ch.isdigit())
            if digits:
                table[digits] = camera
                table[digits.lstrip("0") or "0"] = camera

        cls._CAMERA_TABLE = table
        cls._DEFAULT_CAMERA = next(iter(table.values()), None)

    @classmethod
    def _match_sequence(cls, sample_path: str) -> Optional[CameraInfo]:
        lowered = sample_path.replace("\\", "/").lower()
        for key, camera in cls._CAMERA_TABLE.items():
            if key and key in lowered:
                return camera
        # Match tokens such as instrument_dataset_10 explicitly
        match = re.search(r"instrument_dataset_(\d+)", lowered)
        if match:
            key = match.group(0)
            camera = cls._CAMERA_TABLE.get(key)
            if camera is not None:
                return camera
            digits = match.group(1)
            return cls._CAMERA_TABLE.get(digits) or cls._CAMERA_TABLE.get(digits.lstrip("0") or "0")
        return None

    @classmethod
    def get_camera_info(cls, sample_path: Optional[str] = None) -> Optional[CameraInfo]:
        cls._ensure_loaded()
        if not cls._CAMERA_TABLE:
            return None
        if sample_path:
            camera = cls._match_sequence(sample_path)
            if camera is not None:
                return camera
        return cls._DEFAULT_CAMERA


register_camera_provider("endovis2017", EndoVis2017Metadata.get_camera_info)
