from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, Optional

from ..camera_utils import CameraInfo, make_camera_info, register_camera_provider


def _load_legacy_table() -> Dict[str, CameraInfo]:
    """
    Fallback intrinsics sourced from the original camera_calibration.txt files.
    """
    legacy_values = {
        "1": make_camera_info(1280, 1024, 1084.21, 1084.05, 580.02, 506.79),
        "2": make_camera_info(1280, 1024, 1084.21, 1084.05, 580.02, 506.79),
        "3": make_camera_info(1280, 1024, 1084.21, 1084.05, 580.02, 506.79),
        "4": make_camera_info(1280, 1024, 1084.21, 1084.05, 580.02, 506.79),
        "5": make_camera_info(1280, 1024, 1056.33, 1056.11, 594.58, 496.99),
        "6": make_camera_info(1280, 1024, 1084.21, 1084.05, 580.02, 506.79),
        "7": make_camera_info(1280, 1024, 1084.21, 1084.05, 580.02, 506.79),
        "8": make_camera_info(1280, 1024, 1084.21, 1084.05, 580.02, 506.79),
        "9": make_camera_info(1280, 1024, 1051.67, 1051.45, 585.13, 535.54),
        "10": make_camera_info(1280, 1024, 1051.67, 1051.45, 585.13, 535.54),
        "11": make_camera_info(1280, 1024, 1051.67, 1051.45, 585.13, 535.54),
        "12": make_camera_info(1280, 1024, 1051.67, 1051.45, 585.13, 535.54),
        "13": make_camera_info(1280, 1024, 1081.85, 1081.64, 600.38, 508.57),
        "14": make_camera_info(1280, 1024, 1068.39, 1068.19, 600.90, 500.74),
        "15": make_camera_info(1280, 1024, 1081.85, 1081.64, 600.38, 508.57),
        "16": make_camera_info(1280, 1024, 1081.85, 1081.64, 600.38, 508.57),
    }
    table: Dict[str, CameraInfo] = {}
    for key, info in legacy_values.items():
        table[key] = info
        if key.isdigit():
            table[key.zfill(2)] = info
    return table


class EndoVis2018Metadata:
    """
    Camera intrinsics registry for EndoVis2018 sequences.

    Intrinsics will be loaded from ``dataset_metadata/endovis2018_intrinsics.csv`` if
    present (generated via ``tools/extract_endovis2018_intrinsics.py``). When that file
    is unavailable, the metadata falls back to the legacy values embedded previously.
    """

    _CAMERA_TABLE: Dict[str, CameraInfo] = {}
    _DEFAULT_CAMERA: Optional[CameraInfo] = None
    _INTRINSICS_CSV = Path(__file__).resolve().parents[2] / "dataset_metadata" / "endovis2018_intrinsics.csv"

    @classmethod
    def _ensure_loaded(cls) -> None:
        if cls._CAMERA_TABLE:
            return

        table: Dict[str, CameraInfo] = {}
        if cls._INTRINSICS_CSV.exists():
            try:
                with cls._INTRINSICS_CSV.open("r", encoding="utf-8") as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        seq_raw = row.get("sequence")
                        if not seq_raw:
                            continue
                        width = int(float(row["width"]))
                        height = int(float(row["height"]))
                        fx = float(row["fx"])
                        fy = float(row["fy"])
                        cx = float(row["cx"])
                        cy = float(row["cy"])
                        camera = make_camera_info(width, height, fx, fy, cx, cy)

                        candidates = {seq_raw}
                        if seq_raw.isdigit():
                            candidates.add(seq_raw.lstrip("0") or "0")
                            candidates.add(seq_raw.zfill(2))
                        for key in candidates:
                            table[key] = camera
            except OSError:
                table = {}

        if not table:
            table = _load_legacy_table()

        cls._CAMERA_TABLE = table
        cls._DEFAULT_CAMERA = next(iter(table.values()), None)

    @classmethod
    def get_camera_info(cls, sample_path: Optional[str] = None) -> Optional[CameraInfo]:
        cls._ensure_loaded()
        if sample_path:
            lowered = sample_path.replace("\\", "/").lower()
            import re

            match = re.search(r"seq[_/-]?(\d+)", lowered)
            if match:
                seq_digits = match.group(1)
                seq_id = seq_digits.lstrip("0") or "0"
                if seq_id in cls._CAMERA_TABLE:
                    return cls._CAMERA_TABLE[seq_id]
                if seq_digits in cls._CAMERA_TABLE:
                    return cls._CAMERA_TABLE[seq_digits]
        return cls._DEFAULT_CAMERA


register_camera_provider("endovis2018", EndoVis2018Metadata.get_camera_info)
