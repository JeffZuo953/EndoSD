from __future__ import annotations

import csv
import re
from pathlib import Path
from typing import Dict, Optional

from ..camera_utils import CameraInfo, make_camera_info, register_camera_provider


class HamlynMetadata:
    """Camera intrinsics registry for the Hamlyn dataset."""

    _CAMERA_TABLE: Dict[str, CameraInfo] = {}
    _DEFAULT_CAMERA: Optional[CameraInfo] = None
    _INTRINSICS_CSV = Path(__file__).resolve().parents[2] / "dataset_metadata" / "hamlyn_intrinsics.csv"

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
                        seq_raw = (row.get("sequence") or "").strip()
                        if not seq_raw:
                            continue
                        width = int(float(row["width"]))
                        height = int(float(row["height"]))
                        fx = float(row["fx"])
                        fy = float(row["fy"])
                        cx = float(row["cx"])
                        cy = float(row["cy"])
                        camera = make_camera_info(width, height, fx, fy, cx, cy)

                        normalized = seq_raw.lower()
                        candidates = {
                            normalized,
                        }
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
                        for key in {cand for cand in candidates if cand}:
                            table[key] = camera
            except OSError:
                table = {}

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
