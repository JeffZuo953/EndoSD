from __future__ import annotations

import re
from typing import Dict, Optional

from ..camera_utils import CameraInfo, make_camera_info, register_camera_provider


_WIDTH = 1350.0
_HEIGHT = 1080.0


def _make_info(fx: float, fy: float, cx: float, cy: float) -> CameraInfo:
    return make_camera_info(width=_WIDTH, height=_HEIGHT, fx=fx, fy=fy, cx=cx, cy=cy)


class SCAREDMetadata:
    """
    Hard-coded camera metadata for the SCARED dataset.

    Intrinsics were extracted from the official calibration YAML files located
    under ``dataset_XX/keyframe_YY/endoscope_calibration.yaml``. Each entry
    corresponds to the left camera matrix ``M1`` (fx, fy, cx, cy) at resolution
    1350x1080. Values are stored per dataset and keyframe to accommodate the
    small variations across sequences.
    """

    _CALIBRATION_TABLE: Dict[str, Dict[str, CameraInfo]] = {
        "dataset_01": {f"keyframe_{idx:02d}": _make_info(1035.30811, 1035.08765, 596.95502, 520.41003) for idx in range(1, 6)},
        "dataset_02": {f"keyframe_{idx:02d}": _make_info(1035.30811, 1035.08765, 596.95502, 520.41003) for idx in range(1, 6)},
        "dataset_03": {f"keyframe_{idx:02d}": _make_info(1035.30811, 1035.08765, 596.95502, 520.41003) for idx in range(1, 6)},
        "dataset_04": {f"keyframe_{idx:02d}": _make_info(1073.64844, 1073.43433, 577.49969, 524.40979) for idx in range(1, 6)},
        "dataset_05": {f"keyframe_{idx:02d}": _make_info(1073.64844, 1073.43433, 577.49969, 524.40979) for idx in range(1, 6)},
        "dataset_06": {f"keyframe_{idx:02d}": _make_info(1086.97437, 1086.76831, 586.08032, 512.47589) for idx in range(1, 6)},
        "dataset_07": {f"keyframe_{idx:02d}": _make_info(1086.97437, 1086.76831, 586.08032, 512.47589) for idx in range(1, 6)},
        "dataset_08": {f"keyframe_{idx:02d}": _make_info(1024.08777, 1023.89478, 601.80670, 508.13168) for idx in range(0, 5)},
        "dataset_09": {f"keyframe_{idx:02d}": _make_info(1023.42310, 1023.22693, 595.92773, 510.64709) for idx in range(0, 5)},
    }

    _DATASET_DEFAULTS: Dict[str, CameraInfo] = {
        dataset: next(iter(keyframes.values()))
        for dataset, keyframes in _CALIBRATION_TABLE.items()
    }
    _GLOBAL_DEFAULT: Optional[CameraInfo] = next(iter(_DATASET_DEFAULTS.values()), None)

    @classmethod
    def get_camera_info(cls, sample_path: Optional[str] = None) -> Optional[CameraInfo]:
        if sample_path:
            lowered = sample_path.replace("\\", "/").lower()
            dataset_id = None
            keyframe_id = None
            ds_match = None
            for ds in cls._CALIBRATION_TABLE.keys():
                if ds in lowered:
                    ds_match = ds
                    break
            if ds_match:
                dataset_id = ds_match
            key_match = re.search(r"keyfram?e?_(\d+)", lowered)
            if key_match:
                keyframe_id = f"keyframe_{int(key_match.group(1)):02d}"

            if dataset_id:
                dataset_table = cls._CALIBRATION_TABLE.get(dataset_id)
                if dataset_table:
                    if keyframe_id and keyframe_id in dataset_table:
                        return dataset_table[keyframe_id]
                    return cls._DATASET_DEFAULTS.get(dataset_id)

        return cls._GLOBAL_DEFAULT


register_camera_provider("scared", SCAREDMetadata.get_camera_info)
register_camera_provider("scard", SCAREDMetadata.get_camera_info)
