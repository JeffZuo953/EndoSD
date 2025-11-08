from __future__ import annotations

from typing import Dict, Optional

from ..camera_utils import CameraInfo, make_camera_info, register_camera_provider


class StereoMISMetadata:
    """
    Camera intrinsics for StereoMIS sequences.

    Values are extracted from each ``rectification_params.txt`` under the LS StereoMIS
    dataset root (e.g., ``P1/rectification_params.txt``).
    """

    _SEQUENCE_TABLE: Dict[str, CameraInfo] = {
        "p1": make_camera_info(640, 512, 570.89342998, 570.89342998, 327.3314743, 251.32201576),
        "p2_0": make_camera_info(640, 512, 561.826657, 561.826657, 329.73555374, 257.43183327),
        "p2_1": make_camera_info(640, 512, 561.826657, 561.826657, 329.73555374, 257.43183327),
        "p2_2": make_camera_info(640, 512, 561.826657, 561.826657, 329.73555374, 257.43183327),
        "p2_3": make_camera_info(640, 512, 561.826657, 561.826657, 329.73555374, 257.43183327),
        "p2_4": make_camera_info(640, 512, 561.826657, 561.826657, 329.73555374, 257.43183327),
        "p2_5": make_camera_info(640, 512, 561.826657, 561.826657, 329.73555374, 257.43183327),
        "p2_6": make_camera_info(640, 512, 561.826657, 561.826657, 329.73555374, 257.43183327),
        "p2_7": make_camera_info(640, 512, 561.826657, 561.826657, 329.73555374, 257.43183327),
        "p2_8": make_camera_info(640, 512, 561.826657, 561.826657, 329.73555374, 257.43183327),
        "p3": make_camera_info(640, 512, 568.21502988, 568.21502988, 332.94229889, 244.90832329),
    }
    _DEFAULT_CAMERA: CameraInfo = _SEQUENCE_TABLE["p1"]

    @classmethod
    def get_camera_info(cls, sample_path: Optional[str] = None) -> Optional[CameraInfo]:
        if not cls._SEQUENCE_TABLE:
            return None
        if sample_path:
            lowered = sample_path.replace("\\", "/").lower()
            for key, camera in cls._SEQUENCE_TABLE.items():
                if key in lowered:
                    return camera
        return cls._DEFAULT_CAMERA

    @classmethod
    def get_sequence_table(cls) -> Dict[str, CameraInfo]:
        return cls._SEQUENCE_TABLE


register_camera_provider("stereomis", StereoMISMetadata.get_camera_info)
