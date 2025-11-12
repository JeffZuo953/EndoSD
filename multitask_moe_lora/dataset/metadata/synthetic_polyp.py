from __future__ import annotations

from typing import Optional

from ..camera_utils import CameraInfo, make_camera_info, register_camera_provider


class SyntheticPolypMetadata:
    """Camera metadata for SyntheticDatabase polyp dataset."""

    WIDTH = 640
    HEIGHT = 540
    FX = 448.13
    FY = 448.13
    CX = 320.0
    CY = 270.0

    @classmethod
    def get_camera_info(cls, _: Optional[str] = None) -> Optional[CameraInfo]:
        return make_camera_info(
            width=cls.WIDTH,
            height=cls.HEIGHT,
            fx=cls.FX,
            fy=cls.FY,
            cx=cls.CX,
            cy=cls.CY,
        )


register_camera_provider("synthetic_polyp", SyntheticPolypMetadata.get_camera_info)
register_camera_provider("syntheticdatabase_polyp", SyntheticPolypMetadata.get_camera_info)
register_camera_provider("syntheticpolyp", SyntheticPolypMetadata.get_camera_info)
