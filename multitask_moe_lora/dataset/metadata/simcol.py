from __future__ import annotations

from typing import Optional

from ..camera_utils import CameraInfo, make_camera_info, register_camera_provider


class SimColMetadata:
    """Static intrinsics for the SimCol dataset."""

    WIDTH: int = 475
    HEIGHT: int = 475
    CAMERA_INFO: CameraInfo = make_camera_info(
        width=WIDTH,
        height=HEIGHT,
        fx=227.60416,
        fy=227.60416,
        cx=237.5,
        cy=237.5,
    )

    @staticmethod
    def get_camera_info(_: Optional[str] = None) -> Optional[CameraInfo]:
        return SimColMetadata.CAMERA_INFO


register_camera_provider("simcol", SimColMetadata.get_camera_info)
