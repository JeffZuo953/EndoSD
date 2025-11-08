from __future__ import annotations

from typing import Optional

from ..camera_utils import CameraInfo, make_camera_info, register_camera_provider


class EndoNeRFCamera:
    """Metadata provider for EndoNeRF dataset."""

    WIDTH: int = 640
    HEIGHT: int = 512
    CAMERA_INFO: CameraInfo = make_camera_info(
        width=WIDTH,
        height=HEIGHT,
        fx=569.46820041,
        fy=569.46820041,
        cx=320.0,
        cy=256.0,
    )

    @staticmethod
    def get_camera_info(_: Optional[str] = None) -> Optional[CameraInfo]:
        return EndoNeRFCamera.CAMERA_INFO


register_camera_provider("endonerf", EndoNeRFCamera.get_camera_info)
