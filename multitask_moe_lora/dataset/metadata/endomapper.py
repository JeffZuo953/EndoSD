from __future__ import annotations

from typing import Optional

from ..camera_utils import CameraInfo, make_camera_info, register_camera_provider


class EndoMapperCamera:
    """Metadata provider for EndoMapper dataset."""

    CAMERA_WIDTH: int = 960
    CAMERA_HEIGHT: int = 720
    CAMERA_INFO: CameraInfo = make_camera_info(
        width=CAMERA_WIDTH,
        height=CAMERA_HEIGHT,
        fx=472.64955100886374,
        fy=472.64955100886374,
        cx=479.5,
        cy=359.5,
    )

    @staticmethod
    def get_camera_info(_: Optional[str] = None) -> Optional[CameraInfo]:
        return EndoMapperCamera.CAMERA_INFO


register_camera_provider("endomapper", EndoMapperCamera.get_camera_info)
