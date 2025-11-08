from __future__ import annotations

from typing import Optional

from ..camera_utils import CameraInfo, make_camera_info, register_camera_provider


class Kidney3DDataset:
    """Metadata helper for Kidney3D-CT-depth-seg dataset."""

    CAMERA_WIDTH: int = 1920
    CAMERA_HEIGHT: int = 1080
    CAMERA_INFO: CameraInfo = make_camera_info(
        width=CAMERA_WIDTH,
        height=CAMERA_HEIGHT,
        fx=960.0,
        fy=960.0,
        cx=960.0,
        cy=540.0,
    )

    @staticmethod
    def get_camera_info(_: Optional[str] = None) -> Optional[CameraInfo]:
        return Kidney3DDataset.CAMERA_INFO


register_camera_provider("kidney3d", Kidney3DDataset.get_camera_info)
