from __future__ import annotations

from typing import Optional

from .camera_utils import CameraInfo, make_camera_info, register_camera_provider


class Kidney3DDataset:
    """Metadata helper for Kidney3D-CT-depth-seg dataset."""

    @staticmethod
    def get_camera_info(_: Optional[str] = None) -> Optional[CameraInfo]:
        # Provided focal length is 1.0 mm; intrinsics matrix supplied directly.
        return make_camera_info(
            width=1920,
            height=1080,
            fx=960.0,
            fy=960.0,
            cx=960.0,
            cy=540.0,
        )


register_camera_provider("kidney3d", Kidney3DDataset.get_camera_info)
