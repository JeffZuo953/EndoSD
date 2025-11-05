from __future__ import annotations

from typing import Optional

from .camera_utils import CameraInfo, make_camera_info, register_camera_provider


class SimColDataset:
    """Metadata helper for SimCol dataset."""

    @staticmethod
    def get_camera_info(_: Optional[str] = None) -> Optional[CameraInfo]:
        return make_camera_info(
            width=475,
            height=475,
            fx=227.60416,
            fy=227.60416,
            cx=237.5,
            cy=237.5,
        )


register_camera_provider("simcol", SimColDataset.get_camera_info)
