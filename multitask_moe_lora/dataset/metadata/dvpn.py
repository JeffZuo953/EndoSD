from __future__ import annotations

from typing import Optional

from ..camera_utils import CameraInfo, make_camera_info, register_camera_provider


class DVPNMetadata:
    """Camera intrinsics for the dVPN (daVinci) dataset."""

    WIDTH: int = 384
    HEIGHT: int = 192
    CAMERA_INFO: CameraInfo = make_camera_info(
        width=WIDTH,
        height=HEIGHT,
        fx=373.47833252,
        fy=373.47833252,
        cx=182.91804504,
        cy=113.72999573,
    )

    @staticmethod
    def get_camera_info(_: Optional[str] = None) -> Optional[CameraInfo]:
        return DVPNMetadata.CAMERA_INFO


register_camera_provider("dvpn", DVPNMetadata.get_camera_info)
