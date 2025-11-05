from __future__ import annotations

from typing import Optional

from .camera_utils import CameraInfo, make_camera_info, register_camera_provider


class C3VDDataset:
    """Metadata helper for the original C3VD dataset."""

    @staticmethod
    def get_camera_info(_: Optional[str] = None) -> Optional[CameraInfo]:
        return make_camera_info(
            width=1350,
            height=1080,
            fx=769.243600037458,
            fy=769.243600037458,
            cx=678.544839263292,
            cy=542.975887548343,
        )


class C3VDV2Dataset:
    """Metadata helper for the C3VDv2 dataset."""

    @staticmethod
    def get_camera_info(_: Optional[str] = None) -> Optional[CameraInfo]:
        return make_camera_info(
            width=1350,
            height=1080,
            fx=767.733695862103,
            fy=767.733695862103,
            cx=677.739464094188,
            cy=543.057997844875,
        )


register_camera_provider("c3vd", C3VDDataset.get_camera_info)
register_camera_provider("c3vdv2", C3VDV2Dataset.get_camera_info)
