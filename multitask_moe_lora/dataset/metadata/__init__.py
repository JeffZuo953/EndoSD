"""
Dataset camera metadata helpers.

This package centralizes camera intrinsics for datasets that expose static metadata.
Importing submodules registers their providers with :mod:`dataset.camera_utils`.
"""

# Import order matters to ensure providers register on module import.
from . import c3vd  # noqa: F401
from . import kidney3d  # noqa: F401
from . import endomapper  # noqa: F401
from . import endovis2018  # noqa: F401
from . import endovis2017  # noqa: F401
from . import hamlyn  # noqa: F401
from . import scared  # noqa: F401
from . import dvpn  # noqa: F401
from . import endonerf  # noqa: F401
from . import simcol  # noqa: F401
from . import stereomis  # noqa: F401

__all__ = [
    "c3vd",
    "kidney3d",
    "endomapper",
    "endovis2018",
    "endovis2017",
    "hamlyn",
    "scared",
    "dvpn",
    "endonerf",
    "simcol",
    "stereomis",
]
