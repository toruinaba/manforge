"""Material model base classes.

Re-exports MaterialModel and its three stress-state base classes so
that ``from manforge.core.material import MaterialModel3D`` continues to
work after the split into ``base.py`` / ``bases.py``.
"""

from manforge.core.material.base import MaterialModel
from manforge.core.material.bases import (
    MaterialModel3D,
    MaterialModelPS,
    MaterialModel1D,
)

__all__ = [
    "MaterialModel",
    "MaterialModel3D",
    "MaterialModelPS",
    "MaterialModel1D",
]
