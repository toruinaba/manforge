"""Stress-state base classes for 3D / plane-stress / 1D elements.

These classes are thin shims that set the default ``dimension`` and validate
that an appropriate ``StressDimension`` is passed.  Operator implementations
now live in the ``StressDimension`` derived classes; ``MaterialModel`` delegates
to them.  These shims will be removed in Step 2 of the refactor (#77).
"""

from manforge.core.material.base import MaterialModel
from manforge.core.dimension import (
    SOLID_3D,
    PLANE_STRESS,
    UNIAXIAL_1D,
    StressDimension,
)


class MaterialModel3D(MaterialModel):
    """Shim for full-rank stress states (SOLID_3D, PLANE_STRAIN).

    Parameters
    ----------
    dimension : StressDimension, optional
        Must satisfy ``dimension.ndi == dimension.ndi_phys``.
        Defaults to ``SOLID_3D``.

    Raises
    ------
    ValueError
        If ``dimension.ndi != dimension.ndi_phys``.
    """

    def __init__(self, dimension: StressDimension = SOLID_3D):
        if dimension.ndi != dimension.ndi_phys:
            raise ValueError(
                f"MaterialModel3D requires ndi == ndi_phys "
                f"(e.g. SOLID_3D or PLANE_STRAIN). "
                f"Got '{dimension.name}' with ndi={dimension.ndi}, "
                f"ndi_phys={dimension.ndi_phys}."
            )
        super().__init__(dimension=dimension)


class MaterialModelPS(MaterialModel):
    """Shim for plane-stress elements (PLANE_STRESS).

    Parameters
    ----------
    dimension : StressDimension, optional
        Must satisfy ``dimension.is_plane_stress``.
        Defaults to ``PLANE_STRESS``.

    Raises
    ------
    ValueError
        If ``dimension.is_plane_stress`` is ``False``.
    """

    def __init__(self, dimension: StressDimension = PLANE_STRESS):
        if not dimension.is_plane_stress:
            raise ValueError(
                f"MaterialModelPS requires a plane-stress StressDimension "
                f"(is_plane_stress=True). "
                f"Got '{dimension.name}'."
            )
        super().__init__(dimension=dimension)


class MaterialModel1D(MaterialModel):
    """Shim for uniaxial (1D) elements (UNIAXIAL_1D).

    Parameters
    ----------
    dimension : StressDimension, optional
        Must have ``ntens == 1``.  Defaults to ``UNIAXIAL_1D``.

    Raises
    ------
    ValueError
        If ``dimension.ntens != 1``.
    """

    def __init__(self, dimension: StressDimension = UNIAXIAL_1D):
        if dimension.ntens != 1:
            raise ValueError(
                f"MaterialModel1D requires a 1D StressDimension (ntens=1). "
                f"Got '{dimension.name}' with ntens={dimension.ntens}."
            )
        super().__init__(dimension=dimension)
