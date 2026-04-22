"""StressState: dimensionality descriptor for stress/strain analyses."""

import math
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class StressState:
    """Dimensionality descriptor for a stress analysis."""

    name: str
    ntens: int
    ndi: int
    nshr: int
    ndi_phys: int
    mandel_factors: tuple
    is_plane_stress: bool = False

    def __post_init__(self):
        if self.ntens != self.ndi + self.nshr:
            raise ValueError(
                f"StressState '{self.name}': ntens={self.ntens} != "
                f"ndi={self.ndi} + nshr={self.nshr}"
            )
        if len(self.mandel_factors) != self.ntens:
            raise ValueError(
                f"StressState '{self.name}': len(mandel_factors)="
                f"{len(self.mandel_factors)} != ntens={self.ntens}"
            )

    @property
    def n_missing(self) -> int:
        """Number of unstored direct stress components (ndi_phys − ndi)."""
        return self.ndi_phys - self.ndi

    @property
    def mandel_factors_np(self) -> np.ndarray:
        """Mandel scaling factors as a numpy array, shape (ntens,)."""
        return np.array(self.mandel_factors)

    # backward-compat alias
    @property
    def mandel_factors_jnp(self) -> np.ndarray:
        return self.mandel_factors_np

    @property
    def identity_np(self) -> np.ndarray:
        """Voigt identity vector [1,...,1, 0,...,0], shape (ntens,)."""
        return np.array([1.0] * self.ndi + [0.0] * self.nshr)

    # backward-compat alias
    @property
    def identity_jnp(self) -> np.ndarray:
        return self.identity_np


# ---------------------------------------------------------------------------
# Pre-built instances
# ---------------------------------------------------------------------------

_sqrt2 = math.sqrt(2.0)

SOLID_3D = StressState(
    name="3D_SOLID",
    ntens=6,
    ndi=3,
    nshr=3,
    ndi_phys=3,
    mandel_factors=(1.0, 1.0, 1.0, _sqrt2, _sqrt2, _sqrt2),
)

PLANE_STRAIN = StressState(
    name="PLANE_STRAIN",
    ntens=4,
    ndi=3,
    nshr=1,
    ndi_phys=3,
    mandel_factors=(1.0, 1.0, 1.0, _sqrt2),
)

PLANE_STRESS = StressState(
    name="PLANE_STRESS",
    ntens=3,
    ndi=2,
    nshr=1,
    ndi_phys=3,
    mandel_factors=(1.0, 1.0, _sqrt2),
    is_plane_stress=True,
)

UNIAXIAL_1D = StressState(
    name="UNIAXIAL_1D",
    ntens=1,
    ndi=1,
    nshr=0,
    ndi_phys=3,
    mandel_factors=(1.0,),
)
