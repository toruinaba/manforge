"""StressState: dimensionality descriptor for stress/strain analyses.

Encapsulates the ABAQUS NTENS convention so that operators and models can
be written once and work for any element type.

ABAQUS conventions
------------------
Element type        NDI  NSHR  NTENS  Components
3D solid (C3D8)       3     3      6  [11,22,33,12,13,23]
Plane strain (CPE4)   3     1      4  [11,22,33,12]
Axisymmetric (CAX4)   3     1      4  [11,22,33,12]
Plane stress (CPS4)   2     1      3  [11,22,12]  (sigma_33=0 enforced)
1D truss/bar          1     0      1  [11]
"""

import math
from dataclasses import dataclass, field

import jax.numpy as jnp


@dataclass(frozen=True)
class StressState:
    """Dimensionality descriptor for a stress analysis.

    Parameters
    ----------
    name : str
        Human-readable label (e.g. "3D_SOLID").
    ntens : int
        Total number of stored stress/strain components (= ndi + nshr).
    ndi : int
        Number of stored direct (normal) stress components.
    nshr : int
        Number of stored shear stress components.
    ndi_phys : int
        Physical number of spatial dimensions (3 for 3D/plane/axisymmetric,
        1 for 1D truss).  Used when computing hydrostatic pressure so that
        the divisor is always the physical dimensionality, not just the
        number of stored direct components (relevant for plane stress where
        ndi=2 but sigma_33=0 contributes to the mean, so we divide by 3).
    mandel_factors : tuple[float, ...]
        Length ``ntens``.  Multiply a Voigt vector by these to obtain the
        Mandel representation (shear entries scaled by sqrt(2)).
    is_plane_stress : bool
        True only for plane-stress states.  For J2 (von Mises) plasticity
        with isotropic hardening, the reduced-space radial return preserves
        sigma_33=0, so no outer iteration is required.  For pressure-dependent
        models (e.g. Drucker-Prager, Gurson) a ``PlaneStressWrapper`` that
        iterates on eps_33 would be necessary; such a wrapper is not yet
        implemented.
    """

    name: str
    ntens: int
    ndi: int
    nshr: int
    ndi_phys: int
    mandel_factors: tuple
    is_plane_stress: bool = False

    # ------------------------------------------------------------------
    # Cached JAX arrays (computed lazily on first access)
    # ------------------------------------------------------------------

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
    def mandel_factors_jnp(self) -> jnp.ndarray:
        """Mandel scaling factors as a JAX array, shape (ntens,)."""
        return jnp.array(self.mandel_factors)

    @property
    def identity_jnp(self) -> jnp.ndarray:
        """Voigt identity vector [1,...,1, 0,...,0], shape (ntens,)."""
        return jnp.array([1.0] * self.ndi + [0.0] * self.nshr)


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
