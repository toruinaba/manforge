"""StressDimension: dimensionality descriptor for stress/strain analyses."""

import math
from abc import ABC
from dataclasses import dataclass

import autograd.numpy as anp
import numpy as np

from manforge.utils.smooth import smooth_sqrt, smooth_abs


@dataclass(frozen=True)
class StressDimension(ABC):
    """Dimensionality descriptor for a stress analysis.

    Concrete subclasses implement the operator methods (dev, hydrostatic, etc.).
    Direct instantiation of this class is supported for structural validation
    tests, but operators will raise NotImplementedError.
    """

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
                f"StressDimension '{self.name}': ntens={self.ntens} != "
                f"ndi={self.ndi} + nshr={self.nshr}"
            )
        if len(self.mandel_factors) != self.ntens:
            raise ValueError(
                f"StressDimension '{self.name}': len(mandel_factors)="
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

    @property
    def eng_to_phys_strain_factors_np(self) -> np.ndarray:
        """Conversion factors from engineering shear to physical shear, shape (ntens,).

        Divide a strain Voigt vector by these factors to convert from the
        engineering-shear convention (γ12 = 2 ε12, used by drivers, stiffness,
        and the ABAQUS UMAT interface) to physical shear (ε12, used by stress
        and stress-like quantities).  Direct components are unchanged (factor = 1).

        Equivalent to [1, ..., 1 (ndi times), 2, ..., 2 (nshr times)].
        """
        return np.array([1.0] * self.ndi + [2.0] * self.nshr)

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

    # ------------------------------------------------------------------
    # Operator interface — overridden by derived classes
    # ------------------------------------------------------------------

    def hydrostatic(self, stress):
        raise NotImplementedError(f"{type(self).__name__}.hydrostatic is not implemented")

    def dev(self, stress):
        raise NotImplementedError(f"{type(self).__name__}.dev is not implemented")

    def isotropic_C(self, lam, mu):
        raise NotImplementedError(f"{type(self).__name__}.isotropic_C is not implemented")

    def I_vol(self):
        raise NotImplementedError(f"{type(self).__name__}.I_vol is not implemented")

    def I_dev(self):
        raise NotImplementedError(f"{type(self).__name__}.I_dev is not implemented")

    def vonmises_norm(self, s):
        raise NotImplementedError(f"{type(self).__name__}.vonmises_norm is not implemented")

    def missing_dev_components(self, s):
        """Default: zeros(n_missing) for states where all direct components are stored."""
        return anp.zeros(self.n_missing)


# ---------------------------------------------------------------------------
# Derived concrete classes
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class _Solid3DLikeDimension(StressDimension):
    """Full-rank dimension (ndi == ndi_phys): SOLID_3D and PLANE_STRAIN."""

    def __post_init__(self):
        super().__post_init__()
        if self.ndi != self.ndi_phys:
            raise ValueError(
                f"_Solid3DLikeDimension '{self.name}' requires ndi == ndi_phys "
                f"(got ndi={self.ndi}, ndi_phys={self.ndi_phys})."
            )

    def hydrostatic(self, stress):
        return (stress[0] + stress[1] + stress[2]) / 3.0

    def dev(self, stress):
        return stress - self.hydrostatic(stress) * self.identity_np

    def isotropic_C(self, lam, mu):
        delta_6 = anp.array([1.0, 1.0, 1.0, 0.0, 0.0, 0.0])
        scale_6 = anp.array([2.0, 2.0, 2.0, 1.0, 1.0, 1.0])
        C6 = lam * anp.outer(delta_6, delta_6) + mu * anp.diag(scale_6)
        if self.ntens == 6:
            return C6
        idx = anp.array([0, 1, 2, 3])
        return C6[anp.ix_(idx, idx)]

    def I_vol(self):
        delta = self.identity_np
        return anp.outer(delta, delta) / 3.0

    def I_dev(self):
        return anp.eye(self.ntens) - self.I_vol()

    def vonmises_norm(self, s):
        # n_missing == 0, so vonmises = sqrt(3/2 * s_m·s_m). Inline for
        # symmetry with PS/1D (avoids MaterialModel.vonmises round-trip).
        s_m = s * self.mandel_factors_np
        return smooth_sqrt(1.5 * anp.dot(s_m, s_m))


@dataclass(frozen=True)
class _PlaneStressDimension(StressDimension):
    """Plane-stress dimension (PLANE_STRESS): is_plane_stress=True, ntens=3."""

    def __post_init__(self):
        super().__post_init__()
        if not self.is_plane_stress:
            raise ValueError(
                f"_PlaneStressDimension '{self.name}' requires is_plane_stress=True."
            )

    def hydrostatic(self, stress):
        return (stress[0] + stress[1]) / 3.0

    def dev(self, stress):
        return stress - self.hydrostatic(stress) * self.identity_np

    def isotropic_C(self, lam, mu):
        delta_6 = anp.array([1.0, 1.0, 1.0, 0.0, 0.0, 0.0])
        scale_6 = anp.array([2.0, 2.0, 2.0, 1.0, 1.0, 1.0])
        C6 = lam * anp.outer(delta_6, delta_6) + mu * anp.diag(scale_6)
        idx4 = anp.array([0, 1, 2, 3])
        C4 = C6[anp.ix_(idx4, idx4)]
        retain = anp.array([0, 1, 3])
        C_rr = C4[anp.ix_(retain, retain)]
        C_rc = C4[retain, 2]
        C_cc = C4[2, 2]
        return C_rr - anp.outer(C_rc, C_rc) / C_cc

    def I_vol(self):
        delta = self.identity_np
        return anp.outer(delta, delta) / 3.0

    def I_dev(self):
        return anp.eye(self.ntens) - self.I_vol()

    def missing_dev_components(self, s):
        return anp.array([-(s[0] + s[1])])

    def vonmises_norm(self, s):
        s33 = -(s[0] + s[1])
        return smooth_sqrt(1.5 * (s[0]*s[0] + s[1]*s[1] + s33*s33 + 2.0*s[2]*s[2]))


@dataclass(frozen=True)
class _Uniaxial1DDimension(StressDimension):
    """Uniaxial 1D dimension (UNIAXIAL_1D): ntens=1."""

    def __post_init__(self):
        super().__post_init__()
        if self.ntens != 1:
            raise ValueError(
                f"_Uniaxial1DDimension '{self.name}' requires ntens=1 "
                f"(got ntens={self.ntens})."
            )

    def hydrostatic(self, stress):
        return stress[0] / 3.0

    def dev(self, stress):
        return stress - self.hydrostatic(stress) * self.identity_np

    def isotropic_C(self, lam, mu):
        E = mu * (3.0 * lam + 2.0 * mu) / (lam + mu)
        return anp.array([[E]])

    def I_vol(self):
        delta = self.identity_np
        return anp.outer(delta, delta) / 3.0

    def I_dev(self):
        return anp.eye(1) - self.I_vol()

    def missing_dev_components(self, s):
        half = s[0] / 2.0
        return anp.array([-half, -half])

    def vonmises_norm(self, s):
        return smooth_abs(1.5 * s[0])


# ---------------------------------------------------------------------------
# Pre-built instances
# ---------------------------------------------------------------------------

_sqrt2 = math.sqrt(2.0)

SOLID_3D = _Solid3DLikeDimension(
    name="3D_SOLID",
    ntens=6,
    ndi=3,
    nshr=3,
    ndi_phys=3,
    mandel_factors=(1.0, 1.0, 1.0, _sqrt2, _sqrt2, _sqrt2),
)

PLANE_STRAIN = _Solid3DLikeDimension(
    name="PLANE_STRAIN",
    ntens=4,
    ndi=3,
    nshr=1,
    ndi_phys=3,
    mandel_factors=(1.0, 1.0, 1.0, _sqrt2),
)

PLANE_STRESS = _PlaneStressDimension(
    name="PLANE_STRESS",
    ntens=3,
    ndi=2,
    nshr=1,
    ndi_phys=3,
    mandel_factors=(1.0, 1.0, _sqrt2),
    is_plane_stress=True,
)

UNIAXIAL_1D = _Uniaxial1DDimension(
    name="UNIAXIAL_1D",
    ntens=1,
    ndi=1,
    nshr=0,
    ndi_phys=3,
    mandel_factors=(1.0,),
)
