"""Stress-state base classes for 3D / plane-stress / 1D elements."""

import autograd.numpy as anp

from manforge.core.material.base import MaterialModel
from manforge.core.stress_state import (
    SOLID_3D,
    PLANE_STRESS,
    UNIAXIAL_1D,
    StressState,
)


class MaterialModel3D(MaterialModel):
    """Stress-state base class for full-rank stress states (ndi == ndi_phys).

    Valid for ``SOLID_3D`` (ntens=6) and ``PLANE_STRAIN`` (ntens=4), and any
    future stress state that stores all direct components (e.g. axisymmetric).
    For these states the three physical direct stresses σ11, σ22, σ33 are all
    stored explicitly, so deviatoric and von Mises operators need no
    missing-component corrections.

    Provides concrete implementations of the operator methods used by concrete
    material models:

    * :meth:`_hydrostatic` — p = (σ11 + σ22 + σ33) / 3
    * :meth:`_dev`         — s = σ − p δ
    * :meth:`isotropic_C`  — submatrix extraction from the full 6×6 tensor
    * :meth:`_I_vol`       — δ⊗δ / 3
    * :meth:`_I_dev`       — I − P_vol

    :meth:`_vonmises` is inherited from :class:`MaterialModel` (uses
    ``smooth_sqrt`` with n_missing=0, equivalent to √(3/2 s:s)).

    Subclasses still must implement the required material methods:
    :meth:`elastic_stiffness`, :meth:`yield_function`, and either
    :meth:`update_state` (reduced) or :meth:`state_residual` (augmented).

    Parameters
    ----------
    stress_state : StressState, optional
        Must satisfy ``stress_state.ndi == stress_state.ndi_phys``.
        Defaults to ``SOLID_3D``.

    Raises
    ------
    ValueError
        If ``stress_state.ndi != stress_state.ndi_phys``.
    """

    def __init__(self, stress_state: StressState = SOLID_3D):
        if stress_state.ndi != stress_state.ndi_phys:
            raise ValueError(
                f"MaterialModel3D requires ndi == ndi_phys "
                f"(e.g. SOLID_3D or PLANE_STRAIN). "
                f"Got '{stress_state.name}' with ndi={stress_state.ndi}, "
                f"ndi_phys={stress_state.ndi_phys}."
            )
        self.stress_state = stress_state

    # ------------------------------------------------------------------
    # Operator methods — concrete for full-rank stress states
    # ------------------------------------------------------------------

    def _hydrostatic(self, stress: anp.ndarray) -> anp.ndarray:
        """Mean normal stress p = (σ11 + σ22 + σ33) / 3.

        All three direct components are stored, so no correction is needed.
        """
        return (stress[0] + stress[1] + stress[2]) / 3.0

    def _dev(self, stress: anp.ndarray) -> anp.ndarray:
        """Deviatoric stress s = σ − p δ."""
        p = self._hydrostatic(stress)
        return stress - p * self.stress_state.identity_np

    def isotropic_C(self, lam: float, mu: float) -> anp.ndarray:
        """Isotropic elastic stiffness via submatrix extraction.

        Builds the full 6×6 tensor, then extracts the ntens×ntens subblock
        for components [11, 22, 33, 12, ...].  No condensation is required
        because all direct stress components are free.

        Parameters
        ----------
        lam : float
            First Lamé constant λ.
        mu : float
            Shear modulus μ.

        Returns
        -------
        anp.ndarray, shape (ntens, ntens)
        """
        delta_6 = anp.array([1.0, 1.0, 1.0, 0.0, 0.0, 0.0])
        scale_6 = anp.array([2.0, 2.0, 2.0, 1.0, 1.0, 1.0])
        C6 = lam * anp.outer(delta_6, delta_6) + mu * anp.diag(scale_6)
        if self.ntens == 6:
            return C6
        # PLANE_STRAIN / AXISYMMETRIC: components [11, 22, 33, 12]
        idx = anp.array([0, 1, 2, 3])
        return C6[anp.ix_(idx, idx)]

    def _I_vol(self) -> anp.ndarray:
        """Volumetric projection tensor P_vol = δ⊗δ / 3."""
        delta = self.stress_state.identity_np
        return anp.outer(delta, delta) / 3.0

    def _I_dev(self) -> anp.ndarray:
        """Deviatoric projection tensor P_dev = I − P_vol."""
        return anp.eye(self.ntens) - self._I_vol()


class MaterialModelPS(MaterialModel):
    """Stress-state base class for plane-stress elements (PLANE_STRESS).

    For plane stress, σ33 = 0 is enforced by static condensation of the
    elastic stiffness.  Only two direct components (σ11, σ22) are stored,
    so the von Mises computation must account for the physically-present
    but unstored deviatoric contribution of σ33 = 0.

    Provides concrete implementations of the operator methods:

    * :meth:`_hydrostatic` — p = (σ11 + σ22) / 3  (σ33 = 0)
    * :meth:`_dev`         — s = σ − p δ  (stored components only)
    * :meth:`isotropic_C`  — Schur complement (static condensation of σ33)
    * :meth:`_I_vol`       — δ⊗δ / 3  (δ = [1, 1, 0] for ntens=3)
    * :meth:`_I_dev`       — I − P_vol

    :meth:`_vonmises` is inherited from :class:`MaterialModel` (uses
    ``smooth_sqrt`` with n_missing=1, giving √(3/2 (s:s + p²))).

    Parameters
    ----------
    stress_state : StressState, optional
        Must satisfy ``stress_state.is_plane_stress``.
        Defaults to ``PLANE_STRESS``.

    Raises
    ------
    ValueError
        If ``stress_state.is_plane_stress`` is ``False``.
    """

    def __init__(self, stress_state: StressState = PLANE_STRESS):
        if not stress_state.is_plane_stress:
            raise ValueError(
                f"MaterialModelPS requires a plane-stress StressState "
                f"(is_plane_stress=True). "
                f"Got '{stress_state.name}'."
            )
        self.stress_state = stress_state

    # ------------------------------------------------------------------
    # Operator methods — concrete for PLANE_STRESS
    # ------------------------------------------------------------------

    def _hydrostatic(self, stress: anp.ndarray) -> anp.ndarray:
        """Mean normal stress p = (σ11 + σ22) / 3.

        σ33 = 0 is enforced externally; ndi_phys = 3 so we divide by 3.
        """
        return (stress[0] + stress[1]) / 3.0

    def _dev(self, stress: anp.ndarray) -> anp.ndarray:
        """Deviatoric stress of the stored components, s = σ − p δ."""
        p = self._hydrostatic(stress)
        return stress - p * self.stress_state.identity_np  # δ = [1, 1, 0]

    def isotropic_C(self, lam: float, mu: float) -> anp.ndarray:
        """Plane-stress isotropic stiffness via static condensation.

        Starts from the 4×4 plane-strain submatrix and applies the Schur
        complement to enforce σ33 = 0, yielding a 3×3 matrix for
        components [11, 22, 12].

        Parameters
        ----------
        lam : float
        mu : float

        Returns
        -------
        anp.ndarray, shape (3, 3)
        """
        delta_6 = anp.array([1.0, 1.0, 1.0, 0.0, 0.0, 0.0])
        scale_6 = anp.array([2.0, 2.0, 2.0, 1.0, 1.0, 1.0])
        C6 = lam * anp.outer(delta_6, delta_6) + mu * anp.diag(scale_6)
        # 4×4 plane-strain sub-block
        idx4 = anp.array([0, 1, 2, 3])
        C4 = C6[anp.ix_(idx4, idx4)]
        # Schur complement: eliminate σ33 (index 2 of C4)
        retain = anp.array([0, 1, 3])
        C_rr = C4[anp.ix_(retain, retain)]
        C_rc = C4[retain, 2]
        C_cc = C4[2, 2]
        return C_rr - anp.outer(C_rc, C_rc) / C_cc

    def _I_vol(self) -> anp.ndarray:
        """Volumetric projection tensor P_vol = δ⊗δ / 3."""
        delta = self.stress_state.identity_np  # [1, 1, 0]
        return anp.outer(delta, delta) / 3.0

    def _I_dev(self) -> anp.ndarray:
        """Deviatoric projection tensor P_dev = I − P_vol."""
        return anp.eye(self.ntens) - self._I_vol()


class MaterialModel1D(MaterialModel):
    """Stress-state base class for uniaxial (1D) elements (UNIAXIAL_1D).

    Only σ11 is stored; σ22 = σ33 = 0 are enforced by the element
    formulation.  The von Mises computation must account for two missing
    deviatoric components (n_missing = 2).

    Provides concrete implementations of the operator methods:

    * :meth:`_hydrostatic` — p = σ11 / 3  (σ22 = σ33 = 0)
    * :meth:`_dev`         — s = σ − p δ  (stored component only)
    * :meth:`isotropic_C`  — [[E]] where E = μ(3λ + 2μ) / (λ + μ)
    * :meth:`_I_vol`       — [[1/3]]
    * :meth:`_I_dev`       — [[2/3]]

    :meth:`_vonmises` is inherited from :class:`MaterialModel` (uses
    ``smooth_sqrt`` with n_missing=2, giving √(3/2 (s11² + 2p²)) ≈ |σ11|).

    Parameters
    ----------
    stress_state : StressState, optional
        Must have ``ntens == 1``.  Defaults to ``UNIAXIAL_1D``.

    Raises
    ------
    ValueError
        If ``stress_state.ntens != 1``.
    """

    def __init__(self, stress_state: StressState = UNIAXIAL_1D):
        if stress_state.ntens != 1:
            raise ValueError(
                f"MaterialModel1D requires a 1D StressState (ntens=1). "
                f"Got '{stress_state.name}' with ntens={stress_state.ntens}."
            )
        self.stress_state = stress_state

    # ------------------------------------------------------------------
    # Operator methods — concrete for UNIAXIAL_1D
    # ------------------------------------------------------------------

    def _hydrostatic(self, stress: anp.ndarray) -> anp.ndarray:
        """Mean normal stress p = σ11 / 3.

        σ22 = σ33 = 0 are enforced externally; ndi_phys = 3 so we divide by 3.
        """
        return stress[0] / 3.0

    def _dev(self, stress: anp.ndarray) -> anp.ndarray:
        """Deviatoric stress of the stored component, s = σ − p δ."""
        p = self._hydrostatic(stress)
        return stress - p * self.stress_state.identity_np  # δ = [1.0]

    def isotropic_C(self, lam: float, mu: float) -> anp.ndarray:
        """1D elastic stiffness [[E]] where E = μ(3λ + 2μ) / (λ + μ).

        Parameters
        ----------
        lam : float
        mu : float

        Returns
        -------
        anp.ndarray, shape (1, 1)
        """
        E = mu * (3.0 * lam + 2.0 * mu) / (lam + mu)
        return anp.array([[E]])

    def _I_vol(self) -> anp.ndarray:
        """Volumetric projection tensor [[1/3]] for ntens=1."""
        delta = self.stress_state.identity_np  # [1.0]
        return anp.outer(delta, delta) / 3.0

    def _I_dev(self) -> anp.ndarray:
        """Deviatoric projection tensor [[2/3]] for ntens=1."""
        return anp.eye(1) - self._I_vol()
