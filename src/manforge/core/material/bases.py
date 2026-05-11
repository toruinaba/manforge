"""Stress-state base classes for 3D / plane-stress / 1D elements."""

import autograd.numpy as anp

from manforge.core.material.base import MaterialModel
from manforge._typing import FloatArray, Scalar, Stiffness, StressVec
from manforge.core.dimension import (
    SOLID_3D,
    PLANE_STRESS,
    UNIAXIAL_1D,
    StressDimension,
)
from manforge.utils.smooth import smooth_sqrt, smooth_abs


class MaterialModel3D(MaterialModel):
    """Stress-state base class for full-rank stress states (ndi == ndi_phys).

    Valid for ``SOLID_3D`` (ntens=6) and ``PLANE_STRAIN`` (ntens=4), and any
    future stress state that stores all direct components (e.g. axisymmetric).
    For these states the three physical direct stresses σ11, σ22, σ33 are all
    stored explicitly, so deviatoric and von Mises operators need no
    missing-component corrections.

    Provides concrete implementations of the operator methods used by concrete
    material models:

    * :meth:`hydrostatic` — p = (σ11 + σ22 + σ33) / 3
    * :meth:`dev`         — s = σ − p δ
    * :meth:`isotropic_C`  — submatrix extraction from the full 6×6 tensor
    * :meth:`I_vol`       — δ⊗δ / 3
    * :meth:`I_dev`       — I − P_vol

    :meth:`vonmises` is inherited from :class:`MaterialModel` (uses
    ``smooth_sqrt`` with n_missing=0, equivalent to √(3/2 s:s)).

    Subclasses still must implement the required material methods:
    :meth:`elastic_stiffness`, :meth:`yield_function`, and either
    :meth:`update_state` (reduced) or :meth:`state_residual` (augmented).

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
        self.dimension = dimension

    # ------------------------------------------------------------------
    # Operator methods — concrete for full-rank stress states
    # ------------------------------------------------------------------

    def hydrostatic(self, stress: StressVec) -> Scalar:
        """Mean normal stress p = (σ11 + σ22 + σ33) / 3.

        All three direct components are stored, so no correction is needed.
        """
        return (stress[0] + stress[1] + stress[2]) / 3.0

    def dev(self, stress: StressVec) -> StressVec:
        """Deviatoric stress s = σ − p δ."""
        p = self.hydrostatic(stress)
        return stress - p * self.dimension.identity_np

    def isotropic_C(self, lam: float, mu: float) -> Stiffness:
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

    def I_vol(self) -> Stiffness:
        """Volumetric projection tensor P_vol = δ⊗δ / 3."""
        delta = self.dimension.identity_np
        return anp.outer(delta, delta) / 3.0

    def I_dev(self) -> Stiffness:
        """Deviatoric projection tensor P_dev = I − P_vol."""
        return anp.eye(self.ntens) - self.I_vol()

    # ------------------------------------------------------------------
    # Kinematic-hardening operators (α deviatoric assumed)
    # ------------------------------------------------------------------

    def vonmises_relative(self, stress: StressVec, alpha: StressVec) -> Scalar:
        """Von Mises norm of ξ = σ − α (all components stored for 3D/PE)."""
        return self.vonmises(stress - alpha)

    def flow_direction(self, stress: StressVec, alpha: StressVec) -> StressVec:
        """Plastic flow direction n̂ = (3/2) dev(ξ) / σ_vm(ξ)."""
        xi = stress - alpha
        return 1.5 * self.dev(xi) / self.vonmises(xi)

    def alpha_norm(self, alpha: StressVec) -> Scalar:
        """Von Mises norm of deviatoric backstress α (n_missing=0 for 3D/PE)."""
        return self.vonmises(alpha)


class MaterialModelPS(MaterialModel):
    """Stress-state base class for plane-stress elements (PLANE_STRESS).

    For plane stress, σ33 = 0 is enforced by static condensation of the
    elastic stiffness.  Only two direct components (σ11, σ22) are stored,
    so the von Mises computation must account for the physically-present
    but unstored deviatoric contribution of σ33 = 0.

    Provides concrete implementations of the operator methods:

    * :meth:`hydrostatic` — p = (σ11 + σ22) / 3  (σ33 = 0)
    * :meth:`dev`         — s = σ − p δ  (stored components only)
    * :meth:`isotropic_C`  — Schur complement (static condensation of σ33)
    * :meth:`I_vol`       — δ⊗δ / 3  (δ = [1, 1, 0] for ntens=3)
    * :meth:`I_dev`       — I − P_vol

    :meth:`vonmises` is inherited from :class:`MaterialModel` (uses
    ``smooth_sqrt`` with n_missing=1, giving √(3/2 (s:s + p²))).

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
        self.dimension = dimension

    # ------------------------------------------------------------------
    # Operator methods — concrete for PLANE_STRESS
    # ------------------------------------------------------------------

    def hydrostatic(self, stress: StressVec) -> Scalar:
        """Mean normal stress p = (σ11 + σ22) / 3.

        σ33 = 0 is enforced externally; ndi_phys = 3 so we divide by 3.
        """
        return (stress[0] + stress[1]) / 3.0

    def dev(self, stress: StressVec) -> StressVec:
        """Deviatoric stress of the stored components, s = σ − p δ."""
        p = self.hydrostatic(stress)
        return stress - p * self.dimension.identity_np  # δ = [1, 1, 0]

    def isotropic_C(self, lam: float, mu: float) -> Stiffness:
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

    def I_vol(self) -> Stiffness:
        """Volumetric projection tensor P_vol = δ⊗δ / 3."""
        delta = self.dimension.identity_np  # [1, 1, 0]
        return anp.outer(delta, delta) / 3.0

    def I_dev(self) -> Stiffness:
        """Deviatoric projection tensor P_dev = I − P_vol."""
        return anp.eye(self.ntens) - self.I_vol()

    # ------------------------------------------------------------------
    # Kinematic-hardening operators (α deviatoric assumed)
    # ------------------------------------------------------------------

    def vonmises_relative(self, stress: StressVec, alpha: StressVec) -> Scalar:
        """Von Mises norm of ξ = σ − α with α deviatoric (α33 = −(α11+α22)).

        For plane stress σ33 = 0, so ξ33 = −α33 = α11+α22.  The hydrostatic
        part of ξ depends only on σ (the α contributions cancel), so all
        required quantities are computable from the three stored components.
        """
        p = (stress[0] + stress[1]) / 3.0   # p_ξ = p_σ  (α deviatoric ⇒ tr α = 0)
        s11 = stress[0] - alpha[0] - p
        s22 = stress[1] - alpha[1] - p
        s33 = (alpha[0] + alpha[1]) - p      # ξ33 − p = −α33 − p
        s12 = stress[2] - alpha[2]
        return smooth_sqrt(1.5 * (s11*s11 + s22*s22 + s33*s33 + 2.0*s12*s12))

    def flow_direction(self, stress: StressVec, alpha: StressVec) -> StressVec:
        """Plastic flow direction n̂ = ∂f/∂σ = (3/2) s_ξ / σ_vm(ξ).

        Returns the three stored PS components of n̂.  Uses the same
        α-deviatoric identity as :meth:`vonmises_relative`.
        """
        p = (stress[0] + stress[1]) / 3.0
        s11 = stress[0] - alpha[0] - p
        s22 = stress[1] - alpha[1] - p
        s12 = stress[2] - alpha[2]
        vm = self.vonmises_relative(stress, alpha)
        return 1.5 * anp.array([s11, s22, s12]) / vm

    def alpha_norm(self, alpha: StressVec) -> Scalar:
        """Von Mises norm of deviatoric backstress α (α33 = −(α11+α22))."""
        a33 = -(alpha[0] + alpha[1])
        return smooth_sqrt(1.5 * (
            alpha[0]*alpha[0] + alpha[1]*alpha[1] + a33*a33 + 2.0*alpha[2]*alpha[2]
        ))


class MaterialModel1D(MaterialModel):
    """Stress-state base class for uniaxial (1D) elements (UNIAXIAL_1D).

    Only σ11 is stored; σ22 = σ33 = 0 are enforced by the element
    formulation.  The von Mises computation must account for two missing
    deviatoric components (n_missing = 2).

    Provides concrete implementations of the operator methods:

    * :meth:`hydrostatic` — p = σ11 / 3  (σ22 = σ33 = 0)
    * :meth:`dev`         — s = σ − p δ  (stored component only)
    * :meth:`isotropic_C`  — [[E]] where E = μ(3λ + 2μ) / (λ + μ)
    * :meth:`I_vol`       — [[1/3]]
    * :meth:`I_dev`       — [[2/3]]

    :meth:`vonmises` is inherited from :class:`MaterialModel` (uses
    ``smooth_sqrt`` with n_missing=2, giving √(3/2 (s11² + 2p²)) ≈ |σ11|).

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
        self.dimension = dimension

    # ------------------------------------------------------------------
    # Operator methods — concrete for UNIAXIAL_1D
    # ------------------------------------------------------------------

    def hydrostatic(self, stress: StressVec) -> Scalar:
        """Mean normal stress p = σ11 / 3.

        σ22 = σ33 = 0 are enforced externally; ndi_phys = 3 so we divide by 3.
        """
        return stress[0] / 3.0

    def dev(self, stress: StressVec) -> StressVec:
        """Deviatoric stress of the stored component, s = σ − p δ."""
        p = self.hydrostatic(stress)
        return stress - p * self.dimension.identity_np  # δ = [1.0]

    def isotropic_C(self, lam: float, mu: float) -> Stiffness:
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

    def I_vol(self) -> Stiffness:
        """Volumetric projection tensor [[1/3]] for ntens=1."""
        delta = self.dimension.identity_np  # [1.0]
        return anp.outer(delta, delta) / 3.0

    def I_dev(self) -> Stiffness:
        """Deviatoric projection tensor [[2/3]] for ntens=1."""
        return anp.eye(1) - self.I_vol()

    # ------------------------------------------------------------------
    # Kinematic-hardening operators (α deviatoric assumed)
    # ------------------------------------------------------------------

    def vonmises_relative(self, stress: StressVec, alpha: StressVec) -> Scalar:
        """Von Mises norm of ξ = σ − (3/2)α_dev.

        For 1D, α stores the deviatoric component α11_dev.  The missing
        components α22 = α33 = −α11_dev/2 contribute to the 3D relative
        stress ξ22 = ξ33 = α11_dev/2, so the effective relative stress is
        σ_vm(ξ) = |σ11 − (3/2)·α11_dev|.
        """
        return smooth_abs(stress[0] - 1.5 * alpha[0])

    def flow_direction(self, stress: StressVec, alpha: StressVec) -> StressVec:
        """Plastic flow direction n̂_11 = sign(σ11 − (3/2)·α11_dev)."""
        diff = stress[0] - 1.5 * alpha[0]
        return anp.array([diff / smooth_abs(diff)])

    def alpha_norm(self, alpha: StressVec) -> Scalar:
        """Von Mises norm of deviatoric backstress α (α22=α33=−α/2)."""
        return smooth_abs(1.5 * alpha[0])
