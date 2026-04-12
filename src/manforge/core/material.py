"""Abstract base class for constitutive material models."""

from abc import ABC, abstractmethod

import jax.numpy as jnp

from manforge.autodiff.operators import identity_voigt
from manforge.core.stress_state import SOLID_3D, StressState


class MaterialModel(ABC):
    """Abstract base class for constitutive material models.

    Subclasses must define :attr:`param_names` and :attr:`state_names`,
    and implement the three abstract methods.  The framework then provides
    return mapping, consistent tangent computation, and parameter fitting.

    Attributes
    ----------
    param_names : list[str]
        Names of material parameters (keys expected in ``params`` dicts).
    state_names : list[str]
        Names of internal state variables (keys in ``state`` dicts).
    stress_state : StressState
        Dimensionality descriptor (default: ``SOLID_3D``, 6-component 3D).
    ntens : int
        Read-only property; returns ``self.stress_state.ntens``.
    """

    param_names: list[str]
    state_names: list[str]
    stress_state: StressState = SOLID_3D

    @property
    def ntens(self) -> int:
        """Number of stress/strain components (derived from stress_state)."""
        return self.stress_state.ntens

    # ------------------------------------------------------------------
    # Abstract interface — must be implemented by subclasses
    # ------------------------------------------------------------------

    @abstractmethod
    def elastic_stiffness(self, params: dict) -> jnp.ndarray:
        """Return the elastic stiffness tensor in Voigt notation.

        Parameters
        ----------
        params : dict
            Material parameters keyed by :attr:`param_names`.

        Returns
        -------
        jnp.ndarray, shape (ntens, ntens)
            Elastic stiffness C (σ = C : ε, engineering shear convention).
        """

    @abstractmethod
    def yield_function(
        self,
        stress: jnp.ndarray,
        state: dict,
        params: dict,
    ) -> jnp.ndarray:
        """Evaluate the yield function f(σ, q, params).

        The material is in the elastic domain when f ≤ 0.

        Parameters
        ----------
        stress : jnp.ndarray, shape (ntens,)
            Stress in Voigt notation.
        state : dict
            Internal state variables.
        params : dict
            Material parameters.

        Returns
        -------
        jnp.ndarray, scalar
            Yield function value.
        """

    @abstractmethod
    def hardening_increment(
        self,
        dlambda: jnp.ndarray,
        stress: jnp.ndarray,
        state: dict,
        params: dict,
    ) -> dict:
        """Return updated state variables after a plastic increment.

        Parameters
        ----------
        dlambda : jnp.ndarray, scalar
            Plastic multiplier increment Δλ ≥ 0.
        stress : jnp.ndarray, shape (ntens,)
            Current stress within the NR iteration (Voigt notation).
            Models that depend only on dlambda (e.g. isotropic hardening)
            may ignore this argument.
        state : dict
            State at the beginning of the increment.
        params : dict
            Material parameters.

        Returns
        -------
        dict
            Updated state ``state_{n+1}``.
        """

    # ------------------------------------------------------------------
    # Default helpers provided by the framework
    # ------------------------------------------------------------------

    def isotropic_C(self, lam: float, mu: float) -> jnp.ndarray:
        """Build the isotropic elastic stiffness tensor.

        Builds the full 3D 6×6 stiffness first, then extracts/condenses to
        the dimensionality specified by ``self.stress_state``:

        * **3D solid (NTENS=6)**: returns the full 6×6 tensor.
          C = λ δ⊗δ + μ diag([2, 2, 2, 1, 1, 1])
        * **Plane strain / axisymmetric (NTENS=4)**: returns the 4×4
          submatrix for components [11, 22, 33, 12].
        * **Plane stress (NTENS=3)**: applies static condensation to enforce
          σ33 = 0, returning a 3×3 matrix for [11, 22, 12].
        * **1D truss (NTENS=1)**: returns [[E]] where E = μ(3λ+2μ)/(λ+μ).

        Parameters
        ----------
        lam : float
            First Lamé constant λ.
        mu : float
            Shear modulus μ.

        Returns
        -------
        jnp.ndarray, shape (ntens, ntens)
        """
        # Build the full 3D 6×6 tensor
        delta_6 = identity_voigt()  # 6-component, no ss
        scale_6 = jnp.array([2.0, 2.0, 2.0, 1.0, 1.0, 1.0])
        C6 = lam * jnp.outer(delta_6, delta_6) + mu * jnp.diag(scale_6)

        ss = self.stress_state

        if ss.ntens == 6:
            return C6

        if ss.ntens == 4:
            # Plane strain / axisymmetric: components [11,22,33,12]
            idx = jnp.array([0, 1, 2, 3])
            return C6[jnp.ix_(idx, idx)]

        if ss.ntens == 3:
            # Plane stress: condense out sigma_33 (index 2 of the 4×4 plane-
            # strain submatrix) to enforce sigma_33 = 0.
            # Retain indices [0,1,3] of the 6×6 → [s11, s22, s12].
            # Step 1: 4×4 plane-strain sub-block
            idx4 = jnp.array([0, 1, 2, 3])
            C4 = C6[jnp.ix_(idx4, idx4)]
            # Step 2: static condensation — eliminate row/col 2 (s33)
            # D_ps = D_00 - D_02 * D_22^{-1} * D_20  (Schur complement)
            # Retained dof within C4: [0, 1, 3] → mapped to [0, 1, 2]
            retain = jnp.array([0, 1, 3])
            C_rr = C4[jnp.ix_(retain, retain)]
            C_rc = C4[retain, 2]          # shape (3,)
            C_cc = C4[2, 2]               # scalar
            return C_rr - jnp.outer(C_rc, C_rc) / C_cc

        if ss.ntens == 1:
            # 1D truss (uniaxial stress): C = [[E]]
            E = mu * (3.0 * lam + 2.0 * mu) / (lam + mu)
            return jnp.array([[E]])

        raise ValueError(
            f"isotropic_C: unsupported ntens={ss.ntens} for stress_state '{ss.name}'"
        )

    def initial_state(self) -> dict:
        """Return zero-initialised state dict.

        Returns
        -------
        dict
            ``{name: jnp.array(0.0) for name in self.state_names}``
        """
        return {name: jnp.array(0.0) for name in self.state_names}

    def plastic_corrector(
        self,
        stress_trial: jnp.ndarray,
        C: jnp.ndarray,
        state_n: dict,
        params: dict,
    ):
        """Closed-form plastic correction (optional).

        Override to provide an analytical return-mapping algorithm.
        The default returns ``None``, causing
        :func:`~manforge.core.return_mapping.return_mapping` to fall back
        to the generic Newton-Raphson + autodiff path.

        Parameters
        ----------
        stress_trial : jnp.ndarray, shape (ntens,)
            Elastic trial stress σ_trial = σ_n + C Δε.
        C : jnp.ndarray, shape (ntens, ntens)
            Elastic stiffness tensor (already computed by the caller).
        state_n : dict
            Internal state at the beginning of the increment.
        params : dict
            Material parameters.

        Returns
        -------
        None
            Return ``None`` to signal that no analytical corrector is
            available and the generic path should be used.
        tuple[jnp.ndarray, dict, jnp.ndarray]
            ``(stress_new, state_new, dlambda)`` — converged stress,
            updated state dict, and plastic multiplier increment Δλ.
        """
        return None

    def analytical_tangent(
        self,
        stress: jnp.ndarray,
        state: dict,
        dlambda: jnp.ndarray,
        C: jnp.ndarray,
        state_n: dict,
        params: dict,
    ):
        """Closed-form consistent tangent (optional).

        Override to provide an analytical expression for dσ_{n+1}/dΔε.
        The default returns ``None``, causing
        :func:`~manforge.core.return_mapping.return_mapping` to fall back
        to the generic implicit-differentiation + autodiff path.

        Parameters
        ----------
        stress : jnp.ndarray, shape (ntens,)
            Converged stress σ_{n+1}.
        state : dict
            Converged internal state at step n+1.
        dlambda : jnp.ndarray, scalar
            Converged plastic multiplier increment Δλ.
        C : jnp.ndarray, shape (ntens, ntens)
            Elastic stiffness tensor (already computed by the caller).
        state_n : dict
            Internal state at the beginning of the increment.
        params : dict
            Material parameters.

        Returns
        -------
        None
            Return ``None`` to signal that no analytical tangent is
            available and the generic path should be used.
        jnp.ndarray, shape (ntens, ntens)
            Consistent tangent dσ_{n+1}/dΔε.
        """
        return None


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
    * :meth:`_vonmises`    — √(3/2 s:s)  (Mandel norm, no correction)
    * :meth:`isotropic_C`  — submatrix extraction from the full 6×6 tensor
    * :meth:`_I_vol`       — δ⊗δ / 3
    * :meth:`_I_dev`       — I − P_vol

    Subclasses still must implement the three abstract material methods:
    :meth:`elastic_stiffness`, :meth:`yield_function`,
    :meth:`hardening_increment`.

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

    def _hydrostatic(self, stress: jnp.ndarray) -> jnp.ndarray:
        """Mean normal stress p = (σ11 + σ22 + σ33) / 3.

        All three direct components are stored, so no correction is needed.
        """
        return (stress[0] + stress[1] + stress[2]) / 3.0

    def _dev(self, stress: jnp.ndarray) -> jnp.ndarray:
        """Deviatoric stress s = σ − p δ."""
        p = self._hydrostatic(stress)
        return stress - p * self.stress_state.identity_jnp

    def _vonmises(self, stress: jnp.ndarray) -> jnp.ndarray:
        """Von Mises equivalent stress √(3/2 s:s).

        Uses Mandel scaling for the inner product.  No missing-component
        correction is required because ndi == ndi_phys.
        """
        s = self._dev(stress)
        s_m = s * self.stress_state.mandel_factors_jnp
        return jnp.sqrt(1.5 * jnp.dot(s_m, s_m))

    def isotropic_C(self, lam: float, mu: float) -> jnp.ndarray:
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
        jnp.ndarray, shape (ntens, ntens)
        """
        delta_6 = jnp.array([1.0, 1.0, 1.0, 0.0, 0.0, 0.0])
        scale_6 = jnp.array([2.0, 2.0, 2.0, 1.0, 1.0, 1.0])
        C6 = lam * jnp.outer(delta_6, delta_6) + mu * jnp.diag(scale_6)
        if self.ntens == 6:
            return C6
        # PLANE_STRAIN / AXISYMMETRIC: components [11, 22, 33, 12]
        idx = jnp.array([0, 1, 2, 3])
        return C6[jnp.ix_(idx, idx)]

    def _I_vol(self) -> jnp.ndarray:
        """Volumetric projection tensor P_vol = δ⊗δ / 3."""
        delta = self.stress_state.identity_jnp
        return jnp.outer(delta, delta) / 3.0

    def _I_dev(self) -> jnp.ndarray:
        """Deviatoric projection tensor P_dev = I − P_vol."""
        return jnp.eye(self.ntens) - self._I_vol()
