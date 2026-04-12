"""J2 plasticity with isotropic hardening (reference implementation).

Model parameters
----------------
E        : Young's modulus
nu       : Poisson's ratio
sigma_y0 : Initial yield stress
H        : Isotropic hardening modulus (linear)

Internal state
--------------
ep : equivalent plastic strain (scalar, ≥ 0)

Yield function
--------------
f(σ, ep) = σ_vm(σ) - (σ_y0 + H * ep)

where σ_vm is the von Mises equivalent stress.

Flow rule (associative)
-----------------------
dε_p = Δλ · n,  n = df/dσ = (3/2) s / σ_vm  (unit normal in Mandel sense)
"""

import jax.numpy as jnp

from manforge.autodiff.operators import vonmises
from manforge.core.material import MaterialModel, MaterialModel3D, MaterialModelPS, MaterialModel1D
from manforge.core.stress_state import SOLID_3D, PLANE_STRESS, UNIAXIAL_1D, StressState


class J2IsotropicHardening(MaterialModel):
    """J2 plasticity model with linear isotropic hardening.

    Parameters
    ----------
    stress_state : StressState, optional
        Dimensionality descriptor.  Defaults to ``SOLID_3D`` (6-component 3D).
        Pass ``PLANE_STRAIN`` or ``PLANE_STRESS`` etc. for other element types.
    """

    param_names = ["E", "nu", "sigma_y0", "H"]
    state_names = ["ep"]  # equivalent plastic strain

    def __init__(self, stress_state: StressState = SOLID_3D):
        self.stress_state = stress_state

    def elastic_stiffness(self, params: dict) -> jnp.ndarray:
        """Isotropic elastic stiffness tensor.

        Parameters
        ----------
        params : dict
            Must contain keys ``E`` and ``nu``.

        Returns
        -------
        jnp.ndarray, shape (ntens, ntens)
        """
        E = params["E"]
        nu = params["nu"]
        mu = E / (2.0 * (1.0 + nu))
        lam = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
        return self.isotropic_C(lam, mu)

    def yield_function(
        self,
        stress: jnp.ndarray,
        state: dict,
        params: dict,
    ) -> jnp.ndarray:
        """J2 yield function.

        f = σ_vm - (σ_y0 + H · ep)

        Elastic domain: f ≤ 0.

        Parameters
        ----------
        stress : jnp.ndarray, shape (ntens,)
        state : dict with key ``ep``
        params : dict with keys ``sigma_y0``, ``H``

        Returns
        -------
        jnp.ndarray, scalar
        """
        ep = state["ep"]
        sigma_y = params["sigma_y0"] + params["H"] * ep
        return vonmises(stress, self.stress_state) - sigma_y

    def hardening_increment(
        self,
        dlambda: jnp.ndarray,
        stress: jnp.ndarray,
        state: dict,
        params: dict,
    ) -> dict:
        """Update equivalent plastic strain.

        For J2 with associative flow, Δep = Δλ (von Mises consistency).

        Parameters
        ----------
        dlambda : jnp.ndarray, scalar
            Plastic multiplier increment.
        stress : jnp.ndarray, shape (ntens,)
            Current stress (unused for isotropic hardening).
        state : dict with key ``ep``
        params : dict (unused here, kept for interface consistency)

        Returns
        -------
        dict
            ``{"ep": ep + dlambda}``
        """
        return {"ep": state["ep"] + dlambda}

class J2Isotropic3D(MaterialModel3D):
    """J2 plasticity with analytical radial return for full-rank stress states.

    Inherits operator methods from :class:`~manforge.core.material.MaterialModel3D`
    (``_dev``, ``_vonmises``, ``isotropic_C``, ``_I_vol``, ``_I_dev``), which
    provide branch-free implementations valid when all direct stress components
    are stored (``ndi == ndi_phys``).

    Provides closed-form ``plastic_corrector`` and ``analytical_tangent``
    using the identity ``C @ n_dev = 2μ · n_dev``, which holds exactly for
    SOLID_3D and PLANE_STRAIN but not for statically condensed stress states
    (PLANE_STRESS, UNIAXIAL_1D).  For those, use :class:`J2IsotropicHardening`
    with ``method="autodiff"``.

    Parameters
    ----------
    stress_state : StressState, optional
        Must satisfy ``stress_state.ndi == stress_state.ndi_phys``.
        Defaults to ``SOLID_3D``.  The guard is enforced by the parent.

    Raises
    ------
    ValueError
        If ``stress_state.ndi != stress_state.ndi_phys`` (raised by
        :class:`~manforge.core.material.MaterialModel3D`).
    """

    param_names = ["E", "nu", "sigma_y0", "H"]
    state_names = ["ep"]

    def __init__(self, stress_state: StressState = SOLID_3D):
        super().__init__(stress_state)

    # ------------------------------------------------------------------
    # Material physics — implement the three abstract methods
    # ------------------------------------------------------------------

    def elastic_stiffness(self, params: dict) -> jnp.ndarray:
        """Isotropic elastic stiffness tensor."""
        E, nu = params["E"], params["nu"]
        mu = E / (2.0 * (1.0 + nu))
        lam = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
        return self.isotropic_C(lam, mu)

    def yield_function(self, stress: jnp.ndarray, state: dict, params: dict) -> jnp.ndarray:
        """J2 yield function f = σ_vm − (σ_y0 + H · ep)."""
        sigma_y = params["sigma_y0"] + params["H"] * state["ep"]
        return self._vonmises(stress) - sigma_y

    def hardening_increment(self, dlambda, stress, state, params) -> dict:
        """Δep = Δλ (von Mises associative flow)."""
        return {"ep": state["ep"] + dlambda}

    # ------------------------------------------------------------------
    # Analytical solver hooks
    # ------------------------------------------------------------------

    def plastic_corrector(self, stress_trial, C, state_n, params):
        """J2 radial return — closed-form plastic correction.

        Notes
        -----
        For isotropic J2 with linear hardening, ``C n_voigt = 3μ s_trial / σ_vm``
        for all Voigt components (the volumetric part of the gradient vanishes),
        so the radial return reduces to a single scalar equation in Δλ.
        This identity holds when ``ndi == ndi_phys`` (SOLID_3D, PLANE_STRAIN).

        Parameters
        ----------
        stress_trial : jnp.ndarray, shape (ntens,)
        C : jnp.ndarray, shape (ntens, ntens)
        state_n : dict with key ``ep``
        params : dict with keys ``E``, ``nu``, ``sigma_y0``, ``H``

        Returns
        -------
        tuple[jnp.ndarray, dict, jnp.ndarray]
            ``(stress_new, state_new, dlambda)``
        """
        E, nu = params["E"], params["nu"]
        mu = E / (2.0 * (1.0 + nu))
        H, sigma_y0, ep_n = params["H"], params["sigma_y0"], state_n["ep"]

        sigma_y = sigma_y0 + H * ep_n
        s_trial = self._dev(stress_trial)
        sigma_vm_trial = self._vonmises(stress_trial)

        # Δλ = (σ_vm_trial − σ_y) / (3μ + H)
        dlambda = (sigma_vm_trial - sigma_y) / (3.0 * mu + H)

        # Radial return: σ_new = σ_trial − (3μΔλ/σ_vm) s_trial
        stress_new = stress_trial - (3.0 * mu * dlambda / sigma_vm_trial) * s_trial

        return stress_new, {"ep": ep_n + dlambda}, jnp.asarray(dlambda)

    def analytical_tangent(self, stress, state, dlambda, C, state_n, params):
        """J2 algorithmic consistent tangent — closed-form (de Souza Neto).

            D^ep = I_vol C + θ I_dev C − β (s_trial ⊗ s_trial)

        where θ = 1 − 3μΔλ/σ_vm_trial and β = 9μ²σ_y / ((3μ+H) σ_vm_trial³).

        Parameters
        ----------
        stress : jnp.ndarray, shape (ntens,)
        state : dict  (unused; ep taken from state_n)
        dlambda : jnp.ndarray, scalar
        C : jnp.ndarray, shape (ntens, ntens)
        state_n : dict with key ``ep``
        params : dict with keys ``E``, ``nu``, ``sigma_y0``, ``H``

        Returns
        -------
        jnp.ndarray, shape (ntens, ntens)
        """
        E, nu = params["E"], params["nu"]
        mu = E / (2.0 * (1.0 + nu))
        H, sigma_y0, ep_n = params["H"], params["sigma_y0"], state_n["ep"]

        sigma_y = sigma_y0 + H * ep_n
        sigma_vm_trial = sigma_y + (3.0 * mu + H) * dlambda
        theta = 1.0 - 3.0 * mu * dlambda / sigma_vm_trial

        # dev(σ_new) = θ · s_trial  →  s_trial = dev(σ_new) / θ
        s_trial = self._dev(stress) / theta
        beta = 9.0 * mu ** 2 * sigma_y / ((3.0 * mu + H) * sigma_vm_trial ** 3)

        I_vol = self._I_vol()
        I_dev = self._I_dev()

        return I_vol @ C + theta * (I_dev @ C) - beta * jnp.outer(s_trial, s_trial)


class J2IsotropicPS(MaterialModelPS):
    """J2 plasticity with isotropic hardening for plane-stress elements.

    Inherits operator methods from
    :class:`~manforge.core.material.MaterialModelPS` (``_dev``,
    ``_vonmises``, ``isotropic_C``, ``_I_vol``, ``_I_dev``), which implement
    the static condensation and missing-component correction specific to
    plane-stress kinematics.

    Uses the autodiff return-mapping path (``method="autodiff"``).  A
    closed-form plane-stress corrector (which requires an iterative σ33 = 0
    enforcement loop) is not yet implemented.

    Parameters
    ----------
    stress_state : StressState, optional
        Must satisfy ``stress_state.is_plane_stress``.
        Defaults to ``PLANE_STRESS``.
    """

    param_names = ["E", "nu", "sigma_y0", "H"]
    state_names = ["ep"]

    def __init__(self, stress_state: StressState = PLANE_STRESS):
        super().__init__(stress_state)

    def elastic_stiffness(self, params: dict) -> jnp.ndarray:
        """Plane-stress isotropic stiffness (3×3 condensed)."""
        E, nu = params["E"], params["nu"]
        mu = E / (2.0 * (1.0 + nu))
        lam = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
        return self.isotropic_C(lam, mu)

    def yield_function(self, stress: jnp.ndarray, state: dict, params: dict) -> jnp.ndarray:
        """J2 yield function f = σ_vm − (σ_y0 + H · ep)."""
        sigma_y = params["sigma_y0"] + params["H"] * state["ep"]
        return self._vonmises(stress) - sigma_y

    def hardening_increment(self, dlambda, stress, state, params) -> dict:
        """Δep = Δλ (von Mises associative flow)."""
        return {"ep": state["ep"] + dlambda}


class J2Isotropic1D(MaterialModel1D):
    """J2 plasticity with isotropic hardening for uniaxial (1D) elements.

    Inherits operator methods from
    :class:`~manforge.core.material.MaterialModel1D`.

    Provides a closed-form ``plastic_corrector`` and ``analytical_tangent``
    using the 1D radial return mapping.

    Parameters
    ----------
    stress_state : StressState, optional
        Must have ``ntens == 1``.  Defaults to ``UNIAXIAL_1D``.
    """

    param_names = ["E", "nu", "sigma_y0", "H"]
    state_names = ["ep"]

    def __init__(self, stress_state: StressState = UNIAXIAL_1D):
        super().__init__(stress_state)

    # ------------------------------------------------------------------
    # Material physics — implement the three abstract methods
    # ------------------------------------------------------------------

    def elastic_stiffness(self, params: dict) -> jnp.ndarray:
        """1D elastic stiffness [[E]]."""
        E, nu = params["E"], params["nu"]
        mu = E / (2.0 * (1.0 + nu))
        lam = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
        return self.isotropic_C(lam, mu)

    def yield_function(self, stress: jnp.ndarray, state: dict, params: dict) -> jnp.ndarray:
        """J2 yield function f = σ_vm − (σ_y0 + H · ep)."""
        sigma_y = params["sigma_y0"] + params["H"] * state["ep"]
        return self._vonmises(stress) - sigma_y

    def hardening_increment(self, dlambda, stress, state, params) -> dict:
        """Δep = Δλ (von Mises associative flow)."""
        return {"ep": state["ep"] + dlambda}

    # ------------------------------------------------------------------
    # Analytical solver hooks
    # ------------------------------------------------------------------

    def plastic_corrector(self, stress_trial, C, state_n, params):
        """1D J2 radial return — closed-form.

        Δλ = (|σ_trial| − σ_y) / (E + H)
        σ_new = σ_trial − E · Δλ · sign(σ_trial)

        Parameters
        ----------
        stress_trial : jnp.ndarray, shape (1,)
        C : jnp.ndarray, shape (1, 1)
        state_n : dict with key ``ep``
        params : dict with keys ``E``, ``nu``, ``sigma_y0``, ``H``

        Returns
        -------
        tuple[jnp.ndarray, dict, jnp.ndarray]
        """
        E = C[0, 0]
        H, sigma_y0, ep_n = params["H"], params["sigma_y0"], state_n["ep"]
        sigma_y = sigma_y0 + H * ep_n
        sigma_vm_trial = jnp.abs(stress_trial[0])
        dlambda = (sigma_vm_trial - sigma_y) / (E + H)
        n = stress_trial / sigma_vm_trial  # sign(σ_trial) as length-1 array
        stress_new = stress_trial - E * dlambda * n
        return stress_new, {"ep": ep_n + dlambda}, jnp.asarray(dlambda)

    def analytical_tangent(self, stress, state, dlambda, C, state_n, params):
        """1D consistent tangent D^ep = [[E · H / (E + H)]].

        Parameters
        ----------
        stress : jnp.ndarray, shape (1,)
        state : dict  (unused)
        dlambda : jnp.ndarray, scalar
        C : jnp.ndarray, shape (1, 1)
        state_n : dict  (unused)
        params : dict with keys ``H``

        Returns
        -------
        jnp.ndarray, shape (1, 1)
        """
        E = C[0, 0]
        H = params["H"]
        return jnp.array([[E * H / (E + H)]])
