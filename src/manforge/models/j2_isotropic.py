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

from manforge.autodiff.operators import dev, vonmises, I_dev_voigt, I_vol_voigt
from manforge.core.material import MaterialModel
from manforge.core.stress_state import SOLID_3D, StressState


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

    def plastic_corrector(self, stress_trial, C, state_n, params):
        """J2 radial return — closed-form plastic correction.

        Computes the converged stress, updated state, and plastic multiplier
        analytically without Newton-Raphson iteration.

        Notes
        -----
        For isotropic J2 with linear hardening, ``C n_voigt = 3μ s_trial / σ_vm``
        for all Voigt components (the volumetric part of the gradient vanishes),
        so the radial return reduces to a single scalar equation in Δλ.

        Parameters
        ----------
        stress_trial : jnp.ndarray, shape (ntens,)
            Elastic trial stress.
        C : jnp.ndarray, shape (ntens, ntens)
            Elastic stiffness (passed from return_mapping, not recomputed).
        state_n : dict with key ``ep``
            State at the beginning of the increment.
        params : dict with keys ``E``, ``nu``, ``sigma_y0``, ``H``

        Returns
        -------
        tuple[jnp.ndarray, dict, jnp.ndarray]
            ``(stress_new, state_new, dlambda)``
        """
        E = params["E"]
        nu = params["nu"]
        mu = E / (2.0 * (1.0 + nu))
        H = params["H"]
        sigma_y0 = params["sigma_y0"]
        ep_n = state_n["ep"]

        sigma_y = sigma_y0 + H * ep_n
        s_trial = dev(stress_trial, self.stress_state)
        sigma_vm_trial = vonmises(stress_trial, self.stress_state)

        # Closed-form plastic multiplier: Δλ = (σ_vm_trial - σ_y) / (3μ + H)
        dlambda = (sigma_vm_trial - sigma_y) / (3.0 * mu + H)

        # Radial return: σ_new = σ_trial − Δλ (C n) = σ_trial − (3μΔλ/σ_vm) s_trial
        # because C n_voigt = 3μ s_trial / σ_vm_trial for all 6 components.
        stress_new = stress_trial - (3.0 * mu * dlambda / sigma_vm_trial) * s_trial

        state_new = {"ep": ep_n + dlambda}

        return stress_new, state_new, jnp.asarray(dlambda)

    def analytical_tangent(self, stress, state, dlambda, C, state_n, params):
        """J2 algorithmic consistent tangent — closed-form.

        Uses the well-known expression (de Souza Neto et al. 2008):

            D^ep = I_vol C + θ I_dev C − β (s_trial ⊗ s_trial)

        where θ = 1 − 3μΔλ/σ_vm_trial and β = 9μ²σ_y / ((3μ+H) σ_vm_trial³).

        Parameters
        ----------
        stress : jnp.ndarray, shape (ntens,)
            Converged stress σ_{n+1}.
        state : dict
            Converged state (unused here; ep is taken from ``state_n``).
        dlambda : jnp.ndarray, scalar
            Converged plastic multiplier increment Δλ.
        C : jnp.ndarray, shape (ntens, ntens)
            Elastic stiffness.
        state_n : dict with key ``ep``
            State at the beginning of the increment.
        params : dict with keys ``E``, ``nu``, ``sigma_y0``, ``H``

        Returns
        -------
        jnp.ndarray, shape (ntens, ntens)
            Consistent tangent dσ_{n+1}/dΔε.
        """
        E = params["E"]
        nu = params["nu"]
        mu = E / (2.0 * (1.0 + nu))
        H = params["H"]
        sigma_y0 = params["sigma_y0"]
        ep_n = state_n["ep"]

        sigma_y = sigma_y0 + H * ep_n

        # Reconstruct trial quantities from converged Δλ
        sigma_vm_trial = sigma_y + (3.0 * mu + H) * dlambda
        theta = 1.0 - 3.0 * mu * dlambda / sigma_vm_trial

        # Deviatoric trial stress: dev(σ_new) = θ · s_trial → s_trial = dev(σ_new)/θ
        ss = self.stress_state
        s_trial = dev(stress, ss) / theta

        # Tangent: D^ep = A @ C  where A = I_vol + θ I_dev − coeff · s_trial ⊗ n_voigt
        # and n_voigt @ C = C @ n_voigt = 3μ s_trial / σ_vm_trial
        # → D^ep = I_vol @ C + θ I_dev @ C − (9μ²σ_y/((3μ+H)σ_vm³)) · s_trial ⊗ s_trial
        beta = 9.0 * mu ** 2 * sigma_y / ((3.0 * mu + H) * sigma_vm_trial ** 3)

        I_vol = I_vol_voigt(ss)
        I_dev = I_dev_voigt(ss)

        return I_vol @ C + theta * (I_dev @ C) - beta * jnp.outer(s_trial, s_trial)
