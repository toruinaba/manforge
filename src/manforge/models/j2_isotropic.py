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
from manforge.core.material import MaterialModel


class J2IsotropicHardening(MaterialModel):
    """J2 plasticity model with linear isotropic hardening.

    Parameters
    ----------
    (none at construction — all parameters are passed via ``params`` dicts)
    """

    param_names = ["E", "nu", "sigma_y0", "H"]
    state_names = ["ep"]  # equivalent plastic strain

    def elastic_stiffness(self, params: dict) -> jnp.ndarray:
        """Isotropic elastic stiffness tensor.

        Parameters
        ----------
        params : dict
            Must contain keys ``E`` and ``nu``.

        Returns
        -------
        jnp.ndarray, shape (6, 6)
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
        stress : jnp.ndarray, shape (6,)
        state : dict with key ``ep``
        params : dict with keys ``sigma_y0``, ``H``

        Returns
        -------
        jnp.ndarray, scalar
        """
        ep = state["ep"]
        sigma_y = params["sigma_y0"] + params["H"] * ep
        return vonmises(stress) - sigma_y

    def hardening_increment(
        self,
        dlambda: jnp.ndarray,
        state: dict,
        params: dict,
    ) -> dict:
        """Update equivalent plastic strain.

        For J2 with associative flow, Δep = Δλ (von Mises consistency).

        Parameters
        ----------
        dlambda : jnp.ndarray, scalar
            Plastic multiplier increment.
        state : dict with key ``ep``
        params : dict (unused here, kept for interface consistency)

        Returns
        -------
        dict
            ``{"ep": ep + dlambda}``
        """
        return {"ep": state["ep"] + dlambda}
