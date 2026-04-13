"""J2 plasticity with Armstrong-Frederick nonlinear kinematic hardening.

Model parameters
----------------
E        : Young's modulus
nu       : Poisson's ratio
sigma_y0 : Initial yield stress (constant — no isotropic hardening)
C_k      : Kinematic hardening modulus
gamma    : Dynamic recovery parameter (nonlinearity; gamma=0 → Prager linear rule)

Internal state
--------------
alpha : backstress tensor (shape (ntens,))
ep    : equivalent plastic strain (scalar, ≥ 0)

Yield function
--------------
f(σ, α) = σ_vm(σ − α) − σ_y0

where σ_vm is the von Mises equivalent stress of the *relative* stress.

Flow rule (associative)
-----------------------
dε_p = Δλ · n,  n = df/dσ = (3/2) (s_ξ / σ_vm(ξ)),  ξ = σ − α

Backstress evolution (Armstrong-Frederick, implicit backward Euler)
-------------------------------------------------------------------
Δα = (2/3) C_k Δλ n − γ Δλ α_{n+1}
   = C_k Δλ (s_ξ / σ_vm(ξ)) − γ Δλ α_{n+1}

where the (2/3) and (3/2) factors cancel.  Solving for α_{n+1}:

    α_{n+1} = (α_n + C_k Δλ ŝ) / (1 + γ Δλ)

with ŝ = dev(σ − α_n) / σ_vm(σ − α_n) evaluated at the beginning-of-step
backstress.  The generic NR in return_mapping iterates to self-consistency.

Notes
-----
* gamma=0 reduces to Prager's linear kinematic hardening rule with modulus C_k.
* The saturated backstress magnitude under monotonic loading is C_k / gamma.
* No analytical plastic_corrector or analytical_tangent is provided; the
  generic Newton-Raphson + JAX autodiff path is used, which is exactly what
  this model is designed to test.
"""

import jax.numpy as jnp

from manforge.core.material import MaterialModel3D, MaterialModelPS, MaterialModel1D
from manforge.utils.smooth import smooth_abs


class AFKinematic3D(MaterialModel3D):
    """J2 + Armstrong-Frederick kinematic hardening for full-rank stress states.

    Inherits operator methods from :class:`~manforge.core.material.MaterialModel3D`.
    Uses the generic NR + autodiff return-mapping path (no analytical hooks).

    Parameters
    ----------
    stress_state : StressState, optional
        Must satisfy ``stress_state.ndi == stress_state.ndi_phys``.
        Defaults to ``SOLID_3D``.
    """

    param_names = ["E", "nu", "sigma_y0", "C_k", "gamma"]
    state_names = ["alpha", "ep"]

    def initial_state(self) -> dict:
        """Return virgin state: zero backstress tensor and zero plastic strain."""
        return {
            "alpha": jnp.zeros(self.ntens),
            "ep": jnp.array(0.0),
        }

    def elastic_stiffness(self, params: dict) -> jnp.ndarray:
        """Isotropic elastic stiffness tensor."""
        E, nu = params["E"], params["nu"]
        mu = E / (2.0 * (1.0 + nu))
        lam = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
        return self.isotropic_C(lam, mu)

    def yield_function(self, stress: jnp.ndarray, state: dict, params: dict) -> jnp.ndarray:
        """J2 yield function in relative stress space: f = σ_vm(σ − α) − σ_y0."""
        xi = stress - state["alpha"]
        return self._vonmises(xi) - params["sigma_y0"]

    def hardening_increment(self, dlambda, stress, state, params) -> dict:
        """Armstrong-Frederick backstress update (implicit, analytical).

        α_{n+1} = (α_n + C_k Δλ ŝ) / (1 + γ Δλ)

        where ŝ = dev(σ − α_n) / σ_vm(σ − α_n).
        """
        alpha_n = state["alpha"]
        xi = stress - alpha_n
        s_xi = self._dev(xi)
        vm_safe = smooth_abs(self._vonmises(xi))
        n_hat = s_xi / vm_safe
        alpha_new = (alpha_n + params["C_k"] * dlambda * n_hat) / (1.0 + params["gamma"] * dlambda)
        return {"alpha": alpha_new, "ep": state["ep"] + dlambda}


class AFKinematicPS(MaterialModelPS):
    """J2 + Armstrong-Frederick kinematic hardening for plane-stress elements.

    Inherits operator methods from
    :class:`~manforge.core.material.MaterialModelPS` (including the
    missing-component correction in ``_vonmises``).

    Uses the generic NR + autodiff return-mapping path.

    Parameters
    ----------
    stress_state : StressState, optional
        Must satisfy ``stress_state.is_plane_stress``.
        Defaults to ``PLANE_STRESS``.
    """

    param_names = ["E", "nu", "sigma_y0", "C_k", "gamma"]
    state_names = ["alpha", "ep"]

    def initial_state(self) -> dict:
        """Return virgin state: zero backstress tensor and zero plastic strain."""
        return {
            "alpha": jnp.zeros(self.ntens),
            "ep": jnp.array(0.0),
        }

    def elastic_stiffness(self, params: dict) -> jnp.ndarray:
        """Plane-stress isotropic stiffness (3×3 condensed)."""
        E, nu = params["E"], params["nu"]
        mu = E / (2.0 * (1.0 + nu))
        lam = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
        return self.isotropic_C(lam, mu)

    def yield_function(self, stress: jnp.ndarray, state: dict, params: dict) -> jnp.ndarray:
        """J2 yield function in relative stress space: f = σ_vm(σ − α) − σ_y0."""
        xi = stress - state["alpha"]
        return self._vonmises(xi) - params["sigma_y0"]

    def hardening_increment(self, dlambda, stress, state, params) -> dict:
        """Armstrong-Frederick backstress update."""
        alpha_n = state["alpha"]
        xi = stress - alpha_n
        s_xi = self._dev(xi)
        vm_safe = smooth_abs(self._vonmises(xi))
        n_hat = s_xi / vm_safe
        alpha_new = (alpha_n + params["C_k"] * dlambda * n_hat) / (1.0 + params["gamma"] * dlambda)
        return {"alpha": alpha_new, "ep": state["ep"] + dlambda}


class AFKinematic1D(MaterialModel1D):
    """J2 + Armstrong-Frederick kinematic hardening for uniaxial elements.

    Inherits operator methods from
    :class:`~manforge.core.material.MaterialModel1D`.

    Uses the generic NR + autodiff return-mapping path.

    Parameters
    ----------
    stress_state : StressState, optional
        Must have ``ntens == 1``.  Defaults to ``UNIAXIAL_1D``.
    """

    param_names = ["E", "nu", "sigma_y0", "C_k", "gamma"]
    state_names = ["alpha", "ep"]

    def initial_state(self) -> dict:
        """Return virgin state: zero backstress tensor and zero plastic strain."""
        return {
            "alpha": jnp.zeros(self.ntens),
            "ep": jnp.array(0.0),
        }

    def elastic_stiffness(self, params: dict) -> jnp.ndarray:
        """1D elastic stiffness [[E]]."""
        E, nu = params["E"], params["nu"]
        mu = E / (2.0 * (1.0 + nu))
        lam = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
        return self.isotropic_C(lam, mu)

    def yield_function(self, stress: jnp.ndarray, state: dict, params: dict) -> jnp.ndarray:
        """J2 yield function in relative stress space: f = σ_vm(σ − α) − σ_y0."""
        xi = stress - state["alpha"]
        return self._vonmises(xi) - params["sigma_y0"]

    def hardening_increment(self, dlambda, stress, state, params) -> dict:
        """Armstrong-Frederick backstress update."""
        alpha_n = state["alpha"]
        xi = stress - alpha_n
        s_xi = self._dev(xi)
        vm_safe = smooth_abs(self._vonmises(xi))
        n_hat = s_xi / vm_safe
        alpha_new = (alpha_n + params["C_k"] * dlambda * n_hat) / (1.0 + params["gamma"] * dlambda)
        return {"alpha": alpha_new, "ep": state["ep"] + dlambda}
