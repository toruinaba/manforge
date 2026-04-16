"""J2 plasticity with Ohno-Wang modified Armstrong-Frederick kinematic hardening.

This model is the canonical example of a *genuinely implicit* hardening law
in manforge.  Unlike the standard Armstrong-Frederick model, the backward-Euler
discretization of the Ohno-Wang evolution equation cannot be solved in closed
form, so the model overrides ``hardening_residual`` and triggers the augmented
(ntens+1+n_state) residual system automatically.

Model parameters
----------------
E        : Young's modulus
nu       : Poisson's ratio
sigma_y0 : Initial yield stress (constant — no isotropic hardening)
C_k      : Kinematic hardening modulus
gamma    : Dynamic recovery parameter (Ohno-Wang nonlinearity)

Internal state
--------------
alpha : backstress tensor (shape (ntens,))
ep    : equivalent plastic strain (scalar, ≥ 0)

Yield function
--------------
f(σ, α) = σ_vm(σ − α) − σ_y0

Ohno-Wang backstress evolution
-------------------------------
Standard Armstrong-Frederick has linear dynamic recovery:

    dα = C_k dp n̂ − γ dp α

The Ohno-Wang variant weights the recovery by the current backstress norm:

    dα = C_k dp n̂ − γ dp ‖α‖ α

Physically, this gives softer dynamic recovery when ‖α‖ is small and stronger
saturation behaviour when ‖α‖ is large.

Backward-Euler discretisation of the OW equation:

    α_{n+1} = α_n + C_k Δλ n̂ − γ Δλ ‖α_{n+1}‖ α_{n+1}

Rearranged as a residual (the form used by ``hardening_residual``):

    R_α = α_{n+1} − α_n − C_k Δλ n̂ + γ Δλ ‖α_{n+1}‖ α_{n+1} = 0

Because ‖α_{n+1}‖ appears nonlinearly, this cannot be solved for α_{n+1} in
closed form.  Setting ``hardening_type = 'implicit'`` activates the augmented
Newton-Raphson path and the augmented consistent tangent automatically.

Saturation backstress
---------------------
Under monotonic loading the steady-state condition dα/dp = 0 gives:

    C_k n̂ = γ ‖α_sat‖ α_sat  →  ‖α_sat‖_vm = √(C_k / γ)

This differs from the standard AF saturation ‖α_sat‖ = C_k / γ.  To obtain
the same saturation backstress amplitude with both models, choose:

    γ_OW = C_k / ‖α_sat‖²   vs   γ_AF = C_k / ‖α_sat‖

Flow direction consistency
--------------------------
The augmented residual system (``core/residual.py``) computes the flow
direction as ``n = ∂f/∂σ`` evaluated at ``(σ, state_new)``.  For this model
``f = σ_vm(σ − α)``, so ``n`` depends on ``α_{n+1}``.  For full backward-Euler
consistency, ``hardening_residual`` also evaluates ``n̂`` from the relative
stress ``σ − α_{n+1}`` (using ``state_new["alpha"]``), making the entire
system fully coupled and implicit.

Notes
-----
* gamma=0 reduces to Prager's linear kinematic hardening rule with modulus C_k
  (same limit as standard AF).
* The plastic increment is always solved via the augmented NR using
  ``hardening_residual``.  No ``hardening_increment`` is defined because
  ``hardening_type = 'implicit'`` does not require one.
"""

import jax.numpy as jnp

from manforge.core.material import MaterialModel1D, MaterialModel3D, MaterialModelPS


class OWKinematic3D(MaterialModel3D):
    """J2 + Ohno-Wang kinematic hardening for full-rank stress states.

    Inherits operator methods from :class:`~manforge.core.material.MaterialModel3D`.
    Uses the augmented residual path (``hardening_type = 'implicit'``) because the
    backward-Euler discretisation of the Ohno-Wang evolution equation is genuinely
    implicit.

    Parameters
    ----------
    stress_state : StressState, optional
        Must satisfy ``stress_state.ndi == stress_state.ndi_phys``.
        Defaults to ``SOLID_3D``.
    """

    hardening_type = "implicit"
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

    def hardening_residual(self, state_new, dlambda, stress, state_n, params) -> dict:
        """Ohno-Wang implicit backstress residual.

        R_α = α_{n+1} − α_n − C_k Δλ n̂ + γ Δλ ‖α_{n+1}‖ α_{n+1} = 0

        The flow direction ``n̂`` is evaluated at the *new* relative stress
        ``σ − α_{n+1}`` for full backward-Euler consistency with the flow
        direction used in the stress residual equation.

        Required because ``hardening_type = 'implicit'``, which activates the
        augmented (ntens+1+n_state) Newton-Raphson solver and the augmented
        consistent tangent automatically.
        """
        alpha_new = state_new["alpha"]

        # Flow direction from new relative stress (fully implicit backward-Euler)
        xi = stress - alpha_new
        s_xi = self._dev(xi)
        vm_safe = self._vonmises(xi)
        n_hat = s_xi / vm_safe

        alpha_vm = self._vonmises(alpha_new)

        R_alpha = (
            alpha_new
            - state_n["alpha"]
            - params["C_k"] * dlambda * n_hat
            + params["gamma"] * dlambda * alpha_vm * alpha_new
        )
        R_ep = state_new["ep"] - state_n["ep"] - dlambda
        return {"alpha": R_alpha, "ep": R_ep}


class OWKinematicPS(MaterialModelPS):
    """J2 + Ohno-Wang kinematic hardening for plane-stress elements.

    Inherits operator methods from
    :class:`~manforge.core.material.MaterialModelPS` (including the
    missing-component correction in ``_vonmises``).

    Uses the augmented residual path (``hardening_type = 'implicit'``).

    Parameters
    ----------
    stress_state : StressState, optional
        Must satisfy ``stress_state.is_plane_stress``.
        Defaults to ``PLANE_STRESS``.
    """

    hardening_type = "implicit"
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

    def hardening_residual(self, state_new, dlambda, stress, state_n, params) -> dict:
        """Ohno-Wang implicit backstress residual (plane stress)."""
        alpha_new = state_new["alpha"]
        xi = stress - alpha_new
        s_xi = self._dev(xi)
        vm_safe = self._vonmises(xi)
        n_hat = s_xi / vm_safe
        alpha_vm = self._vonmises(alpha_new)
        R_alpha = (
            alpha_new
            - state_n["alpha"]
            - params["C_k"] * dlambda * n_hat
            + params["gamma"] * dlambda * alpha_vm * alpha_new
        )
        R_ep = state_new["ep"] - state_n["ep"] - dlambda
        return {"alpha": R_alpha, "ep": R_ep}


class OWKinematic1D(MaterialModel1D):
    """J2 + Ohno-Wang kinematic hardening for uniaxial elements.

    Inherits operator methods from
    :class:`~manforge.core.material.MaterialModel1D`.

    Uses the augmented residual path (``hardening_type = 'implicit'``).

    Parameters
    ----------
    stress_state : StressState, optional
        Must have ``ntens == 1``.  Defaults to ``UNIAXIAL_1D``.
    """

    hardening_type = "implicit"
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

    def hardening_residual(self, state_new, dlambda, stress, state_n, params) -> dict:
        """Ohno-Wang implicit backstress residual (1D)."""
        alpha_new = state_new["alpha"]
        xi = stress - alpha_new
        s_xi = self._dev(xi)
        vm_safe = self._vonmises(xi)
        n_hat = s_xi / vm_safe
        alpha_vm = self._vonmises(alpha_new)
        R_alpha = (
            alpha_new
            - state_n["alpha"]
            - params["C_k"] * dlambda * n_hat
            + params["gamma"] * dlambda * alpha_vm * alpha_new
        )
        R_ep = state_new["ep"] - state_n["ep"] - dlambda
        return {"alpha": R_alpha, "ep": R_ep}
