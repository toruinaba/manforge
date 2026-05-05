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
f(state) = σ_vm(state.stress − state.alpha) − σ_y0

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
backstress.  The generic NR in stress_update iterates to self-consistency.

Notes
-----
* gamma=0 reduces to Prager's linear kinematic hardening rule with modulus C_k.
* The saturated backstress magnitude under monotonic loading is C_k / gamma.
* No user_defined_return_mapping or user_defined_tangent is provided; the
  generic numerical_newton + autodiff path is used, which is exactly what
  this model is designed to test.
"""

import autograd.numpy as anp

from manforge.core.material import MaterialModel3D, MaterialModelPS, MaterialModel1D
from manforge.core.state import Explicit, NTENS
from manforge.core.stress_state import SOLID_3D, PLANE_STRESS, UNIAXIAL_1D, StressState


class AFKinematic3D(MaterialModel3D):
    """J2 + Armstrong-Frederick kinematic hardening for full-rank stress states.

    Uses the scalar NR path (all non-stress states explicit via ``update_state``).
    Uses the generic NR + autodiff return-mapping path (no analytical hooks).

    Parameters
    ----------
    stress_state : StressState, optional
        Must satisfy ``stress_state.ndi == stress_state.ndi_phys``.
        Defaults to ``SOLID_3D``.
    E : float
        Young's modulus.
    nu : float
        Poisson's ratio.
    sigma_y0 : float
        Initial yield stress.
    C_k : float
        Kinematic hardening modulus.
    gamma : float
        Dynamic recovery parameter.
    """

    param_names = ["E", "nu", "sigma_y0", "C_k", "gamma"]
    alpha = Explicit(shape=NTENS, doc="backstress tensor")
    ep = Explicit(shape=(), doc="equivalent plastic strain")

    def __init__(self, stress_state: StressState = SOLID_3D, *,
                 E: float, nu: float, sigma_y0: float, C_k: float, gamma: float):
        super().__init__(stress_state)
        self.E = E
        self.nu = nu
        self.sigma_y0 = sigma_y0
        self.C_k = C_k
        self.gamma = gamma

    def yield_function(self, state) -> anp.ndarray:
        """J2 yield function in relative stress space: f = σ_vm(σ − α) − σ_y0."""
        xi = state["stress"] - state["alpha"]
        return self._vonmises(xi) - self.sigma_y0

    def update_state(self, dlambda, state_n, state_trial) -> list:
        """Armstrong-Frederick backstress update (implicit, analytical).

        α_{n+1} = (α_n + C_k Δλ ŝ) / (1 + γ Δλ)

        where ŝ = dev(σ − α_n) / σ_vm(σ − α_n).
        """
        stress = state_trial["stress"]
        alpha_n = state_n["alpha"]
        xi = stress - alpha_n
        s_xi = self._dev(xi)
        vm_safe = self._vonmises(xi)
        n_hat = s_xi / vm_safe
        alpha_new = (alpha_n + self.C_k * dlambda * n_hat) / (1.0 + self.gamma * dlambda)
        return [self.alpha(alpha_new), self.ep(state_n["ep"] + dlambda)]


class AFKinematicPS(MaterialModelPS):
    """J2 + Armstrong-Frederick kinematic hardening for plane-stress elements.

    Uses the scalar NR path (all non-stress states explicit via ``update_state``).
    Uses the generic NR + autodiff return-mapping path.

    Parameters
    ----------
    stress_state : StressState, optional
        Must satisfy ``stress_state.is_plane_stress``.
        Defaults to ``PLANE_STRESS``.
    E : float
        Young's modulus.
    nu : float
        Poisson's ratio.
    sigma_y0 : float
        Initial yield stress.
    C_k : float
        Kinematic hardening modulus.
    gamma : float
        Dynamic recovery parameter.
    """

    param_names = ["E", "nu", "sigma_y0", "C_k", "gamma"]
    alpha = Explicit(shape=NTENS, doc="backstress tensor")
    ep = Explicit(shape=(), doc="equivalent plastic strain")

    def __init__(self, stress_state: StressState = PLANE_STRESS, *,
                 E: float, nu: float, sigma_y0: float, C_k: float, gamma: float):
        super().__init__(stress_state)
        self.E = E
        self.nu = nu
        self.sigma_y0 = sigma_y0
        self.C_k = C_k
        self.gamma = gamma

    def yield_function(self, state) -> anp.ndarray:
        """J2 yield function in relative stress space: f = σ_vm(σ − α) − σ_y0."""
        xi = state["stress"] - state["alpha"]
        return self._vonmises(xi) - self.sigma_y0

    def update_state(self, dlambda, state_n, state_trial) -> list:
        """Armstrong-Frederick backstress update."""
        stress = state_trial["stress"]
        alpha_n = state_n["alpha"]
        xi = stress - alpha_n
        s_xi = self._dev(xi)
        vm_safe = self._vonmises(xi)
        n_hat = s_xi / vm_safe
        alpha_new = (alpha_n + self.C_k * dlambda * n_hat) / (1.0 + self.gamma * dlambda)
        return [self.alpha(alpha_new), self.ep(state_n["ep"] + dlambda)]


class AFKinematic1D(MaterialModel1D):
    """J2 + Armstrong-Frederick kinematic hardening for uniaxial elements.

    Uses the scalar NR path (all non-stress states explicit via ``update_state``).
    Uses the generic NR + autodiff return-mapping path.

    Parameters
    ----------
    stress_state : StressState, optional
        Must have ``ntens == 1``.  Defaults to ``UNIAXIAL_1D``.
    E : float
        Young's modulus.
    nu : float
        Poisson's ratio.
    sigma_y0 : float
        Initial yield stress.
    C_k : float
        Kinematic hardening modulus.
    gamma : float
        Dynamic recovery parameter.
    """

    param_names = ["E", "nu", "sigma_y0", "C_k", "gamma"]
    alpha = Explicit(shape=NTENS, doc="backstress tensor")
    ep = Explicit(shape=(), doc="equivalent plastic strain")

    def __init__(self, stress_state: StressState = UNIAXIAL_1D, *,
                 E: float, nu: float, sigma_y0: float, C_k: float, gamma: float):
        super().__init__(stress_state)
        self.E = E
        self.nu = nu
        self.sigma_y0 = sigma_y0
        self.C_k = C_k
        self.gamma = gamma

    def yield_function(self, state) -> anp.ndarray:
        """J2 yield function in relative stress space: f = σ_vm(σ − α) − σ_y0."""
        xi = state["stress"] - state["alpha"]
        return self._vonmises(xi) - self.sigma_y0

    def update_state(self, dlambda, state_n, state_trial) -> list:
        """Armstrong-Frederick backstress update."""
        stress = state_trial["stress"]
        alpha_n = state_n["alpha"]
        xi = stress - alpha_n
        s_xi = self._dev(xi)
        vm_safe = self._vonmises(xi)
        n_hat = s_xi / vm_safe
        alpha_new = (alpha_n + self.C_k * dlambda * n_hat) / (1.0 + self.gamma * dlambda)
        return [self.alpha(alpha_new), self.ep(state_n["ep"] + dlambda)]
