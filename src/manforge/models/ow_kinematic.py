"""J2 plasticity with Ohno-Wang modified Armstrong-Frederick kinematic hardening.

This model is the canonical example of a *genuinely implicit* hardening law
in manforge.  Unlike the standard Armstrong-Frederick model, the backward-Euler
discretization of the Ohno-Wang evolution equation cannot be solved in closed
form, so the model overrides ``state_residual`` and triggers the vector NR path
automatically.

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
f(state) = σ_vm(state["stress"] − state["alpha"]) − σ_y0

Ohno-Wang backstress evolution
-------------------------------
Standard Armstrong-Frederick has linear dynamic recovery:

    dα = C_k dp n̂ − γ dp α

The Ohno-Wang variant weights the recovery by the current backstress norm:

    dα = C_k dp n̂ − γ dp ‖α‖ α

Backward-Euler discretisation of the OW equation:

    α_{n+1} = α_n + C_k Δλ n̂ − γ Δλ ‖α_{n+1}‖ α_{n+1}

Rearranged as a residual (the form used by ``state_residual``):

    R_α = α_{n+1} − α_n − C_k Δλ n̂ + γ Δλ ‖α_{n+1}‖ α_{n+1} = 0

Because ‖α_{n+1}‖ appears nonlinearly, this cannot be solved for α_{n+1} in
closed form.  Declaring ``stress`` as ``Implicit`` activates the vector
Newton-Raphson path (σ included as NR unknown) and the correct consistent
tangent automatically.

Notes
-----
* gamma=0 reduces to Prager's linear kinematic hardening rule with modulus C_k
  (same limit as standard AF).
* The plastic increment is always solved via the vector NR using
  ``state_residual``.
"""

import autograd.numpy as anp

from manforge.core.material import MaterialModel1D, MaterialModel3D, MaterialModelPS
from manforge.core.state import Implicit, NTENS
from manforge.core.dimension import SOLID_3D, PLANE_STRESS, UNIAXIAL_1D, StressState


class OWKinematic3D(MaterialModel3D):
    """J2 + Ohno-Wang kinematic hardening for full-rank stress states.

    Inherits operator methods from :class:`~manforge.core.material.MaterialModel3D`.
    Uses the vector NR path because the backward-Euler discretisation of the
    Ohno-Wang evolution equation is genuinely implicit.  σ is declared as
    ``Implicit`` to include it as an NR unknown.

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
        Dynamic recovery parameter (Ohno-Wang nonlinearity).
    """

    stress = Implicit(shape=NTENS, doc="Cauchy stress")
    param_names = ["E", "nu", "sigma_y0", "C_k", "gamma"]
    alpha = Implicit(shape=NTENS, doc="backstress tensor")
    ep = Implicit(shape=(), doc="equivalent plastic strain")

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

    def state_residual(self, state_new, dlambda, state_n, state_trial) -> list:
        """Ohno-Wang implicit backstress residual.

        R_α = α_{n+1} − α_n − C_k Δλ n̂ + γ Δλ ‖α_{n+1}‖ α_{n+1} = 0

        The flow direction ``n̂`` is evaluated at the *new* relative stress
        ``σ − α_{n+1}`` for full backward-Euler consistency.
        """
        alpha_new = state_new["alpha"]
        stress_new = state_new["stress"]

        xi = stress_new - alpha_new
        s_xi = self._dev(xi)
        vm_safe = self._vonmises(xi)
        n_hat = s_xi / vm_safe

        alpha_vm = self._vonmises(alpha_new)

        R_stress = self.default_stress_residual(state_new, dlambda, state_trial)
        R_alpha = (
            alpha_new
            - state_n["alpha"]
            - self.C_k * dlambda * n_hat
            + self.gamma * dlambda * alpha_vm * alpha_new
        )
        R_ep = state_new["ep"] - state_n["ep"] - dlambda
        return [self.stress(R_stress), self.alpha(R_alpha), self.ep(R_ep)]


class OWKinematicPS(MaterialModelPS):
    """J2 + Ohno-Wang kinematic hardening for plane-stress elements.

    Uses the vector NR path (stress and all states implicit).

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
        Dynamic recovery parameter (Ohno-Wang nonlinearity).
    """

    stress = Implicit(shape=NTENS, doc="Cauchy stress")
    param_names = ["E", "nu", "sigma_y0", "C_k", "gamma"]
    alpha = Implicit(shape=NTENS, doc="backstress tensor")
    ep = Implicit(shape=(), doc="equivalent plastic strain")

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

    def state_residual(self, state_new, dlambda, state_n, state_trial) -> list:
        """Ohno-Wang implicit backstress residual (plane stress)."""
        alpha_new = state_new["alpha"]
        stress_new = state_new["stress"]
        xi = stress_new - alpha_new
        s_xi = self._dev(xi)
        vm_safe = self._vonmises(xi)
        n_hat = s_xi / vm_safe
        alpha_vm = self._vonmises(alpha_new)
        R_stress = self.default_stress_residual(state_new, dlambda, state_trial)
        R_alpha = (
            alpha_new
            - state_n["alpha"]
            - self.C_k * dlambda * n_hat
            + self.gamma * dlambda * alpha_vm * alpha_new
        )
        R_ep = state_new["ep"] - state_n["ep"] - dlambda
        return [self.stress(R_stress), self.alpha(R_alpha), self.ep(R_ep)]


class OWKinematic1D(MaterialModel1D):
    """J2 + Ohno-Wang kinematic hardening for uniaxial elements.

    Uses the vector NR path (stress and all states implicit).

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
        Dynamic recovery parameter (Ohno-Wang nonlinearity).
    """

    stress = Implicit(shape=NTENS, doc="Cauchy stress")
    param_names = ["E", "nu", "sigma_y0", "C_k", "gamma"]
    alpha = Implicit(shape=NTENS, doc="backstress tensor")
    ep = Implicit(shape=(), doc="equivalent plastic strain")

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

    def state_residual(self, state_new, dlambda, state_n, state_trial) -> list:
        """Ohno-Wang implicit backstress residual (1D)."""
        alpha_new = state_new["alpha"]
        stress_new = state_new["stress"]
        xi = stress_new - alpha_new
        s_xi = self._dev(xi)
        vm_safe = self._vonmises(xi)
        n_hat = s_xi / vm_safe
        alpha_vm = self._vonmises(alpha_new)
        R_stress = self.default_stress_residual(state_new, dlambda, state_trial)
        R_alpha = (
            alpha_new
            - state_n["alpha"]
            - self.C_k * dlambda * n_hat
            + self.gamma * dlambda * alpha_vm * alpha_new
        )
        R_ep = state_new["ep"] - state_n["ep"] - dlambda
        return [self.stress(R_stress), self.alpha(R_alpha), self.ep(R_ep)]
