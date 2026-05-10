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

Dimensional notes
-----------------
* **3D** (OWKinematic3D): α is a 6-component deviatoric tensor.  The formula
  applies directly; (2/3)·(3/2) = 1 is exact and ||α|| is the 3D von Mises
  norm of α.
* **1D** (OWKinematic1D): α is the *effective backstress* (yield: |σ−α|=σ_y0).
  The residual uses n_voigt = ξ/|ξ| and ||α|| = |α| (1D effective norm).
* **PS** (OWKinematicPS): α stores [α11, α22, α12].  Both ŝ and ||α|| are
  evaluated via a 3D lift using α33 = −(α11+α22), matching the 3D model
  under uniaxial loading.

Notes
-----
* gamma=0 reduces to Prager's linear kinematic hardening rule with modulus C_k
  (same limit as standard AF).
* The plastic increment is always solved via the vector NR using
  ``state_residual``.
"""

import autograd.numpy as anp

from manforge.core.material import MaterialModel1D, MaterialModel3D, MaterialModelPS
from manforge.core.state import Implicit, NTENS, SCALAR
from manforge.core.dimension import SOLID_3D, PLANE_STRESS, UNIAXIAL_1D, StressDimension


class OWKinematic3D(MaterialModel3D):
    """J2 + Ohno-Wang kinematic hardening for full-rank stress states.

    Inherits operator methods from :class:`~manforge.core.material.MaterialModel3D`.
    Uses the vector NR path because the backward-Euler discretisation of the
    Ohno-Wang evolution equation is genuinely implicit.  σ is declared as
    ``Implicit`` to include it as an NR unknown.

    Parameters
    ----------
    dimension : StressDimension, optional
        Must satisfy ``dimension.ndi == dimension.ndi_phys``.
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
    ep = Implicit(shape=SCALAR, doc="equivalent plastic strain")

    def __init__(self, dimension: StressDimension = SOLID_3D, *,
                 E: float, nu: float, sigma_y0: float, C_k: float, gamma: float):
        super().__init__(dimension)
        self.E = E
        self.nu = nu
        self.sigma_y0 = sigma_y0
        self.C_k = C_k
        self.gamma = gamma

    def yield_function(self, state) -> anp.ndarray:
        """J2 yield function in relative stress space: f = σ_vm(σ − α) − σ_y0."""
        xi = state["stress"] - state["alpha"]
        return self.vonmises(xi) - self.sigma_y0

    def state_residual(self, state_new, dlambda, state_n, state_trial, *, stress_trial) -> list:
        """Ohno-Wang implicit backstress residual.

        R_α = α_{n+1} − α_n − C_k Δλ n̂ + γ Δλ ‖α_{n+1}‖ α_{n+1} = 0

        The flow direction ``n̂`` is evaluated at the *new* relative stress
        ``σ − α_{n+1}`` for full backward-Euler consistency.
        """
        alpha_new = state_new["alpha"]
        stress_new = state_new["stress"]

        xi = stress_new - alpha_new
        s_xi = self.dev(xi)
        vm_safe = self.vonmises(xi)
        s_hat = s_xi / vm_safe

        alpha_vm = self.vonmises(alpha_new)

        R_stress = self.default_stress_residual(state_new, dlambda, stress_trial)
        R_alpha = (
            alpha_new
            - state_n["alpha"]
            - self.C_k * dlambda * s_hat
            + self.gamma * dlambda * alpha_vm * alpha_new
        )
        R_ep = state_new["ep"] - state_n["ep"] - dlambda
        return [self.stress(R_stress), self.alpha(R_alpha), self.ep(R_ep)]


class OWKinematicPS(MaterialModelPS):
    """J2 + Ohno-Wang kinematic hardening for plane-stress elements.

    Uses the vector NR path (stress and all states implicit).

    Parameters
    ----------
    dimension : StressDimension, optional
        Must satisfy ``dimension.is_plane_stress``.
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
    ep = Implicit(shape=SCALAR, doc="equivalent plastic strain")

    def __init__(self, dimension: StressDimension = PLANE_STRESS, *,
                 E: float, nu: float, sigma_y0: float, C_k: float, gamma: float):
        super().__init__(dimension)
        self.E = E
        self.nu = nu
        self.sigma_y0 = sigma_y0
        self.C_k = C_k
        self.gamma = gamma

    def yield_function(self, state) -> anp.ndarray:
        """J2 yield function with α33-aware von Mises: f = σ_vm_3D(σ − α) − σ_y0.

        Lifts ξ = σ − α to 6 components using α33 = −(α11 + α22) before
        evaluating the von Mises norm.
        """
        xi6 = self.lift_kin_to_3d(state["stress"], state["alpha"])
        return self.vonmises_kin(xi6) - self.sigma_y0

    def state_residual(self, state_new, dlambda, state_n, state_trial, *, stress_trial) -> list:
        """Ohno-Wang implicit backstress residual (PS, α33-aware).

        Computes ŝ_ps = dev_3D(ξ_lifted)_ps / σ_vm_3D, where [0,1,3] of the
        6-component deviatoric are extracted (PS Voigt = [σ11, σ22, σ12]).
        ||α|| uses the 3D deviatoric lift α33 = −(α11+α22).
        """
        alpha_new = state_new["alpha"]
        stress_new = state_new["stress"]
        xi6 = self.lift_kin_to_3d(stress_new, alpha_new)
        vm_safe = self.vonmises_kin(xi6)
        p = (xi6[0] + xi6[1] + xi6[2]) / 3.0
        delta6 = anp.array([1.0, 1.0, 1.0, 0.0, 0.0, 0.0])
        s_xi6 = xi6 - p * delta6
        s_hat6 = s_xi6 / vm_safe
        # PS Voigt = [σ11, σ22, σ12] → 3D indices [0, 1, 3]
        s_hat_ps = anp.array([s_hat6[0], s_hat6[1], s_hat6[3]])

        # ||α||_3D: lift α to 6 components then take von Mises
        alpha6 = anp.array([
            alpha_new[0], alpha_new[1], -(alpha_new[0] + alpha_new[1]),
            alpha_new[2], 0.0, 0.0,
        ])
        alpha_vm = self.vonmises_kin(alpha6)

        R_stress = self.default_stress_residual(state_new, dlambda, stress_trial)
        R_alpha = (
            alpha_new
            - state_n["alpha"]
            - self.C_k * dlambda * s_hat_ps
            + self.gamma * dlambda * alpha_vm * alpha_new
        )
        R_ep = state_new["ep"] - state_n["ep"] - dlambda
        return [self.stress(R_stress), self.alpha(R_alpha), self.ep(R_ep)]


class OWKinematic1D(MaterialModel1D):
    """J2 + Ohno-Wang kinematic hardening for uniaxial elements.

    Uses the vector NR path (stress and all states implicit).

    Parameters
    ----------
    dimension : StressDimension, optional
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
    ep = Implicit(shape=SCALAR, doc="equivalent plastic strain")

    def __init__(self, dimension: StressDimension = UNIAXIAL_1D, *,
                 E: float, nu: float, sigma_y0: float, C_k: float, gamma: float):
        super().__init__(dimension)
        self.E = E
        self.nu = nu
        self.sigma_y0 = sigma_y0
        self.C_k = C_k
        self.gamma = gamma

    def yield_function(self, state) -> anp.ndarray:
        """J2 yield function in relative stress space: f = σ_vm(σ − α) − σ_y0."""
        xi = state["stress"] - state["alpha"]
        return self.vonmises(xi) - self.sigma_y0

    def state_residual(self, state_new, dlambda, state_n, state_trial, *, stress_trial) -> list:
        """Ohno-Wang implicit backstress residual (1D).

        α is interpreted as the *effective* backstress (yield: |σ−α|=σ_y0).
        The flow direction n_voigt = ξ/|ξ| = sign(σ−α) is used directly so
        that the (2/3)·(3/2)=1 cancellation matches the 3D formulation.
        ||α|| for the recovery term is |α| (absolute value, 1D effective norm).
        """
        alpha_new = state_new["alpha"]
        stress_new = state_new["stress"]
        xi = stress_new - alpha_new
        vm_safe = self.vonmises(xi)
        n_voigt = xi / vm_safe  # sign(σ−α) for 1D
        alpha_vm = self.vonmises(alpha_new)  # |α| for 1D
        R_stress = self.default_stress_residual(state_new, dlambda, stress_trial)
        R_alpha = (
            alpha_new
            - state_n["alpha"]
            - self.C_k * dlambda * n_voigt
            + self.gamma * dlambda * alpha_vm * alpha_new
        )
        R_ep = state_new["ep"] - state_n["ep"] - dlambda
        return [self.stress(R_stress), self.alpha(R_alpha), self.ep(R_ep)]
