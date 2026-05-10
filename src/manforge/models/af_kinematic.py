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
f(state) = σ_vm(state["stress"] − state["alpha"]) − σ_y0

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

Dimensional notes
-----------------
* **3D** (AFKinematic3D): α is a 6-component deviatoric tensor.  The formula
  above applies directly; (2/3)·(3/2) = 1 is exact.
* **1D** (AFKinematic1D): α is the *effective backstress* (yield surface
  centre): |σ−α| = σ_y0.  The update uses n_voigt = ξ/|ξ| = sign(σ−α)
  instead of ŝ = (2/3)·n_voigt, reproducing the same hardening slope C_k as
  the 3D model under uniaxial loading.
* **PS** (AFKinematicPS): α stores the three PS components [α11, α22, α12].
  Von Mises is evaluated on the 6-component lift ξ_3D using the deviatoric
  identity α33 = −(α11+α22), so the yield surface and flow direction match
  the 3D model under uniaxial loading.

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
from manforge.core.state import Explicit, NTENS, SCALAR
from manforge.core.dimension import SOLID_3D, PLANE_STRESS, UNIAXIAL_1D, StressDimension


class AFKinematic3D(MaterialModel3D):
    """J2 + Armstrong-Frederick kinematic hardening for full-rank stress states.

    Uses the scalar NR path (all non-stress states explicit via ``update_state``).
    Uses the generic NR + autodiff return-mapping path (no analytical hooks).

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
        Dynamic recovery parameter.
    """

    param_names = ["E", "nu", "sigma_y0", "C_k", "gamma"]
    alpha = Explicit(shape=NTENS, doc="backstress tensor")
    ep = Explicit(shape=SCALAR, doc="equivalent plastic strain")

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

    def update_state(self, dlambda, state_n, state_trial) -> list:
        """Armstrong-Frederick backstress update (implicit, analytical).

        α_{n+1} = (α_n + C_k Δλ ŝ) / (1 + γ Δλ)

        where ŝ = dev(σ − α_n) / σ_vm(σ − α_n).
        """
        alpha_n = state_n["alpha"]
        xi = state_trial["stress"] - alpha_n
        s_xi = self.dev(xi)
        vm_safe = self.vonmises(xi)
        s_hat = s_xi / vm_safe
        alpha_new = (alpha_n + self.C_k * dlambda * s_hat) / (1.0 + self.gamma * dlambda)
        return [self.alpha(alpha_new), self.ep(state_n["ep"] + dlambda)]


class AFKinematicPS(MaterialModelPS):
    """J2 + Armstrong-Frederick kinematic hardening for plane-stress elements.

    Uses the scalar NR path (all non-stress states explicit via ``update_state``).
    Uses the generic NR + autodiff return-mapping path.

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
        Dynamic recovery parameter.
    """

    param_names = ["E", "nu", "sigma_y0", "C_k", "gamma"]
    alpha = Explicit(shape=NTENS, doc="backstress tensor")
    ep = Explicit(shape=SCALAR, doc="equivalent plastic strain")

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

        Lifts ξ = σ − α to 6 components using the deviatoric identity
        α33 = −(α11 + α22) before evaluating the von Mises norm.  This gives
        the same yield surface as the 3D model under uniaxial loading.
        """
        xi6 = self.lift_kin_to_3d(state["stress"], state["alpha"])
        return self.vonmises_kin(xi6) - self.sigma_y0

    def update_state(self, dlambda, state_n, state_trial) -> list:
        """Armstrong-Frederick backstress update (PS, α33-aware).

        Computes ŝ = dev_3D(ξ_lifted) / σ_vm_3D(ξ_lifted) where the lifted
        vector uses α33 = −(α11+α22).  Only the stored PS components [0,1,2]
        of the 6-component ŝ are applied to update α.

        This reproduces the same (2/3)·(3/2)=1 cancellation as the 3D model.

        α_{n+1} = (α_n + C_k Δλ ŝ_ps) / (1 + γ Δλ)
        """
        alpha_n = state_n["alpha"]
        stress_trial = state_trial["stress"]
        xi6 = self.lift_kin_to_3d(stress_trial, alpha_n)
        vm_safe = self.vonmises_kin(xi6)
        # dev of xi6
        p = (xi6[0] + xi6[1] + xi6[2]) / 3.0
        delta6 = anp.array([1.0, 1.0, 1.0, 0.0, 0.0, 0.0])
        s_xi6 = xi6 - p * delta6
        s_hat6 = s_xi6 / vm_safe           # (2/3)*n_voigt in 3D convention
        # PS Voigt = [σ11, σ22, σ12] → 3D indices [0, 1, 3]
        s_hat_ps = anp.array([s_hat6[0], s_hat6[1], s_hat6[3]])
        alpha_new = (alpha_n + self.C_k * dlambda * s_hat_ps) / (1.0 + self.gamma * dlambda)
        return [self.alpha(alpha_new), self.ep(state_n["ep"] + dlambda)]


class AFKinematic1D(MaterialModel1D):
    """J2 + Armstrong-Frederick kinematic hardening for uniaxial elements.

    Uses the scalar NR path (all non-stress states explicit via ``update_state``).
    Uses the generic NR + autodiff return-mapping path.

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
        Dynamic recovery parameter.
    """

    param_names = ["E", "nu", "sigma_y0", "C_k", "gamma"]
    alpha = Explicit(shape=NTENS, doc="backstress tensor")
    ep = Explicit(shape=SCALAR, doc="equivalent plastic strain")

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

    def update_state(self, dlambda, state_n, state_trial) -> list:
        """Armstrong-Frederick backstress update (1D).

        α is interpreted as the *effective* backstress (yield: |σ−α|=σ_y0).
        The flow direction for 1D is n_voigt = ∂f/∂σ = sign(σ−α), so the
        (2/3)·(3/2)=1 cancellation that holds in 3D is reproduced here by
        using n_voigt directly instead of s_hat = (2/3)·n_voigt.

        α_{n+1} = (α_n + C_k Δλ n_voigt) / (1 + γ Δλ)
        """
        alpha_n = state_n["alpha"]
        xi = state_trial["stress"] - alpha_n
        vm_safe = self.vonmises(xi)
        n_voigt = xi / vm_safe  # = sign(σ−α) for 1D
        alpha_new = (alpha_n + self.C_k * dlambda * n_voigt) / (1.0 + self.gamma * dlambda)
        return [self.alpha(alpha_new), self.ep(state_n["ep"] + dlambda)]
