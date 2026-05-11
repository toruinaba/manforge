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
alpha : backstress tensor (shape (ntens,), deviatoric)
ep    : equivalent plastic strain (scalar, ≥ 0)

Yield function
--------------
f(state) = vonmises_norm(dev(σ) − α) − σ_y0

where α is deviatoric (tr α = 0) and vonmises_norm(s) = √(3/2 s:s).

Flow rule (associative)
-----------------------
n̂ = (3/2) dev(ξ) / vonmises_norm(dev(ξ)),  ξ_dev = dev(σ) − α
dε_p = Δλ · n̂

Backstress evolution (Armstrong-Frederick, original form, backward Euler)
--------------------------------------------------------------------------
α_{n+1} = (α_n + (2/3) C_k Δλ n̂) / (1 + γ Δλ)

The (2/3) factor combined with the (3/2) inside n̂ reproduces the classic
kinematic-hardening slope C_k under uniaxial loading in all dimensions.

Dimensional notes
-----------------
The three concrete classes share identical yield_function / update_state
implementations.  Each stress-state base class provides dimension-specific
dev / vonmises_norm that encapsulate per-dimension arithmetic.

* **3D** (AFKinematic3D): α is a 6-component deviatoric tensor.
* **PS** (AFKinematicPS): α stores [α11, α22, α12]; dev uses
  the stored PS components; vonmises_norm reconstructs α33 = −(α11+α22).
* **1D** (AFKinematic1D): α stores the deviatoric component α11_dev;
  dev(σ) = [(2/3)σ11], vonmises_norm gives (3/2)|s11_dev|.

Notes
-----
* gamma=0 reduces to Prager's linear kinematic hardening rule with modulus C_k.
* No user_defined_return_mapping or user_defined_tangent is provided; the
  generic numerical_newton + autodiff path is used.
"""

from manforge.core.material import MaterialModel3D, MaterialModelPS, MaterialModel1D
from manforge.core.state import Explicit, NTENS, SCALAR
from manforge.core.dimension import SOLID_3D, PLANE_STRESS, UNIAXIAL_1D, StressDimension


class AFKinematic3D(MaterialModel3D):
    """J2 + Armstrong-Frederick kinematic hardening for full-rank stress states.

    Uses the scalar NR path (all non-stress states explicit via ``update_state``).

    Parameters
    ----------
    dimension : StressDimension, optional
        Must satisfy ``dimension.ndi == dimension.ndi_phys``.
        Defaults to ``SOLID_3D``.
    E, nu, sigma_y0, C_k, gamma : float
        Material parameters.
    """

    param_names = ["E", "nu", "sigma_y0", "C_k", "gamma"]
    alpha = Explicit(shape=NTENS, doc="backstress tensor (deviatoric)")
    ep = Explicit(shape=SCALAR, doc="equivalent plastic strain")

    def __init__(self, dimension: StressDimension = SOLID_3D, *,
                 E: float, nu: float, sigma_y0: float, C_k: float, gamma: float):
        super().__init__(dimension)
        self.E = E
        self.nu = nu
        self.sigma_y0 = sigma_y0
        self.C_k = C_k
        self.gamma = gamma

    def yield_function(self, state):
        s_xi = self.dev(state["stress"]) - state["alpha"]
        return self.vonmises_norm(s_xi) - self.sigma_y0

    def update_state(self, dlambda, state_n, state_trial):
        alpha_n = state_n["alpha"]
        s_xi = self.dev(state_trial["stress"]) - alpha_n
        n_hat = 1.5 * s_xi / self.vonmises_norm(s_xi)
        alpha_new = (alpha_n + (2.0 / 3.0) * self.C_k * dlambda * n_hat) \
                  / (1.0 + self.gamma * dlambda)
        return [self.alpha(alpha_new), self.ep(state_n["ep"] + dlambda)]


class AFKinematicPS(MaterialModelPS):
    """J2 + Armstrong-Frederick kinematic hardening for plane-stress elements.

    Uses the scalar NR path (all non-stress states explicit via ``update_state``).

    Parameters
    ----------
    dimension : StressDimension, optional
        Must satisfy ``dimension.is_plane_stress``.
        Defaults to ``PLANE_STRESS``.
    E, nu, sigma_y0, C_k, gamma : float
        Material parameters.
    """

    param_names = ["E", "nu", "sigma_y0", "C_k", "gamma"]
    alpha = Explicit(shape=NTENS, doc="backstress tensor (deviatoric)")
    ep = Explicit(shape=SCALAR, doc="equivalent plastic strain")

    def __init__(self, dimension: StressDimension = PLANE_STRESS, *,
                 E: float, nu: float, sigma_y0: float, C_k: float, gamma: float):
        super().__init__(dimension)
        self.E = E
        self.nu = nu
        self.sigma_y0 = sigma_y0
        self.C_k = C_k
        self.gamma = gamma

    def yield_function(self, state):
        s_xi = self.dev(state["stress"]) - state["alpha"]
        return self.vonmises_norm(s_xi) - self.sigma_y0

    def update_state(self, dlambda, state_n, state_trial):
        alpha_n = state_n["alpha"]
        s_xi = self.dev(state_trial["stress"]) - alpha_n
        n_hat = 1.5 * s_xi / self.vonmises_norm(s_xi)
        alpha_new = (alpha_n + (2.0 / 3.0) * self.C_k * dlambda * n_hat) \
                  / (1.0 + self.gamma * dlambda)
        return [self.alpha(alpha_new), self.ep(state_n["ep"] + dlambda)]


class AFKinematic1D(MaterialModel1D):
    """J2 + Armstrong-Frederick kinematic hardening for uniaxial elements.

    Uses the scalar NR path (all non-stress states explicit via ``update_state``).

    The stored ``alpha`` is the deviatoric component α11_dev.  The effective
    backstress (yield-surface centre) is α_eff = (3/2)·α11_dev.

    Parameters
    ----------
    dimension : StressDimension, optional
        Must have ``ntens == 1``.  Defaults to ``UNIAXIAL_1D``.
    E, nu, sigma_y0, C_k, gamma : float
        Material parameters.
    """

    param_names = ["E", "nu", "sigma_y0", "C_k", "gamma"]
    alpha = Explicit(shape=NTENS, doc="backstress tensor (deviatoric component α11_dev)")
    ep = Explicit(shape=SCALAR, doc="equivalent plastic strain")

    def __init__(self, dimension: StressDimension = UNIAXIAL_1D, *,
                 E: float, nu: float, sigma_y0: float, C_k: float, gamma: float):
        super().__init__(dimension)
        self.E = E
        self.nu = nu
        self.sigma_y0 = sigma_y0
        self.C_k = C_k
        self.gamma = gamma

    def yield_function(self, state):
        s_xi = self.dev(state["stress"]) - state["alpha"]
        return self.vonmises_norm(s_xi) - self.sigma_y0

    def update_state(self, dlambda, state_n, state_trial):
        alpha_n = state_n["alpha"]
        s_xi = self.dev(state_trial["stress"]) - alpha_n
        n_hat = 1.5 * s_xi / self.vonmises_norm(s_xi)
        alpha_new = (alpha_n + (2.0 / 3.0) * self.C_k * dlambda * n_hat) \
                  / (1.0 + self.gamma * dlambda)
        return [self.alpha(alpha_new), self.ep(state_n["ep"] + dlambda)]
