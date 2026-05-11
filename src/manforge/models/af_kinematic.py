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
dε_p = Δλ · n̂,  n̂ = ∂f/∂σ = (3/2) s_ξ / σ_vm(ξ),  ξ = σ − α

Backstress evolution (Armstrong-Frederick, original form, backward Euler)
--------------------------------------------------------------------------
Δα = (2/3) C_k Δλ n̂ − γ Δλ α_{n+1}

Solving for α_{n+1}:

    α_{n+1} = (α_n + (2/3) C_k Δλ n̂) / (1 + γ Δλ)

The (2/3) factor combines with the (3/2) inside n̂ to reproduce the classic
kinematic-hardening slope C_k under uniaxial loading in all dimensions.

Dimensional notes
-----------------
The three concrete classes share a single ``update_state`` and
``yield_function`` implementation via ``KinematicAFMixin``.  Each stress-state
base class provides dimension-specific ``vonmises_relative``, ``flow_direction``,
and ``alpha_norm`` operators that encapsulate any per-dimension arithmetic.

* **3D** (AFKinematic3D): α is a 6-component deviatoric tensor.
* **PS** (AFKinematicPS): α stores [α11, α22, α12]; operators use
  α33 = −(α11+α22) without constructing a 6-component intermediate vector.
* **1D** (AFKinematic1D): α stores the deviatoric component α11_dev;
  the effective backstress is α_eff = (3/2)·α11_dev.

Notes
-----
* gamma=0 reduces to Prager's linear kinematic hardening rule with modulus C_k.
* The saturated backstress magnitude under monotonic loading is (2/3)·(C_k/gamma)
  for the stored deviatoric component, equivalent to C_k/gamma for the effective
  backstress in the 1D/uniaxial sense.
* No user_defined_return_mapping or user_defined_tangent is provided; the
  generic numerical_newton + autodiff path is used.
"""

from manforge.core.material import MaterialModel3D, MaterialModelPS, MaterialModel1D
from manforge.core.material.kinematic import KinematicAFMixin
from manforge.core.state import Explicit, NTENS, SCALAR
from manforge.core.dimension import SOLID_3D, PLANE_STRESS, UNIAXIAL_1D, StressDimension


class AFKinematic3D(KinematicAFMixin, MaterialModel3D):
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


class AFKinematicPS(KinematicAFMixin, MaterialModelPS):
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


class AFKinematic1D(KinematicAFMixin, MaterialModel1D):
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
    alpha = Explicit(shape=NTENS, doc="backstress tensor (deviatoric component)")
    ep = Explicit(shape=SCALAR, doc="equivalent plastic strain")

    def __init__(self, dimension: StressDimension = UNIAXIAL_1D, *,
                 E: float, nu: float, sigma_y0: float, C_k: float, gamma: float):
        super().__init__(dimension)
        self.E = E
        self.nu = nu
        self.sigma_y0 = sigma_y0
        self.C_k = C_k
        self.gamma = gamma
