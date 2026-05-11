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

    dα = (2/3) C_k dp n̂ − γ dp α

The Ohno-Wang variant weights the recovery by the current backstress norm:

    dα = (2/3) C_k dp n̂ − γ dp ‖α‖ α

Backward-Euler discretisation as residual:

    R_α = α_{n+1} − α_n − (2/3) C_k Δλ n̂ + γ Δλ ‖α_{n+1}‖ α_{n+1} = 0

The (2/3) factor combines with the (3/2) inside n̂ to reproduce the classic
kinematic-hardening slope C_k under uniaxial loading.

Dimensional notes
-----------------
The three concrete classes share a single ``state_residual`` and
``yield_function`` implementation via ``KinematicOWMixin``.  Each stress-state
base class provides dimension-specific ``vonmises_relative``, ``flow_direction``,
and ``alpha_norm`` operators.

* **3D** (OWKinematic3D): α is a 6-component deviatoric tensor.
* **PS** (OWKinematicPS): α stores [α11, α22, α12]; operators use
  α33 = −(α11+α22) without constructing a 6-component intermediate vector.
* **1D** (OWKinematic1D): α stores the deviatoric component α11_dev;
  operators account for α22 = α33 = −α11_dev/2.

Notes
-----
* gamma=0 reduces to Prager's linear kinematic hardening (same limit as AF).
* The plastic increment is always solved via the vector NR (σ is Implicit).
"""

from manforge.core.material import MaterialModel3D, MaterialModelPS, MaterialModel1D
from manforge.core.material.kinematic import KinematicOWMixin
from manforge.core.state import Implicit, NTENS, SCALAR
from manforge.core.dimension import SOLID_3D, PLANE_STRESS, UNIAXIAL_1D, StressDimension


class OWKinematic3D(KinematicOWMixin, MaterialModel3D):
    """J2 + Ohno-Wang kinematic hardening for full-rank stress states.

    Uses the vector NR path (σ, α, ep all implicit).

    Parameters
    ----------
    dimension : StressDimension, optional
        Must satisfy ``dimension.ndi == dimension.ndi_phys``.
        Defaults to ``SOLID_3D``.
    E, nu, sigma_y0, C_k, gamma : float
        Material parameters.
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


class OWKinematicPS(KinematicOWMixin, MaterialModelPS):
    """J2 + Ohno-Wang kinematic hardening for plane-stress elements.

    Uses the vector NR path (σ, α, ep all implicit).

    Parameters
    ----------
    dimension : StressDimension, optional
        Must satisfy ``dimension.is_plane_stress``.
        Defaults to ``PLANE_STRESS``.
    E, nu, sigma_y0, C_k, gamma : float
        Material parameters.
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


class OWKinematic1D(KinematicOWMixin, MaterialModel1D):
    """J2 + Ohno-Wang kinematic hardening for uniaxial elements.

    Uses the vector NR path (σ, α, ep all implicit).

    The stored ``alpha`` is the deviatoric component α11_dev.  The effective
    backstress (yield-surface centre) is α_eff = (3/2)·α11_dev.

    Parameters
    ----------
    dimension : StressDimension, optional
        Must have ``ntens == 1``.  Defaults to ``UNIAXIAL_1D``.
    E, nu, sigma_y0, C_k, gamma : float
        Material parameters.
    """

    stress = Implicit(shape=NTENS, doc="Cauchy stress")
    param_names = ["E", "nu", "sigma_y0", "C_k", "gamma"]
    alpha = Implicit(shape=NTENS, doc="backstress tensor (deviatoric component)")
    ep = Implicit(shape=SCALAR, doc="equivalent plastic strain")

    def __init__(self, dimension: StressDimension = UNIAXIAL_1D, *,
                 E: float, nu: float, sigma_y0: float, C_k: float, gamma: float):
        super().__init__(dimension)
        self.E = E
        self.nu = nu
        self.sigma_y0 = sigma_y0
        self.C_k = C_k
        self.gamma = gamma
