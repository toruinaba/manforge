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
alpha : backstress tensor (shape (ntens,), deviatoric)
ep    : equivalent plastic strain (scalar, ≥ 0)

Yield function
--------------
f(state) = vonmises_norm(dev(σ) − α) − σ_y0

Ohno-Wang backstress evolution
-------------------------------
Standard Armstrong-Frederick has linear dynamic recovery:

    dα = (2/3) C_k dp n̂ − γ dp α

The Ohno-Wang variant weights the recovery by the current backstress norm:

    dα = (2/3) C_k dp n̂ − γ dp vonmises_norm(α) α

Backward-Euler discretisation as residual:

    R_α = α_{n+1} − α_n − (2/3) C_k Δλ n̂ + γ Δλ vonmises_norm(α_{n+1}) α_{n+1} = 0

Dimensional notes
-----------------
The three concrete classes share identical yield_function / state_residual
implementations via the OWKinematic parent.  Each stress-state dimension
provides dimension-specific dev / vonmises_norm that encapsulate
per-dimension arithmetic.

* **3D** (OWKinematic3D): α is a 6-component deviatoric tensor.
* **PS** (OWKinematicPS): α stores [α11, α22, α12]; vonmises_norm reconstructs
  α33 = −(α11+α22) without constructing a 6-component intermediate vector.
* **1D** (OWKinematic1D): α stores the deviatoric component α11_dev;
  vonmises_norm gives (3/2)|α11_dev|.

Notes
-----
* gamma=0 reduces to Prager's linear kinematic hardening (same limit as AF).
* The plastic increment is always solved via the vector NR (σ is Implicit).
"""

from manforge.core.material import MaterialModel
from manforge.core.state import Implicit, NTENS, SCALAR
from manforge.core.dimension import SOLID_3D, PLANE_STRESS, UNIAXIAL_1D, StressDimension


class OWKinematic(MaterialModel):
    """J2 + Ohno-Wang kinematic hardening — common physics across stress states.

    Subclass and pass ``dimension=`` to select the stress state, or use one of the
    pre-built concrete classes (:class:`OWKinematic3D`, :class:`OWKinematicPS`,
    :class:`OWKinematic1D`) which set appropriate defaults.

    Uses the vector NR path (σ, α, ep all implicit).

    Parameters
    ----------
    dimension : StressDimension, optional
        Defaults to ``SOLID_3D``.
    E, nu, sigma_y0, C_k, gamma : float
        Material parameters.
    """

    stress = Implicit(shape=NTENS, doc="Cauchy stress")
    param_names = ["E", "nu", "sigma_y0", "C_k", "gamma"]
    alpha = Implicit(shape=NTENS, doc="backstress tensor (deviatoric)")
    ep = Implicit(shape=SCALAR, doc="equivalent plastic strain")

    def __init__(self, dimension: StressDimension = SOLID_3D, *,
                 E: float, nu: float, sigma_y0: float, C_k: float, gamma: float):
        super().__init__(dimension=dimension)
        self.E = E
        self.nu = nu
        self.sigma_y0 = sigma_y0
        self.C_k = C_k
        self.gamma = gamma

    def yield_function(self, state):
        s_xi = self.dev(state["stress"]) - state["alpha"]
        return self.vonmises_norm(s_xi) - self.sigma_y0

    def state_residual(self, state_new, dlambda, state_n, *, stress_trial, strain_inc=None):
        alpha_new = state_new["alpha"]
        s_xi = self.dev(state_new["stress"]) - alpha_new
        n_hat = 1.5 * s_xi / self.vonmises_norm(s_xi)
        a_norm = self.vonmises_norm(alpha_new)
        R_stress = self.default_stress_residual(state_new, dlambda, stress_trial)
        R_alpha = (
            alpha_new - state_n["alpha"]
            - (2.0 / 3.0) * self.C_k * dlambda * n_hat
            + self.gamma * dlambda * a_norm * alpha_new
        )
        R_ep = state_new["ep"] - state_n["ep"] - dlambda
        return [self.stress(R_stress), self.alpha(R_alpha), self.ep(R_ep)]


class OWKinematic3D(OWKinematic):
    """J2 + Ohno-Wang kinematic hardening for full-rank stress states.

    Uses the vector NR path (σ, α, ep all implicit).

    Parameters
    ----------
    dimension : StressDimension, optional
        Defaults to ``SOLID_3D``.
    E, nu, sigma_y0, C_k, gamma : float
        Material parameters.
    """

    def __init__(self, dimension: StressDimension = SOLID_3D, *,
                 E: float, nu: float, sigma_y0: float, C_k: float, gamma: float):
        super().__init__(dimension=dimension, E=E, nu=nu, sigma_y0=sigma_y0, C_k=C_k, gamma=gamma)


class OWKinematicPS(OWKinematic):
    """J2 + Ohno-Wang kinematic hardening for plane-stress elements.

    Uses the vector NR path (σ, α, ep all implicit).

    Parameters
    ----------
    dimension : StressDimension, optional
        Defaults to ``PLANE_STRESS``.
    E, nu, sigma_y0, C_k, gamma : float
        Material parameters.
    """

    def __init__(self, dimension: StressDimension = PLANE_STRESS, *,
                 E: float, nu: float, sigma_y0: float, C_k: float, gamma: float):
        super().__init__(dimension=dimension, E=E, nu=nu, sigma_y0=sigma_y0, C_k=C_k, gamma=gamma)


class OWKinematic1D(OWKinematic):
    """J2 + Ohno-Wang kinematic hardening for uniaxial elements.

    Uses the vector NR path (σ, α, ep all implicit).

    The stored ``alpha`` is the deviatoric component α11_dev.  The effective
    backstress (yield-surface centre) is α_eff = (3/2)·α11_dev.

    Parameters
    ----------
    dimension : StressDimension, optional
        Defaults to ``UNIAXIAL_1D``.
    E, nu, sigma_y0, C_k, gamma : float
        Material parameters.
    """

    def __init__(self, dimension: StressDimension = UNIAXIAL_1D, *,
                 E: float, nu: float, sigma_y0: float, C_k: float, gamma: float):
        super().__init__(dimension=dimension, E=E, nu=nu, sigma_y0=sigma_y0, C_k=C_k, gamma=gamma)
