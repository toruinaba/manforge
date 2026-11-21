"""Chaboche model with one backstress: J2 plasticity with Armstrong-Frederick
kinematic hardening and saturating isotropic hardening.

Model parameters
----------------
E        : Young's modulus
nu       : Poisson's ratio
sigma_y0 : Initial yield stress
C        : Kinematic hardening modulus
gamma    : Dynamic recovery parameter (gamma=0 → Prager linear rule)
Qinf     : Saturation value of the isotropic hardening
b        : Rate at which the isotropic hardening saturates

Internal state
--------------
alpha : backstress tensor (shape (ntens,), deviatoric)
R     : isotropic hardening — growth of the yield-surface radius (scalar)
ep    : equivalent plastic strain (scalar, ≥ 0)

Yield function
--------------
f(state) = vonmises_norm(dev(σ) − α) − (σ_y0 + R)

Flow rule (associative)
-----------------------
n̂ = (3/2) ξ / vonmises_norm(ξ),  ξ = dev(σ) − α,   dε_p = Δλ · n̂,   Δp = Δλ

Evolution laws and their backward-Euler updates
-----------------------------------------------
dα = (2/3) C n̂ dp − γ α dp        →  α_{n+1} = (α_n + (2/3) C Δλ n̂) / (1 + γ Δλ)
dR = b (Q∞ − R) dp                →  R_{n+1} = (R_n + b Q∞ Δλ) / (1 + b Δλ)

Both are closed forms, so every state is explicit and the Newton-Raphson
unknowns are ``[σ, Δλ]``.  With ``Qinf = 0`` the model is
:class:`~returnmap.models.AFKinematic`.

Dimensional notes
-----------------
The backstress is carried as the stored components of a 3D deviator, as in
:class:`~returnmap.models.AFKinematic`: six components in 3D,
``[α11, α22, α12]`` with ``α33 = −(α11 + α22)`` in plane stress, and the
deviatoric component ``α11`` in 1D (the shift of the yield-surface centre is
3/2 of it).

Notes
-----
No ``user_defined_return_mapping`` or ``user_defined_tangent`` is provided;
the generic numerical Newton-Raphson and the autodiff tangent are used.
"""

from returnmap.core.material import MaterialModel, ParamMeta
from returnmap.core.state import Explicit, NTENS, SCALAR
from returnmap.core.dimension import SOLID_3D, PLANE_STRESS, UNIAXIAL_1D


class ChabocheKinematic(MaterialModel):
    """Chaboche model with one backstress — common physics across stress states.

    Armstrong-Frederick kinematic hardening plus saturating isotropic
    hardening.  Pass ``dimension=`` to select the stress state, or use one of
    the concrete classes (:class:`ChabocheKinematic3D`,
    :class:`ChabocheKinematicPS`, :class:`ChabocheKinematic1D`), which fix it.

    Parameters
    ----------
    dimension : StressDimension, optional
        Defaults to ``SOLID_3D``.
    E, nu, sigma_y0, C, gamma, Qinf, b : float
        Material parameters.
    """

    param_names = ["E", "nu", "sigma_y0", "C", "gamma", "Qinf", "b"]
    param_meta = {
        "E": ParamMeta(doc="Young's modulus", unit="MPa",
                       default=210_000.0, bounds=(1e3, 1e6)),
        "nu": ParamMeta(doc="Poisson's ratio", default=0.3, bounds=(0.0, 0.49)),
        "sigma_y0": ParamMeta(doc="Initial yield stress", unit="MPa",
                              default=250.0, bounds=(1.0, 5e3)),
        "C": ParamMeta(doc="Kinematic hardening modulus", unit="MPa",
                       default=20_000.0, bounds=(0.0, 1e6)),
        "gamma": ParamMeta(doc="Dynamic recovery coefficient",
                           default=100.0, bounds=(0.0, 1e4)),
        "Qinf": ParamMeta(doc="Saturation value of the isotropic hardening", unit="MPa",
                          default=100.0, bounds=(-1e3, 5e3)),
        "b": ParamMeta(doc="Rate at which the isotropic hardening saturates",
                       default=10.0, bounds=(0.0, 1e4)),
    }
    alpha = Explicit(shape=NTENS, doc="backstress tensor (deviatoric)")
    R = Explicit(shape=SCALAR, doc="isotropic hardening (growth of the yield-surface radius)")
    ep = Explicit(shape=SCALAR, doc="equivalent plastic strain")

    def yield_function(self, state):
        s_xi = self.dev(state["stress"]) - state["alpha"]
        return self.vonmises_norm(s_xi) - (self.sigma_y0 + state["R"])

    def update_state(self, new, dlambda, inc):
        state_n = inc.state_n
        # α is stress-like, so the evolution law takes the physical-shear flow.
        n_hat = self.flow(new).stress_like
        alpha = (state_n["alpha"] + (2.0 / 3.0) * self.C * dlambda * n_hat) \
            / (1.0 + self.gamma * dlambda)
        R = (state_n["R"] + self.b * self.Qinf * dlambda) / (1.0 + self.b * dlambda)
        return [self.alpha(alpha), self.R(R), self.ep(state_n["ep"] + dlambda)]


class ChabocheKinematic3D(ChabocheKinematic):
    """Chaboche model with one backstress for 3D solids.

    Every state is explicit, so the Newton-Raphson unknowns are ``[σ, Δλ]``.

    Parameters
    ----------
    E, nu, sigma_y0, C, gamma, Qinf, b : float
        Material parameters.
    """

    dimension = SOLID_3D


class ChabocheKinematicPS(ChabocheKinematic):
    """Chaboche model with one backstress for plane-stress elements.

    Every state is explicit, so the Newton-Raphson unknowns are ``[σ, Δλ]``.

    Parameters
    ----------
    E, nu, sigma_y0, C, gamma, Qinf, b : float
        Material parameters.
    """

    dimension = PLANE_STRESS


class ChabocheKinematic1D(ChabocheKinematic):
    """Chaboche model with one backstress for uniaxial stress.

    Every state is explicit, so the Newton-Raphson unknowns are ``[σ, Δλ]``.
    The stored ``alpha`` is the deviatoric component α11; the shift of the
    yield-surface centre is (3/2)·α11.

    Parameters
    ----------
    E, nu, sigma_y0, C, gamma, Qinf, b : float
        Material parameters.
    """

    dimension = UNIAXIAL_1D
