"""Chaboche model with superposed backstresses: J2 plasticity with three
Armstrong-Frederick backstresses and three saturating isotropic terms.

One backstress saturates at one rate; a measured hysteresis loop usually
needs several — a fast one for the knee after yielding, slower ones for the
rest of the curve.  The model superposes three.

Model parameters
----------------
E, nu          : elastic constants
sigma_y0       : Initial yield stress
C1, gamma1     : modulus and dynamic recovery of the first backstress
C2, gamma2     : … of the second
C3, gamma3     : … of the third
Qinf1, b1      : saturation value and rate of the first isotropic term
Qinf2, b2      : … of the second
Qinf3, b3      : … of the third

A term is switched off by its modulus or saturation value: ``C_i = 0``,
``Qinf_i = 0``.

Internal state
--------------
alpha1, alpha2, alpha3 : the backstresses (shape (ntens,), deviatoric)
alpha                  : their sum
R1, R2, R3             : the isotropic terms (scalars)
R                      : their sum
ep                     : equivalent plastic strain

``alpha`` and ``R`` repeat what the parts add up to; they are carried as
states so that a result can be read without summing.

Yield function
--------------
f(state) = vonmises_norm(dev(σ) − Σ α_i) − (σ_y0 + Σ R_i)

Evolution laws and their backward-Euler updates  (i = 1, 2, 3)
----------------------------------------------------------------
dα_i = (2/3) C_i n̂ dp − γ_i α_i dp      →  α_i = (α_i,n + (2/3) C_i Δλ n̂) / (1 + γ_i Δλ)
dR_i = b_i (Q∞_i − R_i) dp              →  R_i = (R_i,n + b_i Q∞_i Δλ) / (1 + b_i Δλ)

with the associative flow direction n̂ = (3/2) ξ / vonmises_norm(ξ),
ξ = dev(σ) − Σ α_i, and Δp = Δλ.  Every update is a closed form, so every
state is explicit and the Newton-Raphson unknowns are ``[σ, Δλ]``.

Notes
-----
* With only the first backstress and the first isotropic term active the
  model is :class:`~returnmap.models.ChabocheKinematic`.
* The backstresses are carried as in :class:`~returnmap.models.AFKinematic`
  (see the dimensional notes there).
* No ``user_defined_return_mapping`` or ``user_defined_tangent`` is provided.
"""

from returnmap.core.material import MaterialModel, ParamMeta
from returnmap.core.state import Explicit, NTENS, SCALAR
from returnmap.core.dimension import SOLID_3D, PLANE_STRESS, UNIAXIAL_1D

_N_TERMS = 3


def _kinematic_meta(i: int, C: float, gamma: float) -> dict[str, ParamMeta]:
    return {
        f"C{i}": ParamMeta(doc=f"Kinematic hardening modulus of backstress {i}", unit="MPa",
                           default=C, bounds=(0.0, 1e6)),
        f"gamma{i}": ParamMeta(doc=f"Dynamic recovery coefficient of backstress {i}",
                               default=gamma, bounds=(0.0, 1e4)),
    }


def _isotropic_meta(i: int, Qinf: float, b: float) -> dict[str, ParamMeta]:
    return {
        f"Qinf{i}": ParamMeta(doc=f"Saturation value of isotropic term {i}", unit="MPa",
                              default=Qinf, bounds=(-1e3, 5e3)),
        f"b{i}": ParamMeta(doc=f"Rate at which isotropic term {i} saturates",
                           default=b, bounds=(0.0, 1e4)),
    }


class ChabocheKinematicMulti(MaterialModel):
    """Chaboche model with three backstresses — common physics across stress states.

    Three Armstrong-Frederick backstresses and three saturating isotropic
    terms.  Pass ``dimension=`` to select the stress state, or use one of the
    concrete classes (:class:`ChabocheKinematicMulti3D`,
    :class:`ChabocheKinematicMultiPS`, :class:`ChabocheKinematicMulti1D`),
    which fix it.

    Parameters
    ----------
    dimension : StressDimension, optional
        Defaults to ``SOLID_3D``.
    E, nu, sigma_y0 : float
        Elastic constants and initial yield stress.
    C1, gamma1, C2, gamma2, C3, gamma3 : float
        Modulus and dynamic recovery coefficient of each backstress.
    Qinf1, b1, Qinf2, b2, Qinf3, b3 : float
        Saturation value and rate of each isotropic term.
    """

    param_names = [
        "E", "nu", "sigma_y0", "C1", "gamma1", "C2", "gamma2", "C3", "gamma3",
        "Qinf1", "b1", "Qinf2", "b2", "Qinf3", "b3",
    ]
    param_meta = {
        "E": ParamMeta(doc="Young's modulus", unit="MPa",
                       default=210_000.0, bounds=(1e3, 1e6)),
        "nu": ParamMeta(doc="Poisson's ratio", default=0.3, bounds=(0.0, 0.49)),
        "sigma_y0": ParamMeta(doc="Initial yield stress", unit="MPa",
                              default=250.0, bounds=(1.0, 5e3)),
        # Three rates a decade apart, each saturating at 100 MPa.
        **_kinematic_meta(1, 80_000.0, 800.0),
        **_kinematic_meta(2, 10_000.0, 100.0),
        **_kinematic_meta(3, 1_000.0, 10.0),
        **_isotropic_meta(1, 40.0, 40.0),
        **_isotropic_meta(2, 30.0, 5.0),
        **_isotropic_meta(3, 0.0, 1.0),
    }
    alpha1 = Explicit(shape=NTENS, doc="backstress 1 (deviatoric)")
    alpha2 = Explicit(shape=NTENS, doc="backstress 2 (deviatoric)")
    alpha3 = Explicit(shape=NTENS, doc="backstress 3 (deviatoric)")
    alpha = Explicit(shape=NTENS, doc="backstress: the sum of the three")
    R1 = Explicit(shape=SCALAR, doc="isotropic term 1")
    R2 = Explicit(shape=SCALAR, doc="isotropic term 2")
    R3 = Explicit(shape=SCALAR, doc="isotropic term 3")
    R = Explicit(shape=SCALAR, doc="isotropic hardening: the sum of the three")
    ep = Explicit(shape=SCALAR, doc="equivalent plastic strain")

    def yield_function(self, state):
        alpha = state["alpha1"] + state["alpha2"] + state["alpha3"]
        R = state["R1"] + state["R2"] + state["R3"]
        s_xi = self.dev(state["stress"]) - alpha
        return self.vonmises_norm(s_xi) - (self.sigma_y0 + R)

    def update_state(self, new, dlambda, inc):
        state_n = inc.state_n
        # The backstresses are stress-like, so their evolution law takes the
        # physical-shear flow.
        n_hat = self.flow(new).stress_like
        alphas, Rs = [], []
        for i in range(1, _N_TERMS + 1):
            C, gamma = getattr(self, f"C{i}"), getattr(self, f"gamma{i}")
            Qinf, b = getattr(self, f"Qinf{i}"), getattr(self, f"b{i}")
            alphas.append(
                (state_n[f"alpha{i}"] + (2.0 / 3.0) * C * dlambda * n_hat)
                / (1.0 + gamma * dlambda)
            )
            Rs.append((state_n[f"R{i}"] + b * Qinf * dlambda) / (1.0 + b * dlambda))
        return [
            self.alpha1(alphas[0]), self.alpha2(alphas[1]), self.alpha3(alphas[2]),
            self.alpha(alphas[0] + alphas[1] + alphas[2]),
            self.R1(Rs[0]), self.R2(Rs[1]), self.R3(Rs[2]),
            self.R(Rs[0] + Rs[1] + Rs[2]),
            self.ep(state_n["ep"] + dlambda),
        ]


class ChabocheKinematicMulti3D(ChabocheKinematicMulti):
    """Chaboche model with three backstresses for 3D solids.

    Every state is explicit, so the Newton-Raphson unknowns are ``[σ, Δλ]``.

    Parameters
    ----------
    E, nu, sigma_y0, C1, gamma1, C2, gamma2, C3, gamma3, Qinf1, b1, Qinf2, b2, Qinf3, b3 : float
        Material parameters.
    """

    dimension = SOLID_3D


class ChabocheKinematicMultiPS(ChabocheKinematicMulti):
    """Chaboche model with three backstresses for plane-stress elements.

    Every state is explicit, so the Newton-Raphson unknowns are ``[σ, Δλ]``.

    Parameters
    ----------
    E, nu, sigma_y0, C1, gamma1, C2, gamma2, C3, gamma3, Qinf1, b1, Qinf2, b2, Qinf3, b3 : float
        Material parameters.
    """

    dimension = PLANE_STRESS


class ChabocheKinematicMulti1D(ChabocheKinematicMulti):
    """Chaboche model with three backstresses for uniaxial stress.

    Every state is explicit, so the Newton-Raphson unknowns are ``[σ, Δλ]``.
    The stored backstresses are the deviatoric components α11; the shift of
    the yield-surface centre is (3/2)·α11.

    Parameters
    ----------
    E, nu, sigma_y0, C1, gamma1, C2, gamma2, C3, gamma3, Qinf1, b1, Qinf2, b2, Qinf3, b3 : float
        Material parameters.
    """

    dimension = UNIAXIAL_1D
