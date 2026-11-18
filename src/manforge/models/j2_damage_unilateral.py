"""J2 + weakly coupled Lemaitre damage with quasi-unilateral (crack-closure) elasticity.

Stage 2 of the tension/compression split: the *stress* is split, not only the
damage driving force, so the stiffness recovers when cracks close.

Formulation (Lemaitre & Desmorat)
---------------------------------
Complementary energy on the principal *nominal* stresses, with the
hydrostatic part split on the sign of tr σ::

    ρψ*(σ, D) = (1+ν)/(2E) [ σ₊:σ₊/(1−D) + σ₋:σ₋/(1−hD) ]
              −    ν/(2E) [ ⟨tr σ⟩₊²/(1−D) + ⟨tr σ⟩₋²/(1−hD) ]

    ε_e = ∂ψ*/∂σ,        Y = ∂ψ*/∂D  (≥ 0)

Elasticity derives from a potential, so the elastic tangent is symmetric and
closed elastic cycles do no work — which the simpler split σ = (1−D)σ̃₊ +
(1−hD)σ̃₋ violates for ν ≠ 0.  Uniaxially the stiffness is exactly (1−D)E in
tension and (1−hD)E in compression.

Weak coupling
-------------
Plasticity lives in the effective stress σ̃ = C0 : ε_e (strain equivalence)
and never sees D: σ̃ and ep follow the undamaged J2 history exactly.  D enters
only the map between σ̃ and the nominal σ, which is *implicit* —
σ̃ = C0 : ∂ψ*/∂σ(σ, D) has no closed-form inverse — and the damage rate
through Y(σ, D).

Unknowns and residuals
----------------------
``x = [σ, Δλ, σ̃, D]``, ep explicit (Δep = Δλ)::

    R_σ  = σ̃ − C0 : ∂ψ*/∂σ(σ, D_{n+1})            (genuinely coupled: implicit)
    R_Δλ = f(σ̃, ep) = σ̃_vm − (σ_y0 + H ep)
    R_σ̃ = σ̃ − σ̃_n − C0 Δε + Δλ C0 ñ(σ̃)          (undamaged J2)
    R_D  = E · [D − D_n − (Y(σ, D)/S)^s Δp_D]

ε_e and Y are both taken from ψ* by autodiff, so the potential is the single
source of truth.  Their Jacobians need ∂²ψ*/∂σ², i.e. P± — provided by
:mod:`manforge.utils.spectral` with the zero-eigenvalue convention H(0) = c.

Elastic predictor
-----------------
σ̃_trial = σ̃_n + C0 Δε with D frozen, and σ_trial solves
C0 : ∂ψ*/∂σ(σ_trial, D_n) = σ̃_trial.  That inversion is a small Newton
solve wrapped as an autograd primitive whose derivative is the implicit-
function result dσ = [C0 : ∂²ψ*/∂σ²]⁻¹ dσ̃ — so the elastic tangent and the
plastic consistent tangent differentiate through it exactly.

Limitations
-----------
- Zero-eigenvalue convention c is provisional (spectral.DEFAULT_HEAVISIDE_AT_ZERO).
- No failure criterion: as D → 1 the tension branch of ψ* blows up.
- ``elastic_stiffness`` returns (1−D) C0 as a *reference* stiffness for
  drivers' initial guesses and user hooks; the true elastic tangent is
  state- and direction-dependent and comes from the predictor.
"""

import autograd
import autograd.numpy as anp
import numpy as np
from autograd.extend import defvjp, primitive

from manforge.core.dimension import PLANE_STRESS, SOLID_3D, UNIAXIAL_1D, StressDimension
from manforge.core.material import MaterialModel, ParamMeta
from manforge.core.state import Explicit, Implicit, NTENS, SCALAR
from manforge.models.j2_damage import J2IsotropicDamage
from manforge.utils.smooth import smooth_macaulay
from manforge.utils.spectral import (
    DEFAULT_HEAVISIDE_AT_ZERO, macaulay_sq, neg_macaulay_sq, negative_part_sq,
    positive_part_sq,
)


# ---------------------------------------------------------------------------
# σ̃ → σ inversion as a differentiable primitive
# ---------------------------------------------------------------------------

_INVERT_MAX_ITER = 50


def _invert_numpy(model, s_eff, D):
    """Solve C0 : ∂ψ*/∂σ(σ, D) = σ̃ for σ (numpy values only).

    Plain Newton on this piecewise-linear system cycles between sign regions
    for some mixed states.  The system is the stationarity condition of the
    strictly convex Φ(σ) = ψ*(σ, D) − σ · ε_e (ε_e = C0⁻¹ σ̃), so Newton with
    Armijo backtracking on Φ converges globally — and still in a handful of
    steps, since Φ is quadratic within each sign region.
    """
    C0 = np.asarray(model.undamaged_stiffness())
    psi = model.complementary_energy
    grad = autograd.grad(psi, 0)
    hess = autograd.jacobian(grad, 0)
    s_eff = np.asarray(s_eff, dtype=float)
    eps_e = np.linalg.solve(C0, s_eff)
    tol = 1e-12 * max(1.0, float(np.max(np.abs(s_eff))))

    def phi(x):
        return float(psi(x, D)) - float(x @ eps_e)

    sigma = (1.0 - float(D)) * s_eff          # tension-branch start
    for _ in range(_INVERT_MAX_ITER):
        g = np.asarray(grad(sigma, D)) - eps_e
        if float(np.max(np.abs(C0 @ g))) <= tol:
            return sigma
        step = -np.linalg.solve(np.asarray(hess(sigma, D)), g)
        t, phi0, slope = 1.0, phi(sigma), float(g @ step)
        while phi(sigma + t * step) > phi0 + 1e-4 * t * slope and t > 1e-12:
            t *= 0.5
        sigma = sigma + t * step
    raise RuntimeError(
        f"{type(model).__name__}: σ̃ → σ inversion did not converge in "
        f"{_INVERT_MAX_ITER} iterations (|r|∞ = {float(np.max(np.abs(C0 @ g))):.3e})"
    )


@primitive
def _nominal_from_effective(s_eff, D, model):
    return _invert_numpy(model, s_eff, D)


def _nominal_from_effective_vjp(ans, s_eff, D, model):
    C0 = np.asarray(model.undamaged_stiffness())
    hess = autograd.jacobian(autograd.grad(model.complementary_energy, 0), 0)
    K = C0 @ np.asarray(hess(np.asarray(ans), D))   # dσ̃ = K dσ
    return lambda g: np.linalg.solve(K.T, g)


# Derivative w.r.t. s_eff only: the predictor holds D at D_n.  Asking for
# ∂/∂D raises instead of silently returning zero.
defvjp(_nominal_from_effective, _nominal_from_effective_vjp)


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class J2IsotropicDamageUnilateral(MaterialModel):
    """J2 + weakly coupled Lemaitre damage, quasi-unilateral (crack-closure) stress.

    See the module docstring for the formulation.  Use the concrete classes
    (:class:`J2IsotropicDamageUnilateral3D` / ``PS`` / ``1D``) or pass
    ``dimension=``.

    Parameters
    ----------
    E, nu, sigma_y0, H, S, s, p_D
        As in :class:`~manforge.models.J2IsotropicDamage`.
    h : float
        Crack-closure parameter: compressive stiffness (1 − hD)E, and the
        compressive part of Y is weighted by h.
    c : float
        Value of the Heaviside step at a zero principal stress (see
        :mod:`manforge.utils.spectral`).  Provisional default 1/2.
    """

    param_names = ["E", "nu", "sigma_y0", "H", "S", "s", "p_D", "h"]
    param_meta = {
        **J2IsotropicDamage.param_meta,
        "h": ParamMeta(doc="Crack-closure parameter (compressive stiffness (1−hD)E)",
                       default=0.2, bounds=(0.0, 1.0)),
    }

    stress = Implicit(shape=NTENS, doc="nominal Cauchy stress")
    stress_eff = Implicit(shape=NTENS, doc="effective stress σ̃ = C0 : ε_e")
    ep = Explicit(shape=SCALAR, doc="equivalent plastic strain")
    D = Implicit(shape=SCALAR, doc="scalar damage (0 = intact)")

    def __init__(self, dimension: StressDimension = SOLID_3D, *,
                 E: float, nu: float, sigma_y0: float, H: float,
                 S: float, s: float, p_D: float = 0.0, h: float = 0.2,
                 c: float = DEFAULT_HEAVISIDE_AT_ZERO):
        super().__init__(dimension=dimension)
        self.E, self.nu, self.sigma_y0, self.H = E, nu, sigma_y0, H
        self.S, self.s, self.p_D, self.h = S, s, p_D, h
        self.c = c

    # ------------------------------------------------------------------
    # Potential and its derivatives
    # ------------------------------------------------------------------

    def undamaged_stiffness(self):
        return self.isotropic_stiffness()

    def complementary_energy(self, stress, D):
        """ρψ*(σ, D) — Lemaitre's quasi-unilateral complementary energy."""
        dim, c = self.dimension, self.c
        pos = positive_part_sq(stress, dim, c)
        neg = negative_part_sq(stress, dim, c)
        tr = 3.0 * self.hydrostatic(stress)
        a_t, a_c = 1.0 / (1.0 - D), 1.0 / (1.0 - self.h * D)
        return ((1.0 + self.nu) * (pos * a_t + neg * a_c)
                - self.nu * (macaulay_sq(tr, c) * a_t + neg_macaulay_sq(tr, c) * a_c)
                ) / (2.0 * self.E)

    def elastic_strain(self, stress, D):
        """ε_e = ∂ψ*/∂σ (engineering shear)."""
        return autograd.grad(self.complementary_energy, 0)(stress, D)

    def energy_release_rate(self, stress, D):
        """Y = ∂ψ*/∂D."""
        return autograd.grad(self.complementary_energy, 1)(stress, D)

    def nominal_from_effective(self, stress_eff, D):
        """σ solving C0 : ε_e(σ, D) = σ̃ — differentiable in σ̃ (D held fixed)."""
        return _nominal_from_effective(stress_eff, D, self)

    # ------------------------------------------------------------------
    # MaterialModel interface
    # ------------------------------------------------------------------

    def elastic_stiffness(self, state):
        """Reference stiffness (1−D) C0 — not the elastic tangent (see module doc)."""
        return (1.0 - state["D"]) * self.undamaged_stiffness()

    def yield_function(self, state):
        return self.vonmises(state["stress_eff"]) - (self.sigma_y0 + self.H * state["ep"])

    def flow(self, state):
        """Effective-space normal ∂f/∂σ̃."""
        return self.strain_flow(autograd.grad(self.vonmises)(state["stress_eff"]))

    def elastic_predictor(self, state_n, *, strain_inc):
        s_eff = state_n["stress_eff"] + self.undamaged_stiffness() @ strain_inc
        sigma = self.nominal_from_effective(s_eff, state_n["D"])
        return [self.stress.update(sigma), self.stress_eff.update(s_eff)]

    def update_state(self, new, dlambda, inc):
        return [self.ep(inc.state_n["ep"] + dlambda)]

    def state_residual(self, new, dlambda, inc):
        state_n, strain_inc = inc.state_n, inc.strain_inc
        C0 = self.undamaged_stiffness()
        sigma, s_eff, D_1 = new["stress"], new["stress_eff"], new["D"]

        R_stress = s_eff - C0 @ self.elastic_strain(sigma, D_1)
        R_eff = (s_eff - state_n["stress_eff"] - C0 @ strain_inc
                 + dlambda * (C0 @ self.flow(new).strain_like))
        dp_D = (smooth_macaulay(new["ep"] - self.p_D)
                - smooth_macaulay(state_n["ep"] - self.p_D))
        Y = self.energy_release_rate(sigma, D_1)
        # E-scaled into stress units: the NR tolerance is stress-scaled.
        R_D = self.E * (D_1 - state_n["D"] - (Y / self.S) ** self.s * dp_D)
        return [self.stress(R_stress), self.stress_eff(R_eff), self.D(R_D)]


class J2IsotropicDamageUnilateral3D(J2IsotropicDamageUnilateral):
    """3D solid variant of :class:`J2IsotropicDamageUnilateral`."""

    dimension = SOLID_3D

    def __init__(self, **kwargs):
        super().__init__(dimension=SOLID_3D, **kwargs)


class J2IsotropicDamageUnilateralPS(J2IsotropicDamageUnilateral):
    """Plane-stress variant of :class:`J2IsotropicDamageUnilateral`."""

    dimension = PLANE_STRESS

    def __init__(self, **kwargs):
        super().__init__(dimension=PLANE_STRESS, **kwargs)


class J2IsotropicDamageUnilateral1D(J2IsotropicDamageUnilateral):
    """Uniaxial (1D) variant of :class:`J2IsotropicDamageUnilateral`."""

    dimension = UNIAXIAL_1D

    def __init__(self, **kwargs):
        super().__init__(dimension=UNIAXIAL_1D, **kwargs)
