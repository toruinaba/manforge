"""J2 isotropic hardening with weakly coupled Lemaitre ductile damage.

Effective-stress concept (reduction of the load-bearing cross-section)::

    σ = (1 − D) σ̃

Plasticity lives entirely in the effective (undamaged) space and does not see
D — *weak coupling*.  D only scales the nominal stress, so σ̃ and ep follow
exactly the undamaged :class:`~returnmap.models.J2Isotropic` history, and
σ = (1 − D) σ̃ on top of it.

Model parameters
----------------
E, nu    : elastic constants of the undamaged material
sigma_y0 : initial yield stress (effective space)
H        : linear isotropic hardening modulus (effective space)
S, s     : Lemaitre damage strength [stress] and exponent [-]
p_D      : damage threshold on the equivalent plastic strain

Internal state
--------------
ep : equivalent plastic strain (Explicit, Δep = Δλ)
D  : scalar damage (Implicit — its rate depends on σ̃ = σ/(1 − D))

Equations (backward Euler, step n → n+1)
----------------------------------------
Yield function (effective space)::

    f = σ̃_vm − (σ_y0 + H ep),    σ̃ = σ / (1 − D)

Damage (Lemaitre, gated by the threshold p_D)::

    Y  = σ̃ : C0⁻¹ : σ̃ / 2 = [ (2/3)(1+ν) σ̃_vm² + 3(1−2ν) σ̃_H² ] / (2E)
    ΔD = (Y/S)^s · Δp_D,     Δp_D = ⟨ep_{n+1} − p_D⟩ − ⟨ep_n − p_D⟩

The D residual row is multiplied by E so that it carries stress units like
the other rows (the NR tolerance is stress-scaled).

Y is written without the triaxiality ratio σ̃_H/σ̃_vm so that it stays smooth
at σ̃_vm → 0; the two forms are algebraically identical.  ⟨·⟩ is
:func:`~returnmap.utils.smooth_macaulay`, which keeps the tangent free
of the kink at ep = p_D.

Stress residual — why the row is scaled
---------------------------------------
The trial stress is nominal, σ_trial = σ_n + (1 − D_n) C0 Δε =
(1 − D_n) σ̃_trial, because :meth:`elastic_stiffness` returns (1 − D) C0.  The
effective-space return σ̃ = σ̃_trial − Δλ C0 ñ is therefore written as::

    R_σ = (1 − D_n)/(1 − D_{n+1}) · σ − σ_trial + Δλ (1 − D_n) C0 ñ

i.e. the effective return multiplied by (1 − D_n).  The scaling keeps the
residual in nominal stress units and ∂R_σ/∂σ_trial = −I.  (The consistent
tangent no longer requires the latter — it differentiates dR/dΔε — but any
scaling gives the same root and the same tangent.)  The default residual
``σ − σ_trial + Δλ C(D) ∂f/∂σ`` would be wrong here: it misses the stress drop
−ΔD C0 ε_e caused by damage growth at fixed elastic strain.

Because D_{n+1} is an NR unknown, the autodiff tangent automatically contains
the damage term −σ̃ ⊗ ∂D/∂Δε on top of (1 − D) C̃_ep.

Limitations
-----------
- Damage grows only with plastic flow.  Elastic steps (including unloading)
  freeze D, and their tangent (1 − D_n) C0 is exact for that reason.  A
  damage law that evolves during elastic steps is not expressible on the
  elastic branch of :meth:`stress_update`.
- No failure criterion: the model has no critical damage D_c and does not
  delete elements.  As D → 1 the σ row becomes singular (the 1/(1 − D)
  scaling), so keep loading well short of that.
"""

import autograd
import autograd.numpy as anp

from returnmap.core.material import MaterialModel, ParamMeta
from returnmap.core.state import (
    Explicit, Implicit, Increment, NTENS, SCALAR, State, StateResidual, StateUpdate,
)
from returnmap.core.dimension import SOLID_3D, PLANE_STRESS, UNIAXIAL_1D
from returnmap.utils.smooth import smooth_macaulay
from returnmap.utils.spectral import negative_part_sq, positive_part_sq
from returnmap._typing import Scalar as ScalarType, Stiffness, StressVec, StateDict


class J2IsotropicDamage(MaterialModel):
    """J2 isotropic hardening + weakly coupled Lemaitre damage, any stress state.

    Use one of the concrete classes (:class:`J2IsotropicDamage3D`,
    :class:`J2IsotropicDamagePS`, :class:`J2IsotropicDamage1D`) or pass
    ``dimension=`` to this parent.  All dimensions share the autodiff
    return-mapping path.

    Parameters
    ----------
    dimension : StressDimension, optional
        Defaults to ``SOLID_3D``.
    E, nu : float
        Undamaged elastic constants.
    sigma_y0, H : float
        Effective-space yield stress and linear hardening modulus.
    S, s : float
        Lemaitre damage strength and exponent.
    p_D : float
        Damage threshold on ep.
    """

    param_names = ["E", "nu", "sigma_y0", "H", "S", "s", "p_D"]
    param_meta = {
        "E": ParamMeta(doc="Young's modulus (undamaged)", unit="MPa",
                       default=210_000.0, bounds=(1e3, 1e6)),
        "nu": ParamMeta(doc="Poisson's ratio", default=0.3, bounds=(0.0, 0.49)),
        "sigma_y0": ParamMeta(doc="Initial yield stress (effective)", unit="MPa",
                              default=250.0, bounds=(1.0, 5e3)),
        "H": ParamMeta(doc="Isotropic hardening modulus (effective, linear)", unit="MPa",
                       default=1000.0, bounds=(0.0, 1e5)),
        "S": ParamMeta(doc="Lemaitre damage strength", unit="MPa",
                       default=1.0, bounds=(1e-3, 1e3)),
        "s": ParamMeta(doc="Lemaitre damage exponent", default=1.0, bounds=(0.1, 10.0)),
        "p_D": ParamMeta(doc="Damage threshold on equivalent plastic strain",
                         default=0.0, bounds=(0.0, 1.0)),
    }

    stress = Implicit(shape=NTENS, doc="nominal Cauchy stress σ = (1 − D) σ̃")
    ep = Explicit(shape=SCALAR, doc="equivalent plastic strain")
    D = Implicit(shape=SCALAR, doc="scalar damage (0 = intact)")

    # ------------------------------------------------------------------
    # Effective-space helpers
    # ------------------------------------------------------------------

    def undamaged_stiffness(self) -> Stiffness:
        """Undamaged elastic stiffness C0 — the isotropic stiffness from E, ν."""
        return self.isotropic_stiffness()

    def effective_stress(self, state: "State | StateDict") -> StressVec:
        """σ̃ = σ / (1 − D)."""
        return state["stress"] / (1.0 - state["D"])

    def energy_release_rate(self, stress_eff: StressVec) -> ScalarType:
        """Y = σ̃ : C0⁻¹ : σ̃ / 2 in the triaxiality-free form."""
        vm = self.vonmises(stress_eff)
        sh = self.hydrostatic(stress_eff)
        return ((2.0 / 3.0) * (1.0 + self.nu) * vm ** 2
                + 3.0 * (1.0 - 2.0 * self.nu) * sh ** 2) / (2.0 * self.E)

    # ------------------------------------------------------------------
    # MaterialModel interface
    # ------------------------------------------------------------------

    def elastic_stiffness(self, state: "State | StateDict") -> Stiffness:
        """Damaged stiffness (1 − D) C0."""
        return (1.0 - state["D"]) * self.undamaged_stiffness()

    def yield_function(self, state: "State | StateDict") -> ScalarType:
        """f = σ̃_vm − (σ_y0 + H ep), evaluated in effective space."""
        return self.vonmises(self.effective_stress(state)) - (
            self.sigma_y0 + self.H * state["ep"]
        )

    def flow(self, state: "State | StateDict"):
        """Effective-space normal ∂f/∂σ̃ (not ∂f/∂σ, which carries 1/(1 − D))."""
        stress_eff = self.effective_stress(state)
        n = autograd.grad(self.vonmises)(stress_eff)  # type: ignore[call-arg]
        return self.strain_flow(n)

    def update_state(
        self, new: "State | StateDict", dlambda: ScalarType, inc: Increment,
    ) -> list[StateUpdate | StateResidual]:
        """Δep = Δλ (effective-space associative flow)."""
        return [self.ep(inc.state_n["ep"] + dlambda)]

    def state_residual(
        self, new: "State | StateDict", dlambda: ScalarType, inc: Increment,
    ) -> list[StateResidual]:
        """Scaled effective-space return for σ and Lemaitre backward Euler for D."""
        state_n = inc.state_n
        D_n, D_1 = state_n["D"], new["D"]
        C0 = self.undamaged_stiffness()
        n = self.flow(new).strain_like

        R_stress = ((1.0 - D_n) / (1.0 - D_1)) * new["stress"] - inc.trial["stress"] \
            + dlambda * (1.0 - D_n) * (C0 @ n)

        dp_D = (smooth_macaulay(new["ep"] - self.p_D)
                - smooth_macaulay(state_n["ep"] - self.p_D))
        Y = self.energy_release_rate(self.effective_stress(new))
        # Scaled by E into stress units: the NR tolerance is tol·max(1, ‖σ_trial‖∞)
        # (stress units), so an unscaled dimensionless D row would stop with D
        # accurate only to ~tol·‖σ‖ — around 1e-8 at MPa scale.
        R_D = self.E * (D_1 - D_n - (Y / self.S) ** self.s * dp_D)

        return [self.stress(R_stress), self.D(R_D)]


class J2IsotropicDamage3D(J2IsotropicDamage):
    """3D solid variant of :class:`J2IsotropicDamage`."""

    dimension = SOLID_3D


class J2IsotropicDamagePS(J2IsotropicDamage):
    """Plane-stress variant of :class:`J2IsotropicDamage`."""

    dimension = PLANE_STRESS


class J2IsotropicDamage1D(J2IsotropicDamage):
    """Uniaxial (1D) variant of :class:`J2IsotropicDamage`."""

    dimension = UNIAXIAL_1D


# ===========================================================================
# Crack-closure variant: tension/compression split of the damage driving force
# ===========================================================================

class J2IsotropicDamageCrackClosure(J2IsotropicDamage):
    """:class:`J2IsotropicDamage` whose Y is split into tensile/compressive parts.

    Lemaitre's quasi-unilateral (crack-closure) energy release rate, with the
    split taken on the principal effective stresses σ̃_i::

        Y = (1+ν)/(2E) · [ Σ⟨σ̃_i⟩₊² + h Σ⟨σ̃_i⟩₋² ]
          −    ν/(2E) · [ ⟨tr σ̃⟩₊² + h ⟨tr σ̃⟩₋² ]

    h ∈ [0, 1] is the crack-closure parameter: compression drives damage h
    times as hard as tension.  h = 1 recovers the parent's Y = σ̃ : C0⁻¹ : σ̃ / 2
    exactly; h = 0 means purely compressive states do not damage at all.
    Both brackets are non-negative for ν < 1/2, so Y ≥ 0.

    **Scope (stage 1).**  Only the damage *driving force* is split.  The
    nominal stress is still σ = (1 − D) σ̃ in both tension and compression, so
    the stiffness does *not* recover when cracks close.  Splitting the stress
    itself makes the elastic response piecewise linear, which the framework's
    linear elastic branch (σ = σ_n + C_n Δε) cannot represent — that needs a
    separate design.

    **Autodiff.**  Σ⟨σ̃_i⟩±² comes from
    :func:`returnmap.utils.spectral_trace`, whose gradient is the matrix
    function f'(σ̃) rather than the eigh derivative.  Uniaxial, plane-stress and
    pure-shear states all have repeated principal values, where the eigh
    derivative is NaN; the spectral form is exact there.

    Parameters
    ----------
    h : float
        Crack-closure parameter (0.2 is a typical metal value).
    Other parameters as in :class:`J2IsotropicDamage`.
    """

    param_names = J2IsotropicDamage.param_names + ["h"]
    param_meta = {
        **J2IsotropicDamage.param_meta,
        "h": ParamMeta(doc="Crack-closure parameter (compression/tension damage ratio)",
                       default=0.2, bounds=(0.0, 1.0)),
    }

    def energy_release_rate(self, stress_eff: StressVec) -> ScalarType:
        """Crack-closure Y from the principal split of σ̃."""
        dim = self.dimension
        pos = positive_part_sq(stress_eff, dim)
        neg = negative_part_sq(stress_eff, dim)
        tr = 3.0 * self.hydrostatic(stress_eff)
        tr_pos = anp.maximum(tr, 0.0) ** 2
        tr_neg = anp.minimum(tr, 0.0) ** 2
        return ((1.0 + self.nu) * (pos + self.h * neg)
                - self.nu * (tr_pos + self.h * tr_neg)) / (2.0 * self.E)


class J2IsotropicDamageCrackClosure3D(J2IsotropicDamageCrackClosure):
    """3D solid variant of :class:`J2IsotropicDamageCrackClosure`."""

    dimension = SOLID_3D


class J2IsotropicDamageCrackClosurePS(J2IsotropicDamageCrackClosure):
    """Plane-stress variant of :class:`J2IsotropicDamageCrackClosure`."""

    dimension = PLANE_STRESS


class J2IsotropicDamageCrackClosure1D(J2IsotropicDamageCrackClosure):
    """Uniaxial (1D) variant of :class:`J2IsotropicDamageCrackClosure`."""

    dimension = UNIAXIAL_1D
