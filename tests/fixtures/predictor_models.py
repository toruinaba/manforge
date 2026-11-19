"""Test double for the ``elastic_predictor`` hook: bilinear (unilateral) elasticity.

1D J2 plasticity lives in an effective stress σ̃ = E ε_e; the nominal stress is

    σ = k_t ⟨σ̃⟩₊ + k_c ⟨σ̃⟩₋

with fixed factors (no damage evolution).  This is the stage-2 crack-closure
structure stripped to its framework-relevant core: the elastic law is
piecewise linear, so no constant C_n reproduces an elastic increment that
changes sign, and σ̃ is an implicit field that must move on elastic steps.

Reference solution: run undamaged ``J2Isotropic1D`` on the same strain path
(that *is* the σ̃ history) and map σ = 𝒩(σ̃).
"""

import autograd
import autograd.numpy as anp

from manforge.core.dimension import UNIAXIAL_1D
from manforge.core.material import MaterialModel
from manforge.core.state import Explicit, Implicit, NTENS


class BilinearPlastic1D(MaterialModel):
    param_names = ["E", "nu", "sigma_y0", "H", "k_t", "k_c"]
    stress = Implicit(shape=NTENS, doc="nominal stress σ = 𝒩(σ̃)")
    stress_eff = Implicit(shape=NTENS, doc="effective stress σ̃ = E ε_e")
    ep = Explicit(shape=(), doc="equivalent plastic strain")

    def __init__(self, *, E=200000.0, nu=0.3, sigma_y0=250.0, H=2000.0, k_t=0.6, k_c=0.95):
        super().__init__(dimension=UNIAXIAL_1D)
        self.E, self.nu, self.sigma_y0, self.H = E, nu, sigma_y0, H
        self.k_t, self.k_c = k_t, k_c

    def nominal(self, s_eff):
        return self.k_t * anp.maximum(s_eff, 0.0) + self.k_c * anp.minimum(s_eff, 0.0)

    def yield_function(self, state):
        return self.vonmises(state["stress_eff"]) - (self.sigma_y0 + self.H * state["ep"])

    def flow(self, state):
        return self.strain_flow(autograd.grad(self.vonmises)(state["stress_eff"]))

    def elastic_predictor(self, state_n, *, strain_inc):
        C0 = self.isotropic_stiffness()
        s_eff = state_n["stress_eff"] + C0 @ strain_inc
        return [self.stress.update(self.nominal(s_eff)), self.stress_eff.update(s_eff)]

    def update_state(self, new, dlambda, inc):
        return [self.ep(inc.state_n["ep"] + dlambda)]

    def state_residual(self, new, dlambda, inc):
        C0 = self.isotropic_stiffness()
        s_eff = new["stress_eff"]
        R_eff = s_eff - inc.trial["stress_eff"] \
            + dlambda * (C0 @ self.flow(new).strain_like)
        R_sig = new["stress"] - self.nominal(s_eff)
        return [self.stress(R_sig), self.stress_eff(R_eff)]


class BilinearPlasticTrialResidual1D(BilinearPlastic1D):
    """Same root, but the σ row is written against ``stress_trial``.

    R_σ = σ − σ_trial − 𝒩(σ̃) + 𝒩(σ̃_n + E Δε), and σ_trial = 𝒩(σ̃_n + E Δε), so
    the solution is unchanged.  Here dR/dΔε depends on how σ_trial varies with
    Δε: only a tangent that differentiates *through the predictor* gets it
    right (linearising σ_trial with C_n = E would be off by the factor k).
    """

    def state_residual(self, new, dlambda, inc):
        C0 = self.isotropic_stiffness()
        items = super().state_residual(new, dlambda, inc)
        R_eff = [i for i in items if i.name == "stress_eff"][0].value
        s_trial_eff = inc.state_n["stress_eff"] + C0 @ inc.strain_inc
        R_sig = (new["stress"] - inc.trial["stress"]
                 - self.nominal(new["stress_eff"]) + self.nominal(s_trial_eff))
        return [self.stress(R_sig), self.stress_eff(R_eff)]
