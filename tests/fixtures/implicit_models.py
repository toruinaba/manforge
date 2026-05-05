"""AF kinematic hardening models recast as implicit residual systems.

These are test doubles used to validate the vector NR machinery. Mathematically
identical to the corresponding explicit-path models at convergence.

The explicit update for alpha is:
    alpha_new = (alpha_n + C_k * dlambda * n_hat) / (1 + gamma * dlambda)

Rearranged as a residual:
    R_alpha = alpha_new * (1 + gamma * dlambda) - alpha_n - C_k * dlambda * n_hat = 0

n_hat is evaluated at (stress - alpha_n) — i.e., the OLD backstress — so both
paths are algebraically identical at convergence.

Uses MRO override: re-declare ``stress``, ``alpha`` and ``ep`` as ``Implicit`` to
activate the vector NR path with σ as an NR unknown.
"""

from manforge.core.state import Implicit, NTENS
from manforge.models.af_kinematic import AFKinematic3D, AFKinematicPS
from manforge.core.stress_state import PLANE_STRAIN


class _AFKinematicImplicit3D(AFKinematic3D):
    """AF kinematic 3D model with hardening expressed as an implicit residual."""

    stress = Implicit(shape=NTENS, doc="Cauchy stress (implicit override)")
    alpha = Implicit(shape=NTENS, doc="backstress tensor (implicit override)")
    ep = Implicit(shape=(), doc="equivalent plastic strain (implicit override)")

    def state_residual(self, state_new, dlambda, state_n, state_trial):
        alpha_n = state_n["alpha"]
        stress = state_new["stress"]
        xi = stress - alpha_n
        s_xi = self._dev(xi)
        vm_safe = self._vonmises(xi)
        n_hat = s_xi / vm_safe

        scale = 1.0 + self.gamma * dlambda
        R_stress = self.default_stress_residual(state_new, dlambda, state_trial)
        R_alpha = state_new["alpha"] * scale - alpha_n - self.C_k * dlambda * n_hat
        R_ep = state_new["ep"] - state_n["ep"] - dlambda
        return [self.stress(R_stress), self.alpha(R_alpha), self.ep(R_ep)]


class _AFKinematicImplicitPS(AFKinematicPS):
    """Plane-stress variant of the implicit AF model."""

    stress = Implicit(shape=NTENS, doc="Cauchy stress (implicit override)")
    alpha = Implicit(shape=NTENS, doc="backstress tensor (implicit override)")
    ep = Implicit(shape=(), doc="equivalent plastic strain (implicit override)")

    def state_residual(self, state_new, dlambda, state_n, state_trial):
        alpha_n = state_n["alpha"]
        stress = state_new["stress"]
        xi = stress - alpha_n
        s_xi = self._dev(xi)
        vm_safe = self._vonmises(xi)
        n_hat = s_xi / vm_safe

        scale = 1.0 + self.gamma * dlambda
        R_stress = self.default_stress_residual(state_new, dlambda, state_trial)
        R_alpha = state_new["alpha"] * scale - alpha_n - self.C_k * dlambda * n_hat
        R_ep = state_new["ep"] - state_n["ep"] - dlambda
        return [self.stress(R_stress), self.alpha(R_alpha), self.ep(R_ep)]


class _AFKinematicImplicitPE(AFKinematic3D):
    """Plane-strain variant of the implicit AF model (uses MaterialModel3D with PLANE_STRAIN)."""

    stress = Implicit(shape=NTENS, doc="Cauchy stress (implicit override)")
    alpha = Implicit(shape=NTENS, doc="backstress tensor (implicit override)")
    ep = Implicit(shape=(), doc="equivalent plastic strain (implicit override)")

    def __init__(self):
        super().__init__(stress_state=PLANE_STRAIN,
                         E=210000.0, nu=0.3, sigma_y0=250.0, C_k=10000.0, gamma=100.0)

    def state_residual(self, state_new, dlambda, state_n, state_trial):
        alpha_n = state_n["alpha"]
        stress = state_new["stress"]
        xi = stress - alpha_n
        s_xi = self._dev(xi)
        vm_safe = self._vonmises(xi)
        n_hat = s_xi / vm_safe

        scale = 1.0 + self.gamma * dlambda
        R_stress = self.default_stress_residual(state_new, dlambda, state_trial)
        R_alpha = state_new["alpha"] * scale - alpha_n - self.C_k * dlambda * n_hat
        R_ep = state_new["ep"] - state_n["ep"] - dlambda
        return [self.stress(R_stress), self.alpha(R_alpha), self.ep(R_ep)]
