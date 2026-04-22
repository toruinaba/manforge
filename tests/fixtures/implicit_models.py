"""AF kinematic hardening models recast as augmented (implicit) residual systems.

These are test doubles used to validate the augmented NR machinery. Mathematically
identical to the corresponding reduced-path models at convergence.

The explicit update for alpha is:
    alpha_new = (alpha_n + C_k * dlambda * n_hat) / (1 + gamma * dlambda)

Rearranged as a residual:
    R_alpha = alpha_new * (1 + gamma * dlambda) - alpha_n - C_k * dlambda * n_hat = 0

n_hat is evaluated at (stress - alpha_n) — i.e., the OLD backstress — so both
paths are algebraically identical at convergence.
"""

from manforge.models.af_kinematic import AFKinematic3D, AFKinematicPS
from manforge.core.stress_state import PLANE_STRAIN


class _AFKinematicImplicit3D(AFKinematic3D):
    """AF kinematic 3D model with hardening expressed as an implicit residual."""

    hardening_type = "augmented"

    def state_residual(self, state_new, dlambda, stress, state_n):
        alpha_n = state_n["alpha"]
        xi = stress - alpha_n
        s_xi = self._dev(xi)
        vm_safe = self._vonmises(xi)
        n_hat = s_xi / vm_safe

        scale = 1.0 + self.gamma * dlambda
        R_alpha = state_new["alpha"] * scale - alpha_n - self.C_k * dlambda * n_hat
        R_ep = state_new["ep"] - state_n["ep"] - dlambda
        return {"alpha": R_alpha, "ep": R_ep}


class _AFKinematicImplicitPS(AFKinematicPS):
    """Plane-stress variant of the implicit AF model."""

    hardening_type = "augmented"

    def state_residual(self, state_new, dlambda, stress, state_n):
        alpha_n = state_n["alpha"]
        xi = stress - alpha_n
        s_xi = self._dev(xi)
        vm_safe = self._vonmises(xi)
        n_hat = s_xi / vm_safe

        scale = 1.0 + self.gamma * dlambda
        R_alpha = state_new["alpha"] * scale - alpha_n - self.C_k * dlambda * n_hat
        R_ep = state_new["ep"] - state_n["ep"] - dlambda
        return {"alpha": R_alpha, "ep": R_ep}


class _AFKinematicImplicitPE(AFKinematic3D):
    """Plane-strain variant of the implicit AF model (uses MaterialModel3D with PLANE_STRAIN)."""

    hardening_type = "augmented"

    def __init__(self):
        super().__init__(stress_state=PLANE_STRAIN,
                         E=210000.0, nu=0.3, sigma_y0=250.0, C_k=10000.0, gamma=100.0)

    def state_residual(self, state_new, dlambda, stress, state_n):
        alpha_n = state_n["alpha"]
        xi = stress - alpha_n
        s_xi = self._dev(xi)
        vm_safe = self._vonmises(xi)
        n_hat = s_xi / vm_safe

        scale = 1.0 + self.gamma * dlambda
        R_alpha = state_new["alpha"] * scale - alpha_n - self.C_k * dlambda * n_hat
        R_ep = state_new["ep"] - state_n["ep"] - dlambda
        return {"alpha": R_alpha, "ep": R_ep}
