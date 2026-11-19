"""Ohno-Wang models carrying a few hand-derived Jacobian blocks.

OWKinematic3D has every state implicit (σ, α, ep), so its Jacobian rows are
``stress``, ``dlambda``, ``alpha``, ``ep``.  The blocks below are the ones
that can be written down at sight:

* R_ep = ep − ep_n − Δλ                 → ∂/∂ep = 1, ∂/∂Δλ = −1, ∂/∂σ = ∂/∂α = 0
* R_Δλ = f = ‖dev σ − α‖_vm − σ_y0      → ∂/∂σ = n, ∂/∂α = −n, ∂/∂ep = ∂/∂Δλ = 0

Nine of the sixteen blocks; the σ and α rows are left undeclared on purpose.
"""

import numpy as np

from returnmap.core import hand_block
from returnmap.models import OWKinematic3D

OW_PARAMS = dict(E=210000.0, nu=0.3, sigma_y0=250.0, C_k=8000.0, gamma=80.0)


class OWWithHandBlocks(OWKinematic3D):
    """Exact hand blocks, written with different argument lists on purpose."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.seen: dict = {}

    @hand_block("ep", "ep")
    def dRep_dep(self):                       # needs nothing
        return 1.0

    @hand_block("ep", "dlambda")
    def dRep_dl(self, dlambda):               # value unused; records what it got
        self.seen["dlambda"] = dlambda
        return -1.0

    @hand_block("ep", "stress")
    def dRep_ds(self, state):
        return np.zeros(self.ntens)

    @hand_block("ep", "alpha")
    def dRep_da(self, state, dlambda, state_n):
        self.seen["state_n"] = state_n
        return np.zeros(self.ntens)

    @hand_block("dlambda", "stress")
    def dRy_ds(self, state):
        return self.flow(state).strain_like

    @hand_block("dlambda", "alpha")
    def dRy_da(self, inc, state):             # order of parameters is free
        self.seen["inc"] = inc
        return -self.flow(state).strain_like

    @hand_block("dlambda", "ep")
    def dRy_dep(self, state):
        return 0.0

    @hand_block("dlambda", "dlambda")
    def dRy_dl(self, state):
        return 0.0


class OWWithWrongBlock(OWWithHandBlocks):
    """One sign error, to be found by name."""

    @hand_block("ep", "dlambda")
    def dRep_dl(self, dlambda):
        return +1.0


class OWWithApproxBlock(OWWithHandBlocks):
    """A block that is off by design and says so."""

    @hand_block("ep", "ep", approx=1e-2, note="deliberately off by 1e-3 for the test")
    def dRep_dep(self):
        return 1.0 + 1e-3
