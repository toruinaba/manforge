"""Custom model workflow: define your own model, then verify it step by step.

The workflow an external user follows when implementing a new constitutive
model (README § 5.6):

1. Define the model   — J2 + Voce (nonlinear isotropic) hardening, from scratch
2. NR convergence     — residual_history shows quadratic convergence
3. Jacobian blocks    — JacobianChecker vs hand-derived derivatives
4. FD tangent check   — consistent tangent vs central finite differences
5. Driver run         — full cyclic history

For crosschecking against a second implementation (closed-form hooks or a
Fortran UMAT), see examples/example_fortran_crosscheck.py.

Usage
-----
    uv run python examples/example_custom_model_workflow.py
"""

import autograd
import autograd.numpy as anp
import numpy as np
import numpy.testing as npt

import manforge  # noqa: F401 — enables float64
from manforge.core import Explicit, SCALAR
from manforge.core.material import MaterialModel
from manforge.core.dimension import SOLID_3D, StressDimension
from manforge.simulation.driver import StrainDriver
from manforge.simulation.integrator import PythonIntegrator, PythonNumericalIntegrator
from manforge.simulation.types import FieldHistory
from manforge.verification import JacobianChecker, TangentChecker

# =========================================================================
# 1. Define the model — J2 with Voce hardening:
#    sigma_y(ep) = sigma_y0 + Q * (1 - exp(-b * ep))
#    Explicit type: ep updates in closed form, NR solves scalar dlambda only.
# =========================================================================


class J2Voce(MaterialModel):
    param_names = ["E", "nu", "sigma_y0", "Q", "b"]
    ep = Explicit(shape=SCALAR, doc="equivalent plastic strain")

    def __init__(self, dimension: StressDimension = SOLID_3D, *, E, nu, sigma_y0, Q, b):
        super().__init__(dimension=dimension)
        self.E = E; self.nu = nu; self.sigma_y0 = sigma_y0; self.Q = Q; self.b = b

    def yield_function(self, state):
        sigma_y = self.sigma_y0 + self.Q * (1.0 - anp.exp(-self.b * state["ep"]))
        return self.vonmises(state["stress"]) - sigma_y

    def update_state(self, dlambda, state_new, state_n, *, stress_trial=None, strain_inc=None):
        return [self.ep(state_n["ep"] + dlambda)]


model = J2Voce(E=210_000.0, nu=0.3, sigma_y0=250.0, Q=150.0, b=20.0)
state0 = model.initial_state()

# =========================================================================
# 2. NR convergence — quadratic: residual drops by squaring each iteration
# =========================================================================
print("=" * 60)
print("  Step 2: NR convergence (residual_history)")
print("=" * 60)

deps = np.array([3e-3, 0.0, 0.0, 0.0, 0.0, 0.0])
result = PythonNumericalIntegrator(model).stress_update(deps, np.zeros(6), state0)

assert result.converged and result.is_plastic
print(f"  n_iterations     : {result.n_iterations}")
for k, r in enumerate(result.residual_history):
    print(f"  iter {k}: ||R|| = {r:.3e}")
print()

# =========================================================================
# 3. Jacobian blocks — compare each derivative against hand-derived formulas
# =========================================================================
print("=" * 60)
print("  Step 3: Jacobian blocks vs hand derivation")
print("=" * 60)

jac = JacobianChecker(model).compute(result, state0)

# flow direction: d f / d sigma = (3/2) s / sigma_vm
n_hand = 1.5 * model.dev(result.stress) / model.vonmises(result.stress)
npt.assert_allclose(jac.part["dlambda"]["stress"], n_hand, rtol=1e-10)
print("  part[dlambda][stress] = (3/2) s/sigma_vm      : PASS")

# hardening row: d f / d dlambda = -d sigma_y/d ep = -Q b exp(-b ep)
ep_new = float(result.state["ep"])
dfddl_hand = -model.Q * model.b * np.exp(-model.b * ep_new)
npt.assert_allclose(float(jac.part["dlambda"]["dlambda"]), dfddl_hand, rtol=1e-10)
print(f"  part[dlambda][dlambda] = -Q b exp(-b ep)      : PASS ({dfddl_hand:.2f})")

# cross-check the flow row against autograd directly
n_ad = autograd.grad(
    lambda sig: model.yield_function({**result.state, "stress": sig})
)(result.stress)
npt.assert_allclose(jac.part["dlambda"]["stress"], n_ad, rtol=1e-10)
print("  part[dlambda][stress] == autograd.grad(f)     : PASS")
print()

# =========================================================================
# 4. FD tangent check — consistent tangent vs central finite differences
# =========================================================================
print("=" * 60)
print("  Step 4: consistent tangent vs finite differences")
print("=" * 60)

checker = TangentChecker(PythonIntegrator(model))
for label, d in [("elastic", np.full(6, 1e-5)), ("plastic", deps)]:
    r = checker.check(np.zeros(6), model.initial_state(), d)
    status = "PASS" if r.passed else "FAIL"
    print(f"  [{status}] {label:8s} max rel err = {r.max_rel_err:.2e}")
print()

# =========================================================================
# 5. Driver run — cyclic history, saturation visible in peak stress
# =========================================================================
print("=" * 60)
print("  Step 5: cyclic driver run")
print("=" * 60)

load = FieldHistory.cyclic_strain(
    peaks=[8e-3, -8e-3, 8e-3, -8e-3], n_per_segment=25, ntens=6)
dr = StrainDriver(PythonIntegrator(model)).run(load)

peaks = [float(dr.stress[(i + 1) * 25 - 1, 0]) for i in range(4)]
print(f"  Total steps   : {len(dr.step_results)}")
print(f"  Peak |sigma11|: {[f'{abs(p):.1f}' for p in peaks]}  (Voce saturation)")
print(f"  Final ep      : {float(dr.step_results[-1].state['ep']):.4e}")
print()
print("All verifications passed.")
