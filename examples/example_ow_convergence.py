"""NR convergence history verification for explicit and augmented hardening models.

Shows how to access Newton-Raphson convergence history via
``StressUpdateResult.residual_history`` and verify quadratic convergence.

- J2Isotropic3D (reduced hardening, scalar NR): converges in exactly 1 step
  for linear isotropic hardening (exact linearization).
- OWKinematic3D (augmented hardening, augmented NR): multiple iterations with
  quadratic convergence.

Usage
-----
    uv run python examples/example_ow_convergence.py
"""

import math

import jax.numpy as jnp
import numpy as np

import manforge  # noqa: F401 — enables JAX float64
from manforge.core.stress_update import stress_update
from manforge.models.j2_isotropic import J2Isotropic3D
from manforge.models.ow_kinematic import OWKinematic3D
from manforge.simulation.driver import StrainDriver
from manforge.simulation.integrator import PythonIntegrator
from manforge.simulation.types import FieldHistory, FieldType

deps = jnp.array([3e-3, 0.0, 0.0, 0.0, 0.0, 0.0])


# =========================================================================
# Part 1: J2 reduced hardening — scalar NR, linear → 1 iteration
# =========================================================================
print("=" * 60)
print("  Part 1: J2Isotropic3D — reduced hardening (scalar NR)")
print("=" * 60)

j2 = J2Isotropic3D(E=210_000.0, nu=0.3, sigma_y0=250.0, H=1_000.0)
result_j2 = stress_update(j2, deps, jnp.zeros(6), j2.initial_state(),
                           method="numerical_newton")

print(f"  is_plastic       : {result_j2.is_plastic}")
print(f"  n_iterations     : {result_j2.n_iterations}")
print(f"  residual_history : {[f'{r:.3e}' for r in result_j2.residual_history]}")
print(f"  final residual   : {result_j2.residual_history[-1]:.3e}")
assert result_j2.is_plastic
assert result_j2.n_iterations == 1, (
    f"J2 linear hardening should converge in 1 NR step, got {result_j2.n_iterations}"
)
print("  => Converges in exactly 1 Newton step (linear hardening is exact).")
print()


# =========================================================================
# Part 2: OWKinematic3D — augmented NR, quadratic convergence
# =========================================================================
print("=" * 60)
print("  Part 2: OWKinematic3D — augmented hardening (augmented NR)")
print("=" * 60)

ow = OWKinematic3D(E=210_000.0, nu=0.3, sigma_y0=250.0, C_k=50_000.0, gamma=500.0)
result_ow = stress_update(ow, deps, jnp.zeros(6), ow.initial_state())

history = result_ow.residual_history
print(f"  is_plastic       : {result_ow.is_plastic}")
print(f"  n_iterations     : {result_ow.n_iterations}")
print(f"  residual_history :")
for i, r in enumerate(history):
    label = "initial" if i == 0 else f"iter {i}"
    print(f"    [{label:>8s}]  {r:.6e}")
print()
assert result_ow.is_plastic
assert result_ow.n_iterations >= 1
assert history[-1] < 1e-10, f"final residual {history[-1]:.3e} not converged"

# Estimate local convergence order p ≈ log(e_{k+1}/e_k) / log(e_k/e_{k-1})
if len(history) >= 3:
    orders = []
    for i in range(1, len(history) - 1):
        e_prev, e_curr, e_next = history[i - 1], history[i], history[i + 1]
        if e_prev > 0 and e_curr > 0 and e_next > 0:
            den = math.log(e_curr / e_prev)
            if abs(den) > 1e-30:
                orders.append(math.log(e_next / e_curr) / den)
    if orders:
        print(f"  Local convergence orders : {[f'{p:.2f}' for p in orders]}")
        print(f"  Final order estimate     : {orders[-1]:.2f}  (expected ≈ 2.0)")
        assert orders[-1] > 1.5, (
            f"Expected near-quadratic convergence, got {orders[-1]:.2f}"
        )
        print("  => Quadratic convergence confirmed.")
print()


# =========================================================================
# Part 3: StrainDriver — convergence history across all steps
# =========================================================================
print("=" * 60)
print("  Part 3: StrainDriver — OW history across loading steps")
print("=" * 60)

driver = StrainDriver()
load = FieldHistory(FieldType.STRAIN, "Strain", np.linspace(0.0, 5e-3, 20))
dr = driver.run(PythonIntegrator(ow), load)

elastic_count = sum(1 for rm in dr.step_results if not rm.is_plastic)
plastic_count = sum(1 for rm in dr.step_results if rm.is_plastic)
max_iters = max((rm.n_iterations for rm in dr.step_results if rm.is_plastic), default=0)
max_final_res = max(
    (rm.residual_history[-1] for rm in dr.step_results if rm.is_plastic),
    default=0.0,
)

print(f"  Total steps      : {len(dr.step_results)}")
print(f"  Elastic steps    : {elastic_count}")
print(f"  Plastic steps    : {plastic_count}")
print(f"  Max NR iters     : {max_iters}")
print(f"  Max final resid  : {max_final_res:.3e}")
for rm in dr.step_results:
    if rm.is_plastic:
        assert rm.n_iterations >= 1
        assert rm.residual_history[-1] < 1e-10
    else:
        assert rm.n_iterations == 0
        assert rm.residual_history == []
print("  => All plastic steps converged to tol=1e-10.")
print()

print("=" * 60)
print("  All checks passed.")
print("=" * 60)
