"""Verification of J2 analytical derivatives against AD (autodiff) values.

Demonstrates the step-by-step workflow for verifying closed-form solutions:

1. Run stress_update (numerical_newton) to get the numerical reference
2. Call user_defined_return_mapping / user_defined_tangent to get analytical results
3. Compare element-by-element with numpy.testing
4. Use ad_jacobian_blocks to inspect individual derivative blocks

This is the workflow a developer follows when implementing a new constitutive
model: first derive the analytical formulas, then compare each quantity against
the framework's AD-computed values before porting to Fortran.

Usage
-----
    uv run python examples/example_verify_j2_analytical.py
"""

import autograd
import autograd.numpy as anp
import numpy as np
import numpy.testing as npt

import manforge  # noqa: F401 — enables float64
from manforge.verification.jacobian import ad_jacobian_blocks
from manforge.core.state import _state_with_stress
from manforge.models.j2_isotropic import J2Isotropic3D
from manforge.simulation.driver import StrainDriver
from manforge.simulation.integrator import (
    PythonIntegrator,
    PythonNumericalIntegrator,
    PythonAnalyticalIntegrator,
)
from manforge.simulation.types import FieldHistory, FieldType

# ---------------------------------------------------------------------------
# Model (steel, MPa units)
# ---------------------------------------------------------------------------
model = J2Isotropic3D(E=210_000.0, nu=0.3, sigma_y0=250.0, H=1_000.0)

mu = model.E / (2.0 * (1.0 + model.nu))
state0 = model.initial_state()
C = model.elastic_stiffness(state0)


# =========================================================================
# Part 1: Single-step — numerical (AD) vs analytical (closed-form)
# =========================================================================
print("=" * 60)
print("  Part 1: Single plastic step — AD vs analytical")
print("=" * 60)

deps = anp.array([3e-3, 0.0, 0.0, 0.0, 0.0, 0.0])
stress_n = anp.zeros(6)

# --- numerical (Newton-Raphson) ---
result_ad = PythonNumericalIntegrator(model).stress_update(deps, stress_n, state0)

# --- analytical (user-defined closed-form) ---
result_an = PythonAnalyticalIntegrator(model).stress_update(deps, stress_n, state0)

# Compare stress
npt.assert_allclose(result_ad.stress, result_an.stress, rtol=1e-10)
print("  stress          : PASS")

# Compare dlambda
npt.assert_allclose(result_ad.dlambda, result_an.dlambda, rtol=1e-10)
print(f"  dlambda         : PASS  (= {float(result_ad.dlambda):.6e})")

# Compare state
npt.assert_allclose(result_ad.state["ep"], result_an.state["ep"], rtol=1e-10)
print(f"  state[ep]       : PASS  (= {float(result_ad.state['ep']):.6e})")

# Compare consistent tangent
npt.assert_allclose(result_ad.ddsdde, result_an.ddsdde, rtol=1e-10)
print("  ddsdde (tangent): PASS")

# Verify is_plastic / stress_trial match
assert result_ad.is_plastic is True
npt.assert_allclose(result_ad.stress_trial, result_an.stress_trial, rtol=1e-12)
print("  stress_trial    : PASS")
print()

# You can also call the hooks directly:
rm_an = model.user_defined_return_mapping(result_ad.stress_trial, C, state0)
npt.assert_allclose(result_ad.stress, rm_an.stress, rtol=1e-10)
npt.assert_allclose(result_ad.dlambda, rm_an.dlambda, rtol=1e-10)
print("  user_defined_return_mapping (direct call): PASS")

ddsdde_an = model.user_defined_tangent(
    result_ad.stress, result_ad.state, result_ad.dlambda, C, state0
)
npt.assert_allclose(result_ad.ddsdde, ddsdde_an, rtol=1e-10)
print("  user_defined_tangent (direct call): PASS")
print()


# =========================================================================
# Part 2: Jacobian blocks — verify individual derivative components
# =========================================================================
print("=" * 60)
print("  Part 2: Jacobian block inspection")
print("=" * 60)

jac = ad_jacobian_blocks(model, result_ad, state0)

# --- flow direction: dRdlambda_dsigma = df/dsigma = (3/2) s / sigma_vm ---
s = model._dev(result_ad.stress)
sigma_vm = model._vonmises(result_ad.stress)
n_analytical = (3.0 / 2.0) * s / sigma_vm

npt.assert_allclose(jac.dRdlambda_dsigma, n_analytical, rtol=1e-10)
print("  dRdlambda_dsigma (flow direction n): PASS")

# cross-check: also matches autograd.grad of yield_function
n_ad = autograd.grad(
    lambda sig: model.yield_function(_state_with_stress(result_ad.state, sig))
)(result_ad.stress)
npt.assert_allclose(jac.dRdlambda_dsigma, n_ad, rtol=1e-10)
print("  dRdlambda_dsigma == autograd.grad(f): PASS")

# --- dRsigma_ddlambda = C @ n (return mapping residual structure) ---
Cn = C @ n_analytical
npt.assert_allclose(jac.dRsigma_ddlambda, Cn, rtol=1e-10)
print("  dRsigma_ddlambda (= C @ n)     : PASS")

# --- dRdlambda_ddlambda = -H (for J2 linear isotropic hardening) ---
npt.assert_allclose(float(jac.dRdlambda_ddlambda), -model.H, rtol=1e-10)
print(f"  dRdlambda_ddlambda (= -H = {float(jac.dRdlambda_ddlambda):.1f}): PASS")

# --- reduced hardening: state blocks are empty dicts ---
assert jac.dRstate_dsigma == {}
assert jac.dRstate_dstate == {}
print("  state blocks (reduced model)   : empty dicts (correct)")

# --- full matrix shape ---
assert jac.full.shape == (7, 7)  # ntens + 1 = 6 + 1
print(f"  full Jacobian shape            : {jac.full.shape}")
print()


# =========================================================================
# Part 3: Driver — verify a specific plastic step
# =========================================================================
print("=" * 60)
print("  Part 3: Driver step-by-step verification")
print("=" * 60)

load = FieldHistory.cyclic_strain(
    peaks=[5e-3, -2e-3, 5e-3],
    n_per_segment=10,
    ntens=model.ntens,
    component=0,
)
driver_result = StrainDriver(PythonIntegrator(model)).run(load)

# Find the first plastic step
first_plastic = None
for i, rm in enumerate(driver_result.step_results):
    if rm.is_plastic:
        first_plastic = i
        break

print(f"  Total steps: {len(driver_result.step_results)}  (3 segments × 10 — cyclic)")
print(f"  First plastic step: {first_plastic}")
print()

# Verify a specific plastic step against analytical solution
step_idx = first_plastic + 5  # a few steps into plastic regime
rm = driver_result.step_results[step_idx]
state_prev = driver_result.step_results[step_idx - 1].state

# Call user_defined_return_mapping with the same inputs
rm_an = model.user_defined_return_mapping(rm.stress_trial, C, state_prev)
npt.assert_allclose(rm.stress, rm_an.stress, rtol=1e-10)
npt.assert_allclose(rm.dlambda, rm_an.dlambda, rtol=1e-10)
print(f"  Step {step_idx} user_defined_return_mapping: PASS")

# Call user_defined_tangent
ddsdde_an = model.user_defined_tangent(
    rm.stress, rm.state, rm.dlambda, C, state_prev
)
npt.assert_allclose(rm.ddsdde, ddsdde_an, rtol=1e-10)
print(f"  Step {step_idx} user_defined_tangent: PASS")

# Jacobian blocks for this step
jac = ad_jacobian_blocks(model, rm, state_prev)
n_step = autograd.grad(
    lambda sig: model.yield_function(_state_with_stress(rm.state, sig))
)(rm.stress)
npt.assert_allclose(jac.dRdlambda_dsigma, n_step, rtol=1e-10)
print(f"  Step {step_idx} Jacobian blocks: PASS")
print()

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
print("=" * 60)
print("  All verifications passed.")
print("=" * 60)
