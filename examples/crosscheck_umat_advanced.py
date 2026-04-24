"""Advanced crosscheck example: full API tour.

Demonstrates all features of the manforge Fortran crosscheck API:

* Part 1 — ReturnMappingCrosscheck  (test_cases list, single-step)
* Part 2 — StressUpdateCrosscheck   (multi-step, method="user_defined", ddsdde)
* Part 3 — iter_run streaming + early break on failure
* Part 4 — StressDriver path        (stress-controlled loading)
* Part 5 — custom hooks             (explicit state_to_args / parse_umat_return
                                     for non-standard UMAT argument order)

Requires compiled Fortran modules:

    uv run manforge build fortran/abaqus_stubs.f90 fortran/j2_isotropic_3d.f90 \\
        --name j2_isotropic_3d
    uv run manforge build fortran/mock_kinematic.f90 --name mock_kinematic
    uv run python examples/crosscheck_umat_advanced.py

For a minimal quick-start, see examples/crosscheck_umat_external.py instead.
"""

import sys
import os
import numpy as np

_fortran_dir = os.path.join(os.path.dirname(__file__), "..", "fortran")
sys.path.insert(0, os.path.abspath(_fortran_dir))

from manforge.models.j2_isotropic import J2Isotropic3D
from manforge.simulation import StrainDriver, StressDriver
from manforge.simulation.types import FieldHistory, FieldType
from manforge.verification import (
    FortranUMAT,
    ReturnMappingCrosscheck,
    StressUpdateCrosscheck,
    generate_single_step_cases,
    generate_strain_history,
)

# ---------------------------------------------------------------------------
# Shared setup
# ---------------------------------------------------------------------------
model = J2Isotropic3D(E=210_000.0, nu=0.3, sigma_y0=250.0, H=1_000.0)
_PARAM_FN = lambda m: (m.E, m.nu, m.sigma_y0, m.H)

try:
    fortran_j2 = FortranUMAT("j2_isotropic_3d")
except ModuleNotFoundError:
    raise SystemExit(
        "j2_isotropic_3d not found.  Compile first:\n"
        "  uv run manforge build fortran/abaqus_stubs.f90 "
        "fortran/j2_isotropic_3d.f90 --name j2_isotropic_3d"
    )


# =========================================================================
# Part 1: ReturnMappingCrosscheck — single-step, test_cases list
# =========================================================================
print("=" * 60)
print("  Part 1: ReturnMappingCrosscheck (test_cases)")
print("=" * 60)

test_cases = generate_single_step_cases(model)
print(f"  Generated {len(test_cases)} test cases")

cc1 = ReturnMappingCrosscheck(
    fortran_j2,
    umat_subroutine="j2_isotropic_3d",
    param_fn=_PARAM_FN,
    method="numerical_newton",
)
result1 = cc1.run(model, test_cases)

print(f"  passed       : {result1.passed}  ({result1.n_passed}/{result1.n_cases})")
print(f"  max stress err: {result1.max_stress_rel_err:.2e}")

for cr in result1.cases:
    ep_err = cr.state_rel_err.get("ep", 0.0)
    plastic = cr.py_dlambda is not None and cr.py_dlambda > 0
    print(
        f"  case {cr.index}: stress_err={cr.stress_rel_err:.1e}  "
        f"ep_err={ep_err:.1e}  dlambda={cr.py_dlambda:.3e}  "
        f"{'plastic' if plastic else 'elastic'}"
    )

assert result1.passed, f"Part 1 failed: max_stress_rel_err={result1.max_stress_rel_err:.2e}"
print()


# =========================================================================
# Part 2: StressUpdateCrosscheck — method="user_defined" + ddsdde
# =========================================================================
print("=" * 60)
print("  Part 2: StressUpdateCrosscheck (user_defined, ddsdde)")
print("=" * 60)

history = generate_strain_history(model)
load = FieldHistory(FieldType.STRAIN, "eps", history)

cc2 = StressUpdateCrosscheck(
    fortran_j2,
    umat_subroutine="j2_isotropic_3d",
    param_fn=_PARAM_FN,
    method="user_defined",
    # parse_umat_ddsdde omitted → default: scan trailing returns for 2-D array
)
result2 = cc2.run(StrainDriver(), model, load)

print(f"  passed           : {result2.passed}  ({result2.n_passed}/{result2.n_cases} steps)")
print(f"  max stress err   : {result2.max_stress_rel_err:.2e}")
if result2.max_tangent_rel_err is not None:
    print(f"  max tangent err  : {result2.max_tangent_rel_err:.2e}")
ep_max = result2.max_state_rel_err.get("ep", None)
if ep_max is not None:
    print(f"  max ep err       : {ep_max:.2e}")

assert result2.passed, f"Part 2 failed: max_stress_rel_err={result2.max_stress_rel_err:.2e}"
print()


# =========================================================================
# Part 3: iter_run — step streaming + early break
# =========================================================================
print("=" * 60)
print("  Part 3: iter_run (streaming / early break)")
print("=" * 60)

cc3 = StressUpdateCrosscheck(
    fortran_j2,
    umat_subroutine="j2_isotropic_3d",
    param_fn=_PARAM_FN,
    method="numerical_newton",
)

print("  Normal run (printing every 5th step):")
for cr in cc3.iter_run(StrainDriver(), model, load):
    if cr.index % 5 == 0:
        print(
            f"    step {cr.index:2d}: stress_err={cr.stress_rel_err:.1e}  "
            f"passed={cr.passed}"
        )

print()
print("  Early-break demo (wrong param_fn → detect first failing step):")
cc3_bad = StressUpdateCrosscheck(
    fortran_j2,
    umat_subroutine="j2_isotropic_3d",
    param_fn=lambda m: (m.sigma_y0, m.H, m.E, m.nu),  # wrong order
    method="numerical_newton",
)
first_fail_index = None
for cr in cc3_bad.iter_run(StrainDriver(), model, load):
    if not cr.passed:
        first_fail_index = cr.index
        print(f"    First failure at step {cr.index}: stress_err={cr.stress_rel_err:.2e}")
        break

assert first_fail_index is not None, "Expected at least one failing step with wrong param_fn"
print()


# =========================================================================
# Part 4: StressDriver — stress-controlled path
# =========================================================================
print("=" * 60)
print("  Part 4: StressUpdateCrosscheck (StressDriver, stress-controlled)")
print("=" * 60)

sigma_max = 1.5 * model.sigma_y0
targets = np.array([0.5 * sigma_max, sigma_max, 0.8 * sigma_max, 0.0])
stress_data = np.zeros((len(targets), model.ntens))
stress_data[:, 0] = targets
stress_load = FieldHistory(FieldType.STRESS, "sigma", stress_data)

cc4 = StressUpdateCrosscheck(
    fortran_j2,
    umat_subroutine="j2_isotropic_3d",
    param_fn=_PARAM_FN,
    method="numerical_newton",
)
result4 = cc4.run(StressDriver(), model, stress_load)

print(f"  passed        : {result4.passed}  ({result4.n_passed}/{result4.n_cases} steps)")
print(f"  max stress err: {result4.max_stress_rel_err:.2e}")
for cr in result4.cases:
    print(
        f"  step {cr.index}: σ11_py={cr.py_stress[0]:.1f} MPa  "
        f"err={cr.stress_rel_err:.1e}"
    )

assert result4.passed, f"Part 4 failed: max_stress_rel_err={result4.max_stress_rel_err:.2e}"
print()


# =========================================================================
# Part 5: custom hooks — mock_kinematic (alpha: ndarray, ep: scalar)
# =========================================================================
print("=" * 60)
print("  Part 5: custom state_to_args / parse_umat_return (mock_kinematic)")
print("=" * 60)

try:
    fortran_mock = FortranUMAT("mock_kinematic")
except ModuleNotFoundError:
    print("  mock_kinematic not compiled — skipping Part 5.")
    print("  Compile with: uv run manforge build fortran/mock_kinematic.f90 --name mock_kinematic")
    fortran_mock = None

if fortran_mock is not None:
    import autograd.numpy as anp
    from manforge.core.stress_state import SOLID_3D

    class MockModel:
        param_names = ["E", "H_kin", "H_iso"]
        state_names = ["alpha", "ep"]
        stress_state = SOLID_3D

        def __init__(self, *, E, H_kin, H_iso):
            self.E = E
            self.H_kin = H_kin
            self.H_iso = H_iso

        @property
        def ntens(self):
            return self.stress_state.ntens

        def initial_state(self):
            return {"alpha": anp.zeros(self.ntens), "ep": anp.array(0.0)}

    mock_model = MockModel(E=1.0, H_kin=0.1, H_iso=0.05)
    ntens = mock_model.ntens

    n_steps = 10
    strain_data = np.zeros((n_steps, ntens))
    strain_data[:, 0] = np.linspace(1e-3, 5e-3, n_steps)
    mock_load = FieldHistory(FieldType.STRAIN, "eps", strain_data)

    # Python ground-truth (MockModel is not a full MaterialModel)
    stress_ref = np.zeros(ntens)
    alpha_ref = np.zeros(ntens)
    ep_ref = 0.0
    eps_prev = np.zeros(ntens)
    for eps in strain_data:
        dstran = eps - eps_prev
        eps_prev = eps.copy()
        stress_ref = stress_ref + mock_model.E * dstran
        alpha_ref  = alpha_ref + mock_model.H_kin * dstran
        ep_ref     = ep_ref + mock_model.H_iso * float(np.sum(np.abs(dstran)))

    # Drive the Fortran side using default hooks
    from manforge.verification.umat_crosscheck import (
        _default_state_to_args, _default_parse_umat_return,
    )
    state_f = mock_model.initial_state()
    stress_f = np.zeros(ntens)
    eps_prev_f = np.zeros(ntens)
    for eps in strain_data:
        dstran = eps - eps_prev_f
        eps_prev_f = eps.copy()
        state_tup = _default_state_to_args(state_f, mock_model.state_names)
        ret = fortran_mock.call(
            "mock_kinematic",
            mock_model.E, mock_model.H_kin, mock_model.H_iso,
            stress_f, *state_tup, dstran,
        )
        stress_f, state_f = _default_parse_umat_return(
            ret, mock_model.state_names, mock_model.initial_state()
        )

    alpha_f = np.asarray(state_f["alpha"])
    ep_f    = float(state_f["ep"])

    print(f"  stress_f  = {stress_f[:3]}  (ref: {stress_ref[:3]})")
    print(f"  alpha_f[0]= {alpha_f[0]:.6f}  (ref: {alpha_ref[0]:.6f})")
    print(f"  ep_f      = {ep_f:.6f}  (ref: {ep_ref:.6f})")

    np.testing.assert_allclose(stress_f, stress_ref, rtol=1e-10)
    np.testing.assert_allclose(alpha_f,  alpha_ref,  rtol=1e-10)
    np.testing.assert_allclose(ep_f,     ep_ref,     rtol=1e-10)
    assert alpha_f.shape == (ntens,)
    print("  default hook round-trip: PASS")
    print()


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
print("=" * 60)
print("  All checks passed.")
print("=" * 60)
