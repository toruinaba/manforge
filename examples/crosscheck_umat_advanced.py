"""Advanced crosscheck example: full API tour.

Demonstrates all features of the manforge Fortran crosscheck API:

* Part 1 — CrosscheckStrainDriver   (multi-step, analytical integrator, ddsdde)
* Part 2 — iter_run streaming + early break on failure
* Part 3 — StressDriver path        (stress-controlled loading)
* Part 4 — FortranIntegrator        (explicit state_to_args via default hooks,
                                     for non-standard UMAT with ndarray state)

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
from manforge.simulation import (
    PythonNumericalIntegrator,
    PythonAnalyticalIntegrator,
    FortranIntegrator,
)
from manforge.simulation.types import FieldHistory, FieldType
from manforge.verification import (
    FortranUMAT,
    CrosscheckStrainDriver,
    CrosscheckStressDriver,
    generate_strain_history,
)

# ---------------------------------------------------------------------------
# Shared setup
# ---------------------------------------------------------------------------
model = J2Isotropic3D(E=210_000.0, nu=0.3, sigma_y0=250.0, H=1_000.0)

try:
    fortran_j2 = FortranUMAT("j2_isotropic_3d")
except ModuleNotFoundError:
    raise SystemExit(
        "j2_isotropic_3d not found.  Compile first:\n"
        "  uv run manforge build fortran/abaqus_stubs.f90 "
        "fortran/j2_isotropic_3d.f90 --name j2_isotropic_3d"
    )


def _make_fc_int(fortran):
    """Build a FortranIntegrator for j2_isotropic_3d."""
    return FortranIntegrator(
        fortran,
        "j2_isotropic_3d",
        param_fn=lambda: (model.E, model.nu, model.sigma_y0, model.H),
        state_names=model.state_names,
        initial_state=model.initial_state,
        elastic_stiffness=model.elastic_stiffness,
    )


history = generate_strain_history(model)
load = FieldHistory(FieldType.STRAIN, "eps", history)


# =========================================================================
# Part 1: CrosscheckStrainDriver — method="user_defined" + ddsdde
# =========================================================================
print("=" * 60)
print("  Part 1: CrosscheckStrainDriver (analytical, ddsdde)")
print("=" * 60)

py_int1 = PythonAnalyticalIntegrator(model)
fc_int1 = _make_fc_int(fortran_j2)

cc1 = CrosscheckStrainDriver(py_int1, fc_int1)
result1 = cc1.run(load)

print(f"  passed           : {result1.passed}  ({result1.n_passed}/{result1.n_cases} steps)")
print(f"  max stress err   : {result1.max_stress_rel_err:.2e}")
if result1.max_tangent_rel_err is not None:
    print(f"  max tangent err  : {result1.max_tangent_rel_err:.2e}")
ep_max = result1.max_state_rel_err.get("ep", None)
if ep_max is not None:
    print(f"  max ep err       : {ep_max:.2e}")

assert result1.passed, f"Part 1 failed: max_stress_rel_err={result1.max_stress_rel_err:.2e}"
print()


# =========================================================================
# Part 2: iter_run — step streaming + early break
# =========================================================================
print("=" * 60)
print("  Part 2: iter_run (streaming / early break)")
print("=" * 60)

py_int2 = PythonNumericalIntegrator(model)
fc_int2 = _make_fc_int(fortran_j2)
cc2 = CrosscheckStrainDriver(py_int2, fc_int2)

print("  Normal run (printing every 5th step):")
for cr in cc2.iter_run(load):
    if cr.index % 5 == 0:
        print(
            f"    step {cr.index:2d}: stress_err={cr.stress_rel_err:.1e}  "
            f"passed={cr.passed}"
        )

print()
print("  Early-break demo (wrong param_fn → detect first failing step):")
fc_int2_bad = FortranIntegrator(
    fortran_j2,
    "j2_isotropic_3d",
    param_fn=lambda: (model.sigma_y0, model.H, model.E, model.nu),  # wrong order
    state_names=model.state_names,
    initial_state=model.initial_state,
    elastic_stiffness=model.elastic_stiffness,
)
cc2_bad = CrosscheckStrainDriver(py_int2, fc_int2_bad)
first_fail_index = None
for cr in cc2_bad.iter_run(load):
    if not cr.passed:
        first_fail_index = cr.index
        print(f"    First failure at step {cr.index}: stress_err={cr.stress_rel_err:.2e}")
        # Jacobian inspection available on failure:
        # from manforge.verification import compare_jacobians
        # jac = compare_jacobians(model, cr.result_a, cr.result_b, cr.state_n)
        # print(jac.blocks)
        break

assert first_fail_index is not None, "Expected at least one failing step with wrong param_fn"
print()


# =========================================================================
# Part 3: StressDriver — stress-controlled path
# =========================================================================
print("=" * 60)
print("  Part 3: CrosscheckStressDriver (stress-controlled)")
print("=" * 60)

sigma_max = 1.5 * model.sigma_y0
targets = np.array([0.5 * sigma_max, sigma_max, 0.8 * sigma_max, 0.0])
stress_data = np.zeros((len(targets), model.ntens))
stress_data[:, 0] = targets
stress_load = FieldHistory(FieldType.STRESS, "sigma", stress_data)

py_int3 = PythonNumericalIntegrator(model)
fc_int3 = _make_fc_int(fortran_j2)
cc3 = CrosscheckStressDriver(py_int3, fc_int3)
result3 = cc3.run(stress_load)

print(f"  passed        : {result3.passed}  ({result3.n_passed}/{result3.n_cases} steps)")
print(f"  max stress err: {result3.max_stress_rel_err:.2e}")
for cr in result3.cases:
    print(
        f"  step {cr.index}: σ11_py={cr.py_stress[0]:.1f} MPa  "
        f"err={cr.stress_rel_err:.1e}"
    )

assert result3.passed, f"Part 3 failed: max_stress_rel_err={result3.max_stress_rel_err:.2e}"
print()


# =========================================================================
# Part 4: FortranIntegrator — ndarray state (alpha, ep) via default hooks
# =========================================================================
print("=" * 60)
print("  Part 4: FortranIntegrator (mock_kinematic, ndarray state)")
print("=" * 60)

try:
    fortran_mock = FortranUMAT("mock_kinematic")
except ModuleNotFoundError:
    print("  mock_kinematic not compiled — skipping Part 4.")
    print("  Compile with: uv run manforge build fortran/mock_kinematic.f90 --name mock_kinematic")
    fortran_mock = None

if fortran_mock is not None:
    import autograd.numpy as anp
    from manforge.core.stress_state import SOLID_3D

    class MockModel:
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

    # Python ground-truth (MockModel is not a full MaterialModel)
    stress_ref = np.zeros(ntens)
    alpha_ref  = np.zeros(ntens)
    ep_ref     = 0.0
    eps_prev   = np.zeros(ntens)
    for eps in strain_data:
        dstran     = eps - eps_prev
        eps_prev   = eps.copy()
        stress_ref = stress_ref + mock_model.E * dstran
        alpha_ref  = alpha_ref + mock_model.H_kin * dstran
        ep_ref     = ep_ref + mock_model.H_iso * float(np.sum(np.abs(dstran)))

    # Fortran side via FortranIntegrator (default hooks handle ndarray alpha + scalar ep)
    fc_mock = FortranIntegrator(
        fortran_mock,
        "mock_kinematic",
        param_fn=lambda: (mock_model.E, mock_model.H_kin, mock_model.H_iso),
        state_names=mock_model.state_names,
        initial_state=mock_model.initial_state,
        elastic_stiffness=lambda: mock_model.E * np.eye(ntens),
    )

    stress_f = np.zeros(ntens)
    state_f  = mock_model.initial_state()
    eps_prev_f = np.zeros(ntens)
    for eps in strain_data:
        dstran     = eps - eps_prev_f
        eps_prev_f = eps.copy()
        r          = fc_mock.stress_update(dstran, stress_f, state_f)
        stress_f   = np.asarray(r.stress, dtype=np.float64)
        state_f    = r.state

    alpha_f = np.asarray(state_f["alpha"])
    ep_f    = float(state_f["ep"])

    print(f"  stress_f  = {stress_f[:3]}  (ref: {stress_ref[:3]})")
    print(f"  alpha_f[0]= {alpha_f[0]:.6f}  (ref: {alpha_ref[0]:.6f})")
    print(f"  ep_f      = {ep_f:.6f}  (ref: {ep_ref:.6f})")

    np.testing.assert_allclose(stress_f, stress_ref, rtol=1e-10)
    np.testing.assert_allclose(alpha_f,  alpha_ref,  rtol=1e-10)
    np.testing.assert_allclose(ep_f,     ep_ref,     rtol=1e-10)
    assert alpha_f.shape == (ntens,)
    print("  FortranIntegrator default hook round-trip: PASS")
    print()


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
print("=" * 60)
print("  All checks passed.")
print("=" * 60)
