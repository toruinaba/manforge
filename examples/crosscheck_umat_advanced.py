"""Advanced crosscheck example: full API tour.

Demonstrates all features of the manforge Fortran crosscheck API:

* Part 1 — CrosscheckStrainDriver   (multi-step, analytical integrator, ddsdde)
* Part 2 — iter_run streaming + early break on failure
* Part 3 — Jacobian inspection      (compare_jacobians / ad_jacobian_blocks)
* Part 4 — StressDriver path        (stress-controlled loading)
* Part 5 — FortranIntegrator        (explicit state_to_args via default hooks,
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
    FortranModule,
    CrosscheckStrainDriver,
    CrosscheckStressDriver,
    generate_strain_history,
    compare_jacobians,
)
from manforge.core.jacobian import ad_jacobian_blocks

# ---------------------------------------------------------------------------
# Shared setup
# ---------------------------------------------------------------------------
model = J2Isotropic3D(E=210_000.0, nu=0.3, sigma_y0=250.0, H=1_000.0)

fortran_j2 = FortranModule("j2_isotropic_3d")


def _make_fc_int(fortran):
    """Build a FortranIntegrator for j2_isotropic_3d."""
    return FortranIntegrator.from_model(fortran, "j2_isotropic_3d", model)


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
# Override param_fn to pass parameters in the wrong order — shows that
# from_model() lets you override any auto-derived argument.
fc_int2_bad = FortranIntegrator.from_model(
    fortran_j2,
    "j2_isotropic_3d",
    model,
    param_fn=lambda: (model.sigma_y0, model.H, model.E, model.nu),  # wrong order
)
cc2_bad = CrosscheckStrainDriver(py_int2, fc_int2_bad)
first_fail_index = None
for cr in cc2_bad.iter_run(load):
    if not cr.passed:
        first_fail_index = cr.index
        print(f"    First failure at step {cr.index}: stress_err={cr.stress_rel_err:.2e}")
        break

assert first_fail_index is not None, "Expected at least one failing step with wrong param_fn"
print()


# =========================================================================
# Part 3: Jacobian inspection — compare_jacobians / ad_jacobian_blocks
# =========================================================================
print("=" * 60)
print("  Part 3: Jacobian inspection")
print("=" * 60)

# --- 3a: compare_jacobians で 2 つの結果のブロック誤差を取得 ---
# CrosscheckCaseResult は result_a / result_b / state_n を保持しているので、
# 失敗ステップや検証したいステップで直接呼べる。
py_int3a = PythonNumericalIntegrator(model)
cc3a = CrosscheckStrainDriver(py_int3a, PythonAnalyticalIntegrator(model))
print("  3a: compare_jacobians (numerical_newton vs analytical, plastic step)")
for cr in cc3a.iter_run(load):
    if cr.result_a is not None and cr.result_a.is_plastic:
        jac_cmp = compare_jacobians(model, cr.result_a, cr.result_b, cr.state_n)
        print(f"     passed      : {jac_cmp.passed}")
        print(f"     max_rel_err : {jac_cmp.max_rel_err:.2e}")
        print(f"     blocks      :")
        for name, err in sorted(jac_cmp.blocks.items()):
            print(f"       {name:<30s}: {err:.2e}")
        break

print()

# --- 3b: ad_jacobian_blocks で個々のブロックを直接取り出す ---
# compare_jacobians を使わず、1 つの結果から JacobianBlocks を取得して
# 各微分項を個別に参照したい場合。
from manforge.simulation.integrator import PythonIntegrator as _PyInt

deps_plastic = np.zeros(model.ntens)
deps_plastic[0] = 5e-3          # 十分に塑性域
state_n_base = model.initial_state()
result_plastic = _PyInt(model).stress_update(deps_plastic, np.zeros(model.ntens), state_n_base)

print("  3b: ad_jacobian_blocks — individual block access")
jac = ad_jacobian_blocks(model, result_plastic, state_n_base)

# 応力残差の σ 偏微分ブロック（ntens × ntens）
print(f"     dstress_dsigma  shape : {np.asarray(jac.dstress_dsigma).shape}")
# 降伏面の σ 勾配（ntens,）— 法線ベクトル
print(f"     dyield_dsigma   shape : {np.asarray(jac.dyield_dsigma).shape}")
# Δλ に対する降伏感度（スカラー）
print(f"     dyield_ddlambda       : {float(jac.dyield_ddlambda):.6f}")
# state ブロック（reduced model では None; augmented では dict）
print(f"     dstate_dstate         : {jac.dstate_dstate!r}")
# 完全 Jacobian 行列（ntens+1 × ntens+1）
print(f"     full matrix shape     : {np.asarray(jac.full).shape}")
print()

# --- 3b-2: augmented モデル (OWKinematic3D) での残差 Jacobian state ブロック ---
# reduced モデルでは None だった dstress_dstate / dstate_dstate が
# augmented モデルでは状態変数名をキーとする dict になる。
# フィールド名の意味（"dstress" はすべて応力残差 R_σ の微分）:
#   dstress_dstate['alpha']         = ∂R_σ / ∂q_alpha   (ntens × ntens)
#   dstate_dstate['alpha']['alpha'] = ∂R_{q_alpha} / ∂q_alpha  (backstress 残差の自己微分)
#   dstate_dsigma['alpha']          = ∂R_{q_alpha} / ∂σ
from manforge.models.ow_kinematic import OWKinematic3D

ow_model = OWKinematic3D(E=210_000.0, nu=0.3, sigma_y0=250.0, C_k=5_000.0, gamma=50.0)
deps_ow = np.zeros(ow_model.ntens)
deps_ow[0] = 5e-3
state_n_ow = ow_model.initial_state()
result_ow = _PyInt(ow_model).stress_update(deps_ow, np.zeros(ow_model.ntens), state_n_ow)

print("  3b-2: augmented model (OWKinematic3D) — state residual blocks")
jac_ow = ad_jacobian_blocks(ow_model, result_ow, state_n_ow)
# ∂R_σ/∂q_alpha — 応力残差 の backstress 微分 (ntens × ntens)
print(f"     dstress_dstate['alpha'] shape         : {np.asarray(jac_ow.dstress_dstate['alpha']).shape}")
# ∂R_{q_alpha}/∂q_alpha — backstress 残差の自己微分 (ntens × ntens)
print(f"     dstate_dstate['alpha']['alpha'] shape : {np.asarray(jac_ow.dstate_dstate['alpha']['alpha']).shape}")
# ∂R_{q_alpha}/∂σ — backstress 残差の応力微分 (ntens × ntens)
print(f"     dstate_dsigma['alpha'] shape          : {np.asarray(jac_ow.dstate_dsigma['alpha']).shape}")
# full: ntens+1+n_state = 6+1+(6+1) = 14
print(f"     full matrix shape                     : {np.asarray(jac_ow.full).shape}")
print()

assert jac_cmp.passed, f"Part 3a failed: max_rel_err={jac_cmp.max_rel_err:.2e}"


# =========================================================================
# Part 4: StressDriver — stress-controlled path
# =========================================================================
print("=" * 60)
print("  Part 4: CrosscheckStressDriver (stress-controlled)")
print("=" * 60)

sigma_max = 1.5 * model.sigma_y0
targets = np.array([0.5 * sigma_max, sigma_max, 0.8 * sigma_max, 0.0])
stress_data = np.zeros((len(targets), model.ntens))
stress_data[:, 0] = targets
stress_load = FieldHistory(FieldType.STRESS, "sigma", stress_data)

py_int4 = PythonNumericalIntegrator(model)
fc_int4 = _make_fc_int(fortran_j2)
cc4 = CrosscheckStressDriver(py_int4, fc_int4)
result4 = cc4.run(stress_load)

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
# Part 4: FortranIntegrator — ndarray state (alpha, ep) via default hooks
# =========================================================================
print("=" * 60)
print("  Part 5: FortranIntegrator (mock_kinematic, ndarray state)")
print("=" * 60)

try:
    fortran_mock = FortranModule("mock_kinematic")
except ModuleNotFoundError:
    print("  mock_kinematic not compiled — skipping Part 5.")
    print("  Compile with: uv run manforge build fortran/mock_kinematic.f90 --name mock_kinematic")
    fortran_mock = None

if fortran_mock is not None:
    import autograd.numpy as anp
    from manforge.core.dimension import SOLID_3D

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

    # Fortran side via FortranIntegrator (default hooks handle ndarray alpha + scalar ep).
    fc_mock = FortranIntegrator.from_model(
        fortran_mock,
        "mock_kinematic",
        mock_model,
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
