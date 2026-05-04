"""Multi-step crosscheck: CrosscheckStrainDriver / CrosscheckStressDriver tests.

Requires compiled Fortran modules:
    uv run manforge build fortran/abaqus_stubs.f90 fortran/j2_isotropic_3d.f90 \\
        --name j2_isotropic_3d
    uv run manforge build fortran/mock_kinematic.f90 --name mock_kinematic
"""

import numpy as np
import autograd.numpy as anp
import pytest

pytest.importorskip(
    "j2_isotropic_3d",
    reason="j2_isotropic_3d not compiled -- run: make fortran-build-umat",
)

pytestmark = pytest.mark.fortran

from manforge.verification import (
    CrosscheckStrainDriver,
    CrosscheckStressDriver,
    FortranUMAT,
    generate_strain_history,
)
from manforge.simulation import (
    StrainDriver,
    StressDriver,
    PythonNumericalIntegrator,
    PythonAnalyticalIntegrator,
    FortranIntegrator,
)
from manforge.simulation.types import FieldHistory, FieldType


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def fortran_j2():
    return FortranUMAT("j2_isotropic_3d")


def _j2_load(model):
    history = generate_strain_history(model)
    return FieldHistory(FieldType.STRAIN, "eps", history)


def _make_fc_int(fortran_j2, model):
    """Build a FortranIntegrator for j2_isotropic_3d."""
    return FortranIntegrator(
        fortran_j2,
        "j2_isotropic_3d",
        param_fn=lambda: (model.E, model.nu, model.sigma_y0, model.H),
        state_names=model.state_names,
        initial_state=model.initial_state,
        elastic_stiffness=model.elastic_stiffness,
    )


# ---------------------------------------------------------------------------
# stress_update group — driver-based, multi-step history
# ---------------------------------------------------------------------------

def test_crosscheck_stress_update_numerical_newton(fortran_j2, model):
    """StrainDriver + tension-unload-compression: numerical_newton vs UMAT."""
    py_int = PythonNumericalIntegrator(model)
    fc_int = _make_fc_int(fortran_j2, model)

    cc = CrosscheckStrainDriver(py_int, fc_int)
    result = cc.run(_j2_load(model))

    assert result.passed, f"max_stress_rel_err = {result.max_stress_rel_err:.2e}"
    assert result.max_stress_rel_err < 1e-6
    assert result.n_cases == result.n_passed


def test_crosscheck_stress_update_user_defined(fortran_j2, model):
    """StrainDriver + history: user_defined (analytical) vs UMAT."""
    py_int = PythonAnalyticalIntegrator(model)
    fc_int = _make_fc_int(fortran_j2, model)

    cc = CrosscheckStrainDriver(py_int, fc_int)
    result = cc.run(_j2_load(model))

    assert result.passed, f"max_stress_rel_err = {result.max_stress_rel_err:.2e}"
    assert result.max_stress_rel_err < 1e-6


def test_crosscheck_stress_update_stress_driven(fortran_j2, model):
    """StressDriver: converged dstran replay gives matching Fortran stress."""
    sigma_max = 1.5 * model.sigma_y0
    targets = np.array([0.5 * sigma_max, sigma_max, 0.8 * sigma_max, 0.0])
    stress_data = np.zeros((len(targets), model.ntens))
    stress_data[:, 0] = targets
    load = FieldHistory(FieldType.STRESS, "sigma", stress_data)

    py_int = PythonNumericalIntegrator(model)
    fc_int = _make_fc_int(fortran_j2, model)

    cc = CrosscheckStressDriver(py_int, fc_int)
    result = cc.run(load)

    assert result.passed, f"max_stress_rel_err = {result.max_stress_rel_err:.2e}"


def test_iter_crosscheck_stress_update_breakable(fortran_j2, model):
    """iter_run allows early break on first failing step."""
    load = _j2_load(model)

    py_int = PythonNumericalIntegrator(model)
    fc_int = FortranIntegrator(
        fortran_j2,
        "j2_isotropic_3d",
        param_fn=lambda: (model.sigma_y0, model.H, model.E, model.nu),  # wrong order
        state_names=model.state_names,
        initial_state=model.initial_state,
        elastic_stiffness=model.elastic_stiffness,
    )

    cc = CrosscheckStrainDriver(py_int, fc_int)

    found_failure = False
    for cr in cc.iter_run(load):
        if not cr.passed:
            found_failure = True
            break

    assert found_failure, "Expected at least one failing step with wrong param_fn"


# ---------------------------------------------------------------------------
# single-step group — CrosscheckStrainDriver with 1-step FieldHistory
# ---------------------------------------------------------------------------

def _make_single_step_cases(model):
    """Build single-step test cases spanning elastic and plastic regimes."""
    from manforge.verification.test_cases import estimate_yield_strain
    from manforge.simulation.integrator import PythonIntegrator as _PyInt
    eps_y = estimate_yield_strain(model)
    ntens = model.ntens
    ndi = model.stress_state.ndi
    nshr = model.stress_state.nshr
    state_0 = dict(model.initial_state())
    z = np.zeros(ntens)
    cases = []
    de = np.zeros(ntens); de[0] = 0.5 * eps_y
    cases.append({"strain_inc": de, "stress_n": z.copy(), "state_n": dict(state_0)})
    de = np.zeros(ntens); de[0] = 5.0 * eps_y
    cases.append({"strain_inc": de, "stress_n": z.copy(), "state_n": dict(state_0)})
    if ndi >= 2:
        de = np.zeros(ntens); de[0] = 3.0 * eps_y; de[1] = -1.5 * eps_y
        if ndi >= 3:
            de[2] = -1.5 * eps_y
        cases.append({"strain_inc": de, "stress_n": z.copy(), "state_n": dict(state_0)})
    if nshr >= 1:
        de = np.zeros(ntens); de[ndi] = 3.0 * eps_y
        cases.append({"strain_inc": de, "stress_n": z.copy(), "state_n": dict(state_0)})
    pre_de = np.zeros(ntens); pre_de[0] = 3.0 * eps_y
    _pre = _PyInt(model).stress_update(pre_de, np.zeros(ntens), model.initial_state())
    de2 = np.zeros(ntens); de2[0] = 2.0 * eps_y
    cases.append({"strain_inc": de2, "stress_n": np.array(_pre.stress),
                  "state_n": {k: np.asarray(v) for k, v in _pre.state.items()}})
    return cases


def _run_single_step_crosscheck(py_int, fc_int, model):
    """Drive each single-step case through CrosscheckStrainDriver."""
    cc = CrosscheckStrainDriver(py_int, fc_int)
    failures = []
    for case in _make_single_step_cases(model):
        data = case["strain_inc"][np.newaxis]  # shape (1, ntens)
        load = FieldHistory(FieldType.STRAIN, "eps", data)
        for cr in cc.iter_run(load, initial_stress=case["stress_n"],
                              initial_state=case["state_n"]):
            if not cr.passed:
                failures.append((case, cr))
    return failures


def test_crosscheck_single_step_numerical_newton(fortran_j2, model):
    """Single-step cases (elastic/plastic/multiaxial) with numerical_newton vs UMAT."""
    py_int = PythonNumericalIntegrator(model)
    fc_int = _make_fc_int(fortran_j2, model)
    failures = _run_single_step_crosscheck(py_int, fc_int, model)
    assert not failures, (
        f"{len(failures)} case(s) failed: "
        f"max stress_rel_err = {max(cr.stress_rel_err for _, cr in failures):.2e}"
    )


def test_crosscheck_single_step_user_defined(fortran_j2, model):
    """Single-step cases with user_defined (analytical) return mapping vs UMAT."""
    py_int = PythonAnalyticalIntegrator(model)
    fc_int = _make_fc_int(fortran_j2, model)
    failures = _run_single_step_crosscheck(py_int, fc_int, model)
    assert not failures, (
        f"{len(failures)} case(s) failed: "
        f"max stress_rel_err = {max(cr.stress_rel_err for _, cr in failures):.2e}"
    )


# ---------------------------------------------------------------------------
# Negative cases
# ---------------------------------------------------------------------------

def test_param_fn_order_sensitivity(fortran_j2, model):
    """Passing material params in wrong order produces a failed crosscheck."""
    py_int = PythonNumericalIntegrator(model)
    fc_int = FortranIntegrator(
        fortran_j2,
        "j2_isotropic_3d",
        param_fn=lambda: (model.sigma_y0, model.H, model.E, model.nu),  # wrong order
        state_names=model.state_names,
        initial_state=model.initial_state,
        elastic_stiffness=model.elastic_stiffness,
    )

    cc = CrosscheckStrainDriver(py_int, fc_int)
    result = cc.run(_j2_load(model))

    assert not result.passed, (
        "Expected crosscheck to fail with wrong param_fn order, "
        f"but max_stress_rel_err = {result.max_stress_rel_err:.2e}"
    )


# ---------------------------------------------------------------------------
# Multi-state mock UMAT (alpha: ndarray, ep: scalar)
# ---------------------------------------------------------------------------

class MockKinematicModel:
    """Python twin of fortran/mock_kinematic.f90 (non-physical linear update).

    stress_out = stress_in + E * dstran
    alpha_out  = alpha_in  + H_kin * dstran
    ep_out     = ep_in     + H_iso * sum(abs(dstran))
    """

    param_names = ["E", "H_kin", "H_iso"]
    state_names = ["alpha", "ep"]

    def __init__(self, *, E: float, H_kin: float, H_iso: float):
        self.E = E
        self.H_kin = H_kin
        self.H_iso = H_iso
        from manforge.core.stress_state import SOLID_3D
        self.stress_state = SOLID_3D

    @property
    def ntens(self):
        return self.stress_state.ntens

    def initial_state(self):
        return {
            "alpha": anp.zeros(self.ntens),
            "ep": anp.array(0.0),
        }


@pytest.fixture
def mock_fortran():
    mock_kinematic = pytest.importorskip(
        "mock_kinematic",
        reason="mock_kinematic not compiled -- run: "
               "uv run manforge build fortran/mock_kinematic.f90 --name mock_kinematic",
    )
    return FortranUMAT("mock_kinematic")


def test_crosscheck_multi_state_mock(mock_fortran):
    """Default state_to_args/parse_umat_return handle ndarray state (alpha, ep)."""
    model = MockKinematicModel(E=1.0, H_kin=0.1, H_iso=0.05)
    ntens = model.ntens

    n_steps = 10
    strain_data = np.zeros((n_steps, ntens))
    strain_data[:, 0] = np.linspace(1e-3, 5e-3, n_steps)

    # Drive Python side manually (MockKinematicModel is not a full MaterialModel)
    stress_py = np.zeros(ntens)
    state_py: dict = model.initial_state()
    eps_prev = np.zeros(ntens)
    for eps in strain_data:
        dstran = eps - eps_prev
        eps_prev = eps.copy()
        stress_py = stress_py + model.E * dstran
        alpha = np.asarray(state_py["alpha"]) + model.H_kin * dstran
        ep = float(state_py["ep"]) + model.H_iso * float(np.sum(np.abs(dstran)))
        state_py = {"alpha": anp.array(alpha), "ep": anp.array(ep)}

    # Drive Fortran side via FortranIntegrator using default hooks
    fc_int = FortranIntegrator(
        mock_fortran,
        "mock_kinematic",
        param_fn=lambda: (model.E, model.H_kin, model.H_iso),
        state_names=model.state_names,
        initial_state=model.initial_state,
        elastic_stiffness=lambda: model.E * np.eye(ntens),
    )

    stress_f = np.zeros(ntens)
    state_f: dict = model.initial_state()
    eps_prev_f = np.zeros(ntens)
    for eps in strain_data:
        dstran = eps - eps_prev_f
        eps_prev_f = eps.copy()
        result = fc_int.stress_update(dstran, stress_f, state_f)
        stress_f = np.asarray(result.stress, dtype=np.float64)
        state_f = result.state

    assert np.asarray(state_f["alpha"]).shape == (ntens,), (
        f"alpha shape mismatch: {np.asarray(state_f['alpha']).shape}"
    )
    assert np.ndim(state_f["ep"]) == 0, (
        f"ep should be scalar, got shape {np.asarray(state_f['ep']).shape}"
    )

    np.testing.assert_allclose(
        stress_f, stress_py, rtol=1e-10,
        err_msg="Fortran and Python stress diverge"
    )

    expected_alpha = model.H_kin * np.sum(
        np.diff(strain_data, axis=0, prepend=np.zeros((1, ntens))), axis=0
    )
    np.testing.assert_allclose(
        np.asarray(state_f["alpha"]), expected_alpha, rtol=1e-10
    )
