"""Multi-step crosscheck: SolverCrosscheck and StressUpdateCrosscheck tests.

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
    SolverCrosscheck,
    StressUpdateCrosscheck,
    FortranUMAT,
    generate_single_step_cases,
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

    cc = StressUpdateCrosscheck(py_int, fc_int)
    result = cc.run(StrainDriver(), _j2_load(model))

    assert result.passed, f"max_stress_rel_err = {result.max_stress_rel_err:.2e}"
    assert result.max_stress_rel_err < 1e-6
    assert result.n_cases == result.n_passed


def test_crosscheck_stress_update_user_defined(fortran_j2, model):
    """StrainDriver + history: user_defined (analytical) vs UMAT."""
    py_int = PythonAnalyticalIntegrator(model)
    fc_int = _make_fc_int(fortran_j2, model)

    cc = StressUpdateCrosscheck(py_int, fc_int)
    result = cc.run(StrainDriver(), _j2_load(model))

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

    cc = StressUpdateCrosscheck(py_int, fc_int)
    result = cc.run(StressDriver(), load)

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

    cc = StressUpdateCrosscheck(py_int, fc_int)

    found_failure = False
    for cr in cc.iter_run(StrainDriver(), load):
        if not cr.passed:
            found_failure = True
            break

    assert found_failure, "Expected at least one failing step with wrong param_fn"


# ---------------------------------------------------------------------------
# single-step group — SolverCrosscheck with FortranIntegrator
# ---------------------------------------------------------------------------

def test_crosscheck_single_step_numerical_newton(fortran_j2, model):
    """Single-step cases (elastic/plastic/multiaxial) with numerical_newton vs UMAT."""
    py_int = PythonNumericalIntegrator(model)
    fc_int = _make_fc_int(fortran_j2, model)
    cs = SolverCrosscheck(py_int, fc_int)
    result = cs.run(generate_single_step_cases(model))

    assert result.passed, (
        f"max_stress_rel_err = {result.max_stress_rel_err:.2e}, "
        f"failed cases: {[c.index for c in result.cases if not c.passed]}"
    )
    assert result.max_stress_rel_err < 1e-6


def test_crosscheck_single_step_user_defined(fortran_j2, model):
    """Single-step cases with user_defined (analytical) return mapping vs UMAT."""
    py_int = PythonAnalyticalIntegrator(model)
    fc_int = _make_fc_int(fortran_j2, model)
    cs = SolverCrosscheck(py_int, fc_int)
    result = cs.run(generate_single_step_cases(model))

    assert result.passed, f"max_stress_rel_err = {result.max_stress_rel_err:.2e}"


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

    cc = StressUpdateCrosscheck(py_int, fc_int)
    result = cc.run(StrainDriver(), _j2_load(model))

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
