"""Multi-step crosscheck: crosscheck_return_mapping and crosscheck_stress_update tests.

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
    crosscheck_return_mapping,
    crosscheck_stress_update,
    iter_crosscheck_stress_update,
    FortranUMAT,
    generate_single_step_cases,
    generate_strain_history,
)
from manforge.simulation import StrainDriver, StressDriver
from manforge.simulation.types import FieldHistory, FieldType


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def fortran_j2():
    return FortranUMAT("j2_isotropic_3d")


_J2_PARAM_FN = lambda m: (m.E, m.nu, m.sigma_y0, m.H)


def _j2_load(model):
    history = generate_strain_history(model)
    return FieldHistory(FieldType.STRAIN, "eps", history)


# ---------------------------------------------------------------------------
# stress_update group — driver-based, multi-step history
# ---------------------------------------------------------------------------

def test_crosscheck_stress_update_numerical_newton(fortran_j2, model):
    """StrainDriver + tension-unload-compression: numerical_newton vs UMAT."""
    result = crosscheck_stress_update(
        StrainDriver(), model, fortran_j2, _j2_load(model),
        umat_subroutine="j2_isotropic_3d",
        param_fn=_J2_PARAM_FN,
        method="numerical_newton",
    )

    assert result.passed, (
        f"max_stress_rel_err = {result.max_stress_rel_err:.2e}"
    )
    assert result.max_stress_rel_err < 1e-6
    assert result.n_cases == result.n_passed


def test_crosscheck_stress_update_user_defined(fortran_j2, model):
    """StrainDriver + history: user_defined (analytical) vs UMAT."""
    result = crosscheck_stress_update(
        StrainDriver(), model, fortran_j2, _j2_load(model),
        umat_subroutine="j2_isotropic_3d",
        param_fn=_J2_PARAM_FN,
        method="user_defined",
    )

    assert result.passed, (
        f"max_stress_rel_err = {result.max_stress_rel_err:.2e}"
    )
    assert result.max_stress_rel_err < 1e-6


def test_crosscheck_stress_update_stress_driven(fortran_j2, model):
    """StressDriver: converged dstran replay gives matching Fortran stress."""
    sigma_max = 1.5 * model.sigma_y0
    targets = np.array([0.5 * sigma_max, sigma_max, 0.8 * sigma_max, 0.0])
    stress_data = np.zeros((len(targets), model.ntens))
    stress_data[:, 0] = targets
    load = FieldHistory(FieldType.STRESS, "sigma", stress_data)

    result = crosscheck_stress_update(
        StressDriver(), model, fortran_j2, load,
        umat_subroutine="j2_isotropic_3d",
        param_fn=_J2_PARAM_FN,
        method="numerical_newton",
    )

    assert result.passed, (
        f"max_stress_rel_err = {result.max_stress_rel_err:.2e}"
    )


def test_iter_crosscheck_stress_update_breakable(fortran_j2, model):
    """iter_* variant allows early break on first failing step."""
    history = generate_strain_history(model)
    load = FieldHistory(FieldType.STRAIN, "eps", history)

    # Use a wrong param_fn to force failure on first plastic step
    def bad_param_fn(m):
        return (m.sigma_y0, m.H, m.E, m.nu)  # intentionally wrong order

    found_failure = False
    for cr in iter_crosscheck_stress_update(
        StrainDriver(), model, fortran_j2, load,
        umat_subroutine="j2_isotropic_3d",
        param_fn=bad_param_fn,
        method="numerical_newton",
    ):
        if not cr.passed:
            found_failure = True
            break

    assert found_failure, "Expected at least one failing step with wrong param_fn"


# ---------------------------------------------------------------------------
# return_mapping group — test_cases based, single-step
# ---------------------------------------------------------------------------

def test_crosscheck_return_mapping_numerical_newton(fortran_j2, model):
    """Single-step cases (elastic/plastic/multiaxial) with numerical_newton."""
    test_cases = generate_single_step_cases(model)

    result = crosscheck_return_mapping(
        model, fortran_j2, test_cases,
        umat_subroutine="j2_isotropic_3d",
        param_fn=_J2_PARAM_FN,
        method="numerical_newton",
    )

    assert result.passed, (
        f"max_stress_rel_err = {result.max_stress_rel_err:.2e}, "
        f"failed cases: {[c.index for c in result.cases if not c.passed]}"
    )
    assert result.max_stress_rel_err < 1e-6


def test_crosscheck_return_mapping_user_defined(fortran_j2, model):
    """Single-step cases with user_defined (analytical) return mapping."""
    test_cases = generate_single_step_cases(model)

    result = crosscheck_return_mapping(
        model, fortran_j2, test_cases,
        umat_subroutine="j2_isotropic_3d",
        param_fn=_J2_PARAM_FN,
        method="user_defined",
    )

    assert result.passed, (
        f"max_stress_rel_err = {result.max_stress_rel_err:.2e}"
    )


# ---------------------------------------------------------------------------
# Negative cases
# ---------------------------------------------------------------------------

def test_method_required(fortran_j2, model):
    """Omitting method raises TypeError (it is a keyword-only required arg)."""
    history = generate_strain_history(model)
    load = FieldHistory(FieldType.STRAIN, "eps", history)

    with pytest.raises(TypeError):
        crosscheck_stress_update(
            StrainDriver(), model, fortran_j2, load,
            umat_subroutine="j2_isotropic_3d",
            param_fn=_J2_PARAM_FN,
            # method intentionally omitted
        )


def test_param_fn_order_sensitivity(fortran_j2, model):
    """Passing material params in wrong order produces a failed crosscheck."""
    result = crosscheck_stress_update(
        StrainDriver(), model, fortran_j2, _j2_load(model),
        umat_subroutine="j2_isotropic_3d",
        # Intentionally wrong order: UMAT expects (E, nu, sigma_y0, H)
        param_fn=lambda m: (m.sigma_y0, m.H, m.E, m.nu),
        method="numerical_newton",
    )

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

    def iter_run_steps(self, strain_history):
        stress = np.zeros(self.ntens)
        state = self.initial_state()
        eps_prev = np.zeros(self.ntens)
        for eps in strain_history:
            dstran = np.asarray(eps) - eps_prev
            eps_prev = np.asarray(eps).copy()
            stress = stress + self.E * dstran
            alpha = np.asarray(state["alpha"]) + self.H_kin * dstran
            ep = float(state["ep"]) + self.H_iso * float(np.sum(np.abs(dstran)))
            state = {"alpha": anp.array(alpha), "ep": anp.array(ep)}
            yield stress.copy(), state


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
    load = FieldHistory(FieldType.STRAIN, "eps", strain_data)

    # Drive Python side manually (MockKinematicModel is not a full MaterialModel)
    stress_py_list = []
    state_py: dict = model.initial_state()
    stress_py = np.zeros(ntens)
    eps_prev = np.zeros(ntens)
    for eps in strain_data:
        dstran = eps - eps_prev
        eps_prev = eps.copy()
        stress_py = stress_py + model.E * dstran
        alpha = np.asarray(state_py["alpha"]) + model.H_kin * dstran
        ep = float(state_py["ep"]) + model.H_iso * float(np.sum(np.abs(dstran)))
        state_py = {"alpha": anp.array(alpha), "ep": anp.array(ep)}
        stress_py_list.append(stress_py.copy())

    # Drive Fortran side using default hooks
    from manforge.verification.umat_crosscheck import (
        _default_state_to_args, _default_parse_umat_return,
    )
    state_f: dict = model.initial_state()
    stress_f_arr = np.zeros(ntens)
    eps_prev_f = np.zeros(ntens)
    for eps in strain_data:
        dstran = eps - eps_prev_f
        eps_prev_f = eps.copy()

        state_tup = _default_state_to_args(state_f, model.state_names)
        ret = mock_fortran.call(
            "mock_kinematic",
            model.E, model.H_kin, model.H_iso,
            stress_f_arr, *state_tup, dstran,
        )
        stress_f_arr, state_f = _default_parse_umat_return(
            ret, model.state_names, model.initial_state()
        )

    # Verify state shapes round-tripped correctly
    assert np.asarray(state_f["alpha"]).shape == (ntens,), (
        f"alpha shape mismatch: {np.asarray(state_f['alpha']).shape}"
    )
    assert np.ndim(state_f["ep"]) == 0, (
        f"ep should be scalar, got shape {np.asarray(state_f['ep']).shape}"
    )

    # Verify stress trajectories agree
    stress_py_arr = np.vstack(stress_py_list)
    np.testing.assert_allclose(
        stress_f_arr, stress_py_arr[-1], rtol=1e-10,
        err_msg="Fortran and Python stress diverge"
    )

    # Verify alpha matches non-physical linear formula
    expected_alpha = model.H_kin * np.sum(
        np.diff(strain_data, axis=0, prepend=np.zeros((1, ntens))), axis=0
    )
    np.testing.assert_allclose(
        np.asarray(state_f["alpha"]), expected_alpha, rtol=1e-10
    )
