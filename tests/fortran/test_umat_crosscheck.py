"""Multi-step crosscheck: crosscheck_umat harness tests.

Requires compiled Fortran modules:
    make fortran-build-umat
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

from manforge.verification import crosscheck_umat, FortranUMAT
from manforge.simulation import StrainDriver, StressDriver
from manforge.simulation.types import FieldHistory, FieldType
from manforge.verification import generate_strain_history


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def fortran_j2():
    return FortranUMAT("j2_isotropic_3d")


# ---------------------------------------------------------------------------
# J2 Isotropic — strain-driven
# ---------------------------------------------------------------------------

def test_crosscheck_j2_strain_driven(fortran_j2, model):
    """StrainDriver + tension-unload-compression history: stress trajectories match."""
    history = generate_strain_history(model)
    load = FieldHistory(FieldType.STRAIN, "eps", history)

    result = crosscheck_umat(
        StrainDriver(), model, fortran_j2,
        umat_subroutine="j2_isotropic_3d",
        load=load,
        param_fn=lambda m: (m.E, m.nu, m.sigma_y0, m.H),
    )

    assert result.passed, (
        f"max_stress_rel_err = {result.max_stress_rel_err:.2e} (tol 1e-6)"
    )
    assert result.max_stress_rel_err < 1e-6
    assert result.stress_py.shape == (len(history), model.ntens)
    assert result.stress_f.shape == result.stress_py.shape
    assert result.strain.shape == result.stress_py.shape


# ---------------------------------------------------------------------------
# J2 Isotropic — stress-driven
# ---------------------------------------------------------------------------

def test_crosscheck_j2_stress_driven(fortran_j2, model):
    """StressDriver: converged dstran replay gives matching Fortran stress."""
    eps_y = model.sigma_y0 / model.E
    # Build a small stress history: elastic → plastic → unload
    sigma_max = 1.5 * model.sigma_y0
    targets = np.array([
        0.5 * sigma_max,
        sigma_max,
        0.8 * sigma_max,
        0.0,
    ])
    stress_data = np.zeros((len(targets), model.ntens))
    stress_data[:, 0] = targets

    load = FieldHistory(FieldType.STRESS, "sigma", stress_data)

    result = crosscheck_umat(
        StressDriver(), model, fortran_j2,
        umat_subroutine="j2_isotropic_3d",
        load=load,
        param_fn=lambda m: (m.E, m.nu, m.sigma_y0, m.H),
    )

    assert result.passed, (
        f"max_stress_rel_err = {result.max_stress_rel_err:.2e} (tol 1e-6)"
    )


# ---------------------------------------------------------------------------
# param_fn order sensitivity
# ---------------------------------------------------------------------------

def test_param_fn_order_sensitivity(fortran_j2, model):
    """Passing material params in wrong order produces a failed crosscheck."""
    history = generate_strain_history(model)
    load = FieldHistory(FieldType.STRAIN, "eps", history)

    result = crosscheck_umat(
        StrainDriver(), model, fortran_j2,
        umat_subroutine="j2_isotropic_3d",
        load=load,
        # Intentionally wrong order: UMAT expects (E, nu, sigma_y0, H)
        param_fn=lambda m: (m.sigma_y0, m.H, m.E, m.nu),
    )

    assert not result.passed, (
        "Expected crosscheck to fail with wrong param_fn order, "
        f"but max_stress_rel_err = {result.max_stress_rel_err:.2e}"
    )


# ---------------------------------------------------------------------------
# Multi-state mock UMAT (ndarray state variable)
# ---------------------------------------------------------------------------

class MockKinematicModel:
    """Python twin of fortran/mock_kinematic.f90 (non-physical linear update).

    stress_out = stress_in + E * dstran
    alpha_out  = alpha_in  + H_kin * dstran
    ep_out     = ep_in     + H_iso * sum(abs(dstran))
    """

    from manforge.core.stress_state import SOLID_3D as _default_ss

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
        """Simple step-through matching mock_kinematic.f90 logic."""
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
    from manforge.simulation.types import FieldHistory, FieldType

    model = MockKinematicModel(E=1.0, H_kin=0.1, H_iso=0.05)

    # Build a simple uniaxial strain history
    ntens = model.ntens
    n_steps = 10
    strain_data = np.zeros((n_steps, ntens))
    strain_data[:, 0] = np.linspace(1e-3, 5e-3, n_steps)
    load = FieldHistory(FieldType.STRAIN, "eps", strain_data)

    # MockKinematicModel doesn't implement the full MaterialModel interface, so
    # we drive the Python side manually and build a DriverStep-like iterator.
    # Instead, use a thin adapter that wraps iter_run_steps as a StrainDriver.
    # Since MockKinematicModel is not a real MaterialModel, test the function
    # via a direct call using a custom driver shim.

    # Build Python stress history manually (ground truth)
    stress_py_list = []
    eps_prev = np.zeros(ntens)
    stress_f_init = np.zeros(ntens)
    state_f = model.initial_state()

    for i, eps in enumerate(strain_data):
        dstran = eps - eps_prev
        eps_prev = eps.copy()

        # Fortran call using default hooks
        from manforge.verification.umat_crosscheck import (
            _default_state_to_args, _default_parse_umat_return,
        )
        state_tup = _default_state_to_args(state_f, model.state_names)
        ret = mock_fortran.call(
            "mock_kinematic",
            model.E, model.H_kin, model.H_iso,
            stress_f_init, *state_tup, dstran,
        )
        stress_f_new, state_f = _default_parse_umat_return(
            ret, model.state_names, model.initial_state()
        )
        stress_f_init = stress_f_new

    # Verify state shapes are correctly round-tripped
    assert np.asarray(state_f["alpha"]).shape == (ntens,), (
        f"alpha shape mismatch: {np.asarray(state_f['alpha']).shape}"
    )
    assert np.ndim(state_f["ep"]) == 0, (
        f"ep should be scalar, got shape {np.asarray(state_f['ep']).shape}"
    )

    # Verify alpha values match the non-physical linear formula
    expected_alpha = model.H_kin * np.sum(
        np.diff(strain_data, axis=0, prepend=np.zeros((1, ntens))), axis=0
    )
    np.testing.assert_allclose(
        np.asarray(state_f["alpha"]), expected_alpha, rtol=1e-10
    )
