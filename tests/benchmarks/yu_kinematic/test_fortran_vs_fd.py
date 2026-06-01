"""Fortran analytical solution vs Python finite-difference verification.

This is the independent verification axis that catches "shared bugs" —
bugs that exist identically in both Python and Fortran, which Python==Fortran
tests (test_numerical_vs_fortran.py) cannot detect.

TestFortranJacobianVsFd:
  Fortran yu_calc_jacobian (analytical) vs FD of Fortran yu_calc_residual (truth).

TestFortranDdsddeVsFd:
  FortranIntegrator.stress_update.ddsdde (analytical) vs FD of Fortran stress_update (truth).
"""
import numpy as np
import pytest

pytest.importorskip(
    "yu_kinematic_3d",
    reason="yu_kinematic_3d module not compiled — run `make fortran-build-yu`",
)

pytestmark = pytest.mark.fortran

from manforge.models import YUKinematic3D
from manforge.simulation.integrator import (
    FortranModule,
    FortranIntegrator,
    PythonAnalyticalIntegrator,
    PythonNumericalIntegrator,
)
from tests.fixtures.yu_fd import (
    PARAMS,
    FD_RTOL_JAC,
    FD_ATOL_JAC,
    FD_RTOL,
    FD_ATOL,
    FD_RTOL_BAND,
    fd_jacobian,
    fd_ddsdde,
    run_steps,
    pick_largest_dlambda,
    pick_min_gstag,
    uniaxial_plastic_step,
    pure_shear_step,
    load_reversal_step,
    stagnation_active_step,
    stagnation_transition_band_history,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def fortran_mod():
    return FortranModule("yu_kinematic_3d")


@pytest.fixture(scope="module")
def yu_model():
    return YUKinematic3D(**PARAMS)


def _make_fc_int(fortran_mod, model):
    return FortranIntegrator.from_model(fortran_mod, "yu_kinematic_3d", model)


def _props(model):
    return (
        model.E, model.nu, model.Y, model.B, model.C_1, model.C_2,
        model.Rsat, model.k, model.b, model.h, model.Ea, model.xi,
    )


# ---------------------------------------------------------------------------
# TestFortranJacobianVsFd
# ---------------------------------------------------------------------------

class TestFortranJacobianVsFd:
    """Fortran calc_jacobian analytical vs FD of Fortran calc_residual."""

    def _run_scenario(self, model, fortran_mod, strain_history):
        """Return (state_new, state_n, dlambda, stress_trial) for the max-dlambda plastic step."""
        integrator = PythonNumericalIntegrator(model)
        step_data = run_steps(model, integrator, strain_history)
        picked = pick_largest_dlambda(model, step_data)
        if picked is None:
            return None
        result, stress_n, state_n, strain_inc = picked
        if not result.is_plastic:
            return None
        state_new = result.state
        dlambda = float(result.dlambda)
        C0 = model.elastic_stiffness(state_n)
        stress_trial = stress_n + C0 @ strain_inc
        return state_new, state_n, dlambda, stress_trial

    def _compare(self, model, fortran_mod, strain_history, label):
        out = self._run_scenario(model, fortran_mod, strain_history)
        if out is None:
            pytest.skip(f"Scenario '{label}' produced no plastic step")
        state_new, state_n, dlambda, stress_trial = out

        props = _props(model)

        # Analytical Jacobian from Fortran
        J_fortran = np.asarray(fortran_mod.call(
            "yu_calc_jacobian",
            *props,
            state_new["stress"], state_new["theta"], state_new["beta"],
            float(state_new["R"]), float(state_new["eps_eq"]),
            float(state_n["theta_max"]), float(state_n["R"]),
            state_n["q"], float(state_n["r"]), dlambda,
        ), dtype=float)

        # FD Jacobian — differencing Fortran yu_calc_residual
        def fortran_residual_fn(sn, dl):
            return np.asarray(fortran_mod.call(
                "yu_calc_residual",
                *props,
                sn["stress"], sn["theta"], sn["beta"],
                float(sn["R"]), float(sn["eps_eq"]),
                state_n["theta"], state_n["beta"], float(state_n["theta_max"]),
                stress_trial, dl,
            ), dtype=float)

        J_fd = fd_jacobian(model, state_new, state_n, dlambda, fortran_residual_fn)

        np.testing.assert_allclose(
            J_fortran, J_fd,
            rtol=FD_RTOL_JAC, atol=FD_ATOL_JAC,
            err_msg=f"Fortran Jacobian vs FD mismatch for scenario '{label}'"
        )

    def test_uniaxial(self, yu_model, fortran_mod):
        self._compare(yu_model, fortran_mod, uniaxial_plastic_step(), "uniaxial")

    def test_pure_shear(self, yu_model, fortran_mod):
        self._compare(yu_model, fortran_mod, pure_shear_step(), "pure_shear")

    def test_load_reversal(self, yu_model, fortran_mod):
        self._compare(yu_model, fortran_mod, load_reversal_step(), "load_reversal")

    def test_stagnation_active(self, yu_model, fortran_mod):
        self._compare(yu_model, fortran_mod, stagnation_active_step(), "stagnation_active")


# ---------------------------------------------------------------------------
# TestFortranDdsddeVsFd
# ---------------------------------------------------------------------------

class TestFortranDdsddeVsFd:
    """FortranIntegrator.stress_update.ddsdde (analytical) vs FD of Fortran stress_update.

    FortranIntegrator does not expose is_plastic/dlambda (UMAT convention).
    We use PythonAnalyticalIntegrator to identify the plastic step, then
    re-run the same (stress_n, state_n, strain_inc) through FortranIntegrator
    to obtain the Fortran DDSDDE and compare against FD of Fortran stress_update.
    """

    def _get_plastic_step(self, model, scenario):
        """Return (stress_n, state_n, strain_inc) for max-dlambda plastic step via Python."""
        py_int = PythonAnalyticalIntegrator(model)
        step_data = run_steps(model, py_int, scenario())
        picked = pick_largest_dlambda(model, step_data)
        return picked  # (result, stress_n, state_n, strain_inc) or None

    def _get_min_gstag_step(self, model, scenario):
        """Return ((result, stress_n, state_n, strain_inc), gstag) for min-gstag step."""
        py_int = PythonAnalyticalIntegrator(model)
        step_data = run_steps(model, py_int, scenario())
        return pick_min_gstag(model, step_data)

    @pytest.mark.parametrize("scenario,label", [
        (uniaxial_plastic_step,   "uniaxial"),
        (pure_shear_step,         "pure_shear"),
        (load_reversal_step,      "load_reversal"),
        (stagnation_active_step,  "stagnation_active"),
    ])
    def test_ddsdde_matches_fd(self, yu_model, fortran_mod, scenario, label):
        """Fortran DDSDDE must match central-FD dσ/dε within tolerance."""
        picked = self._get_plastic_step(yu_model, scenario)
        if picked is None:
            pytest.skip(f"Scenario '{label}' produced no plastic step")

        _result, stress_n, state_n, strain_inc = picked
        fc_int = _make_fc_int(fortran_mod, yu_model)

        D_an = np.array(fc_int.stress_update(strain_inc, stress_n, state_n).ddsdde, dtype=float)
        D_fd = fd_ddsdde(fc_int, strain_inc, stress_n, state_n)

        np.testing.assert_allclose(
            D_an, D_fd, rtol=FD_RTOL, atol=FD_ATOL,
            err_msg=f"Fortran DDSDDE vs FD mismatch for scenario '{label}'"
        )

    def test_ddsdde_stagnation_transition_band(self, yu_model, fortran_mod):
        """Fortran DDSDDE at the stagnation boundary must match FD (A1+A2 fix regression guard)."""
        picked, gstag = self._get_min_gstag_step(yu_model, stagnation_transition_band_history)
        if picked is None:
            pytest.skip("No plastic step found in stagnation transition band scenario")

        _result, stress_n, state_n, strain_inc = picked
        fc_int = _make_fc_int(fortran_mod, yu_model)

        D_an = np.array(fc_int.stress_update(strain_inc, stress_n, state_n).ddsdde, dtype=float)
        D_fd = fd_ddsdde(fc_int, strain_inc, stress_n, state_n)

        np.testing.assert_allclose(
            D_an, D_fd, rtol=FD_RTOL_BAND, atol=FD_ATOL,
            err_msg=f"Fortran DDSDDE vs FD mismatch in transition band (|g_stag|={gstag:.4f})"
        )
