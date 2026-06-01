"""Path B benchmarks: YUKinematic3D Python vs Fortran subroutines.

Requires the compiled yu_kinematic_3d Fortran extension:
    make fortran-build-yu
or:
    uv run manforge build fortran/abaqus_stubs.f90 fortran/yu_kinematic_3d.f90 --name yu_kinematic_3d

Test classes:
    TestHelpers              -- elastic_stiffness, calc_norm_n_flow,
                                _prepare_Rstress, _prepare_Rtheta
    TestJacobianBlocks       -- one test per dRxx_dyy block (16 total)
    TestResidualAndJacobian  -- calc_residual + calc_jacobian over trajectories
    TestReturnMapping        -- yu_kinematic_3d core (return mapping + ddsdde)
    TestCrosscheckTrajectory -- PythonAnalyticalIntegrator vs FortranIntegrator
    TestUmatShim             -- ABAQUS umat entry point (PROPS/STATEV packing, PNEWDT)
    TestParamOrder           -- param_names order matches yu_kinematic_3d dummy args
"""

import re
import numpy as np
import pytest

pytest.importorskip(
    "yu_kinematic_3d",
    reason=(
        "yu_kinematic_3d not compiled -- run: "
        "make fortran-build-yu"
    ),
)

pytestmark = pytest.mark.fortran

from manforge.simulation.integrator import FortranModule, PythonNumericalIntegrator, PythonAnalyticalIntegrator, FortranIntegrator
from manforge.simulation.types import FieldHistory, FieldType
from manforge.verification.crosscheck_driver import CrosscheckStrainDriver
from manforge.models import YUKinematic3D

from .conftest import PARAMS


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def fortran_mod():
    return FortranModule("yu_kinematic_3d")


@pytest.fixture
def model():
    return YUKinematic3D(**PARAMS)


@pytest.fixture
def plastic_state(model):
    """Run one plastic step; return (model, state, state_n, dlambda)."""
    integrator = PythonNumericalIntegrator(model)
    stress_n = np.zeros(6)
    state_n = model.initial_state()
    strain_inc = np.array([2e-3, -6e-4, -6e-4, 0.0, 0.0, 0.0])
    result = integrator.stress_update(strain_inc, stress_n, state_n)
    assert result.is_plastic
    return model, result.state, state_n, float(result.dlambda)


# ---------------------------------------------------------------------------
# TestHelpers: helper subroutines at a single plastic step
# ---------------------------------------------------------------------------

class TestHelpers:
    """Compare helper subroutines Python vs Fortran at a single plastic step."""

    def test_elastic_stiffness(self, plastic_state, fortran_mod):
        m, state, _, _ = plastic_state
        py = m.elastic_stiffness(state)
        f = fortran_mod.call(
            "yu_kinematic_3d_elastic_stiffness",
            m.E, m.nu, float(state["eps_eq"]), m.Ea, m.xi,
        )
        np.testing.assert_allclose(py, f, rtol=1e-12, atol=1e-12)

    def test_calc_norm_n_flow(self, plastic_state, fortran_mod):
        m, state, _, _ = plastic_state
        xi = m.dev(state["stress"]) - state["theta"] - state["beta"]
        xi_norm_py, flow_py = m.calc_norm_n_flow(xi)
        xi_norm_f, flow_f = fortran_mod.call("yu_calc_norm_n_flow", xi)
        np.testing.assert_allclose(xi_norm_py, xi_norm_f, rtol=1e-12)
        np.testing.assert_allclose(flow_py, flow_f, rtol=1e-12)

    def test_prepare_rstress(self, plastic_state, fortran_mod):
        m, state, _, _ = plastic_state
        xi = m.dev(state["stress"]) - state["theta"] - state["beta"]
        py = m._prepare_Rstress(xi)
        f = fortran_mod.call("yu_prepare_rstress", xi)
        np.testing.assert_allclose(py, f, atol=1e-12)

    def test_prepare_rtheta(self, plastic_state, fortran_mod):
        m, state, state_n, dlambda = plastic_state
        theta = state["theta"]
        theta_max = float(state_n["theta_max"])  # state_n: consistent with residual C_k
        R = float(state["R"])
        R_n = float(state_n["R"])
        py = m._prepare_Rtheta(theta, theta_max, R, R_n, dlambda)
        theta_bar_py, theta_flow_py, C_k_py, s_py, a_py, a_prime_py = py
        f = fortran_mod.call(
            "yu_prepare_rtheta",
            m.B, m.Y, m.k, m.Rsat, m.C_1, m.C_2,
            theta, theta_max, R, R_n, dlambda,
        )
        theta_bar_f, theta_flow_f, C_k_f, s_f, a_f, a_prime_f = f
        np.testing.assert_allclose(theta_bar_py, theta_bar_f, rtol=1e-12)
        np.testing.assert_allclose(theta_flow_py, theta_flow_f, rtol=1e-12)
        np.testing.assert_allclose(C_k_py, C_k_f, rtol=1e-12)
        np.testing.assert_allclose(s_py, s_f, rtol=1e-12)
        np.testing.assert_allclose(a_py, a_f, rtol=1e-12)
        np.testing.assert_allclose(a_prime_py, a_prime_f, rtol=1e-12)


# ---------------------------------------------------------------------------
# TestJacobianBlocks: one test per dRxx_dyy block (16 total)
# ---------------------------------------------------------------------------

class TestJacobianBlocks:
    """Compare each Jacobian block Python vs Fortran at a single plastic step."""

    # --- R_stress blocks ---

    def test_dRstress_dstress(self, plastic_state, fortran_mod):
        m, state, _, dlambda = plastic_state
        C = m.elastic_stiffness(state)
        xi = m.dev(state["stress"]) - state["theta"] - state["beta"]
        py = m.dRstress_dstress(C, xi, dlambda)
        f = fortran_mod.call("yu_drs_dstress", C, xi, dlambda)
        np.testing.assert_allclose(py, f, rtol=1e-10, atol=1e-12)

    def test_dRstress_dbeta(self, plastic_state, fortran_mod):
        m, state, _, dlambda = plastic_state
        C = m.elastic_stiffness(state)
        xi = m.dev(state["stress"]) - state["theta"] - state["beta"]
        py = m.dRstress_dbeta(C, xi, dlambda)
        f = fortran_mod.call("yu_drs_dbeta", C, xi, dlambda)
        np.testing.assert_allclose(py, f, rtol=1e-10, atol=1e-12)

    def test_dRstress_dtheta(self, plastic_state, fortran_mod):
        m, state, _, dlambda = plastic_state
        C = m.elastic_stiffness(state)
        xi = m.dev(state["stress"]) - state["theta"] - state["beta"]
        py = m.dRstress_dtheta(C, xi, dlambda)
        f = fortran_mod.call("yu_drs_dtheta", C, xi, dlambda)
        np.testing.assert_allclose(py, f, rtol=1e-10, atol=1e-12)

    def test_dRstress_dlambda(self, plastic_state, fortran_mod):
        m, state, _, dlambda = plastic_state
        C = m.elastic_stiffness(state)
        xi = m.dev(state["stress"]) - state["theta"] - state["beta"]
        eps_eq = float(state["eps_eq"])
        py = m.dRstress_dlambda(C, xi, eps_eq, dlambda)
        f = fortran_mod.call("yu_drs_dlambda", m.E, m.Ea, m.xi, C, xi, eps_eq, dlambda)
        np.testing.assert_allclose(py, f, rtol=1e-10, atol=1e-12)

    # --- R_beta blocks ---

    def test_dRbeta_dstress(self, plastic_state, fortran_mod):
        m, state, _, dlambda = plastic_state
        py = m.dRbeta_dstress(dlambda)
        f = fortran_mod.call("yu_drb_dstress", m.k, m.b, m.Y, dlambda)
        np.testing.assert_allclose(py, f, rtol=1e-10, atol=1e-12)

    def test_dRbeta_dbeta(self, plastic_state, fortran_mod):
        m, state, _, dlambda = plastic_state
        py = m.dRbeta_dbeta(dlambda)
        f = fortran_mod.call("yu_drb_dbeta", m.k, m.b, m.Y, dlambda)
        np.testing.assert_allclose(py, f, rtol=1e-10, atol=1e-12)

    def test_dRbeta_dtheta(self, plastic_state, fortran_mod):
        m, state, _, dlambda = plastic_state
        py = m.dRbeta_dtheta(dlambda)
        f = fortran_mod.call("yu_drb_dtheta", m.k, m.b, m.Y, dlambda)
        np.testing.assert_allclose(py, f, rtol=1e-10, atol=1e-12)

    def test_dRbeta_dlambda(self, plastic_state, fortran_mod):
        m, state, _, dlambda = plastic_state
        xi = m.dev(state["stress"]) - state["theta"] - state["beta"]
        beta = state["beta"]
        py = m.dRbeta_dlambda(xi, beta, dlambda)
        f = fortran_mod.call("yu_drb_dlambda", m.k, m.b, m.Y, xi, beta, dlambda)
        np.testing.assert_allclose(py, f, rtol=1e-10, atol=1e-12)

    # --- R_theta blocks ---

    def test_dRtheta_dstress(self, plastic_state, fortran_mod):
        m, state, state_n, dlambda = plastic_state
        theta = state["theta"]
        theta_max = float(state["theta_max"])
        R = float(state["R"])
        R_n = float(state_n["R"])
        py = m.dRtheta_dstress(theta, theta_max, R, R_n, dlambda)
        f = fortran_mod.call(
            "yu_drt_dstress",
            m.B, m.Y, m.k, m.Rsat, m.C_1, m.C_2,
            theta, theta_max, R, R_n, dlambda,
        )
        np.testing.assert_allclose(py, f, rtol=1e-10, atol=1e-12)

    def test_dRtheta_dbeta(self, plastic_state, fortran_mod):
        m, state, state_n, dlambda = plastic_state
        theta = state["theta"]
        theta_max = float(state["theta_max"])
        R = float(state["R"])
        R_n = float(state_n["R"])
        py = m.dRtheta_dbeta(theta, theta_max, R, R_n, dlambda)
        f = fortran_mod.call(
            "yu_drt_dbeta",
            m.B, m.Y, m.k, m.Rsat, m.C_1, m.C_2,
            theta, theta_max, R, R_n, dlambda,
        )
        np.testing.assert_allclose(py, f, rtol=1e-10, atol=1e-12)

    def test_dRtheta_dtheta(self, plastic_state, fortran_mod):
        m, state, state_n, dlambda = plastic_state
        theta = state["theta"]
        theta_max = float(state["theta_max"])
        R = float(state["R"])
        R_n = float(state_n["R"])
        py = m.dRtheta_dtheta(theta, theta_max, R, R_n, dlambda)
        f = fortran_mod.call(
            "yu_drt_dtheta",
            m.B, m.Y, m.k, m.Rsat, m.C_1, m.C_2,
            theta, theta_max, R, R_n, dlambda,
        )
        np.testing.assert_allclose(py, f, rtol=1e-10, atol=1e-12)

    def test_dRtheta_dlambda(self, plastic_state, fortran_mod):
        m, state, state_n, dlambda = plastic_state
        xi = m.dev(state["stress"]) - state["theta"] - state["beta"]
        theta = state["theta"]
        theta_max = float(state["theta_max"])
        R = float(state["R"])
        R_n = float(state_n["R"])
        py = m.dRtheta_dlambda(xi, theta, theta_max, R, R_n, dlambda)
        f = fortran_mod.call(
            "yu_drt_dlambda",
            m.B, m.Y, m.k, m.Rsat, m.C_1, m.C_2,
            xi, theta, theta_max, R, R_n, dlambda,
        )
        np.testing.assert_allclose(py, f, rtol=1e-10, atol=1e-12)

    # --- R_yield blocks ---

    def test_dRyield_dstress(self, plastic_state, fortran_mod):
        m, state, _, _ = plastic_state
        xi = m.dev(state["stress"]) - state["theta"] - state["beta"]
        py = m.dRyield_dstress(xi)
        f = fortran_mod.call("yu_drl_dstress", xi)
        np.testing.assert_allclose(py, f, rtol=1e-12)

    def test_dRyield_dbeta(self, plastic_state, fortran_mod):
        m, state, _, _ = plastic_state
        xi = m.dev(state["stress"]) - state["theta"] - state["beta"]
        py = m.dRyield_dbeta(xi)
        f = fortran_mod.call("yu_drl_dbeta", xi)
        np.testing.assert_allclose(py, f, rtol=1e-12)

    def test_dRyield_dtheta(self, plastic_state, fortran_mod):
        m, state, _, _ = plastic_state
        xi = m.dev(state["stress"]) - state["theta"] - state["beta"]
        py = m.dRyield_dtheta(xi)
        f = fortran_mod.call("yu_drl_dtheta", xi)
        np.testing.assert_allclose(py, f, rtol=1e-12)

    def test_dRyield_dlambda(self, plastic_state, fortran_mod):
        m, _, _, _ = plastic_state
        py = m.dRyield_dlambda()
        f = fortran_mod.call("yu_drl_dlambda")
        np.testing.assert_allclose(py, f, rtol=1e-12)


# ---------------------------------------------------------------------------
# TestResidualAndJacobian: integrated residual + Jacobian over trajectories
# ---------------------------------------------------------------------------

class TestResidualAndJacobian:
    """Compare calc_residual and calc_jacobian Python vs Fortran over trajectories."""

    def _run_and_check(self, m, history, fortran_mod):
        integrator = PythonNumericalIntegrator(m)
        stress_n = np.zeros(6)
        state_n = m.initial_state()
        n_plastic = 0

        for strain_total_n, strain_total_np1 in zip(history[:-1], history[1:]):
            strain_inc = strain_total_np1 - strain_total_n
            result = integrator.stress_update(strain_inc, stress_n, state_n)

            if result.is_plastic:
                n_plastic += 1
                state = result.state
                dl = float(result.dlambda)

                C0 = m.elastic_stiffness(state_n)
                stress_trial = stress_n + C0 @ strain_inc

                props = (
                    m.E, m.nu, m.Y, m.B, m.C_1, m.C_2,
                    m.Rsat, m.k, m.b, m.h, m.Ea, m.xi,
                )

                py_r = m.calc_residual(state, state_n, stress_trial, dl)
                f_r = fortran_mod.call(
                    "yu_calc_residual",
                    *props,
                    state["stress"], state["theta"], state["beta"],
                    float(state["R"]), float(state["eps_eq"]),
                    state_n["theta"], state_n["beta"], float(state_n["theta_max"]),
                    stress_trial, dl,
                )
                np.testing.assert_allclose(
                    np.asarray(py_r), np.asarray(f_r),
                    rtol=1e-12, atol=1e-12,
                    err_msg=f"calc_residual mismatch at plastic step {n_plastic}",
                )

                py_j = m.calc_jacobian(state, state_n, stress_trial, dl)
                f_j = fortran_mod.call(
                    "yu_calc_jacobian",
                    *props,
                    state["stress"], state["theta"], state["beta"],
                    float(state["R"]), float(state["eps_eq"]),
                    float(state_n["theta_max"]), float(state_n["R"]), dl,
                )
                np.testing.assert_allclose(
                    np.asarray(py_j), np.asarray(f_j),
                    rtol=1e-10, atol=1e-12,
                    err_msg=f"calc_jacobian mismatch at plastic step {n_plastic}",
                )

            stress_n = result.stress
            state_n = result.state

        if n_plastic == 0:
            pytest.skip("No plastic steps in this scenario")

    def test_residual_and_jacobian_match_fortran(self, yu_3d_scenario, fortran_mod):
        m, history = yu_3d_scenario
        self._run_and_check(m, history, fortran_mod)


# ---------------------------------------------------------------------------
# FortranIntegrator helper (shared by TestReturnMapping + TestCrosscheckTrajectory)
# ---------------------------------------------------------------------------

def _make_fc_int(fortran_mod, model):
    """Build FortranIntegrator for yu_kinematic_3d.

    state_names order (from YUKinematic declaration, "stress" excluded):
        theta(6), beta(6), R(scalar), q(6), r(scalar), eps_eq(scalar), theta_max(scalar)
    Fortran argument order matches this order exactly.
    Trailing outputs n_iter/converged are ignored by the default ddsdde parser
    (it scans for the first 2-D array, which is ddsdde at position 8).
    """
    return FortranIntegrator.from_model(fortran_mod, "yu_kinematic_3d", model)


# ---------------------------------------------------------------------------
# TestReturnMapping: single-step comparison at a plastic step
# ---------------------------------------------------------------------------

class TestReturnMapping:
    """Compare yu_kinematic_3d (full return mapping) vs PythonAnalyticalIntegrator."""

    def test_single_plastic_step_stress(self, plastic_state, fortran_mod):
        """Stress at a single plastic step matches between Python and Fortran."""
        m, _, _, _ = plastic_state
        # Re-run the step with the analytical integrator (user_defined_return_mapping)
        py_int = PythonAnalyticalIntegrator(m)
        stress_n_arr = np.zeros(6)
        state_n_fresh = m.initial_state()
        strain_inc = np.array([2e-3, -6e-4, -6e-4, 0.0, 0.0, 0.0])
        py_result = py_int.stress_update(strain_inc, stress_n_arr, state_n_fresh)
        fc_int = _make_fc_int(fortran_mod, m)
        fc_result = fc_int.stress_update(strain_inc, stress_n_arr, state_n_fresh)
        np.testing.assert_allclose(
            np.asarray(py_result.stress),
            np.asarray(fc_result.stress),
            rtol=1e-6, atol=1e-8,
        )

    def test_single_plastic_step_state(self, plastic_state, fortran_mod):
        """State variables at a single plastic step match between Python and Fortran."""
        m, _, _, _ = plastic_state
        py_int = PythonAnalyticalIntegrator(m)
        stress_n_arr = np.zeros(6)
        state_n_fresh = m.initial_state()
        strain_inc = np.array([2e-3, -6e-4, -6e-4, 0.0, 0.0, 0.0])
        py_result = py_int.stress_update(strain_inc, stress_n_arr, state_n_fresh)
        fc_int = _make_fc_int(fortran_mod, m)
        fc_result = fc_int.stress_update(strain_inc, stress_n_arr, state_n_fresh)
        for key in ["theta", "beta", "q"]:
            np.testing.assert_allclose(
                np.asarray(py_result.state[key]),
                np.asarray(fc_result.state[key]),
                rtol=1e-6, atol=1e-8,
                err_msg=f"state[{key!r}] mismatch",
            )
        for key in ["R", "r", "eps_eq", "theta_max"]:
            np.testing.assert_allclose(
                float(py_result.state[key]),
                float(fc_result.state[key]),
                rtol=1e-6, atol=1e-8,
                err_msg=f"state[{key!r}] mismatch",
            )

    def test_single_plastic_step_ddsdde(self, plastic_state, fortran_mod):
        """DDSDDE at a single plastic step matches (relaxed tolerance: 1e-4)."""
        m, _, _, _ = plastic_state
        py_int = PythonAnalyticalIntegrator(m)
        stress_n_arr = np.zeros(6)
        state_n_fresh = m.initial_state()
        strain_inc = np.array([2e-3, -6e-4, -6e-4, 0.0, 0.0, 0.0])
        py_result = py_int.stress_update(strain_inc, stress_n_arr, state_n_fresh)
        fc_int = _make_fc_int(fortran_mod, m)
        fc_result = fc_int.stress_update(strain_inc, stress_n_arr, state_n_fresh)
        np.testing.assert_allclose(
            np.asarray(py_result.ddsdde),
            np.asarray(fc_result.ddsdde),
            rtol=1e-5, atol=1e-5,
        )


# ---------------------------------------------------------------------------
# TestCrosscheckTrajectory: multi-step trajectory crosscheck
# ---------------------------------------------------------------------------

class TestCrosscheckTrajectory:
    """PythonAnalyticalIntegrator vs FortranIntegrator over full strain trajectories."""

    def test_analytical_vs_fortran(self, yu_3d_scenario, fortran_mod):
        """Strict crosscheck: analytical Python == Fortran (stress<1e-6, state<1e-6, tangent<1e-5)."""
        model, strain_history = yu_3d_scenario
        fc_int = _make_fc_int(fortran_mod, model)
        py_int = PythonAnalyticalIntegrator(model)
        load = FieldHistory(FieldType.STRAIN, "eps", strain_history)

        cc = CrosscheckStrainDriver(py_int, fc_int, stress_tol=1e-6, tangent_tol=1e-5, state_tol=1e-6)
        result = cc.run(load)

        assert result.passed, (
            f"CrosscheckStrainDriver failed: "
            f"max_stress_rel_err={result.max_stress_rel_err:.2e}, "
            f"n_passed={result.n_passed}/{result.n_cases}"
        )
        assert result.max_stress_rel_err < 1e-6


# ---------------------------------------------------------------------------
# Helpers for umat calls
# ---------------------------------------------------------------------------

def _pack_statev(state):
    """Pack a state dict into STATEV(22) following the Fortran layout.

    Layout (1-indexed):
        1..6  = theta(6)
        7..12 = beta(6)
        13    = R
        14..19= q(6)
        20    = r
        21    = eps_eq
        22    = theta_max
    """
    sv = np.zeros(22)
    sv[0:6]   = np.asarray(state["theta"])
    sv[6:12]  = np.asarray(state["beta"])
    sv[12]    = float(state["R"])
    sv[13:19] = np.asarray(state["q"])
    sv[19]    = float(state["r"])
    sv[20]    = float(state["eps_eq"])
    sv[21]    = float(state["theta_max"])
    return sv


def _unpack_statev(sv):
    """Unpack STATEV(22) back to a dict."""
    return {
        "theta":     sv[0:6].copy(),
        "beta":      sv[6:12].copy(),
        "R":         sv[12],
        "q":         sv[13:19].copy(),
        "r":         sv[19],
        "eps_eq":    sv[20],
        "theta_max": sv[21],
    }


def _call_umat(fortran_mod, model, stress_n, state_n, strain_inc, pnewdt_in=1.0):
    """Call the ABAQUS umat entry point and return (stress_out, state_out, ddsdde, pnewdt_out)."""
    import yu_kinematic_3d as fm
    NTENS = 6
    STRESS = np.asarray(stress_n, dtype=np.float64).copy()
    STATEV = _pack_statev(state_n)
    PROPS  = np.array([model.E, model.nu, model.Y, model.B,
                       model.C_1, model.C_2, model.Rsat, model.k,
                       model.b, model.h, model.Ea, model.xi], dtype=np.float64)
    DSTRAN = np.asarray(strain_inc, dtype=np.float64)
    STRAN  = np.zeros(NTENS)
    TIME   = np.zeros(2); DTIME = 1.0; TEMP = 0.0; DTEMP = 0.0
    PREDEF = np.zeros(1); DPRED = np.zeros(1)
    CMNAME = b'YU      '
    COORDS = np.zeros(3); DROT = np.eye(3)
    CELENT = 1.0; DFGRD0 = np.eye(3); DFGRD1 = np.eye(3)
    pnewdt = np.array(pnewdt_in, dtype=np.float64)

    # umat returns: ddsdde, sse, spd, scd, rpl, ddsddt, drplde, drpldt
    # STRESS and STATEV are modified in-place (intent(inout))
    ret = fm.umat(STRESS, STATEV, STRAN, DSTRAN, TIME, DTIME, TEMP, DTEMP,
                  PREDEF, DPRED, CMNAME, 3, 3,
                  PROPS, COORDS, DROT, pnewdt, CELENT,
                  DFGRD0, DFGRD1, 1, 1, 1, 1, 1, 1)
    ddsdde = ret[0]
    return STRESS.copy(), _unpack_statev(STATEV), ddsdde, float(pnewdt)


# ---------------------------------------------------------------------------
# TestUmatShim: ABAQUS umat entry point
# ---------------------------------------------------------------------------

class TestUmatShim:
    """Verify the ABAQUS umat shim: PROPS/STATEV packing and PNEWDT logic."""

    def test_umat_matches_core(self, plastic_state, fortran_mod):
        """umat output (STRESS, STATEV, DDSDDE) matches yu_kinematic_3d core."""
        m, _, _, _ = plastic_state
        stress_n = np.zeros(6)
        state_n = m.initial_state()
        strain_inc = np.array([2e-3, -6e-4, -6e-4, 0.0, 0.0, 0.0])

        # umat path
        stress_u, state_u, ddsdde_u, _ = _call_umat(fortran_mod, m, stress_n, state_n, strain_inc)

        # core path
        ret_c = fortran_mod.call(
            "yu_kinematic_3d",
            m.E, m.nu, m.Y, m.B, m.C_1, m.C_2, m.Rsat, m.k, m.b, m.h, m.Ea, m.xi,
            stress_n,
            np.asarray(state_n["theta"]), np.asarray(state_n["beta"]),
            float(state_n["R"]), np.asarray(state_n["q"]),
            float(state_n["r"]), float(state_n["eps_eq"]), float(state_n["theta_max"]),
            strain_inc,
        )
        # ret_c: (stress_out, theta_out, beta_out, R_out, q_out, r_out, eps_eq_out, theta_max_out, ddsdde, n_iter, converged)
        stress_c = np.asarray(ret_c[0])
        state_c = {
            "theta": np.asarray(ret_c[1]), "beta": np.asarray(ret_c[2]),
            "R": float(ret_c[3]), "q": np.asarray(ret_c[4]),
            "r": float(ret_c[5]), "eps_eq": float(ret_c[6]), "theta_max": float(ret_c[7]),
        }
        ddsdde_c = np.asarray(ret_c[8])

        np.testing.assert_allclose(stress_u, stress_c, rtol=1e-12, atol=1e-12,
                                   err_msg="umat STRESS != core stress_out")
        np.testing.assert_allclose(ddsdde_u, ddsdde_c, rtol=1e-12, atol=1e-12,
                                   err_msg="umat DDSDDE != core ddsdde")
        for key in ["theta", "beta", "q"]:
            np.testing.assert_allclose(state_u[key], state_c[key], rtol=1e-12, atol=1e-12,
                                       err_msg=f"umat STATEV {key} != core")
        for key in ["R", "r", "eps_eq", "theta_max"]:
            np.testing.assert_allclose(state_u[key], state_c[key], rtol=1e-12, atol=1e-12,
                                       err_msg=f"umat STATEV {key} != core")

    def test_umat_statev_roundtrip(self, plastic_state, fortran_mod):
        """PROPS/STATEV packing roundtrip: umat matches PythonAnalyticalIntegrator."""
        m, _, _, _ = plastic_state
        stress_n = np.zeros(6)
        state_n = m.initial_state()
        strain_inc = np.array([2e-3, -6e-4, -6e-4, 0.0, 0.0, 0.0])

        py_int = PythonAnalyticalIntegrator(m)
        py_result = py_int.stress_update(strain_inc, stress_n, state_n)

        stress_u, state_u, _, _ = _call_umat(fortran_mod, m, stress_n, state_n, strain_inc)

        np.testing.assert_allclose(stress_u, np.asarray(py_result.stress),
                                   rtol=1e-6, atol=1e-8,
                                   err_msg="umat STRESS != Python stress")
        for key in ["theta", "beta", "q"]:
            np.testing.assert_allclose(state_u[key], np.asarray(py_result.state[key]),
                                       rtol=1e-6, atol=1e-8,
                                       err_msg=f"umat STATEV {key} != Python state")
        for key in ["R", "r", "eps_eq", "theta_max"]:
            np.testing.assert_allclose(state_u[key], float(py_result.state[key]),
                                       rtol=1e-6, atol=1e-8,
                                       err_msg=f"umat STATEV {key} != Python state")

    def test_umat_pnewdt_elastic(self, model, fortran_mod):
        """Elastic step: PNEWDT stays at 1.0 (no cutback requested)."""
        strain_inc = np.array([1e-5, 0.0, 0.0, 0.0, 0.0, 0.0])
        _, _, _, pnewdt = _call_umat(fortran_mod, model, np.zeros(6), model.initial_state(), strain_inc)
        assert pnewdt == 1.0

    def test_umat_pnewdt_nonconvergence(self, model, fortran_mod):
        """Non-converging input (extremely large strain): PNEWDT reduced to 0.5."""
        strain_inc = np.array([10.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        _, _, _, pnewdt = _call_umat(fortran_mod, model, np.zeros(6), model.initial_state(), strain_inc)
        assert pnewdt == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# TestParamOrder: param_names order matches yu_kinematic_3d dummy args
# ---------------------------------------------------------------------------

def _extract_yu_dummy_args(fortran_mod):
    """Parse first 12 arg names from yu_kinematic_3d f2py __doc__."""
    import yu_kinematic_3d as fm
    doc = fm.yu_kinematic_3d.__doc__ or ""
    first_line = doc.split("\n")[0]
    m = re.search(r"\(([^)]+)\)", first_line)
    if not m:
        return []
    return [a.strip().lower() for a in m.group(1).split(",")]


class TestParamOrder:
    """param_names order matches the first 12 dummy args of yu_kinematic_3d.

    Fortran renames two params to avoid keyword collisions:
        B (Python)  → b_bnd  (Fortran)
        xi (Python) → xi_param (Fortran)
    The test maps these before comparison.
    """

    # Python param_names → Fortran dummy arg name (only entries that differ)
    _FORTRAN_NAME = {"B": "b_bnd", "b": "b_kin", "xi": "xi_param"}

    def test_param_names_match_fortran_arg_order(self, fortran_mod):
        from manforge.models import YUKinematic3D
        model = YUKinematic3D(**PARAMS)
        dummy_args = _extract_yu_dummy_args(fortran_mod)
        n = len(model.param_names)
        assert len(dummy_args) >= n, (
            f"yu_kinematic_3d: expected at least {n} dummy args, got {len(dummy_args)}: {dummy_args}"
        )
        expected = [self._FORTRAN_NAME.get(p, p).lower() for p in model.param_names]
        actual = dummy_args[:n]
        assert actual == expected, (
            f"param_names order mismatch:\n"
            f"  expected (with Fortran renames): {expected}\n"
            f"  actual   (from __doc__):         {actual}"
        )

