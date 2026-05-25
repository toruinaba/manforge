"""Integration tests: YUKinematic3D Fortran binding registry + check_bindings.

Requires the compiled yu_kinematic_3d Fortran extension:
    make fortran-build-yu
"""

import numpy as np
import pytest

from manforge.verification import FortranModule

_PARAMS = dict(
    E=206_000, nu=0.3, Y=360.0, C_1=2000.0, C_2=200.0,
    B=435.0, Rsat=255.0, k=26.0, b=66.0, h=0.4, Ea=159_000, xi=61.0,
)

_SKIP_MSG = "yu_kinematic_3d not compiled -- run: make fortran-build-yu"


@pytest.fixture
def yu_model():
    from manforge.models import YUKinematic3D
    return YUKinematic3D(**_PARAMS)


@pytest.fixture
def fortran():
    pytest.importorskip("yu_kinematic_3d", reason=_SKIP_MSG)
    return FortranModule("yu_kinematic_3d")


@pytest.fixture
def plastic_args(yu_model):
    """Run one plastic step and return convenient argument bundles."""
    from manforge.simulation.integrator import PythonNumericalIntegrator
    integrator = PythonNumericalIntegrator(yu_model)
    stress_n = np.zeros(6)
    state_n = yu_model.initial_state()
    strain_inc = np.array([2e-3, -6e-4, -6e-4, 0.0, 0.0, 0.0])
    result = integrator.stress_update(strain_inc, stress_n, state_n)
    assert result.is_plastic

    state = result.state
    m = yu_model
    C = m.elastic_stiffness(state)
    stress = state["stress"]
    theta = state["theta"]
    beta = state["beta"]
    xi = m.dev(stress) - theta - beta
    eps_eq = float(state["eps_eq"])
    R = float(state["R"])
    R_n = float(state_n["R"])
    theta_max = float(state["theta_max"])
    dlambda = float(result.dlambda)

    return dict(
        state=state, state_n=state_n,
        C=C, xi=xi, stress=stress,
        theta=theta, beta=beta,
        eps_eq=eps_eq, R=R, R_n=R_n,
        theta_max=theta_max, dlambda=dlambda,
    )


# ---------------------------------------------------------------------------
# Registry checks
# ---------------------------------------------------------------------------

@pytest.mark.fortran
def test_all_bindings_registered():
    pytest.importorskip("yu_kinematic_3d", reason=_SKIP_MSG)
    from manforge.models import YUKinematic3D
    bindings = YUKinematic3D._fortran_bindings
    expected = {
        "elastic_stiffness":    "yu_kinematic_3d_elastic_stiffness",
        "calc_norm_n_flow":     "yu_calc_norm_n_flow",
        "_prepare_Rstress":     "yu_prepare_rstress",
        "_prepare_Rtheta":      "yu_prepare_rtheta",
        "dRstress_dstress":     "yu_drs_dstress",
        "dRstress_dbeta":       "yu_drs_dbeta",
        "dRstress_dtheta":      "yu_drs_dtheta",
        "dRstress_dlambda":     "yu_drs_dlambda",
        "dRbeta_dstress":       "yu_drb_dstress",
        "dRbeta_dbeta":         "yu_drb_dbeta",
        "dRbeta_dtheta":        "yu_drb_dtheta",
        "dRbeta_dlambda":       "yu_drb_dlambda",
        "dRtheta_dstress":      "yu_drt_dstress",
        "dRtheta_dbeta":        "yu_drt_dbeta",
        "dRtheta_dtheta":       "yu_drt_dtheta",
        "dRtheta_dlambda":      "yu_drt_dlambda",
        "dRyield_dstress":      "yu_drl_dstress",
        "dRyield_dbeta":        "yu_drl_dbeta",
        "dRyield_dtheta":       "yu_drl_dtheta",
        "dRyield_dlambda":      "yu_drl_dlambda",
    }
    for method, subroutine in expected.items():
        assert method in bindings, f"{method} not in _fortran_bindings"
        assert bindings[method].subroutine == subroutine, (
            f"{method}: expected subroutine={subroutine!r}, got {bindings[method].subroutine!r}"
        )


# ---------------------------------------------------------------------------
# check_bindings: single-array-output methods (14 matrix / vector blocks)
# ---------------------------------------------------------------------------

@pytest.mark.fortran
def test_check_bindings(yu_model, fortran, plastic_args):
    from manforge.verification import check_bindings
    m = yu_model
    a = plastic_args

    cases = {
        "elastic_stiffness": (
            (a["state"],),
            (m.E, m.nu, a["eps_eq"], m.Ea, m.xi),
        ),
        "dRstress_dstress": (
            (a["C"], a["xi"], a["dlambda"]),
            (a["C"], a["xi"], a["dlambda"]),
        ),
        "dRstress_dbeta": (
            (a["C"], a["xi"], a["dlambda"]),
            (a["C"], a["xi"], a["dlambda"]),
        ),
        "dRstress_dtheta": (
            (a["C"], a["xi"], a["dlambda"]),
            (a["C"], a["xi"], a["dlambda"]),
        ),
        "dRstress_dlambda": (
            (a["C"], a["xi"], a["eps_eq"], a["dlambda"]),
            (m.E, m.Ea, m.xi, a["C"], a["xi"], a["eps_eq"], a["dlambda"]),
        ),
        "dRbeta_dstress": (
            (a["dlambda"],),
            (m.k, m.b, m.Y, a["dlambda"]),
        ),
        "dRbeta_dbeta": (
            (a["dlambda"],),
            (m.k, m.b, m.Y, a["dlambda"]),
        ),
        "dRbeta_dtheta": (
            (a["dlambda"],),
            (m.k, m.b, m.Y, a["dlambda"]),
        ),
        "dRbeta_dlambda": (
            (a["xi"], a["beta"], a["dlambda"]),
            (m.k, m.b, m.Y, a["xi"], a["beta"], a["dlambda"]),
        ),
        "dRtheta_dstress": (
            (a["theta"], a["theta_max"], a["R"], a["R_n"], a["dlambda"]),
            (m.B, m.Y, m.k, m.Rsat, m.C_1, m.C_2,
             a["theta"], a["theta_max"], a["R"], a["R_n"], a["dlambda"]),
        ),
        "dRtheta_dbeta": (
            (a["theta"], a["theta_max"], a["R"], a["R_n"], a["dlambda"]),
            (m.B, m.Y, m.k, m.Rsat, m.C_1, m.C_2,
             a["theta"], a["theta_max"], a["R"], a["R_n"], a["dlambda"]),
        ),
        "dRtheta_dtheta": (
            (a["theta"], a["theta_max"], a["R"], a["R_n"], a["dlambda"]),
            (m.B, m.Y, m.k, m.Rsat, m.C_1, m.C_2,
             a["theta"], a["theta_max"], a["R"], a["R_n"], a["dlambda"]),
        ),
        "dRtheta_dlambda": (
            (a["xi"], a["theta"], a["theta_max"], a["R"], a["R_n"], a["dlambda"]),
            (m.B, m.Y, m.k, m.Rsat, m.C_1, m.C_2,
             a["xi"], a["theta"], a["theta_max"], a["R"], a["R_n"], a["dlambda"]),
        ),
        "dRyield_dstress": (
            (a["xi"],),
            (a["xi"],),
        ),
        "dRyield_dbeta": (
            (a["xi"],),
            (a["xi"],),
        ),
        "dRyield_dtheta": (
            (a["xi"],),
            (a["xi"],),
        ),
        "dRyield_dlambda": (
            (),
            (),
        ),
    }

    results = check_bindings(yu_model, fortran, cases, rtol=1e-10, atol=1e-12)
    for name, (ok, err) in results.items():
        assert ok, f"{name}: max_rel_err={err:.3e}"


# ---------------------------------------------------------------------------
# Individual tests for tuple-return helpers
# ---------------------------------------------------------------------------

@pytest.mark.fortran
def test_prepare_rstress(yu_model, fortran, plastic_args):
    a = plastic_args
    py_out = yu_model._prepare_Rstress(a["xi"])
    f_out = fortran.call("yu_prepare_rstress", a["xi"])
    np.testing.assert_allclose(py_out, f_out, atol=1e-12,
                               err_msg="_prepare_Rstress: matrix mismatch")


@pytest.mark.fortran
def test_calc_norm_n_flow(yu_model, fortran, plastic_args):
    a = plastic_args
    xi_norm_py, flow_py = yu_model.calc_norm_n_flow(a["xi"])
    xi_norm_f, flow_f = fortran.call("yu_calc_norm_n_flow", a["xi"])
    np.testing.assert_allclose(xi_norm_py, xi_norm_f, rtol=1e-12,
                               err_msg="calc_norm_n_flow: xi_norm mismatch")
    np.testing.assert_allclose(flow_py, flow_f, rtol=1e-12,
                               err_msg="calc_norm_n_flow: flow mismatch")


@pytest.mark.fortran
def test_prepare_rtheta(yu_model, fortran, plastic_args):
    a = plastic_args
    m = yu_model
    py_out = m._prepare_Rtheta(
        a["theta"], a["theta_max"], a["R"], a["R_n"], a["dlambda"]
    )
    theta_bar_py, theta_flow_py, C_k_py, s_py, a_py, a_prime_py = py_out

    f_out = fortran.call(
        "yu_prepare_rtheta",
        m.B, m.Y, m.k, m.Rsat, m.C_1, m.C_2,
        a["theta"], a["theta_max"], a["R"], a["R_n"], a["dlambda"],
    )
    theta_bar_f, theta_flow_f, C_k_f, s_f, a_f, a_prime_f = f_out

    np.testing.assert_allclose(theta_bar_py, theta_bar_f, rtol=1e-12)
    np.testing.assert_allclose(theta_flow_py, theta_flow_f, rtol=1e-12)
    np.testing.assert_allclose(C_k_py, C_k_f, rtol=1e-12)
    np.testing.assert_allclose(s_py, s_f, rtol=1e-12)
    np.testing.assert_allclose(a_py, a_f, rtol=1e-12)
    np.testing.assert_allclose(a_prime_py, a_prime_f, rtol=1e-12)
