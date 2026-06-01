"""Finite-difference check of the YUKinematic3D consistent tangent (DDSDDE).

Compares result.ddsdde (from PythonAnalyticalIntegrator / calc_ddsdde) against
a central finite-difference approximation of dσ/dε for several stress paths.

The stagnation_transition_band test is the critical regression guard for the
A1+A2 chain-rule coefficient fix in calc_ddsdde (Stage 2 of the convergence
improvement plan).
"""
import numpy as np
import pytest

from manforge.models import YUKinematic3D
from manforge.simulation.integrator import PythonAnalyticalIntegrator
from tests.fixtures.yu_fd import (
    PARAMS,
    FD_RTOL,
    FD_ATOL,
    FD_RTOL_BAND,
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


def _make_integrator():
    model = YUKinematic3D(**PARAMS)
    return model, PythonAnalyticalIntegrator(model)


@pytest.mark.parametrize("scenario,label", [
    (uniaxial_plastic_step,   "uniaxial"),
    (pure_shear_step,         "pure_shear"),
    (load_reversal_step,      "load_reversal"),
    (stagnation_active_step,  "stagnation_active"),
])
def test_ddsdde_matches_fd(scenario, label):
    """result.ddsdde must match central-FD dσ/dε within tolerance."""
    model, integrator = _make_integrator()
    step_data = run_steps(model, integrator, scenario())
    picked = pick_largest_dlambda(model, step_data)
    if picked is None:
        pytest.skip(f"Scenario '{label}' produced no plastic step")

    result, stress_n, state_n, strain_inc = picked
    D_an = np.array(result.ddsdde, dtype=float)
    D_fd = fd_ddsdde(integrator, strain_inc, stress_n, state_n)

    np.testing.assert_allclose(
        D_an, D_fd, rtol=FD_RTOL, atol=FD_ATOL,
        err_msg=f"DDSDDE mismatch for scenario '{label}' (rtol={FD_RTOL}, atol={FD_ATOL})"
    )


def test_ddsdde_stagnation_transition_band():
    """Chain-rule correction block (A1+A2 fixed) is verified at the stagnation boundary.

    This test specifically targets the step where |g_stag| is minimal — the
    transition band where smooth_heaviside has its largest derivative and the
    chain-rule correction term in calc_ddsdde is most significant.
    """
    model, integrator = _make_integrator()
    step_data = run_steps(model, integrator, stagnation_transition_band_history())
    picked, gstag = pick_min_gstag(model, step_data)
    if picked is None:
        pytest.skip("No plastic step found in stagnation transition band scenario")

    result, stress_n, state_n, strain_inc = picked
    D_an = np.array(result.ddsdde, dtype=float)
    D_fd = fd_ddsdde(integrator, strain_inc, stress_n, state_n)

    np.testing.assert_allclose(
        D_an, D_fd, rtol=FD_RTOL_BAND, atol=FD_ATOL,
        err_msg=f"DDSDDE mismatch in transition band (|g_stag|={gstag:.4f})"
    )
