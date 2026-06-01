"""Finite-difference check of the YUKinematic3D analytical Jacobian.

Compares the analytical Jacobian (calc_jacobian) against a central
finite-difference approximation of calc_residual for several stress
paths, including pure shear and load-reversal scenarios that were
previously untested.

All tests use the analytical NR path (user_defined_return_mapping) to
obtain a converged state, then evaluate the Jacobian at that state.
"""
import numpy as np
import pytest

from manforge.models import YUKinematic3D
from tests.fixtures.yu_fd import (
    PARAMS,
    FD_RTOL_JAC as FD_RTOL,
    FD_ATOL_JAC as FD_ATOL,
    fd_jacobian,
    get_converged_state,
    uniaxial_plastic_step,
    pure_shear_step,
    load_reversal_step,
    stagnation_active_step,
)


@pytest.mark.parametrize("scenario,label", [
    (uniaxial_plastic_step,   "uniaxial"),
    (pure_shear_step,         "pure_shear"),
    (load_reversal_step,      "load_reversal"),
    (stagnation_active_step,  "stagnation_active"),
])
def test_analytical_jacobian_matches_fd(scenario, label):
    """Analytical calc_jacobian must match central-FD approximation within tolerance."""
    model, result, state_n = get_converged_state(scenario())

    if result is None or not result.is_plastic:
        pytest.skip(f"Scenario '{label}' did not yield a plastic step")

    rm           = result.return_mapping
    state_new    = rm.state
    dlambda      = float(rm.dlambda)
    stress_trial = result.stress_trial

    def python_residual_fn(sn, dl):
        return model.calc_residual(sn, state_n, stress_trial, dl)

    J_analytical = model.calc_jacobian(state_new, state_n, stress_trial, dlambda)
    J_fd         = fd_jacobian(model, state_new, state_n, dlambda, python_residual_fn)

    np.testing.assert_allclose(
        J_analytical, J_fd,
        rtol=FD_RTOL, atol=FD_ATOL,
        err_msg=f"Jacobian mismatch for scenario '{label}'"
    )
