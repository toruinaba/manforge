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
from manforge.simulation.driver import StrainDriver
from manforge.simulation.integrator import PythonIntegrator
from manforge.simulation.types import FieldHistory, FieldType

PARAMS = dict(
    E=206_000, nu=0.3, Y=360.0, C_1=2000.0, C_2=200.0,
    B=435.0, Rsat=255.0, k=26.0, b=66.0, h=0.4, Ea=159_000, xi=61.0,
)

FD_EPS   = 1e-5   # central-difference step
FD_RTOL  = 1e-4   # relative tolerance for analytical vs fd comparison
FD_ATOL  = 1e-3   # absolute tolerance (some blocks may be near-zero)


def _fd_jacobian(model, state_new, state_n, stress_trial, dlambda):
    """Central finite-difference of the **total** derivative used in the NR loop.

    When dlambda is perturbed, eps_eq and R are updated consistently with the
    NR iteration (eps_eq = eps_eq_n + dlambda, R = R_n + g_flag*delta_R(dlambda)).
    This gives the total derivative that calc_jacobian must match for 2nd-order
    NR convergence.  For all other columns (sigma, theta, beta), R/eps_eq are
    held at their converged iterate values.
    """
    from manforge.utils.smooth import smooth_heaviside

    x0 = np.hstack([
        state_new["stress"],
        [dlambda],
        state_new["theta"],
        state_new["beta"],
    ])
    n = len(x0)
    J_fd = np.zeros((n, n))

    # g_flag evaluated at converged beta (dlambda perturbation holds beta fixed)
    g_xi_0     = np.array(state_new["beta"]) - np.array(state_n["q"])
    g_flag_0   = float(smooth_heaviside(float(model.vonmises_norm(g_xi_0)) - float(state_n["r"])))
    R_n_val    = float(state_n["R"])
    eps_eq_n   = float(state_n["eps_eq"])

    def residual(x):
        sn           = dict(state_new)
        sn["stress"] = x[0:6].copy()
        sn["theta"]  = x[7:13].copy()
        sn["beta"]   = x[13:19].copy()
        dl_val       = float(x[6])
        # Update eps_eq and R consistently with the NR-loop update
        sn["eps_eq"] = eps_eq_n + dl_val
        s            = 1.0 / (1.0 + model.k * dl_val)
        delta_R      = s * (R_n_val + model.k * model.Rsat * dl_val) - R_n_val
        sn["R"]      = R_n_val + g_flag_0 * delta_R
        return np.array(model.calc_residual(sn, state_n, stress_trial, dl_val), dtype=float)

    for j in range(n):
        xp = x0.copy(); xp[j] += FD_EPS
        xm = x0.copy(); xm[j] -= FD_EPS
        J_fd[:, j] = (residual(xp) - residual(xm)) / (2.0 * FD_EPS)

    return J_fd


def _get_converged_state(strain_history):
    """Run the driver; return (model, result, state_n) for the plastic step with max dlambda."""
    model      = YUKinematic3D(**PARAMS)
    integrator = PythonIntegrator(model)
    history    = FieldHistory(type=FieldType.STRAIN, name="strain", data=strain_history)
    driver     = StrainDriver(integrator)
    steps      = list(driver.iter_run(history))
    assert steps, "No steps produced"

    best_step    = None
    best_prev    = None
    best_dlambda = -1.0
    initial      = model.initial_state()
    for i, step in enumerate(steps):
        if step.result.is_plastic:
            dl = float(step.result.dlambda)
            if dl > best_dlambda:
                best_dlambda = dl
                best_step    = step
                best_prev    = steps[i - 1] if i > 0 else None

    if best_step is None:
        return model, None, None

    state_n = best_prev.result.state if best_prev is not None else initial
    return model, best_step.result, state_n


# ---------------------------------------------------------------------------
# Scenario builders
# ---------------------------------------------------------------------------

def _uniaxial_plastic_step():
    n = 10
    data = np.zeros((n, 6))
    data[:, 0] = np.linspace(0.0, 4e-3, n)
    return data


def _pure_shear_step():
    n = 10
    data = np.zeros((n, 6))
    data[:, 3] = np.linspace(0.0, 4e-3, n)  # gamma12 only
    return data


def _load_reversal_step():
    n = 20
    data = np.zeros((n, 6))
    data[:10, 0] = np.linspace(0.0, 5e-3, 10)
    data[10:, 0] = np.linspace(5e-3, -2e-3, 10)
    return data


def _stagnation_active_step():
    """Large cycle then small reversal to make stagnation surface active."""
    large = np.zeros((20, 6))
    large[:, 0] = np.linspace(0.0, 0.03, 20)
    small = np.zeros((10, 6))
    small[:, 0] = np.linspace(0.03, 0.02, 10)
    return np.vstack([large, small])


@pytest.mark.parametrize("scenario,label", [
    (_uniaxial_plastic_step,   "uniaxial"),
    (_pure_shear_step,         "pure_shear"),
    (_load_reversal_step,      "load_reversal"),
    (_stagnation_active_step,  "stagnation_active"),
])
def test_analytical_jacobian_matches_fd(scenario, label):
    """Analytical calc_jacobian must match central-FD approximation within tolerance."""
    history = scenario()
    model, result, state_n = _get_converged_state(history)

    if result is None or not result.is_plastic:
        pytest.skip(f"Scenario '{label}' did not yield a plastic step")

    rm           = result.return_mapping
    stress_new   = rm.stress
    state_new    = rm.state
    dlambda      = float(rm.dlambda)
    stress_trial = result.stress_trial

    J_analytical = model.calc_jacobian(state_new, state_n, stress_trial, dlambda)
    J_fd         = _fd_jacobian(model, state_new, state_n, stress_trial, dlambda)

    np.testing.assert_allclose(
        J_analytical, J_fd,
        rtol=FD_RTOL, atol=FD_ATOL,
        err_msg=f"Jacobian mismatch for scenario '{label}'"
    )
