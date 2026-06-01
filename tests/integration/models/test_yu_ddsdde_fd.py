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
from manforge.simulation.driver import StrainDriver
from manforge.simulation.integrator import PythonAnalyticalIntegrator
from manforge.simulation.types import FieldHistory, FieldType

PARAMS = dict(
    E=206_000, nu=0.3, Y=360.0, C_1=2000.0, C_2=200.0,
    B=435.0, Rsat=255.0, k=26.0, b=66.0, h=0.4, Ea=159_000, xi=61.0,
)

FD_H         = 1e-5   # central-difference step size
FD_RTOL      = 2e-2   # relative tolerance for large-increment scenarios (Stage 5 will tighten)
FD_ATOL      = 1e-1   # absolute tolerance (scaled to stiffness magnitudes ~1e5)
FD_RTOL_BAND = 5e-3   # tighter tolerance for stagnation transition band test


def _fd_ddsdde(integrator, strain_inc, stress_n, state_n, h=FD_H):
    """Central finite-difference of dσ/dε via independent stress_update calls.

    Each perturbed call is independent (same stress_n, state_n base point),
    so this gives the consistent tangent FD truth at the given step start.
    """
    ntens = len(strain_inc)
    J = np.zeros((ntens, ntens))
    for j in range(ntens):
        e_j = np.zeros(ntens)
        e_j[j] = 1.0
        sp = np.asarray(integrator.stress_update(strain_inc + h * e_j, stress_n, state_n).stress, dtype=float)
        sm = np.asarray(integrator.stress_update(strain_inc - h * e_j, stress_n, state_n).stress, dtype=float)
        J[:, j] = (sp - sm) / (2.0 * h)
    return J


def _make_integrator():
    model = YUKinematic3D(**PARAMS)
    return model, PythonAnalyticalIntegrator(model)


def _run_steps(model, integrator, strain_data):
    """Run driver; return list of (result, stress_n, state_n, strain_inc) for every step."""
    history = FieldHistory(type=FieldType.STRAIN, name="strain", data=strain_data)
    driver = StrainDriver(integrator)
    steps = list(driver.iter_run(history))
    initial = model.initial_state()

    out = []
    for i, step in enumerate(steps):
        prev = steps[i - 1] if i > 0 else None
        state_n = prev.result.state if prev is not None else initial
        stress_n = np.asarray(prev.result.stress, dtype=float) if prev is not None else np.zeros(6)
        prev_strain = np.asarray(prev.strain, dtype=float) if prev is not None else np.zeros(6)
        strain_inc = np.asarray(step.strain, dtype=float) - prev_strain
        out.append((step.result, stress_n, state_n, strain_inc))
    return out


def _pick_largest_dlambda(model, step_data, gstag_min=0.05):
    """Select plastic step with maximum dlambda, outside the stagnation transition band.

    Prefers steps where |g_stag| > gstag_min so that the smooth-heaviside
    derivative term in the chain-rule correction is negligible (g_flag fully
    saturated at 0 or 1).  Falls back to all plastic steps if none qualify.
    """
    def gstag(result, state_n):
        beta = np.asarray(result.state["beta"])
        q = np.asarray(state_n["q"])
        return abs(float(model.vonmises_norm(beta - q)) - float(state_n["r"]))

    best, best_dl = None, -1.0
    for result, stress_n, state_n, strain_inc in step_data:
        if result.is_plastic and gstag(result, state_n) > gstag_min:
            dl = float(result.dlambda)
            if dl > best_dl:
                best_dl = dl
                best = (result, stress_n, state_n, strain_inc)
    if best is None:
        # Fall back: any plastic step
        for result, stress_n, state_n, strain_inc in step_data:
            if result.is_plastic and float(result.dlambda) > best_dl:
                best_dl = float(result.dlambda)
                best = (result, stress_n, state_n, strain_inc)
    return best


def _pick_min_gstag(model, step_data):
    """Select plastic step with minimum |g_stag| — targets the stagnation transition band."""
    best, best_abs = None, np.inf
    for result, stress_n, state_n, strain_inc in step_data:
        if not result.is_plastic:
            continue
        beta = np.asarray(result.state["beta"])
        q = np.asarray(state_n["q"])
        gstag = abs(float(model.vonmises_norm(beta - q)) - float(state_n["r"]))
        if gstag < best_abs:
            best_abs = gstag
            best = (result, stress_n, state_n, strain_inc)
    return best, best_abs


# ---------------------------------------------------------------------------
# Scenario builders (cumulative total-strain histories)
# ---------------------------------------------------------------------------

def _uniaxial_plastic_step():
    data = np.zeros((10, 6))
    data[:, 0] = np.linspace(0.0, 4e-3, 10)
    return data


def _pure_shear_step():
    data = np.zeros((10, 6))
    data[:, 3] = np.linspace(0.0, 4e-3, 10)
    return data


def _load_reversal_step():
    data = np.zeros((20, 6))
    data[:10, 0] = np.linspace(0.0, 5e-3, 10)
    data[10:, 0] = np.linspace(5e-3, -2e-3, 10)
    return data


def _stagnation_active_step():
    large = np.zeros((20, 6))
    large[:, 0] = np.linspace(0.0, 0.03, 20)
    small = np.zeros((10, 6))
    small[:, 0] = np.linspace(0.03, 0.02, 10)
    return np.vstack([large, small])


def _stagnation_transition_band_history():
    """Pure-shear ramp that crosses the stagnation boundary (g_stag changes sign).

    At ~step 45 (gamma≈0.036) g_stag passes through 0, exercising the
    smooth-heaviside derivative term in the chain-rule correction block.
    """
    data = np.zeros((100, 6))
    data[:, 3] = np.linspace(0.0, 0.08, 100)
    return data


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("scenario,label", [
    (_uniaxial_plastic_step,   "uniaxial"),
    (_pure_shear_step,         "pure_shear"),
    (_load_reversal_step,      "load_reversal"),
    (_stagnation_active_step,  "stagnation_active"),
])
def test_ddsdde_matches_fd(scenario, label):
    """result.ddsdde must match central-FD dσ/dε within tolerance."""
    model, integrator = _make_integrator()
    step_data = _run_steps(model, integrator, scenario())
    picked = _pick_largest_dlambda(model, step_data)
    if picked is None:
        pytest.skip(f"Scenario '{label}' produced no plastic step")

    result, stress_n, state_n, strain_inc = picked
    D_an = np.array(result.ddsdde, dtype=float)
    D_fd = _fd_ddsdde(integrator, strain_inc, stress_n, state_n)

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
    step_data = _run_steps(model, integrator, _stagnation_transition_band_history())
    picked, gstag = _pick_min_gstag(model, step_data)
    if picked is None:
        pytest.skip("No plastic step found in stagnation transition band scenario")

    result, stress_n, state_n, strain_inc = picked
    D_an = np.array(result.ddsdde, dtype=float)
    D_fd = _fd_ddsdde(integrator, strain_inc, stress_n, state_n)

    np.testing.assert_allclose(
        D_an, D_fd, rtol=FD_RTOL_BAND, atol=FD_ATOL,
        err_msg=f"DDSDDE mismatch in transition band (|g_stag|={gstag:.4f})"
    )
