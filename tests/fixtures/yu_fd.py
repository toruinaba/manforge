"""Shared FD (finite-difference) helpers for YUKinematic3D tests.

Used by:
  tests/integration/models/test_yu_jacobian_fd.py
  tests/integration/models/test_yu_ddsdde_fd.py
  tests/benchmarks/yu_kinematic/test_fortran_vs_fd.py
"""
import numpy as np

from manforge.models import YUKinematic3D
from manforge.simulation.driver import StrainDriver
from manforge.simulation.integrator import PythonIntegrator
from manforge.simulation.types import FieldHistory, FieldType

# ---------------------------------------------------------------------------
# Standard YUKinematic3D parameters
# ---------------------------------------------------------------------------

PARAMS = dict(
    E=206_000, nu=0.3, Y=360.0, C_1=2000.0, C_2=200.0,
    B=435.0, Rsat=255.0, k=26.0, b=66.0, h=0.4, Ea=159_000, xi=61.0,
)

# ---------------------------------------------------------------------------
# FD constants
# ---------------------------------------------------------------------------

FD_EPS        = 1e-5   # step for Jacobian FD (residual-based)
FD_H          = 1e-5   # step for DDSDDE FD (stress_update-based)
FD_RTOL       = 2e-2   # relative tolerance (large-increment scenarios; Stage 5 will tighten)
FD_ATOL       = 1e-1   # absolute tolerance (scaled to stiffness ~1e5)
FD_RTOL_JAC   = 1e-4   # relative tolerance for Jacobian tests
FD_ATOL_JAC   = 1e-3   # absolute tolerance for Jacobian tests
FD_RTOL_BAND  = 5e-3   # tighter tolerance for stagnation transition band

# ---------------------------------------------------------------------------
# Scenario builders (cumulative total-strain histories, shape (N, 6))
# ---------------------------------------------------------------------------

def uniaxial_plastic_step():
    data = np.zeros((10, 6))
    data[:, 0] = np.linspace(0.0, 4e-3, 10)
    return data


def pure_shear_step():
    data = np.zeros((10, 6))
    data[:, 3] = np.linspace(0.0, 4e-3, 10)
    return data


def load_reversal_step():
    data = np.zeros((20, 6))
    data[:10, 0] = np.linspace(0.0, 5e-3, 10)
    data[10:, 0] = np.linspace(5e-3, -2e-3, 10)
    return data


def stagnation_active_step():
    """Large cycle then small reversal to make stagnation surface active."""
    large = np.zeros((20, 6))
    large[:, 0] = np.linspace(0.0, 0.03, 20)
    small = np.zeros((10, 6))
    small[:, 0] = np.linspace(0.03, 0.02, 10)
    return np.vstack([large, small])


def stagnation_transition_band_history():
    """Pure-shear ramp that crosses the stagnation boundary (g_stag changes sign).

    At ~step 45 (gamma≈0.036) g_stag passes through 0, exercising the
    smooth-heaviside derivative term in the chain-rule correction block.
    """
    data = np.zeros((100, 6))
    data[:, 3] = np.linspace(0.0, 0.08, 100)
    return data


# ---------------------------------------------------------------------------
# Step-running helpers
# ---------------------------------------------------------------------------

def run_steps(model, integrator, strain_data):
    """Run driver; return list of (result, stress_n, state_n, strain_inc) for each step."""
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


def get_converged_state(strain_history, integrator_cls=PythonIntegrator):
    """Run driver; return (model, result, state_n) for plastic step with max dlambda."""
    model = YUKinematic3D(**PARAMS)
    integrator = integrator_cls(model)
    history = FieldHistory(type=FieldType.STRAIN, name="strain", data=strain_history)
    driver = StrainDriver(integrator)
    steps = list(driver.iter_run(history))
    assert steps, "No steps produced"

    best_step = None
    best_prev = None
    best_dlambda = -1.0
    initial = model.initial_state()
    for i, step in enumerate(steps):
        if step.result.is_plastic:
            dl = float(step.result.dlambda)
            if dl > best_dlambda:
                best_dlambda = dl
                best_step = step
                best_prev = steps[i - 1] if i > 0 else None

    if best_step is None:
        return model, None, None

    state_n = best_prev.result.state if best_prev is not None else initial
    return model, best_step.result, state_n


def pick_largest_dlambda(model, step_data, gstag_min=0.05):
    """Select plastic step with max dlambda, preferring steps outside the transition band."""
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
        for result, stress_n, state_n, strain_inc in step_data:
            if result.is_plastic and float(result.dlambda) > best_dl:
                best_dl = float(result.dlambda)
                best = (result, stress_n, state_n, strain_inc)
    return best


def pick_min_gstag(model, step_data):
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
# FD kernels
# ---------------------------------------------------------------------------

def fd_ddsdde(integrator, strain_inc, stress_n, state_n, h=FD_H):
    """Central finite-difference of dσ/dε via independent stress_update calls.

    Works with any integrator (Python or Fortran) that implements stress_update.
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


def fd_jacobian(model, state_new, state_n, dlambda, residual_fn, eps=FD_EPS):
    """Central finite-difference of the NR Jacobian.

    Parameters
    ----------
    model       : YUKinematic3D instance (used only for vonmises_norm / scalar params k, Rsat)
    state_new   : converged state at step n+1
    state_n     : state at step n (start of step)
    dlambda     : converged plastic multiplier
    residual_fn : callable (state_dict, dlambda_val) -> array(19)
                  Use Python:  lambda sn, dl: np.array(model.calc_residual(sn, state_n, stress_trial, dl))
                  Use Fortran: see test_fortran_vs_fd.py
    eps         : central-difference step size

    The dlambda column updates eps_eq and R consistently with the NR loop:
      eps_eq = eps_eq_n + dlambda_perturbed
      R      = R_n + g_flag * delta_R(dlambda_perturbed)
    All other columns hold eps_eq/R at their converged values.
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

    g_xi_0   = np.array(state_new["beta"]) - np.array(state_n["q"])
    g_flag_0 = float(smooth_heaviside(float(model.vonmises_norm(g_xi_0)) - float(state_n["r"])))
    R_n_val  = float(state_n["R"])
    eps_eq_n = float(state_n["eps_eq"])

    def _residual(x):
        sn           = dict(state_new)
        sn["stress"] = x[0:6].copy()
        sn["theta"]  = x[7:13].copy()
        sn["beta"]   = x[13:19].copy()
        dl_val       = float(x[6])
        sn["eps_eq"] = eps_eq_n + dl_val
        s            = 1.0 / (1.0 + model.k * dl_val)
        delta_R      = s * (R_n_val + model.k * model.Rsat * dl_val) - R_n_val
        sn["R"]      = R_n_val + g_flag_0 * delta_R
        return np.array(residual_fn(sn, dl_val), dtype=float)

    for j in range(n):
        xp = x0.copy(); xp[j] += eps
        xm = x0.copy(); xm[j] -= eps
        J_fd[:, j] = (_residual(xp) - _residual(xm)) / (2.0 * eps)

    return J_fd
