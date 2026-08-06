"""Where the hand-derived Jacobian disagrees with autograd, and why.

Both YU formulations omit the same term: R is gated by
``Gg = smooth_heaviside(g_stag)`` and ``g_stag`` is a function of beta, so
``a = B + R - Y`` carries a beta dependence that ``calc_ft_fb`` treats as
constant.  The omission is invisible on ordinary histories because Gg saturates
at 0 or 1, where its derivative underflows -- converged solutions simply do not
sit in the 0.01 MPa transition band.  Forcing them there exposes it.

This matters for the projected variant specifically: with mu in closed form,
dGg/dbeta is now expressible analytically, so the term could be supplied rather
than dropped.  These tests record the size of what is being dropped so that a
future analytical derivation has something to beat.
"""

import numpy as np
import pytest

from manforge.models import YUKinematicPS, YUKinematicProjPS
from manforge.simulation.integrator import PythonNumericalIntegrator
from manforge.utils.smooth import smooth_heaviside
from manforge.verification.jacobian import JacobianChecker

from .conftest import PARAMS

MODELS = [YUKinematicPS, YUKinematicProjPS]

# Reaching the transition band takes a state whose stagnation radius nearly
# equals where beta lands, which no natural history produces; r_n is nudged by
# these fractions instead.
_R_FRACTIONS = np.linspace(0.990, 1.010, 41)


def _plastic_base(model, n_steps=15):
    """March into the plastic regime; return (stress, state)."""
    integrator = PythonNumericalIntegrator(model)
    stress, state = np.zeros(3), model.initial_state()
    for _ in range(n_steps):
        result = integrator.stress_update(np.array([1.5e-3, -5e-4, 0.0]), stress, state)
        stress, state = np.asarray(result.stress), dict(result.state)
    return stress, state


def _transition_cases(model):
    """Yield (result, state_n, g_stag, Gg) for steps landing inside the band."""
    integrator = PythonNumericalIntegrator(model)
    stress, state = _plastic_base(model)
    reached = model.vonmises_norm(np.asarray(state["beta"]) - np.asarray(state["q"]))
    for fraction in _R_FRACTIONS:
        state_n = dict(state)
        state_n["r"] = float(reached) * fraction
        result = integrator.stress_update(np.array([2e-5, -7e-6, 0.0]), stress, state_n)
        if not result.is_plastic:
            continue
        g_xi = np.asarray(result.state["beta"]) - np.asarray(state_n["q"])
        g_stag = float(model.vonmises_norm(g_xi) - state_n["r"])
        Gg = float(smooth_heaviside(g_stag + 1.0e-10))
        if 1e-9 < Gg < 1.0 - 1e-9:
            yield result, state_n, g_stag, Gg


@pytest.mark.parametrize("cls", MODELS)
def test_ft_fb_is_exact_away_from_the_transition_band(cls):
    """Off the band the omitted term vanishes, so the block must be exact.

    This is the regime every ordinary history stays in, which is why the
    omission went unnoticed.
    """
    model = cls(**PARAMS)
    checker = JacobianChecker(model)
    integrator = PythonNumericalIntegrator(model)
    stress, state = np.zeros(3), model.initial_state()
    n_checked = 0

    for _ in range(30):
        state_n = dict(state)
        result = integrator.stress_update(np.array([1.5e-3, -5e-4, 0.0]), stress, state)
        stress, state = np.asarray(result.stress), dict(result.state)
        if not result.is_plastic:
            continue
        g_xi = np.asarray(state["beta"]) - np.asarray(state_n["q"])
        Gg = float(smooth_heaviside(float(model.vonmises_norm(g_xi) - state_n["r"]) + 1e-10))
        if not (Gg < 1e-9 or Gg > 1.0 - 1e-9):
            continue
        n_checked += 1
        ref = np.asarray(checker.compute(result, state_n).part["theta"]["beta"], float)
        got = np.asarray(model.calc_ft_fb(state, result.dlambda, state_n), float)
        err = np.abs(got - ref).max() / max(np.abs(ref).max(), 1.0)
        assert err < 1e-8, f"ft_fb off the band should be exact, got {err:.3e}"

    assert n_checked > 0, "history never left the transition band -- test proves nothing"


@pytest.mark.parametrize("cls", MODELS)
def test_ft_fb_error_tracks_the_gate_derivative(cls):
    """Inside the band the error is proportional to dGg/dg_stag.

    That proportionality is the evidence that the discrepancy is the missing
    gate-derivative term and not an unrelated algebra slip: the error follows
    the sigmoid's slope across four orders of magnitude.
    """
    model = cls(**PARAMS)
    checker = JacobianChecker(model)
    samples = []

    for result, state_n, g_stag, _Gg in _transition_cases(model):
        ref = np.asarray(checker.compute(result, state_n).part["theta"]["beta"], float)
        got = np.asarray(model.calc_ft_fb(dict(result.state), result.dlambda, state_n), float)
        err = np.abs(got - ref).max() / max(np.abs(ref).max(), 1.0)
        slope = 0.5 * 500.0 * (1.0 - np.tanh(0.5 * 500.0 * g_stag) ** 2)
        samples.append((slope, err))

    assert len(samples) >= 5, "not enough transition-band samples"
    steep = max(samples, key=lambda s: s[0])
    flat = min(samples, key=lambda s: s[0])
    assert steep[1] > 1e-5, f"steepest gate should show a clear error, got {steep[1]:.3e}"
    assert steep[1] > flat[1] * 10, "error does not follow the gate derivative"


@pytest.mark.parametrize("cls", MODELS)
def test_transition_band_error_is_bounded(cls):
    """Record the worst case so a future analytical term can be compared.

    ~2% on theta::beta is what the projected variant stands to recover by
    differentiating its closed-form mu; the bound is deliberately loose so the
    test documents the magnitude without pinning a specific number.
    """
    model = cls(**PARAMS)
    checker = JacobianChecker(model)
    worst = 0.0

    for result, state_n, _g_stag, _Gg in _transition_cases(model):
        ref = np.asarray(checker.compute(result, state_n).part["theta"]["beta"], float)
        got = np.asarray(model.calc_ft_fb(dict(result.state), result.dlambda, state_n), float)
        worst = max(worst, np.abs(got - ref).max() / max(np.abs(ref).max(), 1.0))

    assert 1e-4 < worst < 0.1, f"transition-band error moved out of range: {worst:.3e}"
