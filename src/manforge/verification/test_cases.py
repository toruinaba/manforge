"""Test case generation for constitutive model verification.

Provides model-agnostic utilities for generating strain increments and
loading histories suitable for verifying constitutive model implementations.

Functions
---------
estimate_yield_strain
    Estimate the uniaxial yield strain via bisection on the yield surface.
generate_strain_history
    Generate a default tension-unload-compression strain history for
    multi-step comparison.
make_increment
    Assemble an Increment by hand, to call a model's hooks at a chosen point.
"""

import numpy as np

from manforge.core.state import Increment, State
from manforge.simulation.types import FieldHistory


def make_increment(model, state_n, *, stress_trial, strain_inc=None) -> Increment:
    """Assemble an :class:`~manforge.core.Increment` by hand.

    For calling a model's hooks directly — ``state_residual``,
    ``update_state``, ``elastic_check`` — at a point no integrator produced.
    In a real step the integrator builds the increment
    (``integrator.increment(strain_inc, stress_n, state_n)``) and the result
    carries it (``result.increment``); prefer those when a step exists.

    The trial state is ``state_n`` with σ replaced by ``stress_trial``, which
    is what the default (linear) elastic predictor produces: only the stress
    moves.

    Parameters
    ----------
    model : MaterialModel
    state_n : State or dict
        State at the beginning of the increment.
    stress_trial : array, shape (ntens,), keyword-only
        Elastic trial stress.
    strain_inc : array, shape (ntens,), optional
        Strain increment.  Zero when omitted — for hooks that do not read it.
        It is not derived from ``stress_trial``, so pass it whenever the hook
        under test does.
    """
    data_n = dict(state_n.items())
    fields = tuple(model.state_names)
    if strain_inc is None:
        strain_inc = np.zeros(model.ntens)
    return Increment(
        state_n=State(data_n, fields),
        strain_inc=strain_inc,
        trial=State({**data_n, "stress": stress_trial}, fields),
    )


def estimate_yield_strain(model) -> float:
    """Estimate the yield strain in the first component direction.

    Applies a unit strain in the ``e_1 = [1, 0, ..., 0]`` direction and
    uses bisection to find the scalar multiplier ``eps_y`` such that::

        yield_function(eps_y * C @ e_1, initial_state) = 0

    Parameters
    ----------
    model : MaterialModel
        Constitutive model instance.

    Returns
    -------
    float
        Approximate yield strain in the ``e_1`` direction.
    """
    ntens = model.ntens
    state_0 = model.initial_state()
    C = model.elastic_stiffness(state_0)

    e1 = (lambda _a: (_a.__setitem__(0, 1.0), _a)[1])(np.zeros(ntens))
    sigma_unit = C @ e1

    from manforge.core.state import _state_with_stress
    eps_hi = 1e-3
    for _ in range(60):
        state_hi = _state_with_stress(state_0, eps_hi * sigma_unit)
        f_hi = float(model.yield_function(state_hi))
        if f_hi > 0.0:
            break
        eps_hi *= 10.0
    else:
        C_diag_max = float(np.max(np.abs(np.diag(C))))
        return 1.0 / C_diag_max

    eps_lo = 0.0
    for _ in range(60):
        eps_mid = (eps_lo + eps_hi) / 2.0
        state_mid = _state_with_stress(state_0, eps_mid * sigma_unit)
        f_mid = float(model.yield_function(state_mid))
        if f_mid > 0.0:
            eps_hi = eps_mid
        else:
            eps_lo = eps_mid

    return (eps_lo + eps_hi) / 2.0



def generate_strain_history(model, eps_y=None) -> np.ndarray:
    """Generate a uniaxial tension-unload-compression strain history.

    Parameters
    ----------
    model : MaterialModel
        Constitutive model instance.
    eps_y : float, optional
        Characteristic yield strain.  If *None*, estimated via
        :func:`estimate_yield_strain`.

    Returns
    -------
    np.ndarray, shape (N, ntens)
        Cumulative total strain at each step (``N = 35``).
    """
    if eps_y is None:
        eps_y = estimate_yield_strain(model)

    peaks = [
        0.5 * eps_y,
        5.0 * eps_y,
        2.0 * eps_y,
        0.0,
        -5.0 * eps_y,
        -2.0 * eps_y,
        0.0,
    ]
    return FieldHistory.cyclic_strain(
        peaks, n_per_segment=5, ntens=model.ntens, component=0
    ).data
