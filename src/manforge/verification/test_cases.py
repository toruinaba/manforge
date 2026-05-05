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
"""

import numpy as np


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

    ntens = model.ntens
    steps_per_segment = 5

    targets = [
        0.5 * eps_y,
        5.0 * eps_y,
        2.0 * eps_y,
        0.0,
        -5.0 * eps_y,
        -2.0 * eps_y,
        0.0,
    ]

    axial = [0.0]
    for target in targets:
        segment = np.linspace(axial[-1], target, steps_per_segment + 1)[1:]
        axial.extend(segment.tolist())

    axial = np.array(axial[1:])

    history = np.zeros((len(axial), ntens))
    history[:, 0] = axial
    return history
