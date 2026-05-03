"""Test case generation for constitutive model verification.

Provides model-agnostic utilities for generating strain increments and
loading histories suitable for verifying constitutive model implementations.

Functions
---------
estimate_yield_strain
    Estimate the uniaxial yield strain via bisection on the yield surface.
generate_single_step_cases
    Generate a set of single-increment test cases for use with
    :class:`~manforge.verification.SolverComparison`.
generate_strain_history
    Generate a default tension-unload-compression strain history for
    multi-step comparison.
"""

import numpy as np

from manforge.core.stress_update import stress_update


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
    C = model.elastic_stiffness()
    state_0 = model.initial_state()

    e1 = (lambda _a: (_a.__setitem__(0, 1.0), _a)[1])(np.zeros(ntens))
    sigma_unit = C @ e1

    eps_hi = 1e-3
    for _ in range(60):
        f_hi = float(model.yield_function(eps_hi * sigma_unit, state_0))
        if f_hi > 0.0:
            break
        eps_hi *= 10.0
    else:
        C_diag_max = float(np.max(np.abs(np.diag(C))))
        return 1.0 / C_diag_max

    eps_lo = 0.0
    for _ in range(60):
        eps_mid = (eps_lo + eps_hi) / 2.0
        f_mid = float(model.yield_function(eps_mid * sigma_unit, state_0))
        if f_mid > 0.0:
            eps_hi = eps_mid
        else:
            eps_lo = eps_mid

    return (eps_lo + eps_hi) / 2.0


def generate_single_step_cases(model, eps_y=None) -> list[dict]:
    """Generate single-increment test cases spanning elastic and plastic regimes.

    Parameters
    ----------
    model : MaterialModel
        Constitutive model instance.
    eps_y : float, optional
        Characteristic yield strain.  If *None*, estimated via
        :func:`estimate_yield_strain`.

    Returns
    -------
    list[dict]
        Each dict has keys ``"strain_inc"``, ``"stress_n"``, ``"state_n"``
        — compatible with :class:`~manforge.verification.SolverComparison`.
    """
    if eps_y is None:
        eps_y = estimate_yield_strain(model)

    ntens = model.ntens
    ndi   = model.stress_state.ndi
    nshr  = model.stress_state.nshr
    state_0 = dict(model.initial_state())
    zero_stress = np.zeros(ntens)

    cases = []

    # Case 1: Elastic uniaxial
    de = np.zeros(ntens)
    de[0] = 0.5 * eps_y
    cases.append({"strain_inc": de, "stress_n": zero_stress.copy(), "state_n": dict(state_0)})

    # Case 2: Plastic uniaxial
    de = np.zeros(ntens)
    de[0] = 5.0 * eps_y
    cases.append({"strain_inc": de, "stress_n": zero_stress.copy(), "state_n": dict(state_0)})

    # Case 3: Plastic multiaxial
    if ndi >= 2:
        de = np.zeros(ntens)
        de[0] = 3.0 * eps_y
        de[1] = -1.5 * eps_y
        if ndi >= 3:
            de[2] = -1.5 * eps_y
        cases.append({"strain_inc": de, "stress_n": zero_stress.copy(), "state_n": dict(state_0)})

    # Case 4: Shear-dominant
    if nshr >= 1:
        de = np.zeros(ntens)
        de[ndi] = 3.0 * eps_y
        cases.append({"strain_inc": de, "stress_n": zero_stress.copy(), "state_n": dict(state_0)})

    # Case 5: Pre-stressed starting state
    prestress_de = (lambda _a: (_a.__setitem__(0, 3.0 * eps_y), _a)[1])(np.zeros(ntens))
    _pre = stress_update(model, prestress_de, np.zeros(ntens), model.initial_state())
    stress_pre, state_pre = _pre.stress, _pre.state
    de2 = np.zeros(ntens)
    de2[0] = 2.0 * eps_y
    cases.append({
        "strain_inc": de2,
        "stress_n":   np.array(stress_pre),
        "state_n":    {k: np.asarray(v) for k, v in state_pre.items()},
    })

    return cases


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
