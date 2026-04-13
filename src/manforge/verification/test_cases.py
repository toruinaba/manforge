"""Test case generation for constitutive model verification.

Provides model-agnostic utilities for generating strain increments and
loading histories suitable for verifying constitutive model implementations.

All functions require only the :class:`~manforge.core.material.MaterialModel`
interface -- no specific parameter names (e.g. ``"E"``, ``"sigma_y0"``) are
assumed.

Functions
---------
estimate_yield_strain
    Estimate the uniaxial yield strain via bisection on the yield surface.
generate_single_step_cases
    Generate a set of single-increment test cases for use with
    :func:`~manforge.verification.compare.compare_solvers`.
generate_strain_history
    Generate a default tension-unload-compression strain history for
    multi-step comparison.
"""

import numpy as np
import jax.numpy as jnp

from manforge.core.return_mapping import return_mapping


def estimate_yield_strain(model, params) -> float:
    """Estimate the yield strain in the first component direction.

    Applies a unit strain in the ``e_1 = [1, 0, ..., 0]`` direction and
    uses bisection to find the scalar multiplier ``eps_y`` such that::

        yield_function(eps_y * C @ e_1, initial_state, params) = 0

    Requires only ``model.elastic_stiffness``, ``model.yield_function``,
    and ``model.initial_state`` -- no assumption about parameter names.

    Parameters
    ----------
    model : MaterialModel
        Constitutive model instance.
    params : dict
        Material parameters.

    Returns
    -------
    float
        Approximate yield strain in the ``e_1`` direction.  If the model
        never yields along ``e_1`` (purely elastic), falls back to
        ``1 / max(diag(C))``.

    Notes
    -----
    The bisection searches for an upper bound by repeatedly doubling
    ``eps_hi`` from ``1e-3`` until the yield function is positive, then
    performs 60 bisection iterations (~18 significant digits).
    """
    ntens = model.ntens
    C = model.elastic_stiffness(params)
    state_0 = model.initial_state()

    e1 = jnp.zeros(ntens).at[0].set(1.0)
    sigma_unit = C @ e1  # stress from unit strain in e_1 direction

    # Find an upper bound where the yield function is positive
    eps_hi = 1e-3
    for _ in range(60):
        f_hi = float(model.yield_function(eps_hi * sigma_unit, state_0, params))
        if f_hi > 0.0:
            break
        eps_hi *= 10.0
    else:
        # Model appears purely elastic -- fall back to stiffness-based scale
        C_diag_max = float(jnp.max(jnp.abs(jnp.diag(C))))
        return 1.0 / C_diag_max

    # Bisect: find eps s.t. yield_function(eps * sigma_unit, state_0, params) = 0
    eps_lo = 0.0
    for _ in range(60):
        eps_mid = (eps_lo + eps_hi) / 2.0
        f_mid = float(model.yield_function(eps_mid * sigma_unit, state_0, params))
        if f_mid > 0.0:
            eps_hi = eps_mid
        else:
            eps_lo = eps_mid

    return (eps_lo + eps_hi) / 2.0


def generate_single_step_cases(model, params, eps_y=None) -> list[dict]:
    """Generate single-increment test cases spanning elastic and plastic regimes.

    Produces up to 5 test cases based on a characteristic yield strain,
    covering the elastic regime, plastic uniaxial and multiaxial loading,
    shear-dominant loading, and a pre-stressed starting state.

    Parameters
    ----------
    model : MaterialModel
        Constitutive model instance.  Used to determine stress-state
        dimensionality (``ntens``, ``ndi``, ``nshr``) and to construct
        the pre-stressed case via ``return_mapping``.
    params : dict
        Material parameters.
    eps_y : float, optional
        Characteristic yield strain.  If *None* (default), estimated via
        :func:`estimate_yield_strain`.

    Returns
    -------
    list[dict]
        Each dict has keys ``"strain_inc"``, ``"stress_n"``,
        ``"state_n"``, ``"params"`` -- compatible with
        :func:`~manforge.verification.compare.compare_solvers`.

    Notes
    -----
    Cases generated:

    1. **Elastic uniaxial** (0.5 × eps_y, component 0)
    2. **Plastic uniaxial** (5 × eps_y, component 0)
    3. **Plastic multiaxial** (3 × eps_y, deviatoric mix of direct components)
       -- only when ``ndi >= 2``
    4. **Shear-dominant** (3 × eps_y, first shear component)
       -- only when ``nshr >= 1``
    5. **Pre-stressed** (plastic start state + 2 × eps_y uniaxial increment)

    The pre-stressed case uses ``return_mapping`` on the Python model to
    generate a realistic plastic starting point.
    """
    if eps_y is None:
        eps_y = estimate_yield_strain(model, params)

    ntens = model.ntens
    ndi   = model.stress_state.ndi
    nshr  = model.stress_state.nshr
    state_0 = {k: float(v) for k, v in model.initial_state().items()}
    zero_stress = np.zeros(ntens)

    cases = []

    # Case 1: Elastic uniaxial
    de = np.zeros(ntens)
    de[0] = 0.5 * eps_y
    cases.append({
        "strain_inc": de,
        "stress_n":   zero_stress.copy(),
        "state_n":    dict(state_0),
        "params":     params,
    })

    # Case 2: Plastic uniaxial
    de = np.zeros(ntens)
    de[0] = 5.0 * eps_y
    cases.append({
        "strain_inc": de,
        "stress_n":   zero_stress.copy(),
        "state_n":    dict(state_0),
        "params":     params,
    })

    # Case 3: Plastic multiaxial (deviatoric: opposing signs on direct components)
    if ndi >= 2:
        de = np.zeros(ntens)
        de[0] = 3.0 * eps_y
        de[1] = -1.5 * eps_y
        if ndi >= 3:
            de[2] = -1.5 * eps_y
        cases.append({
            "strain_inc": de,
            "stress_n":   zero_stress.copy(),
            "state_n":    dict(state_0),
            "params":     params,
        })

    # Case 4: Shear-dominant
    if nshr >= 1:
        de = np.zeros(ntens)
        de[ndi] = 3.0 * eps_y   # first shear component index
        cases.append({
            "strain_inc": de,
            "stress_n":   zero_stress.copy(),
            "state_n":    dict(state_0),
            "params":     params,
        })

    # Case 5: Pre-stressed starting state
    # Run Python return_mapping to generate a non-virgin stress/state,
    # then apply a further increment.  Both solvers receive the same
    # non-zero (stress_n, state_n) -- this does not test the path to get
    # there, only that both produce the same output from this starting point.
    prestress_de = jnp.zeros(ntens).at[0].set(3.0 * eps_y)
    stress_pre, state_pre, _ = return_mapping(
        model, prestress_de, jnp.zeros(ntens), model.initial_state(), params
    )
    de2 = np.zeros(ntens)
    de2[0] = 2.0 * eps_y
    cases.append({
        "strain_inc": de2,
        "stress_n":   np.array(stress_pre),
        "state_n":    {k: float(v) for k, v in state_pre.items()},
        "params":     params,
    })

    return cases


def generate_strain_history(model, params, eps_y=None) -> np.ndarray:
    """Generate a uniaxial tension-unload-compression strain history.

    Produces a multi-step strain history that exercises the elastic and
    plastic regimes in both tension and compression, including unloading
    and reverse loading.

    Parameters
    ----------
    model : MaterialModel
        Constitutive model instance.  Used to determine ``ntens`` and
        ``eps_y`` when not supplied explicitly.
    params : dict
        Material parameters.
    eps_y : float, optional
        Characteristic yield strain.  If *None* (default), estimated via
        :func:`estimate_yield_strain`.

    Returns
    -------
    np.ndarray, shape (N, ntens)
        Cumulative total strain at each step (``N = 35``).  Only the first
        component (axial) is non-zero.  Increments are computed as
        ``Δε_i = ε_i − ε_{i-1}`` with ``ε_0 = 0``.

    Notes
    -----
    The history visits the following axial strain targets, with 5 equal
    increments per segment::

        0 → +0.5*eps_y  (elastic loading)
          → +5*eps_y    (plastic loading)
          → +2*eps_y    (elastic unloading, still tensile)
          → 0           (unload to zero)
          → -5*eps_y    (compressive plastic)
          → -2*eps_y    (elastic reverse)
          → 0           (return to zero)

    Total: 7 × 5 = 35 steps.
    """
    if eps_y is None:
        eps_y = estimate_yield_strain(model, params)

    ntens = model.ntens
    steps_per_segment = 5

    targets = [
        0.5 * eps_y,    # elastic loading
        5.0 * eps_y,    # plastic loading
        2.0 * eps_y,    # elastic unloading (tensile)
        0.0,            # unload to zero
        -5.0 * eps_y,   # compressive plastic
        -2.0 * eps_y,   # elastic reverse
        0.0,            # return to zero
    ]

    axial = [0.0]
    for target in targets:
        segment = np.linspace(axial[-1], target, steps_per_segment + 1)[1:]
        axial.extend(segment.tolist())

    axial = np.array(axial[1:])  # drop the leading zero; shape (35,)

    history = np.zeros((len(axial), ntens))
    history[:, 0] = axial
    return history
