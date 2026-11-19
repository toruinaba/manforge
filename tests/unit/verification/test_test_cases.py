"""Smoke tests for verification/test_cases.py.

TODO: cover generate_strain_history shapes, scenario variety, seed reproducibility.
"""

import pytest
from manforge.verification import generate_strain_history
from manforge.models.j2_isotropic import J2Isotropic3D


def test_generate_strain_history_importable():
    assert callable(generate_strain_history)


def test_generate_strain_history_shape():
    model = J2Isotropic3D(E=210000.0, nu=0.3, sigma_y0=250.0, H=1000.0)
    hist = generate_strain_history(model)
    assert hist.ndim == 2
    assert hist.shape[1] == model.ntens


def test_make_increment_trial_is_state_n_with_the_trial_stress():
    import numpy as np
    from manforge.verification import make_increment

    model = J2Isotropic3D(E=210000.0, nu=0.3, sigma_y0=250.0, H=1000.0)
    state_n = {**model.initial_state(), "ep": np.array(0.01)}
    trial = np.array([300.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    inc = make_increment(model, state_n, stress_trial=trial)

    np.testing.assert_array_equal(inc.trial["stress"], trial)
    assert float(inc.trial["ep"]) == 0.01          # only the stress moves
    np.testing.assert_array_equal(inc.state_n["stress"], state_n["stress"])
    np.testing.assert_array_equal(inc.strain_inc, np.zeros(6))

    deps = np.array([1e-3, 0.0, 0.0, 0.0, 0.0, 0.0])
    assert make_increment(model, state_n, stress_trial=trial, strain_inc=deps).strain_inc is deps
