"""Unit tests for the test_cases module (pure-Python, no Fortran needed)."""

import numpy as np
import pytest

from manforge.verification.test_cases import (
    estimate_yield_strain,
    generate_single_step_cases,
    generate_strain_history,
)


# ---------------------------------------------------------------------------
# estimate_yield_strain
# ---------------------------------------------------------------------------

def test_estimate_yield_strain_approx(model):
    """Estimated yield strain matches the analytical J2 3D uniaxial value.

    For J2 under uniaxial strain [eps,0,0,0,0,0], the von Mises stress is
    2*mu*eps (derivable from the deviatoric stress of the uniaxial elastic
    state), so yield occurs at eps_y = sigma_y0 / (2*mu).
    """
    E  = model.E
    nu = model.nu
    mu = E / (2.0 * (1.0 + nu))
    eps_y_expected = model.sigma_y0 / (2.0 * mu)

    eps_y = estimate_yield_strain(model)

    assert abs(eps_y - eps_y_expected) / eps_y_expected < 0.01  # 1% tolerance


def test_estimate_yield_strain_positive(model):
    """Yield strain must be strictly positive."""
    eps_y = estimate_yield_strain(model)
    assert eps_y > 0.0


def test_estimate_yield_strain_respects_sigma_y0(model):
    """Harder material (larger sigma_y0) gives larger yield strain."""
    from manforge.models.j2_isotropic import J2Isotropic3D
    model_soft = J2Isotropic3D(E=model.E, nu=model.nu, sigma_y0=100.0, H=model.H)
    model_hard = J2Isotropic3D(E=model.E, nu=model.nu, sigma_y0=500.0, H=model.H)
    eps_soft = estimate_yield_strain(model_soft)
    eps_hard = estimate_yield_strain(model_hard)
    assert eps_hard > eps_soft


# ---------------------------------------------------------------------------
# generate_single_step_cases
# ---------------------------------------------------------------------------

def test_generate_single_step_cases_count(model):
    """5 test cases are generated for a 3D SOLID model."""
    cases = generate_single_step_cases(model)
    assert len(cases) == 5


def test_generate_single_step_cases_keys(model):
    """Each case has the required keys for SolverCrosscheck."""
    cases = generate_single_step_cases(model)
    required = {"strain_inc", "stress_n", "state_n"}
    for case in cases:
        assert required.issubset(case.keys())


def test_generate_single_step_cases_shapes(model):
    """strain_inc and stress_n have shape (6,) for SOLID_3D."""
    cases = generate_single_step_cases(model)
    for case in cases:
        assert np.asarray(case["strain_inc"]).shape == (6,)
        assert np.asarray(case["stress_n"]).shape   == (6,)


def test_generate_single_step_elastic_case_is_small(model):
    """First case (elastic) uses a strain magnitude smaller than yield."""
    eps_y = estimate_yield_strain(model)
    cases = generate_single_step_cases(model, eps_y=eps_y)
    elastic_de = np.asarray(cases[0]["strain_inc"])
    assert np.linalg.norm(elastic_de) < eps_y


def test_generate_single_step_plastic_case_is_large(model):
    """Second case (plastic uniaxial) uses a strain larger than yield."""
    eps_y = estimate_yield_strain(model)
    cases = generate_single_step_cases(model, eps_y=eps_y)
    plastic_de = np.asarray(cases[1]["strain_inc"])
    assert plastic_de[0] > eps_y


def test_generate_single_step_prestressed_case_has_nonzero_stress(model):
    """Pre-stressed case (last) has a non-zero stress_n."""
    cases = generate_single_step_cases(model)
    last_stress = np.asarray(cases[-1]["stress_n"])
    assert np.any(last_stress != 0.0)


# ---------------------------------------------------------------------------
# generate_strain_history
# ---------------------------------------------------------------------------

def test_generate_strain_history_shape(model):
    """Default strain history has shape (35, ntens)."""
    history = generate_strain_history(model)
    assert history.shape == (35, model.ntens)


def test_generate_strain_history_axial_only(model):
    """Default history applies strain only in the first component."""
    history = generate_strain_history(model)
    # All non-axial columns should be zero
    assert np.all(history[:, 1:] == 0.0)


def test_generate_strain_history_returns_to_zero(model):
    """Default history ends at zero strain."""
    history = generate_strain_history(model)
    assert abs(history[-1, 0]) < 1e-14


def test_generate_strain_history_includes_compression(model):
    """Default history reaches negative strains (compression)."""
    history = generate_strain_history(model)
    assert np.any(history[:, 0] < 0.0)


def test_generate_strain_history_custom_scale(model):
    """Custom eps_y scales the history proportionally."""
    eps_y = 2e-3
    history = generate_strain_history(model, eps_y=eps_y)
    assert abs(np.max(history[:, 0]) - 5.0 * eps_y) < 1e-12
