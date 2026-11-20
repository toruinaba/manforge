"""fit_params — how the search is set up, and what it tells the caller.

Whether a fit recovers the right parameters is covered end to end in
``tests/e2e/test_fitting_pipeline.py``.
"""

import warnings

import numpy as np
import pytest

import returnmap.fitting.optimizer as optimizer
from returnmap.fitting import (
    FitTarget, ParameterSpace, SLOW_EVALUATION_SECONDS, SlowFitWarning, fit_params,
)
from returnmap.fitting.optimizer import _SearchScale
from returnmap.models import AFKinematic1D, J2Isotropic1D
from returnmap.simulation import FieldHistory, FieldType, PythonIntegrator, StrainDriver

ELASTIC = dict(E=206000.0, nu=0.3)
J2 = J2Isotropic1D(**ELASTIC, sigma_y0=250.0, H=2000.0)
AF = AFKinematic1D(**ELASTIC, sigma_y0=250.0, C_k=20000.0, gamma=100.0)

STRAIN = np.linspace(0.0, 6e-3, 20)


def target_of(model):
    """The model's own response, as the curve to fit."""
    result = StrainDriver(PythonIntegrator(model)).run(FieldHistory(FieldType.STRAIN, STRAIN))
    return FitTarget.stress_strain(STRAIN, result.stress[:, 0])


J2_SPACE = ParameterSpace(
    free={"sigma_y0": (220.0, (100.0, 500.0)), "H": (1500.0, (0.0, 1e4))}, fixed=ELASTIC)
AF_SPACE = ParameterSpace(
    free={"C_k": (15000.0, (1000.0, 1e5))},
    fixed=dict(ELASTIC, sigma_y0=250.0, gamma=100.0))


def response(model, strain):
    load = FieldHistory(FieldType.STRAIN, strain)
    return StrainDriver(PythonIntegrator(model)).run(load).stress[:, 0]


# ---------------------------------------------------------------------------
# The search variables: each free variable scaled by the width of its bounds
# ---------------------------------------------------------------------------

def test_every_bounded_variable_spans_zero_to_one():
    space = ParameterSpace(
        free={"sigma_y0": (250.0, (100.0, 500.0)), "C_k": (2e4, (5e3, 8e4)),
              "gamma": (100.0, (20.0, 500.0))},
        fixed=ELASTIC,
    )
    search = _SearchScale.of(space)
    assert search.bounds == [(0.0, 1.0)] * 3
    np.testing.assert_allclose(search.to_search(space.x0), [0.375, 0.2, 1 / 6])
    np.testing.assert_allclose(search.to_params(search.to_search(space.x0)), space.x0)


def test_search_values_map_back_inside_the_bounds():
    """offset + scale * 1.0 can overshoot the upper bound by a rounding step."""
    space = ParameterSpace(free={"a": (0.5, (0.3, 0.9))})
    search = _SearchScale.of(space)
    assert 0.3 + (0.9 - 0.3) * 1.0 > 0.9          # the overshoot being guarded
    assert search.to_params([1.0])[0] == 0.9
    assert search.to_params([0.0])[0] == 0.3


@pytest.mark.parametrize("bounds, initial, scale, searched", [
    (None, 2e4, 2e4, (-np.inf, np.inf)),          # unbounded: its own magnitude
    ((0.0, None), 2e4, 2e4, (0.0, np.inf)),       # one-sided: the same
    (None, 0.0, 1.0, (-np.inf, np.inf)),          # nothing to scale by
    (None, -50.0, 50.0, (-np.inf, np.inf)),
], ids=["unbounded", "one-sided", "zero", "negative"])
def test_a_variable_without_a_finite_range_is_scaled_by_its_initial_value(
    bounds, initial, scale, searched,
):
    search = _SearchScale.of(ParameterSpace(free={"a": (initial, bounds)}))
    assert search.scale[0] == scale
    assert search.offset[0] == 0.0
    assert search.bounds == [searched]
    assert search.to_params(search.to_search([initial]))[0] == initial


def test_results_are_reported_in_the_units_of_the_bounds():
    strain = np.linspace(0.0, 6e-3, 20)
    true = J2Isotropic1D(**ELASTIC, sigma_y0=250.0, H=2000.0)
    fit = fit_params(
        true, [FitTarget.stress_strain(strain, response(true, strain))],
        ParameterSpace(free={"sigma_y0": (220.0, (100.0, 500.0)), "H": (1500.0, (0.0, 1e4))},
                       fixed=ELASTIC),
    )
    assert fit.model_params["sigma_y0"] == pytest.approx(250.0, rel=1e-5)
    assert fit.model_params["H"] == pytest.approx(2000.0, rel=1e-5)
    assert fit.history, "the optimiser's iterates are recorded"
    for point in fit.history:
        assert 100.0 <= point["sigma_y0"] <= 500.0 and 0.0 <= point["H"] <= 1e4
    assert fit.history[-1] == pytest.approx(
        {"sigma_y0": fit.params["sigma_y0"], "H": fit.params["H"]})


def test_nelder_mead_is_scaled_but_not_held_to_the_bounds():
    """Documented as running without bounds; the optimum here lies outside them."""
    strain = np.linspace(0.0, 6e-3, 20)
    true = J2Isotropic1D(**ELASTIC, sigma_y0=250.0, H=2000.0)
    fit = fit_params(
        true, [FitTarget.stress_strain(strain, response(true, strain))],
        ParameterSpace(free={"H": (800.0, (0.0, 1000.0))},
                       fixed=dict(ELASTIC, sigma_y0=250.0)),
        method="Nelder-Mead",
    )
    assert fit.model_params["H"] == pytest.approx(2000.0, rel=1e-4)


@pytest.mark.slow
def test_the_optimum_does_not_depend_on_where_a_large_parameter_starts():
    """C_k is two orders of magnitude larger than gamma and sigma_y0.

    Searching on the raw values, L-BFGS-B started with C_k near its optimum
    reported convergence after 28 evaluations with C_k exactly where it
    began (residual 5.53 against 5.32 from the other start).
    """
    strain = np.concatenate([
        np.linspace(0.0, 0.015, 16)[1:], np.linspace(0.015, -0.015, 31)[1:],
    ])
    true = AFKinematic1D(**ELASTIC, sigma_y0=285.0, C_k=26000.0, gamma=140.0)
    noise = np.random.default_rng(0).normal(0.0, 3.0, len(strain))
    target = FitTarget.stress_strain(strain, response(true, strain) + noise)

    def fit_from(C_k):
        start = dict(sigma_y0=290.0, C_k=C_k, gamma=135.0)
        return fit_params(
            AFKinematic1D(**ELASTIC, **start), [target],
            ParameterSpace(
                free={"sigma_y0": (290.0, (150.0, 450.0)), "C_k": (C_k, (5e3, 8e4)),
                      "gamma": (135.0, (20.0, 500.0))},
                fixed=ELASTIC,
            ),
            tol=1e-6, nr_jacobian="broyden",
        )

    near, far = fit_from(26000.0), fit_from(30000.0)

    assert near.model_params["C_k"] != 26000.0
    assert near.residual == pytest.approx(far.residual, rel=1e-4)
    for name in ("sigma_y0", "C_k", "gamma"):
        assert near.model_params[name] == pytest.approx(far.model_params[name], rel=1e-3)
    # Better than the parameters the data was generated with, as a fit to
    # noisy data should be.
    assert near.residual < target.residual(PythonIntegrator(true), None)


# ---------------------------------------------------------------------------
# progress
# ---------------------------------------------------------------------------

def test_progress_is_called_after_every_evaluation_of_the_search():
    calls = []
    fit = fit_params(J2, [target_of(J2)], J2_SPACE,
                     progress=lambda n, residual, best: calls.append((n, residual, best)))

    assert [n for n, _, _ in calls] == list(range(1, fit.n_evaluations + 1))
    best = [b for _, _, b in calls]
    assert all(later <= earlier for earlier, later in zip(best, best[1:]))
    assert all(b <= r for _, r, b in calls)
    # The optimiser reports the point it settled on, which is one of the
    # evaluations but not necessarily the smallest (a finite-difference probe
    # next to it can come out a hair lower).
    assert fit.residual in [r for _, r, _ in calls]
    assert best[-1] <= fit.residual


def test_progress_print_is_enough(capsys):
    fit_params(J2, [target_of(J2)], J2_SPACE, max_iter=1, progress=print)
    first = capsys.readouterr().out.splitlines()[0].split()
    assert first[0] == "1" and float(first[1]) == float(first[2])


def test_progress_does_not_change_the_fit():
    plain = fit_params(J2, [target_of(J2)], J2_SPACE)
    watched = fit_params(J2, [target_of(J2)], J2_SPACE, progress=lambda *a: None)
    assert watched.model_params == plain.model_params
    assert watched.n_evaluations == plain.n_evaluations


# ---------------------------------------------------------------------------
# SlowFitWarning
# ---------------------------------------------------------------------------

def test_a_quick_fit_does_not_warn():
    assert SLOW_EVALUATION_SECONDS == 0.5
    with warnings.catch_warnings():
        warnings.simplefilter("error", SlowFitWarning)
        fit_params(J2, [target_of(J2)], J2_SPACE, max_iter=1)


def test_a_slow_evaluation_warns_once_before_the_search(monkeypatch):
    monkeypatch.setattr(optimizer, "SLOW_EVALUATION_SECONDS", 0.0)
    with pytest.warns(SlowFitWarning) as caught:
        fit_params(AF, [target_of(AF)], AF_SPACE, max_iter=1)
    assert len(caught) == 1
    message = str(caught[0].message)
    assert "one evaluation of the objective takes" in message
    assert "(20 load steps)" in message
    # Reported at the caller's line, not inside the library.
    assert caught[0].filename == __file__


def test_the_warning_names_automatic_differentiation_as_the_cause(monkeypatch):
    monkeypatch.setattr(optimizer, "SLOW_EVALUATION_SECONDS", 0.0)
    with pytest.warns(SlowFitWarning) as caught:
        fit_params(AF, [target_of(AF)], AF_SPACE, max_iter=1)
    message = str(caught[0].message)
    assert "AFKinematic1D has no hand-written return mapping" in message
    assert 'nr_jacobian="broyden"' in message
    assert "a model that has a hand-written return mapping" in message


def test_the_warning_does_not_suggest_what_is_already_in_use(monkeypatch):
    monkeypatch.setattr(optimizer, "SLOW_EVALUATION_SECONDS", 0.0)
    with pytest.warns(SlowFitWarning) as caught:
        fit_params(AF, [target_of(AF)], AF_SPACE, max_iter=1, nr_jacobian="broyden")
    assert "broyden" not in str(caught[0].message)


def test_a_closed_form_model_is_not_blamed_on_automatic_differentiation(monkeypatch):
    """Slow for another reason (a very long history): only that is suggested."""
    monkeypatch.setattr(optimizer, "SLOW_EVALUATION_SECONDS", 0.0)
    with pytest.warns(SlowFitWarning) as caught:
        fit_params(J2, [target_of(J2)], J2_SPACE, max_iter=1)
    message = str(caught[0].message)
    assert "hand-written" not in message and "broyden" not in message
    assert "fewer load steps" in message
