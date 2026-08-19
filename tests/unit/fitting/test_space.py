"""Unit tests for ParameterSpace — variable/parameter mapping only, no solver."""

import numpy as np
import pytest

from manforge.fitting.space import ParameterSpace


def test_x0_and_bounds_follow_declaration_order():
    space = ParameterSpace(
        free={"H": (800.0, (0.0, 5000.0)), "sigma_y0": (220.0, None)},
    )
    assert space.free_names == ("H", "sigma_y0")
    np.testing.assert_allclose(space.x0, [800.0, 220.0])
    assert space.bounds == [(0.0, 5000.0), (-np.inf, np.inf)]


def test_one_sided_bound():
    space = ParameterSpace(free={"H": (800.0, (0.0, None))})
    assert space.bounds == [(0.0, np.inf)]


def test_resolve_merges_fixed_free_derived():
    space = ParameterSpace(
        free={"Y": (200.0, (50.0, 600.0)), "B_ratio": (1.5, (1.0, 5.0))},
        fixed={"E": 210_000.0},
        derived={"B": lambda p: p["Y"] * p["B_ratio"]},
    )
    values = space.resolve([300.0, 2.0])
    assert values == {"E": 210_000.0, "Y": 300.0, "B_ratio": 2.0, "B": 600.0}


def test_derived_may_depend_on_earlier_derived():
    space = ParameterSpace(
        free={"Y": (100.0, None)},
        derived={
            "B": lambda p: 2.0 * p["Y"],
            "C": lambda p: p["B"] + 1.0,
        },
    )
    assert space.resolve([100.0])["C"] == 201.0


def test_model_params_selects_and_drops_helper_variables():
    space = ParameterSpace(
        free={"Y": (200.0, None), "B_ratio": (1.5, None)},
        fixed={"E": 210_000.0, "nu": 0.3},
        derived={"B": lambda p: p["Y"] * p["B_ratio"]},
    )
    params = space.model_params([200.0, 1.5], ["E", "nu", "Y", "B"])
    assert params == {"E": 210_000.0, "nu": 0.3, "Y": 200.0, "B": 300.0}


def test_model_params_reports_unsupplied_parameter():
    space = ParameterSpace(free={"Y": (200.0, None)})
    with pytest.raises(ValueError, match="no value for model parameter"):
        space.model_params([200.0], ["Y", "B"])


def test_resolve_rejects_wrong_length_vector():
    space = ParameterSpace(free={"Y": (200.0, None), "H": (1.0, None)})
    with pytest.raises(ValueError, match="expected 2 free values"):
        space.resolve([200.0])


def test_duplicate_name_across_groups_rejected():
    with pytest.raises(ValueError, match="declared in both"):
        ParameterSpace(free={"Y": (200.0, None)}, fixed={"Y": 1.0})


def test_empty_free_rejected():
    with pytest.raises(ValueError, match="at least one free variable"):
        ParameterSpace(free={})


def test_coerce_dict_form():
    space = ParameterSpace.coerce(
        {"sigma_y0": (220.0, (50.0, 600.0))}, {"E": 210_000.0}
    )
    assert space.free_names == ("sigma_y0",)
    assert space.fixed == {"E": 210_000.0}


def test_coerce_passes_through_parameter_space():
    original = ParameterSpace(free={"Y": (200.0, None)})
    assert ParameterSpace.coerce(original) is original


def test_coerce_rejects_fixed_params_with_parameter_space():
    space = ParameterSpace(free={"Y": (200.0, None)}, fixed={"E": 1.0})
    with pytest.raises(ValueError, match="fixed_params cannot be combined"):
        ParameterSpace.coerce(space, {"nu": 0.3})
