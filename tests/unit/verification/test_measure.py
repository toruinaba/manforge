"""The one error measure every check uses (verification/_measure.py)."""

import numpy as np
import pytest

from returnmap.verification._measure import array_rel_err, as_error, max_abs, rel_err


def test_largest_difference_over_largest_reference():
    ref = np.array([[2.0, 0.0], [0.0, -4.0]])
    cand = ref + np.array([[1e-3, 0.0], [0.0, 2e-3]])
    assert array_rel_err(cand, ref) == pytest.approx(2e-3 / 4.0)


def test_round_off_on_a_zero_entry_is_not_a_large_error():
    """Component by component this entry would be 100 % off."""
    ref = np.array([1e5, 0.0])
    cand = np.array([1e5, 1e-12])
    assert array_rel_err(cand, ref) == pytest.approx(1e-17)


@pytest.mark.parametrize("unit", [1e-9, 1.0, 1e6])
def test_no_constant_carries_a_unit(unit):
    """The same relative discrepancy reads the same in Pa, MPa or GPa."""
    ref = unit * np.array([210e3, 80e3, 0.0])
    cand = ref * (1.0 + 1e-4)
    assert array_rel_err(cand, ref) == pytest.approx(1e-4)


def test_two_zeros_agree_and_a_nonzero_against_zero_does_not():
    assert array_rel_err(np.zeros(3), np.zeros(3)) == 0.0
    assert array_rel_err(np.array([0.0, 1e-30]), np.zeros(2)) == np.inf


@pytest.mark.parametrize("cand, ref", [
    (np.array([1.0, np.nan]), np.array([1.0, 1.0])),      # NaN in the candidate
    (np.array([1.0, 1.0]), np.array([1.0, np.nan])),      # NaN in the reference
    (np.array([np.inf, 1.0]), np.array([1.0, 1.0])),      # overflow
    (np.array([np.inf, 1.0]), np.array([np.inf, 1.0])),   # inf − inf
])
def test_non_finite_values_are_the_worst_error_not_the_absence_of_one(cand, ref):
    """``nan > x`` is false for every x; a NaN must not read as "no error"."""
    assert array_rel_err(cand, ref) == np.inf


def test_rel_err_and_as_error_agree_on_nan():
    assert rel_err(float("nan"), 1.0) == np.inf
    assert rel_err(1.0, float("nan")) == np.inf
    assert rel_err(0.0, 0.0) == 0.0
    assert as_error(float("nan")) == np.inf
    assert as_error(0.25) == 0.25
    assert np.isnan(max_abs([1.0, np.nan]))
