"""check_tangent on a stand-in integrator (verification/tangent.py).

The check needs only ``ntens`` and ``stress_update``, so a linear update with
a known tangent exercises the report without a material model.  The checks on
real models are in ``tests/integration/verification/test_tangent.py``.
"""

from types import SimpleNamespace

import numpy as np
import pytest

from returnmap.verification import TangentReport, check_tangent

C = np.array([
    [3.0, 1.0, 0.0],
    [1.0, 3.0, 0.0],
    [0.0, 0.0, 1.0],
]) * 1e5


class _Linear:
    """σ_{n+1} = σ_n + C Δε, reporting ``tangent`` as its consistent tangent."""

    def __init__(self, stiffness, tangent=None):
        self.ntens = len(stiffness)
        self._c = np.asarray(stiffness, dtype=float)
        self._tangent = self._c if tangent is None else np.asarray(tangent, dtype=float)
        self.calls: list[tuple] = []

    def stress_update(self, strain_inc, stress_n, state_n):
        self.calls.append((np.array(strain_inc), np.array(stress_n), state_n))
        return SimpleNamespace(stress=stress_n + self._c @ strain_inc, ddsdde=self._tangent)


DEPS, STRESS_N, STATE_N = np.array([1e-3, 0.0, 2e-3]), np.array([10.0, 0.0, 0.0]), {"ep": 0.0}


def test_exact_tangent_passes():
    report = check_tangent(_Linear(C), DEPS, STRESS_N, STATE_N)
    assert isinstance(report, TangentReport)
    assert report.passed is True
    assert report.max_rel_err < 1e-8
    np.testing.assert_array_equal(report.ddsdde, C)
    assert report.ddsdde_fd.shape == (3, 3)
    assert "agrees with central differences" in report.summary()
    assert str(report) == report.summary()


def test_arguments_follow_stress_update():
    """(strain_inc, stress_n, state_n), in the order the integrator takes them."""
    integrator = _Linear(C)
    check_tangent(integrator, DEPS, STRESS_N, STATE_N)
    first_deps, first_stress, first_state = integrator.calls[0]
    np.testing.assert_array_equal(first_deps, DEPS)
    np.testing.assert_array_equal(first_stress, STRESS_N)
    assert first_state is STATE_N
    assert len(integrator.calls) == 1 + 2 * 3            # the update, then ± per component


def test_wrong_entry_fails_and_is_located():
    wrong = C.copy()
    wrong[0, 1] *= 1.5                                   # 0.5e5 off, against max 3e5
    report = check_tangent(_Linear(C, tangent=wrong), DEPS, STRESS_N, STATE_N)
    assert not report.passed
    assert report.max_rel_err == pytest.approx(0.5e5 / 3e5, rel=1e-6)
    assert np.unravel_index(np.argmax(report.rel_err_matrix), (3, 3)) == (0, 1)
    assert report.rel_err_matrix.max() == pytest.approx(report.max_rel_err)
    assert "FAIL ddsdde" in report.summary() and "worst at [0][1]" in report.summary()


@pytest.mark.parametrize("unit", [1e-9, 1.0, 1e6])
def test_error_does_not_depend_on_the_units(unit):
    """A 1e-4 discrepancy is 1e-4 whether the tangent is ~1e-4 or ~1e11.

    With a constant in the denominator (``|fd| + 0.01``) the small-unit case
    passed anything and the large-unit case amplified round-off on the zeros.
    """
    stiffness = C * unit
    report = check_tangent(
        _Linear(stiffness, tangent=stiffness * (1.0 + 1e-4)), DEPS, STRESS_N * unit, STATE_N,
    )
    assert report.max_rel_err == pytest.approx(1e-4, rel=1e-3)
    assert not report.passed


def test_zero_entries_of_the_tangent_are_not_amplified():
    """The exact zeros of C come out of the differences as round-off."""
    report = check_tangent(_Linear(C * 1e6), DEPS, STRESS_N, STATE_N)
    assert report.passed, report.summary()


def test_tolerance_is_inclusive_and_recorded():
    wrong = C * (1.0 + 1e-3)
    loose = check_tangent(_Linear(C, tangent=wrong), DEPS, STRESS_N, STATE_N, tol=1e-2)
    tight = check_tangent(_Linear(C, tangent=wrong), DEPS, STRESS_N, STATE_N, tol=1e-4)
    assert loose.passed and loose.tol == 1e-2
    assert not tight.passed and "tol 1e-04" in tight.summary()
    exact = TangentReport(ddsdde=C, ddsdde_fd=C, tol=0.0)
    assert exact.passed                                   # err == tol passes


def test_nan_tangent_fails():
    bad = C.copy()
    bad[2, 2] = np.nan
    report = check_tangent(_Linear(C, tangent=bad), DEPS, STRESS_N, STATE_N)
    assert not report.passed
    assert report.max_rel_err == np.inf
    assert "FAIL" in report.summary()


# ---------------------------------------------------------------------------
# A tangent that is zero: nothing of its own size to be relative to
# ---------------------------------------------------------------------------

def test_noise_against_a_zero_tangent_is_measured_against_the_elastic_stiffness():
    """Central differences of a perfectly plastic step return round-off, and
    round-off against a tangent of exactly zero is not a 100 % error."""
    from returnmap.verification import TangentReport

    zero = np.zeros((1, 1))
    noise = np.array([[2.8e-7]])
    plain = TangentReport(ddsdde=zero, ddsdde_fd=noise, tol=1e-5)
    assert plain.max_rel_err == 1.0 and not plain.passed

    scaled = TangentReport(ddsdde=zero, ddsdde_fd=noise, tol=1e-5, elastic_scale=2.0e5)
    assert scaled.max_rel_err == pytest.approx(2.8e-7 / 2.0e5)
    assert scaled.passed
    np.testing.assert_allclose(scaled.rel_err_matrix, [[2.8e-7 / 2.0e5]])


def test_an_ordinary_tangent_is_still_measured_against_itself():
    from returnmap.verification import TangentReport

    fd = np.array([[1000.0]])
    report = TangentReport(ddsdde=1.01 * fd, ddsdde_fd=fd, tol=1e-5, elastic_scale=2.0e5)
    assert report.max_rel_err == pytest.approx(0.01)      # not 10 / 2e5


def test_a_wrong_tangent_on_a_plateau_still_fails():
    """Measured against the elastic stiffness, an elastic tangent where zero
    is right is off by 1."""
    from returnmap.verification import TangentReport

    report = TangentReport(
        ddsdde=np.array([[2.0e5]]), ddsdde_fd=np.array([[2.8e-7]]), tol=1e-5,
        elastic_scale=2.0e5,
    )
    assert report.max_rel_err == pytest.approx(1.0)
    assert not report.passed

