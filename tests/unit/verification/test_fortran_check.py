"""check_bindings against a stand-in Fortran module (verification/fortran_check.py).

The compiled-module run is in
``tests/integration/verification/test_fortran_check.py``.
"""

import numpy as np
import pytest

from manforge.core import verified_against_fortran
from manforge.models import J2Isotropic3D
from manforge.verification import BindingReport, check_bindings

PARAMS = dict(E=210000.0, nu=0.3, sigma_y0=250.0, H=1000.0)


class _Bound(J2Isotropic3D):
    @verified_against_fortran("sub_twice")
    def twice(self, x):
        return 2.0 * np.asarray(x, dtype=float)

    @verified_against_fortran("sub_shift")
    def shift(self, x):
        return np.asarray(x, dtype=float) + 1.0


class _Module:
    """Answers ``call(subroutine, *args)`` from a table of Python functions."""

    def __init__(self, **subroutines):
        self._subs = subroutines

    def call(self, name, *args):
        return self._subs[name](*args)


X = np.array([1.0, -3.0, 0.0])


def test_agreeing_bindings_pass_and_the_rest_are_listed():
    model = _Bound(**PARAMS)
    fortran = _Module(sub_twice=lambda x: 2.0 * x)
    report = check_bindings(model, fortran, {"twice": ((X,), (X,))})

    assert isinstance(report, BindingReport)
    assert report.passed, report.summary()
    assert report.checks["twice"].rel_err == 0.0
    assert report.checks["twice"].subroutine == "sub_twice"
    # shift, and the binding J2Isotropic3D itself declares, were not in cases.
    assert "shift" in report.not_checked and "twice" not in report.not_checked
    assert "bindings: 1 checked — all agree" in report.summary()
    assert "not checked:" in report.summary()
    assert str(report) == report.summary()


def test_a_differing_binding_fails_by_name():
    model = _Bound(**PARAMS)
    fortran = _Module(sub_twice=lambda x: 2.0 * x, sub_shift=lambda x: x + 1.0 + 3e-3)
    report = check_bindings(model, fortran, {"twice": ((X,), (X,)), "shift": ((X,), (X,))})

    assert not report.passed
    assert [c.method for c in report.failed] == ["shift"]
    # Python is the reference: 3e-3 against max|x + 1| = 2.
    assert report.checks["shift"].rel_err == pytest.approx(3e-3 / 2.0)
    assert "FAIL shift ↔ sub_shift" in report.summary()


def test_round_off_on_a_zero_entry_does_not_fail():
    """Component by component, 1e-13 against an exact zero is a 100 % error."""
    model = _Bound(**PARAMS)
    big = np.array([2.0e5, 0.0])
    fortran = _Module(sub_twice=lambda x: 2.0 * x + np.array([0.0, 1e-13]))
    report = check_bindings(model, fortran, {"twice": ((big,), (big,))})
    assert report.passed, report.summary()


def test_tolerance_is_recorded_per_method():
    model = _Bound(**PARAMS)
    fortran = _Module(sub_shift=lambda x: x + 1.0 + 3e-3)
    report = check_bindings(model, fortran, {"shift": ((X,), (X,))}, tol=1e-2)
    assert report.passed and report.checks["shift"].tol == 1e-2


def test_nan_from_fortran_fails():
    model = _Bound(**PARAMS)
    fortran = _Module(sub_twice=lambda x: 2.0 * x * np.nan)
    report = check_bindings(model, fortran, {"twice": ((X,), (X,))})
    assert not report.passed and report.checks["twice"].rel_err == np.inf


def test_nothing_checked_is_an_error_not_a_pass():
    with pytest.raises(ValueError, match="no binding was checked"):
        check_bindings(_Bound(**PARAMS), _Module(), {})


def test_unbound_method_and_size_mismatch_are_reported():
    model = _Bound(**PARAMS)
    with pytest.raises(KeyError):
        check_bindings(model, _Module(), {"yield_function": ((), ())})
    fortran = _Module(sub_twice=lambda x: np.zeros(5))
    with pytest.raises(ValueError, match="shape mismatch"):
        check_bindings(model, fortran, {"twice": ((X,), (X,))})
