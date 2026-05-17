"""Integration tests for verification/comparator_base.py (Comparator base class).

TODO: add Comparator subclass contract tests beyond those in test_base.py.
"""

import pytest
from manforge.verification.comparator_base import Comparator
from manforge.verification import CaseResult, ComparisonResult


def test_comparator_base_importable():
    assert Comparator is not None


def test_comparator_run_aggregates(model):
    """Minimal Comparator.run produces a ComparisonResult with correct counts."""
    class _Stub(Comparator):
        def iter_run(self):
            yield CaseResult(index=0, passed=True)
            yield CaseResult(index=1, passed=False)

    result = _Stub().run()
    assert isinstance(result, ComparisonResult)
    assert result.n_cases == 2
    assert result.n_passed == 1
    assert result.passed is False
