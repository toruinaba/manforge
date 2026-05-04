"""P3: base CaseResult exposes a/b_converged; ComparisonResult aggregates non-converged counts."""
import numpy as np
import pytest

import manforge  # noqa: F401
from manforge.models.j2_isotropic import J2Isotropic3D
from manforge.simulation import (
    PythonNumericalIntegrator,
    PythonAnalyticalIntegrator,
)
from manforge.simulation.types import FieldHistory, FieldType
from manforge.verification import (
    CaseResult,
    ComparisonResult,
    CrosscheckStrainDriver,
    generate_strain_history,
)


@pytest.fixture
def model():
    return J2Isotropic3D(E=210_000.0, nu=0.3, sigma_y0=250.0, H=1_000.0)


class TestBaseCaseResult:
    def test_default_converged_true(self):
        cr = CaseResult(index=0)
        assert cr.a_converged is True
        assert cr.b_converged is True

    def test_can_set_false(self):
        cr = CaseResult(index=0, a_converged=False, b_converged=False)
        assert cr.a_converged is False
        assert cr.b_converged is False


class TestBaseComparisonResult:
    def test_default_counters_zero(self):
        cr = ComparisonResult(passed=True, n_cases=0, n_passed=0)
        assert cr.n_a_nonconverged == 0
        assert cr.n_b_nonconverged == 0


class TestCounterAggregation:
    def test_stress_update_crosscheck_all_converged(self, model):
        py_a = PythonNumericalIntegrator(model)
        py_b = PythonAnalyticalIntegrator(model)
        cc = CrosscheckStrainDriver(py_a, py_b)
        history = generate_strain_history(model)
        load = FieldHistory(FieldType.STRAIN, "eps", history)
        result = cc.run(load)
        assert result.n_a_nonconverged == 0
        assert result.n_b_nonconverged == 0

    def test_base_run_counts_nonconverged_via_stub(self):
        """Comparator.run counts a/b_converged=False correctly."""
        from manforge.verification.comparator_base import Comparator

        class _Stub(Comparator):
            def iter_run(self):
                yield CaseResult(index=0, passed=True,
                                 a_converged=False, b_converged=True)
                yield CaseResult(index=1, passed=True,
                                 a_converged=True, b_converged=False)
                yield CaseResult(index=2, passed=True,
                                 a_converged=False, b_converged=False)

        result = _Stub().run()
        assert result.n_a_nonconverged == 2
        assert result.n_b_nonconverged == 2
        assert result.n_cases == 3
