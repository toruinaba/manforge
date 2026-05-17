"""P2: CrosscheckCaseResult exposes inner-NR trajectory fields."""
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
    CrosscheckStrainDriver,
    generate_strain_history,
)


@pytest.fixture
def model():
    return J2Isotropic3D(E=210_000.0, nu=0.3, sigma_y0=250.0, H=1_000.0)


class TestCrosscheckTrajectory:
    def test_stress_update_case_has_ab_trajectory_fields(self, model):
        py_int_a = PythonNumericalIntegrator(model)
        py_int_b = PythonAnalyticalIntegrator(model)
        cc = CrosscheckStrainDriver(py_int_a, py_int_b)
        history = generate_strain_history(model)
        load = FieldHistory(FieldType.STRAIN, "eps", history)
        result = cc.run(load)

        assert result.passed
        for cr in result.cases:
            assert isinstance(cr.a_n_iterations, int)
            assert isinstance(cr.b_n_iterations, int)
            assert isinstance(cr.a_residual_history, list)
            assert isinstance(cr.b_residual_history, list)
            assert cr.a_converged is True
            assert cr.b_converged is True

    def test_numerical_newton_plastic_step_has_nonzero_iterations(self, model):
        py_int_a = PythonNumericalIntegrator(model)
        py_int_b = PythonAnalyticalIntegrator(model)
        cc = CrosscheckStrainDriver(py_int_a, py_int_b)
        history = generate_strain_history(model)
        load = FieldHistory(FieldType.STRAIN, "eps", history)
        result = cc.run(load)

        plastic_cases = [cr for cr in result.cases
                         if cr.py_dlambda is not None and cr.py_dlambda > 0]
        assert len(plastic_cases) > 0
        assert any(cr.a_n_iterations >= 1 for cr in plastic_cases)

    def test_nonconverged_a_converged_flag_propagates(self, model):
        # max_iter=1 forces non-convergence on plastic steps; elastic steps still converge.
        py_int_a = PythonNumericalIntegrator(model, max_iter=1, raise_on_nonconverged=False)
        py_int_b = PythonAnalyticalIntegrator(model)
        cc = CrosscheckStrainDriver(py_int_a, py_int_b)

        # elastic step: converged=True
        elastic_strain = np.zeros((1, model.ntens))
        elastic_strain[0, 0] = 1e-7
        load_e = FieldHistory(FieldType.STRAIN, "eps", elastic_strain)
        for cr in cc.iter_run(load_e):
            assert cr.a_converged is True

        # plastic step: max_iter=1 is insufficient → converged=False propagates
        plastic_strain = np.zeros((1, model.ntens))
        plastic_strain[0, 0] = 5e-3  # well into plastic regime
        load_p = FieldHistory(FieldType.STRAIN, "eps", plastic_strain)
        for cr in cc.iter_run(load_p):
            assert cr.a_converged is False


