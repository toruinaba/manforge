"""ChabocheKinematicMulti: the limits its formulation page states."""

import numpy as np
import pytest

from returnmap.models import (
    ChabocheKinematic1D, ChabocheKinematic3D, ChabocheKinematicMulti1D, ChabocheKinematicMulti3D,
)
from returnmap.simulation import FieldHistory, FieldType, PythonIntegrator, StrainDriver

PARAMS = dict(
    E=210000.0, nu=0.3, sigma_y0=250.0,
    C1=80000.0, gamma1=800.0, C2=10000.0, gamma2=100.0, C3=1000.0, gamma3=10.0,
    Qinf1=40.0, b1=40.0, Qinf2=30.0, b2=5.0, Qinf3=0.0, b3=1.0,
)
OFF = dict(C2=0.0, gamma2=0.0, C3=0.0, gamma3=0.0, Qinf2=0.0, b2=0.0, Qinf3=0.0, b3=0.0)


@pytest.mark.slow
def test_uniaxial_stress_saturates_at_the_sum_of_all_terms():
    """sigma_y0 + sum C_i/gamma_i + sum Qinf_i."""
    model = ChabocheKinematicMulti1D(**PARAMS)
    load = FieldHistory(FieldType.STRAIN, np.linspace(0.0, 2.0, 4001)[1:])
    result = StrainDriver(PythonIntegrator(model)).run(load)
    expected = 250.0 + (100.0 + 100.0 + 100.0) + (40.0 + 30.0 + 0.0)
    assert result.stress[-1, 0] == pytest.approx(expected, rel=1e-4)
    for i in (1, 2, 3):
        assert 1.5 * float(result.history(f"alpha{i}")[-1, 0]) == pytest.approx(100.0, rel=1e-4)


def test_the_backstresses_saturate_at_their_own_rates():
    """By 1 % plastic strain the fast one is saturated and the slow one has barely started."""
    model = ChabocheKinematicMulti1D(**PARAMS)
    load = FieldHistory(FieldType.STRAIN, np.linspace(0.0, 0.012, 241)[1:])
    result = StrainDriver(PythonIntegrator(model)).run(load)
    fast, medium, slow = (1.5 * float(result.history(f"alpha{i}")[-1, 0]) for i in (1, 2, 3))
    assert fast > 99.0
    assert 50.0 < medium < 75.0
    assert slow < 12.0


@pytest.mark.parametrize("multi, single", [
    (ChabocheKinematicMulti1D, ChabocheKinematic1D), (ChabocheKinematicMulti3D, ChabocheKinematic3D),
])
def test_with_one_term_of_each_kind_it_is_the_single_backstress_model(multi, single):
    model = multi(**{**PARAMS, **OFF})
    reference = single(E=210000.0, nu=0.3, sigma_y0=250.0,
                       C=80000.0, gamma=800.0, Qinf=40.0, b=40.0)
    load = FieldHistory.cyclic_strain([0.01, -0.01, 0.01], n_per_segment=20, ntens=model.ntens)
    ours = StrainDriver(PythonIntegrator(model)).run(load)
    theirs = StrainDriver(PythonIntegrator(reference)).run(load)
    np.testing.assert_array_equal(ours.stress, theirs.stress)
    np.testing.assert_array_equal(ours.history("alpha"), theirs.history("alpha"))
    np.testing.assert_array_equal(ours.history("R"), theirs.history("R"))
