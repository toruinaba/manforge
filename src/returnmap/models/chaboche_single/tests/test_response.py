"""ChabocheKinematic: the limits its formulation page states."""

import numpy as np
import pytest

from returnmap.models import AFKinematic1D, AFKinematic3D, ChabocheKinematic1D, ChabocheKinematic3D
from returnmap.simulation import FieldHistory, FieldType, PythonIntegrator, StrainDriver

PARAMS = dict(E=210000.0, nu=0.3, sigma_y0=250.0, C=20000.0, gamma=100.0, Qinf=100.0, b=10.0)


def _monotonic(model, max_strain, n_steps):
    load = FieldHistory(FieldType.STRAIN, np.linspace(0.0, max_strain, n_steps + 1)[1:])
    return StrainDriver(PythonIntegrator(model)).run(load)


@pytest.mark.slow
def test_uniaxial_stress_saturates_at_the_sum_of_the_three_terms():
    """sigma_y0 + C/gamma + Qinf: the backstress and the isotropic term both saturate."""
    model = ChabocheKinematic1D(**PARAMS)
    result = _monotonic(model, 1.0, 2000)
    assert result.stress[-1, 0] == pytest.approx(250.0 + 20000.0 / 100.0 + 100.0, rel=1e-4)
    assert float(result.history("R")[-1]) == pytest.approx(100.0, rel=1e-4)
    # The stored 1D backstress is the deviatoric component: 2/3 of the shift.
    assert 1.5 * float(result.history("alpha")[-1, 0]) == pytest.approx(200.0, rel=1e-6)


def test_the_plastic_modulus_right_after_yield_is_C_plus_b_Qinf():
    model = ChabocheKinematic1D(**PARAMS)
    result = _monotonic(model, 0.0013, 1300)
    ep, stress = result.history("ep"), result.stress[:, 0]
    first = int(np.argmax(ep > 0.0))
    slope = (stress[first + 3] - stress[first + 1]) / (ep[first + 3] - ep[first + 1])
    assert slope == pytest.approx(20000.0 + 10.0 * 100.0, rel=1e-2)


@pytest.mark.parametrize("chaboche, af", [
    (ChabocheKinematic1D, AFKinematic1D), (ChabocheKinematic3D, AFKinematic3D),
])
def test_without_isotropic_hardening_it_is_armstrong_frederick(chaboche, af):
    """Qinf = 0 leaves R at zero, and what remains is the same arithmetic."""
    model = chaboche(**dict(PARAMS, Qinf=0.0))
    reference = af(E=210000.0, nu=0.3, sigma_y0=250.0, C_k=20000.0, gamma=100.0)
    load = FieldHistory.cyclic_strain([0.01, -0.01, 0.01], n_per_segment=20, ntens=model.ntens)
    ours = StrainDriver(PythonIntegrator(model)).run(load)
    theirs = StrainDriver(PythonIntegrator(reference)).run(load)
    np.testing.assert_array_equal(ours.stress, theirs.stress)
    np.testing.assert_array_equal(ours.history("alpha"), theirs.history("alpha"))
    assert np.all(ours.history("R") == 0.0)


def test_negative_Qinf_softens():
    """Cyclic softening is the same law with Qinf < 0."""
    model = ChabocheKinematic1D(**dict(PARAMS, Qinf=-50.0))
    result = _monotonic(model, 0.5, 1000)
    assert float(result.history("R")[-1]) == pytest.approx(-50.0, rel=1e-2)
    assert np.all(np.diff(result.history("R")) <= 0.0)
