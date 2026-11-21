"""What one loading says about a reference model agrees with what another says.

Two properties that follow from a model being isotropic and coming in several
dimensions, checked for every model in ``returnmap.models`` without knowing
anything else about it:

* **Tension and shear follow one curve.**  For a von Mises model, equivalent
  stress against accumulated plastic strain is the same curve in uniaxial
  tension (sigma11) and in simple shear (sqrt(3) tau12).  A hardening law that
  takes the flow direction in the wrong Voigt convention breaks this — and
  only this: the two conventions differ in the shear components alone, so
  nothing uniaxial ever sees it (see ``FlowVector``).
* **Uniaxial stress is the same in every dimension.**  The 1D model, the 3D
  model with the lateral stresses held at zero, and the plane-stress model
  with sigma22 = sigma12 = 0 are one loading.  A plane-stress class declared
  in a convention its law is not written in breaks this.

The damage models are left out of the first: their damage is driven by the
hydrostatic stress as well, so tension and shear are not meant to coincide.
"""

import warnings

import numpy as np
import pytest

import returnmap.models as models
from returnmap.simulation import (
    FieldHistory, FieldType, MixedDriver, PythonIntegrator, StrainDriver,
)

BY_NAME = {
    name: getattr(models, name) for name in models.__all__
    if name.endswith(("3D", "PS", "1D"))
}
VON_MISES_SOLIDS = [name for name in BY_NAME if name.endswith("3D") and "Damage" not in name]
REDUCED = [name for name in BY_NAME if not name.endswith("1D") and name[:-2] + "1D" in BY_NAME]

N_STEPS = 80
MAX_STRAIN = 0.03


def _plastic_strain(result, model):
    name = "eps_eq" if "eps_eq" in model.state_names else "ep"
    return np.asarray(result.history(name))


def _uniaxial_stress(model):
    """Axial strain driven, every other stress component zero."""
    axial = np.linspace(0.0, MAX_STRAIN, N_STEPS + 1)[1:]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        if model.ntens == 1:
            return StrainDriver(PythonIntegrator(model)).run(FieldHistory(FieldType.STRAIN, axial))
        return MixedDriver(PythonIntegrator(model), prescribed_strain_idx=[0]).run(
            FieldHistory(FieldType.STRAIN, axial.reshape(-1, 1)))


def _simple_shear(model):
    """Engineering shear gamma12 driven, up to the same equivalent strain."""
    data = np.zeros((N_STEPS, 6))
    data[:, 3] = np.linspace(0.0, np.sqrt(3.0) * MAX_STRAIN, N_STEPS + 1)[1:]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return StrainDriver(PythonIntegrator(model)).run(FieldHistory(FieldType.STRAIN, data))


def test_there_are_models_to_check():
    assert len(VON_MISES_SOLIDS) >= 6 and len(REDUCED) >= 12


@pytest.mark.slow
@pytest.mark.parametrize("name", VON_MISES_SOLIDS)
def test_tension_and_shear_follow_one_curve(name):
    model = BY_NAME[name].at_defaults()
    tension, shear = _uniaxial_stress(model), _simple_shear(model)
    p_t, s_t = _plastic_strain(tension, model), tension.stress[:, 0]
    p_s, s_s = _plastic_strain(shear, model), np.sqrt(3.0) * shear.stress[:, 3]

    p_max = min(p_t[-1], p_s[-1])
    assert p_max > 0.5 * MAX_STRAIN, "the loading did not go plastic"
    grid = np.linspace(0.05 * p_max, p_max, 40)
    in_tension, in_shear = np.interp(grid, p_t, s_t), np.interp(grid, p_s, s_s)
    difference = np.max(np.abs(in_shear - in_tension)) / np.max(np.abs(in_tension))
    # The two histories sample the curve at different plastic strains, so what
    # is left is the interpolation between steps (a few 1e-3 where the curve
    # has a corner).  The convention error is tens of percent.
    assert difference < 2e-2, (
        f"{name}: equivalent stress in shear differs from tension by {difference:.1e} "
        f"(tension {in_tension[-1]:.1f}, shear {in_shear[-1]:.1f} at p = {p_max:.4f})")


@pytest.mark.slow
@pytest.mark.parametrize("name", REDUCED)
def test_uniaxial_stress_agrees_with_the_1d_model(name):
    model, uniaxial = BY_NAME[name].at_defaults(), BY_NAME[name[:-2] + "1D"].at_defaults()
    reduced, reference = _uniaxial_stress(model), _uniaxial_stress(uniaxial)
    difference = (np.max(np.abs(reduced.stress[:, 0] - reference.stress[:, 0]))
                  / np.max(np.abs(reference.stress[:, 0])))
    # One loading, two solvers' tolerances apart.
    assert difference < 1e-5, (
        f"{name}: uniaxial stress differs from {type(uniaxial).__name__} by {difference:.1e}")
