"""What every reference model owes its users, checked across the whole package.

A model in ``returnmap.models`` is used by people who did not write it: they
read its parameters off ``param_meta`` (what each one means, a value to start
from, the range it plausibly lies in) and expect those starting values to run.
"""

import numpy as np
import pytest

import returnmap.models as models
from returnmap.simulation import FieldHistory, PythonIntegrator, StrainDriver

ALL = [getattr(models, name) for name in models.__all__]
# The classes that fix their dimension are the ones constructed without one.
CONCRETE = [cls for cls in ALL if cls.__name__.endswith(("3D", "PS", "1D"))]


def at_defaults(cls):
    return cls(**{name: cls.param_meta[name].default for name in cls.param_names})


@pytest.mark.parametrize("cls", ALL, ids=lambda cls: cls.__name__)
def test_every_parameter_is_described(cls):
    assert list(cls.param_meta) == list(cls.param_names)
    for name, meta in cls.param_meta.items():
        assert meta.doc, name
        assert meta.default is not None, name
        assert meta.bounds is not None, name
        lower, upper = meta.bounds
        assert lower <= meta.default <= upper, name


@pytest.mark.parametrize("cls", CONCRETE, ids=lambda cls: cls.__name__)
def test_the_default_parameters_run(cls):
    """A tension-compression cycle at the defaults converges and goes plastic."""
    model = at_defaults(cls)
    load = FieldHistory.cyclic_strain([0.01, -0.01], n_per_segment=10, ntens=model.ntens)
    result = StrainDriver(PythonIntegrator(model)).run(load)
    assert result.converged
    assert np.all(np.isfinite(result.stress))
    assert any(step.is_plastic for step in result.step_results)
    assert all(float(step.dlambda) >= 0.0 for step in result.step_results)
