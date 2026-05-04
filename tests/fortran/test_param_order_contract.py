"""Contract test: model.param_names order matches Fortran subroutine argument order.

For each model / compiled module pair, extracts the first ``len(param_names)``
dummy argument names from the f2py-generated ``__doc__`` and asserts they match
``[n.lower() for n in model.param_names]``.

This enforces the project convention (fortran/README.md and
fortran/j2_isotropic_3d.f90:15-18) that PROPS order = param_names order,
so that FortranIntegrator.from_model() can auto-generate param_fn safely.

Requires compiled modules — skipped automatically when binaries are absent.
"""

import re
import pytest

pytest.importorskip(
    "j2_isotropic_3d",
    reason="j2_isotropic_3d not compiled -- run: make fortran-build-umat",
)

pytestmark = pytest.mark.fortran

import autograd.numpy as anp
from manforge.core.stress_state import SOLID_3D
from manforge.models.j2_isotropic import J2Isotropic3D, J2IsotropicPS, J2Isotropic1D
from manforge.verification import FortranModule


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _extract_dummy_args(fortran_module, subroutine_name: str) -> list[str]:
    """Parse the first signature line of the f2py __doc__ for a subroutine.

    f2py generates docs like::

        "result = subroutine_name(arg1,arg2,...)\n\nWrapper ..."

    Returns the argument names in order, lower-cased.
    """
    fn = getattr(fortran_module, subroutine_name)
    doc = fn.__doc__ or ""
    # first line: "retvals = name(arg1,arg2,...)"  or just "name(arg1,...)"
    first_line = doc.split("\n")[0]
    m = re.search(r"\(([^)]+)\)", first_line)
    if not m:
        return []
    return [a.strip().lower() for a in m.group(1).split(",")]


# ---------------------------------------------------------------------------
# Test cases: (model_instance, module_name, subroutine_name)
# ---------------------------------------------------------------------------

def _j2_model():
    return J2Isotropic3D(E=210e3, nu=0.3, sigma_y0=250.0, H=1000.0)


def _j2_ps_model():
    return J2IsotropicPS(E=210e3, nu=0.3, sigma_y0=250.0, H=1000.0)


def _j2_1d_model():
    return J2Isotropic1D(E=210e3, nu=0.3, sigma_y0=250.0, H=1000.0)


class _MockKinematicModel:
    param_names = ["E", "H_kin", "H_iso"]
    state_names = ["alpha", "ep"]

    def __init__(self):
        self.E = 1.0
        self.H_kin = 0.1
        self.H_iso = 0.05
        self.stress_state = SOLID_3D

    def initial_state(self):
        return {"alpha": anp.zeros(self.stress_state.ntens), "ep": anp.array(0.0)}


_MOCK_MODULE = pytest.importorskip(
    "mock_kinematic",
    reason="mock_kinematic not compiled -- run: "
           "uv run manforge build fortran/mock_kinematic.f90 --name mock_kinematic",
)


@pytest.mark.parametrize("model_fn,module_name,subroutine", [
    (_j2_model,       "j2_isotropic_3d", "j2_isotropic_3d"),
    (_j2_ps_model,    "j2_isotropic_3d", "j2_isotropic_3d"),
    (_j2_1d_model,    "j2_isotropic_3d", "j2_isotropic_3d"),
    (_MockKinematicModel, "mock_kinematic", "mock_kinematic"),
])
def test_param_names_match_fortran_arg_order(model_fn, module_name, subroutine):
    """model.param_names order == first N Fortran dummy argument names."""
    model = model_fn()
    fortran = FortranModule(module_name)
    dummy_args = _extract_dummy_args(fortran.module, subroutine)

    n = len(model.param_names)
    assert len(dummy_args) >= n, (
        f"{subroutine}: expected at least {n} dummy args, got {len(dummy_args)}"
    )
    expected = [name.lower() for name in model.param_names]
    actual = dummy_args[:n]
    assert actual == expected, (
        f"{type(model).__name__}.param_names {model.param_names!r} "
        f"does not match first {n} Fortran args {dummy_args[:n]!r} "
        f"of subroutine '{subroutine}'"
    )
