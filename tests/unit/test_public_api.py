"""The public interface: seven packages, one home per name.

``manforge.core`` / ``.models`` / ``.simulation`` / ``.fitting`` /
``.verification`` / ``.service`` / ``.utils`` are the documented import paths.
These tests keep the rule honest: every name a package lists exists, no name
is listed twice, and the names a model author or a user reaches for are where
the documentation says they are.
"""

import importlib
import subprocess
import sys

import pytest

PACKAGES = ["core", "models", "simulation", "fitting", "verification", "service", "utils"]

# Listed by two packages on purpose: declared on the model (core), read off a
# ModelDescriptor by a deployment that imports nothing but the service layer.
SHARED = {"ParamMeta": {"core", "service"}}


def _exports(package):
    return importlib.import_module(f"manforge.{package}").__all__


@pytest.mark.parametrize("package", PACKAGES)
def test_every_listed_name_exists(package):
    module = importlib.import_module(f"manforge.{package}")
    missing = [name for name in module.__all__ if not hasattr(module, name)]
    assert missing == []
    assert len(set(module.__all__)) == len(module.__all__), "listed twice"


def test_each_name_has_one_home():
    homes = {}
    for package in PACKAGES:
        for name in _exports(package):
            homes.setdefault(name, set()).add(package)
    shared = {name: pkgs for name, pkgs in homes.items() if len(pkgs) > 1}
    assert shared == SHARED


@pytest.mark.parametrize("package, names", [
    ("core", ["MaterialModel", "ParamMeta", "Implicit", "Explicit", "NTENS", "SCALAR",
              "Increment", "SOLID_3D", "PLANE_STRESS", "UNIAXIAL_1D", "hand_block",
              "verified_against_fortran", "StressUpdateResult"]),
    ("simulation", ["PythonIntegrator", "StrainDriver", "StressDriver", "MixedDriver",
                    "FieldHistory", "FieldType", "DriverResult", "FortranIntegrator",
                    "FortranModule", "JacobianBlocks", "SubSteppingIntegrator"]),
    ("fitting", ["fit_params", "FitTarget", "Observation", "ParameterSpace", "FitResult"]),
    ("verification", ["JacobianChecker", "TangentChecker", "check_bindings",
                      "CrosscheckStrainDriver", "make_increment"]),
    ("service", ["ModelRegistry", "run_simulation"]),
    ("utils", ["smooth_sqrt", "smooth_macaulay", "smooth_heaviside", "positive_part"]),
])
def test_names_are_where_the_documentation_says(package, names):
    exported = set(_exports(package))
    assert [n for n in names if n not in exported] == []


def test_checks_do_not_re_export_what_they_check():
    """verification holds checks; the declarations live with the model and the solver."""
    verification = set(_exports("verification"))
    assert verification.isdisjoint(
        {"FortranModule", "JacobianBlocks", "verified_against_fortran", "FortranBinding"}
    )


@pytest.mark.parametrize("package", PACKAGES)
def test_each_package_imports_on_its_own(package):
    """No package relies on another having been imported first.

    ``manforge.core.dimension`` imports ``manforge.utils`` while
    ``manforge.utils.spectral`` is written against ``manforge.core``; a fresh
    interpreter per package catches a cycle that import order would hide here.
    """
    proc = subprocess.run(
        [sys.executable, "-c", f"import manforge.{package}"],
        capture_output=True, text=True,
    )
    assert proc.returncode == 0, proc.stderr
