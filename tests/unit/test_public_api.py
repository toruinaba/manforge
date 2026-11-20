"""The public interface: seven packages, one home per name.

``returnmap.core`` / ``.models`` / ``.simulation`` / ``.fitting`` /
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


def _exports(package):
    return importlib.import_module(f"returnmap.{package}").__all__


@pytest.mark.parametrize("package", PACKAGES)
def test_every_listed_name_exists(package):
    module = importlib.import_module(f"returnmap.{package}")
    missing = [name for name in module.__all__ if not hasattr(module, name)]
    assert missing == []
    assert len(set(module.__all__)) == len(module.__all__), "listed twice"


def test_each_name_has_one_home():
    homes = {}
    for package in PACKAGES:
        for name in _exports(package):
            homes.setdefault(name, set()).add(package)
    shared = {name: pkgs for name, pkgs in homes.items() if len(pkgs) > 1}
    assert shared == {}


@pytest.mark.parametrize("package, names", [
    ("core", ["MaterialModel", "ParamMeta", "Implicit", "Explicit", "NTENS", "SCALAR",
              "Increment", "SOLID_3D", "PLANE_STRESS", "UNIAXIAL_1D", "hand_block",
              "verified_against_fortran", "StressUpdateResult"]),
    ("simulation", ["PythonIntegrator", "StrainDriver", "StressDriver", "MixedDriver",
                    "FieldHistory", "FieldType", "DriverResult", "FortranIntegrator",
                    "FortranModule", "JacobianBlocks", "hand_jacobian",
                    "SubSteppingIntegrator", "compare_models"]),
    ("fitting", ["fit_params", "FitTarget", "Observation", "ParameterSpace", "FitResult"]),
    ("verification", ["check_hand_blocks", "check_solution", "check_tangent",
                      "crosscheck", "compare_jacobians", "check_bindings",
                      "make_increment"]),
    ("service", ["ModelRegistry", "run_simulation"]),
    ("utils", ["smooth_sqrt", "smooth_macaulay", "smooth_heaviside", "positive_part"]),
])
def test_names_are_where_the_documentation_says(package, names):
    exported = set(_exports(package))
    assert [n for n in names if n not in exported] == []


def test_every_check_returns_a_report_with_a_verdict_and_a_table():
    """``assert report.passed, report.summary()`` reads the same for every check."""
    import returnmap.verification as verification

    reports = [n for n in _exports("verification") if n.endswith("Report")]
    assert sorted(reports) == [
        "BindingReport", "CrosscheckReport", "HandBlockReport",
        "JacobianComparisonReport", "SolutionReport", "TangentReport",
    ]
    for name in reports:
        cls = getattr(verification, name)
        assert isinstance(cls.passed, property), name
        assert callable(cls.summary), name
        assert cls.__str__ is not object.__str__, name
    # Results are what a computation returns; a check returns a report.
    assert [n for n in _exports("verification") if n.endswith("Result")] == []
    assert [n for n in _exports("verification") if n.endswith("Checker")] == []


def test_checks_do_not_re_export_what_they_check():
    """verification holds checks; the declarations live with the model and the solver."""
    verification = set(_exports("verification"))
    assert verification.isdisjoint(
        {"FortranModule", "JacobianBlocks", "verified_against_fortran", "FortranBinding"}
    )


def test_service_does_not_re_export_the_model_declarations():
    """ParamMeta is declared on the model, so it is imported from returnmap.core.

    A deployment reads it off ``ModelDescriptor.param_meta`` and needs the
    class itself only for type annotations.
    """
    assert "ParamMeta" not in _exports("service")


@pytest.mark.parametrize("package", PACKAGES)
def test_each_package_imports_on_its_own(package):
    """No package relies on another having been imported first.

    ``returnmap.core.dimension`` imports ``returnmap.utils`` while
    ``returnmap.utils.spectral`` is written against ``returnmap.core``; a fresh
    interpreter per package catches a cycle that import order would hide here.
    """
    proc = subprocess.run(
        [sys.executable, "-c", f"import returnmap.{package}"],
        capture_output=True, text=True,
    )
    assert proc.returncode == 0, proc.stderr
