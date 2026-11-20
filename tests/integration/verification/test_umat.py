"""Umat — the declaration of a model's Fortran port, and what reads it.

No compiler is needed: the "compiled module" is a Python stand-in that
answers the way an f2py module would, in a throw-away package laid out like a
model family.
"""

import sys
import textwrap

import pytest

from returnmap.verification import Umat, check_model
from returnmap.verification.umat import declared_umats, umat_of

FAMILY = '''
from returnmap.models import J2Isotropic3D


class PortedJ2(J2Isotropic3D):
    """A model with a UMAT."""


class UnportedJ2(J2Isotropic3D):
    """A model of the same family without one."""
'''

VERIFICATION = '''
from returnmap.verification import Umat

from {package}.model import PortedJ2

UMATS = [
    Umat(PortedJ2, module="{module}", sources=["fortran/ported_j2.f90"],
         libraries=["lapack"], subroutine="j2"),
]
'''

# What an f2py module looks like from Python: a callable per subroutine.
# The J2 step is computed in Python; ``SCALE`` lets a test make it wrong.
STAND_IN = '''
import numpy as np
from returnmap.models import J2Isotropic3D
from returnmap.simulation import PythonIntegrator

SCALE = {scale}


def j2(E, nu, sigma_y0, H, stress_n, ep, dstran):
    model = J2Isotropic3D(E=E, nu=nu, sigma_y0=sigma_y0, H=H)
    state_n = {{"stress": np.asarray(stress_n), "ep": np.array(ep)}}
    r = PythonIntegrator(model).stress_update(np.asarray(dstran), np.asarray(stress_n), state_n)
    return SCALE * np.asarray(r.stress), float(r.state["ep"]), np.asarray(r.ddsdde)
'''


@pytest.fixture
def family(tmp_path, monkeypatch, request):
    """A package ``<name>`` with model.py and verification.py, importable."""
    package = f"fam_{request.node.name.split('[')[0][-24:].strip('_')}"
    module = f"{package}_umat"
    root = tmp_path / package
    (root / "sub").mkdir(parents=True)
    (root / "__init__.py").write_text("")
    (root / "model.py").write_text(textwrap.dedent(FAMILY))
    (root / "verification.py").write_text(
        textwrap.dedent(VERIFICATION).format(package=package, module=module))
    monkeypatch.syspath_prepend(str(tmp_path))
    yield package, module, tmp_path
    for name in [n for n in sys.modules if n == package or n.startswith(f"{package}.")
                 or n == module]:
        del sys.modules[name]


def build(tmp_path, module, scale=1.0):
    (tmp_path / f"{module}.py").write_text(textwrap.dedent(STAND_IN).format(scale=scale))


def models_of(package):
    import importlib
    return importlib.import_module(f"{package}.model")


# ---------------------------------------------------------------------------
# The declaration
# ---------------------------------------------------------------------------

class TestDeclaration:
    def test_sources_are_relative_to_the_model_module(self, family):
        package, module, tmp_path = family
        umat = umat_of(models_of(package).PortedJ2.at_defaults())
        assert isinstance(umat, Umat)
        assert umat.module == module and list(umat.libraries) == ["lapack"]
        assert umat.source_paths() == [tmp_path / package / "fortran" / "ported_j2.f90"]

    def test_only_the_class_it_names_has_a_umat(self, family):
        package, _, _ = family
        assert umat_of(models_of(package).UnportedJ2.at_defaults()) is None

    def test_a_model_outside_a_package_has_none(self):
        from returnmap.models import J2Isotropic3D

        class Local(J2Isotropic3D):
            pass

        assert umat_of(Local.at_defaults()) is None

    def test_declared_umats_of_a_family_and_of_a_package_of_families(self, family):
        package, module, _ = family
        assert [u.module for u in declared_umats(package)] == [module]
        modules = [u.module for u in declared_umats("returnmap.models")]
        assert modules == ["j2_isotropic_3d", "yu_kinematic_3d", "yu_kinematic_ps"]
        assert declared_umats("returnmap.models.af_kinematic") == []

    def test_the_reference_umats_point_at_their_sources(self):
        for umat in declared_umats("returnmap.models"):
            assert all(path.is_file() for path in umat.source_paths()), umat.module

    def test_built_means_importable(self, family):
        package, module, tmp_path = family
        umat = umat_of(models_of(package).PortedJ2.at_defaults())
        assert not umat.is_built()
        build(tmp_path, module)
        assert umat.is_built()


# ---------------------------------------------------------------------------
# check_model
# ---------------------------------------------------------------------------

class TestCheckModel:
    def test_not_declared(self, family):
        package, _, _ = family
        report = check_model(models_of(package).UnportedJ2.at_defaults())
        assert report["UMAT"].status == "not declared"
        assert report.passed

    def test_declared_but_not_built_does_not_count_against_the_model(self, family):
        package, module, _ = family
        report = check_model(models_of(package).PortedJ2.at_defaults())
        check = report["UMAT"]
        assert check.status == "not built" and check.passed and report.passed
        assert check.detail == (
            f"module {module} is not built — returnmap build --family {package}")
        assert "—    UMAT" in report.summary()

    def test_a_built_umat_is_compared_over_every_history(self, family):
        package, module, tmp_path = family
        build(tmp_path, module)
        report = check_model(models_of(package).PortedJ2.at_defaults())
        check = report["UMAT"]
        assert check.status == "passed", report.summary()
        assert set(check.reports) == {"uniaxial", "multiaxial"}
        assert check.detail.endswith(f"over 2 histories ({module})")

    def test_a_umat_that_disagrees_fails_only_its_own_row(self, family):
        package, module, tmp_path = family
        build(tmp_path, module, scale=1.001)          # stress 0.1 % too large
        report = check_model(models_of(package).PortedJ2.at_defaults())
        assert [c.name for c in report.checks if c.status == "failed"] == ["UMAT"]
        assert not report.passed
        assert "stress" in report["UMAT"].detail
