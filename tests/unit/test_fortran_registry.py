"""Unit tests for fortran_registry (Fortran binary not required)."""

import pytest

from manforge.core.material import MaterialModel3D
from manforge.models.j2_isotropic import J2Isotropic3D
from manforge.verification.fortran_registry import (
    FortranBinding,
    collect_bindings,
    verified_against_fortran,
)


def test_decorator_attaches_binding():
    @verified_against_fortran("foo_sub", test="tests/foo.py::bar", notes="hi")
    def f():
        pass

    assert f.__fortran_binding__ == FortranBinding("foo_sub", "tests/foo.py::bar", "hi")


def test_decorator_does_not_change_behavior():
    @verified_against_fortran("foo_sub")
    def add(a, b):
        return a + b

    assert add(2, 3) == 5


def test_collect_bindings_via_mro():
    class Base:
        @verified_against_fortran("base_sub")
        def m(self): ...

    class Child(Base):
        pass

    assert collect_bindings(Child)["m"].subroutine == "base_sub"


def test_collect_bindings_override_wins():
    class Base:
        @verified_against_fortran("base_sub")
        def m(self): ...

    class Child(Base):
        @verified_against_fortran("child_sub")
        def m(self): ...

    assert collect_bindings(Child)["m"].subroutine == "child_sub"


def test_collect_bindings_excludes_undecorated():
    class MyClass:
        def plain(self): ...

        @verified_against_fortran("bound_sub")
        def bound(self): ...

    result = collect_bindings(MyClass)
    assert "bound" in result
    assert "plain" not in result


def test_collect_bindings_excludes_dunders():
    class MyClass:
        @verified_against_fortran("init_sub")
        def __init__(self): ...

    result = collect_bindings(MyClass)
    assert "__init__" not in result


def test_abstract_base_has_no_fortran_bindings():
    """Intermediate abstract classes must not get _fortran_bindings."""
    assert not hasattr(MaterialModel3D, "_fortran_bindings")


def test_j2_concrete_has_fortran_bindings():
    assert hasattr(J2Isotropic3D, "_fortran_bindings")
    assert "elastic_stiffness" in J2Isotropic3D._fortran_bindings


def test_j2_elastic_stiffness_binding_fields():
    binding = J2Isotropic3D._fortran_bindings["elastic_stiffness"]
    assert binding.subroutine == "j2_isotropic_3d_elastic_stiffness"
    assert binding.test is not None
    assert "test_j2_bindings" in binding.test
