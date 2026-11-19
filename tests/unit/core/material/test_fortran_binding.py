"""Unit tests for fortran_registry (Fortran binary not required)."""

import pytest

from manforge.core import MaterialModel, FortranBinding, verified_against_fortran
from manforge.models import J2Isotropic3D
from manforge.core.material.fortran_binding import collect_bindings


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
    assert not hasattr(MaterialModel, "_fortran_bindings")


def test_j2_concrete_has_fortran_bindings():
    assert hasattr(J2Isotropic3D, "_fortran_bindings")


def test_recorded_test_pointers_exist():
    """``test=`` is a pointer to the test that verifies the binding.  It is
    never executed, so nothing notices when the test is moved or renamed: the
    J2 binding pointed at a file that no longer existed."""
    import pathlib
    import re

    import manforge.models as models

    root = pathlib.Path(__file__).resolve().parents[4]
    broken = []
    n_checked = 0
    for name in models.__all__:
        for method, binding in getattr(models, name)._fortran_bindings.items():
            if not binding.test:
                continue
            n_checked += 1
            path, *parts = binding.test.split("::")
            file = root / path
            if not file.exists():
                broken.append(f"{name}.{method}: no file {path}")
                continue
            source = file.read_text()
            for part in parts:
                if not re.search(rf"^\s*(def|class) {re.escape(part)}\b", source, re.M):
                    broken.append(f"{name}.{method}: no {part} in {path}")
    assert n_checked > 0
    assert broken == []
