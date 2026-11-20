"""The build command's own logic: what it hands to f2py.

Running an actual build needs gfortran and is covered in ``tests/e2e``.
"""

import sys
from pathlib import Path

from returnmap.cli import abaqus_stubs_path, f2py_command, main


def test_the_stubs_are_shipped_inside_the_package():
    path = abaqus_stubs_path()
    assert path.name == "abaqus_stubs.f90"
    assert path.is_file()
    import returnmap
    assert Path(returnmap.__file__).parent in path.parents
    source = path.read_text().lower()
    assert all(name in source for name in ("subroutine sinv", "subroutine sprinc", "subroutine rotsig"))


def test_plain_build_is_sources_then_module_name():
    cmd = f2py_command([Path("/src/a.f90"), Path("/src/b.f90")], "my_umat")
    assert cmd == [sys.executable, "-m", "numpy.f2py", "-c", "/src/a.f90", "/src/b.f90",
                   "-m", "my_umat"]


def test_stubs_are_compiled_in_first():
    cmd = f2py_command([Path("/src/umat.f90")], "my_umat", abaqus_stubs=True)
    assert cmd[4:6] == [str(abaqus_stubs_path()), "/src/umat.f90"]


def test_libraries_are_linked_after_the_module_name():
    cmd = f2py_command([Path("/src/umat.f90")], "my_umat", link=["lapack", "blas"])
    assert cmd[-4:] == ["-m", "my_umat", "-llapack", "-lblas"]


def test_stubs_command_prints_the_path(capsys, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["returnmap", "stubs"])
    main()
    assert capsys.readouterr().out.strip() == str(abaqus_stubs_path())


def _parsed_build_args(monkeypatch, *argv):
    """What the build command receives for a command line."""
    import returnmap.cli as cli

    seen = {}
    monkeypatch.setattr(cli, "cmd_build", lambda args: seen.update(vars(args)))
    monkeypatch.setattr(sys, "argv", ["returnmap", "build", *argv])
    cli.main()
    return seen


def test_build_takes_the_stubs_flag_and_repeated_link(monkeypatch):
    args = _parsed_build_args(
        monkeypatch, "umat.f90", "--name", "my_umat",
        "--abaqus-stubs", "--link", "lapack", "-l", "blas",
    )
    assert args["abaqus_stubs"] is True
    assert args["link"] == ["lapack", "blas"]


def test_build_defaults_to_no_stubs_and_no_libraries(monkeypatch):
    """A source that is not a UMAT (the smoke-test module) needs neither."""
    args = _parsed_build_args(monkeypatch, "test_basic.f90", "--name", "basic")
    assert args["abaqus_stubs"] is False
    assert args["link"] == []
