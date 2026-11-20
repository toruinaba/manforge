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


# ---------------------------------------------------------------------------
# build --family: every UMAT a package of models declares
# ---------------------------------------------------------------------------

def _run_build(monkeypatch, *argv):
    """Run ``returnmap build`` with the compiler replaced by a recorder."""
    import returnmap.cli as cli

    built = []
    for check in ("_check_gfortran", "_check_f2py", "_check_meson"):
        monkeypatch.setattr(cli, check, lambda: None)
    monkeypatch.setattr(
        cli, "_build_module",
        lambda sources, name, **options: built.append((name, sources, options)),
    )
    monkeypatch.setattr(sys, "argv", ["returnmap", "build", *argv])
    cli.main()
    return built


def test_family_builds_what_the_models_declare(monkeypatch, tmp_path):
    built = _run_build(monkeypatch, "--family", "returnmap.models.yu_kinematic",
                       "--output-dir", str(tmp_path))
    assert [name for name, _, _ in built] == ["yu_kinematic_3d", "yu_kinematic_ps"]
    for name, sources, options in built:
        assert [p.name for p in sources] == [f"{name}.f90"]
        assert all(p.is_file() for p in sources)
        # A UMAT always needs the stubs; the libraries come from the declaration.
        assert options["abaqus_stubs"] is True
        assert options["link"] == ["lapack", "blas"]
        assert options["output_dir"] == tmp_path


def test_family_can_be_a_package_of_families(monkeypatch, tmp_path):
    built = _run_build(monkeypatch, "--family", "returnmap.models", "-o", str(tmp_path))
    assert [name for name, _, _ in built] == [
        "j2_isotropic_3d", "yu_kinematic_3d", "yu_kinematic_ps"]


def test_family_without_a_umat_is_an_error(monkeypatch, capsys):
    import pytest

    with pytest.raises(SystemExit) as exit_info:
        _run_build(monkeypatch, "--family", "returnmap.models.af_kinematic")
    assert exit_info.value.code == 1
    assert "declares no UMAT" in capsys.readouterr().out


def test_family_and_source_files_are_alternatives(monkeypatch, capsys):
    import pytest

    for argv in (["--family", "returnmap.models", "x.f90"], ["x.f90"], []):
        with pytest.raises(SystemExit) as exit_info:
            _run_build(monkeypatch, *argv)
        assert exit_info.value.code == 2
    assert "--family <package>" in capsys.readouterr().out
