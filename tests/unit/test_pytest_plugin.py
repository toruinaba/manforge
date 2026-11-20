"""The pytest plugin the package ships, run in a project of its own.

Each test starts pytest in a fresh directory, as a repository of models built
on returnmap would: no ``conftest.py``, no marker declarations.
"""

import subprocess
import sys
import textwrap
from importlib.metadata import entry_points

import pytest

# The plugin loads by itself only where returnmap is installed.  A checkout
# that is merely on PYTHONPATH (the Docker image) names it in its conftest.
pytestmark = pytest.mark.skipif(
    not any(ep.value == "returnmap.pytest_plugin" for ep in entry_points(group="pytest11")),
    reason="returnmap is not installed, so its pytest11 entry point is not registered",
)


def run_pytest(tmp_path, *args):
    return subprocess.run(
        [sys.executable, "-m", "pytest", "-p", "no:cacheprovider", *args],
        capture_output=True, text=True, cwd=tmp_path,
    )


def test_markers_are_registered_without_a_conftest(tmp_path):
    (tmp_path / "test_marked.py").write_text(textwrap.dedent("""
        import pytest

        @pytest.mark.slow
        def test_slow(): pass

        @pytest.mark.fortran
        def test_fortran(): pass

        def test_plain(): pass
    """))
    # --strict-markers turns an unregistered marker into an error.
    proc = run_pytest(tmp_path, "--strict-markers", "-m", "not slow and not fortran", "-q")
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "1 passed, 2 deselected" in proc.stdout


def test_compiled_modules_under_fortran_are_importable(tmp_path):
    """``returnmap build`` puts modules in <project>/fortran; a stand-in module
    there is found by name, and a missing one skips."""
    (tmp_path / "fortran").mkdir()
    (tmp_path / "fortran" / "my_umat.py").write_text("ANSWER = 42\n")
    (tmp_path / "test_umat.py").write_text(textwrap.dedent("""
        import pytest

        def test_built():
            assert pytest.importorskip("my_umat").ANSWER == 42

        def test_not_built():
            pytest.importorskip("another_umat")
    """))
    proc = run_pytest(tmp_path, "-q")
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "1 passed, 1 skipped" in proc.stdout


def test_loading_it_twice_is_the_same_plugin(tmp_path):
    """Installed (entry point), named in a conftest, and passed with -p."""
    (tmp_path / "conftest.py").write_text('pytest_plugins = ["returnmap.pytest_plugin"]\n')
    (tmp_path / "test_one.py").write_text("def test_one(): pass\n")
    proc = run_pytest(tmp_path, "-p", "returnmap.pytest_plugin", "--strict-markers", "-q")
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert proc.stdout.count("@pytest.mark.slow") == 0
    markers = run_pytest(tmp_path, "--markers")
    assert markers.stdout.count("@pytest.mark.slow:") == 1
