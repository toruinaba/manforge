"""The example model family (``examples/model_family``) works as a repository of its own.

It is the layout ``docs/guide/adding-a-model.md`` walks through — a package of
models built on returnmap, with its verification declaration, its tests and
its UMAT next to the model.  Run here from a copy outside the checkout, so
that what passes is the example and not something it picks up from the
repository around it.
"""

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).parents[2]
EXAMPLE = REPO / "examples" / "model_family"


@pytest.fixture
def project(tmp_path):
    """A copy of the example, and the environment a subprocess runs it in."""
    root = tmp_path / "model_family"
    shutil.copytree(
        EXAMPLE, root,
        ignore=shutil.ignore_patterns("__pycache__", "fortran", ".venv", "*.lock"),
    )
    # The UMAT source lives in a directory called fortran/ too; only the
    # build directory at the top was left out.
    shutil.copytree(
        EXAMPLE / "src" / "mymodels" / "voce" / "fortran",
        root / "src" / "mymodels" / "voce" / "fortran",
        ignore=shutil.ignore_patterns("*.so", "*.mod", "*.o"),
    )
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join([str(REPO / "src"), str(root / "src")])
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    return root, env


def _run(args, root, env):
    return subprocess.run(
        [sys.executable, *args], capture_output=True, text=True, cwd=root, env=env,
    )


def test_the_family_passes_its_checks(project):
    """Without a compiler: the Python checks run, the UMAT test is skipped."""
    root, env = project
    proc = _run(["-m", "pytest", "-q", "-p", "no:cacheprovider"], root, env)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "2 passed, 1 skipped" in proc.stdout, proc.stdout


def test_what_the_report_says_follows_from_what_each_model_declares(project):
    root, env = project
    code = (
        "from returnmap.verification import check_model\n"
        "from mymodels.voce.verification import MODELS\n"
        "for model in MODELS:\n"
        "    report = check_model(model)\n"
        "    print(type(model).__name__, report.passed,\n"
        "          ' '.join(f'{c.name}={c.status}' for c in report.checks))\n"
    )
    proc = _run(["-c", code], root, env)
    assert proc.returncode == 0, proc.stderr
    hand, plain = proc.stdout.strip().splitlines()
    # J2Voce3D: hand-written return mapping and tangent, UMAT declared.
    assert hand.startswith("J2Voce3D True")
    for row in ("converges", "tangent", "hand-written solution",
                "hand-written vs autodiff", "hand-written tangent"):
        assert f"{row}=passed" in hand, hand
    assert "hand blocks=not declared" in hand
    assert "UMAT=not built" in hand
    # J2Voce1D: the physics only.
    assert plain.startswith("J2Voce1D True")
    assert "converges=passed" in plain and " tangent=passed" in plain
    assert "hand-written solution=not declared" in plain
    assert "UMAT=not declared" in plain


@pytest.mark.fortran
def test_the_umat_builds_and_agrees(project):
    """``returnmap build --family`` reads the declaration; the UMAT row passes."""
    if shutil.which("gfortran") is None:
        pytest.skip("gfortran not found")
    root, env = project
    build = _run(["-m", "returnmap", "build", "--family", "mymodels"], root, env)
    assert build.returncode == 0, build.stdout + build.stderr
    assert list((root / "fortran").glob("voce_3d*.so")), build.stdout

    proc = _run(["-m", "pytest", "-q", "-p", "no:cacheprovider"], root, env)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "3 passed" in proc.stdout, proc.stdout
