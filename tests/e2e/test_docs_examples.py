"""A guide page built from example code shows what that code does.

A scenario page of the guide takes its code from ``examples/`` with
``literalinclude`` — the ``# %%`` cells of a script, or a class of the example
model family — and shows what a script prints.  The scripts are run here, so
that a change to the API that breaks the page's code fails a test, and the
output printed on the page is compared with what the scripts print now:

* a ``text`` block is output that is the same on every run — each of its
  lines is a line the scripts print, character for character;
* a ``none`` block is output with numbers that depend on the run (a time per
  step, the result of an optimisation, an error at round-off level) — its
  lines are compared with the numbers masked, so the rows, the columns and
  the wording are still checked.

The page holds no Python block of its own: every line of code on it comes from
a file under ``examples/``, so there is none that is shown without being run.
"""

import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).parents[2]
FAMILY = REPO / "examples" / "model_family"

# page of the guide -> the scripts whose output it shows, in the order shown
PAGES = {
    "docs/guide/simulation.md": ["examples/example_response.py"],
    "docs/guide/tutorial.md": ["examples/example_identification.py"],
    "docs/guide/comparing.md": ["examples/example_compare_models.py"],
    "docs/guide/model-python.md": ["examples/model_family/report.py"],
    "docs/guide/model-hand-written.md": ["examples/model_family/report.py"],
    "docs/guide/model-umat.md": ["examples/model_family/report.py",
                                 "examples/model_family/umat_crosscheck.py"],
}
# Pages whose scripts take more than a few seconds (the fits of the tutorial).
SLOW = {"docs/guide/tutorial.md"}
# Pages that show the output of a compiled UMAT: the example family is built first.
FORTRAN = {"docs/guide/model-umat.md"}

_NUMBER = re.compile(r"[-+]?\d+(?:\.\d+)?(?:e[-+]?\d+)?")
_INCLUDE = re.compile(r"^```\{literalinclude\} (\S+)\n(.*?)^```$", flags=re.M | re.S)


def _blocks(page: Path, language: str) -> list[list[str]]:
    """The fenced blocks of one language in a page, each as its lines."""
    text = page.read_text(encoding="utf-8")
    return [body.splitlines()
            for body in re.findall(rf"^```{language}\n(.*?)^```$", text, flags=re.M | re.S)]


def _masked(line: str) -> str:
    """A line with its numbers and its column padding taken out."""
    return " ".join(_NUMBER.sub("#", line).split())


def _missing(expected: list[str], printed: list[str]) -> str | None:
    """The first of ``expected`` that is not in ``printed``, reading both in order."""
    remaining = iter(printed)
    for line in expected:
        if not any(line == got for got in remaining):
            return line
    return None


def _marks(page: str) -> list:
    marks = []
    if page in SLOW:
        marks.append(pytest.mark.slow)
    if page in FORTRAN:
        marks.append(pytest.mark.fortran)
    return marks


def _run(args: list[str], cwd: Path, pythonpath: list[Path]) -> subprocess.CompletedProcess:
    env = dict(os.environ, PYTHONPATH=os.pathsep.join(str(p) for p in pythonpath),
               PYTHONDONTWRITEBYTECODE="1", PYTHONIOENCODING="utf-8", MPLBACKEND="Agg")
    return subprocess.run([sys.executable, *args], capture_output=True, text=True,
                          encoding="utf-8", cwd=cwd, env=env)


@pytest.fixture(
    scope="module",
    params=[pytest.param(page, marks=_marks(page)) for page in sorted(PAGES)],
    ids=lambda page: Path(page).stem,
)
def page_and_output(request, tmp_path_factory):
    """A page, and what its scripts print when run outside the checkout.

    A script of ``examples/`` is run in an empty directory.  A script of the
    example model family is run in a copy of the family, as the project of
    its own that it is; for a page that shows a compiled UMAT, the family's
    UMAT is built there first (skipped without gfortran).
    """
    page = REPO / request.param
    scripts = [REPO / script for script in PAGES[request.param]]
    workdir = tmp_path_factory.mktemp("example")
    pythonpath = [REPO / "src"]

    if any(FAMILY in script.parents for script in scripts):
        root = workdir / "model_family"
        shutil.copytree(FAMILY, root, ignore=shutil.ignore_patterns(
            "__pycache__", ".venv", "*.lock", "*.so", "*.mod", "*.o"))
        scripts = [root / script.relative_to(FAMILY) if FAMILY in script.parents else script
                   for script in scripts]
        workdir, pythonpath = root, [REPO / "src", root / "src"]
        if request.param in FORTRAN:
            if shutil.which("gfortran") is None:
                pytest.skip("gfortran not found")
            build = _run(["-m", "returnmap", "build", "--family", "mymodels"], root, pythonpath)
            assert build.returncode == 0, build.stdout + build.stderr

    printed = []
    for script in scripts:
        proc = _run([str(script)], workdir, pythonpath)
        assert proc.returncode == 0, f"{script.name} failed:\n{proc.stdout}\n{proc.stderr}"
        printed += proc.stdout.splitlines()
    return page, printed


def test_the_output_on_the_page_is_what_the_scripts_print(page_and_output):
    """Each line of a ``text`` block is a line of the scripts' output, in order."""
    page, printed = page_and_output
    expected = [line for block in _blocks(page, "text") for line in block]
    line = _missing(expected, printed)
    assert line is None, (
        f"{page.name} shows this line as output, but its scripts do not print it "
        f"(or print it earlier):\n{line}")


def test_run_dependent_output_on_the_page_has_the_shape_the_scripts_print(page_and_output):
    """Each line of a ``none`` block is a line of the output once numbers are masked."""
    page, printed = page_and_output
    expected = [_masked(line) for block in _blocks(page, "none") for line in block]
    line = _missing(expected, [_masked(got) for got in printed])
    assert line is None, (
        f"{page.name} shows a line of this shape as output, but its scripts print "
        f"none like it (or print it earlier):\n{line}")


@pytest.mark.parametrize("page", sorted(PAGES), ids=lambda page: Path(page).stem)
def test_every_line_of_code_on_the_page_comes_from_examples(page):
    path = REPO / page
    text = path.read_text(encoding="utf-8")
    assert not _blocks(path, "python"), (
        f"{path.name} has a Python block of its own; take the code from a file under "
        "examples/ with literalinclude, so that it is run")
    assert _blocks(path, "text") or _blocks(path, "none"), f"{path.name} shows no output"

    includes = _INCLUDE.findall(text)
    assert includes, f"{path.name} includes no code"
    for target, options in includes:
        source = (path.parent / target).resolve()
        assert REPO / "examples" in source.parents, f"{path.name} includes {target}, outside examples/"
        assert source.is_file(), f"{path.name} includes {target}, which does not exist"
        lines = source.read_text(encoding="utf-8")
        for marker in re.findall(r'^:(?:start|end)-(?:after|at|before): "(.*)"$', options, flags=re.M):
            assert marker in lines, f"{path.name} includes {target} from {marker!r}, which is not in it"


def test_every_script_in_examples_belongs_to_a_page():
    """examples/ holds the scripts the guide is built from, and nothing else.

    The other things there are named by a page too: the notebook and the data
    of the tutorial, and the model family of "adding a model".
    """
    guide = "".join(path.read_text(encoding="utf-8") for path in (REPO / "docs").rglob("*.md")
                    if "_build" not in path.parts)
    scripts = sorted(path.name for path in (REPO / "examples").glob("*.py"))
    orphans = [name for name in scripts if f"examples/{name}" not in guide]
    assert not orphans, f"no page of docs/ names these scripts in examples/: {orphans}"
