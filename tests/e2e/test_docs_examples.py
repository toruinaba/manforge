"""A guide page built from an example script shows what the script does.

A scenario page of the guide takes its code from one script in ``examples/``
(``literalinclude`` of the script's ``# %%`` cells), and shows what the script
prints.  The script is run here, so that a change to the API that breaks the
page's code fails a test, and the output printed on the page is compared with
what the script prints now:

* a ``text`` block is output that is the same on every run — each of its
  lines is a line the script prints, character for character;
* a ``none`` block is output with numbers that depend on the run (a time per
  step, the result of an optimisation) — its lines are compared with the
  numbers masked, so the rows, the columns and the wording are still checked.

The page holds no Python block of its own: every line of code on it comes from
the script, so there is none that is shown without being run.
"""

import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).parents[2]

# page of the guide -> the script it is built from
PAGES = {
    "docs/guide/simulation.md": "examples/example_response.py",
    "docs/guide/tutorial.md": "examples/example_identification.py",
    "docs/guide/comparing.md": "examples/example_compare_models.py",
}
# Scripts that take more than a few seconds (the fits of the tutorial).
SLOW = {"docs/guide/tutorial.md"}

_NUMBER = re.compile(r"[-+]?\d+(?:\.\d+)?(?:e[-+]?\d+)?")


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


@pytest.fixture(
    scope="module",
    params=[pytest.param(page, marks=pytest.mark.slow) if page in SLOW else page
            for page in sorted(PAGES)],
    ids=lambda page: Path(page).stem,
)
def page_and_run(request, tmp_path_factory):
    """A page, and its script run in an empty directory outside the checkout."""
    page = REPO / request.param
    script = REPO / PAGES[request.param]
    env = dict(os.environ, PYTHONPATH=str(REPO / "src"), PYTHONDONTWRITEBYTECODE="1",
               PYTHONIOENCODING="utf-8", MPLBACKEND="Agg")
    proc = subprocess.run(
        [sys.executable, str(script)], capture_output=True, text=True, encoding="utf-8",
        cwd=tmp_path_factory.mktemp("example"), env=env,
    )
    return page, script, proc


def test_the_script_runs(page_and_run):
    _, script, proc = page_and_run
    assert proc.returncode == 0, f"{script.name} failed:\n{proc.stdout}\n{proc.stderr}"


def test_the_output_on_the_page_is_what_the_script_prints(page_and_run):
    """Each line of a ``text`` block is a line of the script's output, in order."""
    page, script, proc = page_and_run
    expected = [line for block in _blocks(page, "text") for line in block]
    assert expected, f"{page.name} shows no output of {script.name}"
    line = _missing(expected, proc.stdout.splitlines())
    assert line is None, (
        f"{page.name} shows this line as output of {script.name}, "
        f"but the script does not print it (or prints it earlier):\n{line}")


def test_run_dependent_output_on_the_page_has_the_shape_the_script_prints(page_and_run):
    """Each line of a ``none`` block is a line of the output once numbers are masked."""
    page, script, proc = page_and_run
    expected = [_masked(line) for block in _blocks(page, "none") for line in block]
    line = _missing(expected, [_masked(got) for got in proc.stdout.splitlines()])
    assert line is None, (
        f"{page.name} shows a line of this shape as output of {script.name}, "
        f"but the script prints none like it (or prints it earlier):\n{line}")


def test_every_line_of_code_on_the_page_comes_from_the_script(page_and_run):
    page, script, _ = page_and_run
    assert not _blocks(page, "python"), (
        f"{page.name} has a Python block of its own; take it from {script.name} "
        "with literalinclude, so that it is run")
    cells = re.findall(r'^:start-after: "(# %% .*)"$', page.read_text(encoding="utf-8"), flags=re.M)
    assert cells, f"{page.name} includes no cell of {script.name}"
    source = script.read_text(encoding="utf-8").splitlines()
    for cell in cells:
        assert cell in source, f"{page.name} includes the cell {cell!r}, which {script.name} does not have"


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
