"""A guide page built from an example script shows what the script does.

A scenario page of the guide takes its code from one script in ``examples/``
(``literalinclude`` of the script's ``# %%`` cells), and shows what the script
prints in ``text`` blocks.  The script is run here, so that a change to the
API that breaks the page's code fails a test, and the output printed on the
page is compared with what the script prints now.

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
}


def _blocks(page: Path, language: str) -> list[list[str]]:
    """The fenced blocks of one language in a page, each as its lines."""
    text = page.read_text(encoding="utf-8")
    return [body.splitlines()
            for body in re.findall(rf"^```{language}\n(.*?)^```$", text, flags=re.M | re.S)]


@pytest.fixture(scope="module", params=sorted(PAGES), ids=lambda page: Path(page).stem)
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
    printed = iter(proc.stdout.splitlines())
    for line in expected:
        assert any(line == got for got in printed), (
            f"{page.name} shows this line as output of {script.name}, "
            f"but the script does not print it (or prints it earlier):\n{line}")


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
