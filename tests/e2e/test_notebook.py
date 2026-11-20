"""The marimo tuning notebook runs from top to bottom.

A marimo notebook is a Python file, and running it executes every cell with
the controls at their initial values — enough to catch a cell that no longer
matches the library.  Skipped when marimo is not installed
(``uv sync --extra notebook``).
"""

import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip("marimo")

NOTEBOOK = Path(__file__).parents[2] / "examples" / "notebook_tuning.py"


def test_tuning_notebook_runs(tmp_path):
    # From another directory: the record is found relative to the notebook.
    proc = subprocess.run(
        [sys.executable, str(NOTEBOOK)], capture_output=True, text=True, cwd=tmp_path,
    )
    assert proc.returncode == 0, proc.stderr
