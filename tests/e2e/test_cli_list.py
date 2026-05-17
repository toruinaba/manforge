"""E2E tests for `manforge list` CLI command."""

import shutil
import subprocess
import pytest


def test_list_exits_zero():
    """manforge list exits 0 (even with no compiled modules)."""
    if shutil.which("uv") is None:
        pytest.skip("uv not found in PATH")
    result = subprocess.run(
        ["uv", "run", "manforge", "list"],
        capture_output=True, text=True,
    )
    assert result.returncode == 0


def test_list_help():
    """manforge list --help exits 0."""
    if shutil.which("uv") is None:
        pytest.skip("uv not found in PATH")
    result = subprocess.run(
        ["uv", "run", "manforge", "list", "--help"],
        capture_output=True, text=True,
    )
    assert result.returncode == 0
