"""E2E tests for `returnmap list` CLI command."""

import shutil
import subprocess
import pytest


def test_list_exits_zero():
    """returnmap list exits 0 (even with no compiled modules)."""
    if shutil.which("uv") is None:
        pytest.skip("uv not found in PATH")
    result = subprocess.run(
        ["uv", "run", "returnmap", "list"],
        capture_output=True, text=True,
    )
    assert result.returncode == 0


def test_list_help():
    """returnmap list --help exits 0."""
    if shutil.which("uv") is None:
        pytest.skip("uv not found in PATH")
    result = subprocess.run(
        ["uv", "run", "returnmap", "list", "--help"],
        capture_output=True, text=True,
    )
    assert result.returncode == 0
