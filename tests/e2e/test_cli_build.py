"""E2E tests for `manforge build` CLI command."""

import shutil
import subprocess
import pytest


def _uv():
    if shutil.which("uv") is None:
        pytest.skip("uv not found in PATH")


def test_build_help(_uv=None):
    """manforge build --help exits 0 and mentions 'build'."""
    if shutil.which("uv") is None:
        pytest.skip("uv not found in PATH")
    result = subprocess.run(
        ["uv", "run", "manforge", "build", "--help"],
        capture_output=True, text=True,
    )
    assert result.returncode == 0
    assert "build" in result.stdout.lower()
