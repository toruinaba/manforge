"""E2E tests for `returnmap build` CLI command."""

import shutil
import subprocess
import pytest


def _uv():
    if shutil.which("uv") is None:
        pytest.skip("uv not found in PATH")


def test_build_help(_uv=None):
    """returnmap build --help exits 0 and mentions 'build'."""
    if shutil.which("uv") is None:
        pytest.skip("uv not found in PATH")
    result = subprocess.run(
        ["uv", "run", "returnmap", "build", "--help"],
        capture_output=True, text=True,
    )
    assert result.returncode == 0
    assert "build" in result.stdout.lower()
