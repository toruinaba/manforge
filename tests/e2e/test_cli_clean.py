"""E2E tests for `returnmap clean` CLI command."""

import shutil
import subprocess
import pytest


def test_clean_dry_run_help():
    """returnmap clean --dry-run exits 0."""
    if shutil.which("uv") is None:
        pytest.skip("uv not found in PATH")
    result = subprocess.run(
        ["uv", "run", "returnmap", "clean", "--dry-run"],
        capture_output=True, text=True,
    )
    assert result.returncode == 0


def test_clean_help():
    """returnmap clean --help exits 0."""
    if shutil.which("uv") is None:
        pytest.skip("uv not found in PATH")
    result = subprocess.run(
        ["uv", "run", "returnmap", "clean", "--help"],
        capture_output=True, text=True,
    )
    assert result.returncode == 0
    assert "clean" in result.stdout.lower()
