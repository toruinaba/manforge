"""E2E tests for `manforge clean` CLI command."""

import shutil
import subprocess
import pytest


def test_clean_dry_run_help():
    """manforge clean --dry-run exits 0."""
    if shutil.which("uv") is None:
        pytest.skip("uv not found in PATH")
    result = subprocess.run(
        ["uv", "run", "manforge", "clean", "--dry-run"],
        capture_output=True, text=True,
    )
    assert result.returncode == 0


def test_clean_help():
    """manforge clean --help exits 0."""
    if shutil.which("uv") is None:
        pytest.skip("uv not found in PATH")
    result = subprocess.run(
        ["uv", "run", "manforge", "clean", "--help"],
        capture_output=True, text=True,
    )
    assert result.returncode == 0
    assert "clean" in result.stdout.lower()
