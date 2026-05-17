"""Smoke tests for simulation/_residual.py.

TODO: cover build_residual callable shape, build_state_from_x reconstruction.
"""

import importlib
import pytest


def test_residual_module_importable():
    mod = importlib.import_module("manforge.simulation._residual")
    assert mod is not None


def test_build_residual_importable():
    from manforge.simulation._residual import build_residual
    assert callable(build_residual)
