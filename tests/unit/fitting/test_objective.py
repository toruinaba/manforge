"""Smoke tests for fitting/objective.py.

TODO: cover objective function construction, gradient finite-difference check.
"""

import importlib
import pytest


def test_objective_importable():
    """manforge.fitting.objective must be importable without error."""
    mod = importlib.import_module("manforge.fitting.objective")
    assert mod is not None
