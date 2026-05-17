"""Smoke tests for autodiff backend (src/manforge/autodiff/backend.py).

TODO: cover gradient computation, Hessian, type promotion.
"""

import importlib
import pytest


def test_backend_importable():
    """autodiff.backend must be importable without error."""
    mod = importlib.import_module("manforge.autodiff.backend")
    assert mod is not None
