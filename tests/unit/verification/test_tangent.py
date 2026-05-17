"""Smoke tests for verification/tangent.py TangentCheckResult dataclass.

TODO: cover TangentCheckResult fields, check_tangent result structure.
"""

import importlib
import pytest


def test_tangent_module_importable():
    """manforge.verification.tangent must be importable without error."""
    mod = importlib.import_module("manforge.verification.tangent")
    assert mod is not None


def test_check_tangent_importable():
    from manforge.verification.tangent import check_tangent
    assert callable(check_tangent)
