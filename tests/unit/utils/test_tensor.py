"""Smoke tests for utils/tensor.py.

TODO: cover inner_product, deviatoric_inner_product, strain_norm for all dimensions.
"""

import importlib
import pytest


def test_tensor_utils_importable():
    """manforge.utils.tensor must be importable without error."""
    mod = importlib.import_module("manforge.utils.tensor")
    assert mod is not None
