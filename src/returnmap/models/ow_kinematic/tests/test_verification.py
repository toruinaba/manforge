"""The checks every model owes — :func:`returnmap.verification.check_model`.

The same few lines in every family; what is checked follows from what the
model declares, and on which histories from ``verification.py``.
"""

import pytest

from returnmap.models.ow_kinematic.verification import MODELS
from returnmap.verification import check_model


@pytest.mark.parametrize("model", MODELS, ids=lambda model: type(model).__name__)
def test_passes_every_check_that_applies(model):
    report = check_model(model)
    assert report.passed, report.summary()
