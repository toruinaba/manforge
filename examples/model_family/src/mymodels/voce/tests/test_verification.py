"""The checks every model owes — ``returnmap.verification.check_model``.

The same few lines in every family; what is checked follows from what the
model declares, and on which histories from ``verification.py``.
"""

import pytest

from returnmap.verification import check_model

from mymodels.voce.verification import MODELS, UMATS


@pytest.mark.parametrize("model", MODELS, ids=lambda model: type(model).__name__)
def test_passes_every_check_that_applies(model):
    report = check_model(model)
    assert report.passed, report.summary()


@pytest.mark.fortran
@pytest.mark.parametrize("umat", UMATS, ids=lambda umat: umat.module)
def test_the_umat_agrees_with_the_python_model(umat):
    """The compiled UMAT, step by step over the same histories (needs the
    module built: ``returnmap build --family mymodels``)."""
    pytest.importorskip(umat.module, reason=f"{umat.module} is not built")
    report = check_model(umat.model.at_defaults())
    assert report["UMAT"].status == "passed", report.summary()
