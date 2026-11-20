"""The checks every model owes — :func:`returnmap.verification.check_model`.

What is checked follows from what the model declares, and on which histories
from ``verification.py``.

YUKinematic3D and YUKinematicPS do not pass yet, and are marked so.  The
hand-derived Jacobian and the hand-written tangent treat the stagnation gate
— ``smooth_heaviside(g_stag)``, which switches the growth of ``R`` on — as a
constant.  That is exact wherever the gate is 0 or 1, and wrong on the one
step where the converged ``g_stag`` lands inside the smoothing band and the
gate is in between (measured on the standard multiaxial history: gate 0.0015
at step 13 in 3D, 0.99 and 0.39 at steps 16 and 18 in plane stress).  On
those steps stress and state still agree with autodiff to 1e-11 — the hand
Newton iteration converges to the same root — but the tangent is off by up
to 4e-2 and the ``[theta][beta]`` block by its own size.  Whether to derive
the gate's derivative, declare the blocks approximate, or change the gate is
a decision about the model; until then the mark keeps the gap visible.
"""

import functools

import pytest

from returnmap.models.yu_kinematic.verification import MODELS
from returnmap.verification import check_model

GATE_DERIVATIVE_MISSING = {
    "YUKinematic3D": "hand-written tangent ignores the derivative of the stagnation gate "
                     "(tangent 1.2e-4 off on the step the gate switches on)",
    "YUKinematicPS": "hand-derived [theta][beta] / [theta][dlambda] blocks and the "
                     "hand-written tangent ignore the derivative of the stagnation gate "
                     "(tangent 4.4e-2 off on the step the gate switches on)",
}


BY_NAME = {type(model).__name__: model for model in MODELS}


@functools.lru_cache(maxsize=None)
def report_of(name):
    """One run of the checks per model, shared by the tests below."""
    return check_model(BY_NAME[name])


def _case(name):
    reason = GATE_DERIVATIVE_MISSING.get(name)
    # The 3D and plane-stress models take 5-10 s: three histories, two paths.
    marks = [pytest.mark.slow, pytest.mark.xfail(strict=True, reason=reason)] if reason else []
    return pytest.param(name, marks=marks)


@pytest.mark.parametrize("name", [_case(name) for name in BY_NAME])
def test_passes_every_check_that_applies(name):
    report = report_of(name)
    assert report.passed, report.summary()


@pytest.mark.slow
@pytest.mark.parametrize("name", list(GATE_DERIVATIVE_MISSING))
def test_the_gate_is_the_only_gap(name):
    """Everything but the tangent-related checks passes, so the mark above
    cannot hide a second problem."""
    report = report_of(name)
    assert report["converges"].status == "passed", report.summary()
    assert report["tangent"].status == "passed", report.summary()
    assert report["hand-written solution"].status == "passed", report.summary()
    crosschecks = report["hand-written vs autodiff"].reports
    assert max(r.max_stress_rel_err for r in crosschecks.values()) < 1e-7
    assert max(e for r in crosschecks.values() for e in r.max_state_rel_err.values()) < 1e-7
    # The steps that disagree are few: the gate is fractional on one or two.
    assert sum(r.n_steps - r.n_passed for r in crosschecks.values()) <= 3
