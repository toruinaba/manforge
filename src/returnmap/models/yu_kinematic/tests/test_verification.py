"""The checks every model owes — :func:`returnmap.verification.check_model`.

What is checked follows from what the model declares, and on which histories
from ``verification.py``.

The hand derivations of YUKinematic3D and YUKinematicPS are declared
approximate on one kind of step: where the stagnation gate —
``smooth_heaviside(g_stag)``, which switches the growth of ``R`` on — is
neither 0 nor 1 (``stagnation_gate_in_transition``).  They treat the gate as
a constant, so on such a step the hand-written tangent, and the
``[theta][beta]`` and ``[theta][dlambda]`` blocks in plane stress, miss its
derivative (measured on the standard multiaxial history: tangent off by 1.2e-4
in 3D with the gate at 0.0015, by 4.4e-2 in plane stress with the gate at
0.99).  Stress and state agree with autodiff there as everywhere else.
Those steps are left out of the comparison and counted; the second test
below keeps the count honest.
"""

import functools

import pytest

from returnmap.models.yu_kinematic.verification import MODELS, UMATS
from returnmap.verification import check_model

BY_NAME = {type(model).__name__: model for model in MODELS}
# The 3D and plane-stress models take 5-10 s: three histories, two paths.
SLOW = ("YUKinematic3D", "YUKinematicPS")


@functools.lru_cache(maxsize=None)
def report_of(name):
    """One run of the checks per model, shared by the tests below."""
    return check_model(BY_NAME[name])


def _case(name):
    return pytest.param(name, marks=[pytest.mark.slow] if name in SLOW else [])


@pytest.mark.parametrize("name", [_case(name) for name in BY_NAME])
def test_passes_every_check_that_applies(name):
    report = report_of(name)
    assert report.passed, report.summary()


@pytest.mark.slow
@pytest.mark.parametrize("name", SLOW)
def test_the_declared_approximation_covers_only_the_gate_switching(name):
    """A handful of steps, not a blanket: were the condition to hold on many
    steps, the hand derivations would be checked on few."""
    report = report_of(name)
    assert report.approximations == [
        "the stagnation gate is differentiated as a constant; exact while it is 0 "
        "or 1, not on the step where it switches"
    ]
    n_plastic = sum(report.histories.values())
    crosschecks = report["hand-written vs autodiff"].reports
    n_off = sum(r.n_steps - r.n_passed for r in crosschecks.values())
    assert 1 <= n_off <= 3 and n_plastic > 70
    # Stress and state agree on every step, those included.
    assert max(r.max_stress_rel_err for r in crosschecks.values()) < 1e-7
    assert max(e for r in crosschecks.values() for e in r.max_state_rel_err.values()) < 1e-7
    if name == "YUKinematicPS":
        blocks = report["hand blocks"].reports["all"].blocks
        declared = {key: b.n_approx_points for key, b in blocks.items() if b.n_approx_points}
        assert set(declared) == {("theta", "beta"), ("theta", "dlambda")}
        assert all(n <= 3 for n in declared.values())
        # On those steps the block is off by its own size; elsewhere it is exact.
        assert blocks[("theta", "beta")].approx_err > 0.5
        assert blocks[("theta", "beta")].worst_err < 1e-12


@pytest.mark.fortran
@pytest.mark.parametrize("umat", UMATS, ids=lambda umat: umat.module)
def test_the_umat_agrees_with_the_python_model(umat):
    """The compiled UMAT, step by step over the same histories (needs the
    module built: ``returnmap build --family returnmap.models.yu_kinematic``)."""
    pytest.importorskip(umat.module, reason=f"{umat.module} is not built")
    report = report_of(umat.model.__name__)
    assert report["UMAT"].status == "passed", report.summary()
