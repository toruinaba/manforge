"""How every check in this package measures an error.

One rule, so that a tolerance means the same thing whichever check it is
passed to::

    err    = max|candidate − reference| / max|reference|
    passed = err <= tol

The largest difference, relative to the largest magnitude of the reference —
taken over the whole array (a Jacobian block, a tangent, a stress vector) and,
where a check covers several points, over the points checked.

Why whole arrays and not component by component: an entry that is zero in
exact arithmetic and 1e-17 in floating point is round-off, not a 100 % error.
A component-wise measure needs a constant in the denominator to survive such
entries, and that constant carries a unit: ``|Δ| / (|ref| + 0.01)`` lets
everything through in units where the numbers are small, and turns the
round-off on a zero entry into a failure in units where they are large.
Nothing here depends on the units the model is written in.

A NaN anywhere is the worst possible error, not the absence of one: ``nan >
x`` is false for every ``x``, so left alone a NaN loses every "is this worse
than the worst so far" comparison and drops out of the report.

What "the reference" is when nothing of its own size is there to divide by —
a Jacobian block that is zero in exact arithmetic, a state variable that has
not left zero — is each check's to say, in units of the problem and never as
a bare number: see :data:`NOISE_FLOOR`.
"""

from __future__ import annotations

import numpy as np

#: A quantity this many orders below the scale of what it belongs to is
#: round-off in double precision, whatever the units: it has no size of its
#: own to be relative to.  The check that meets one says what it is measured
#: against instead (``check_hand_blocks``: the size the block's row and
#: column imply; ``crosscheck``: this fraction of the largest quantity of
#: the run).
NOISE_FLOOR = 1e-12


def max_abs(a) -> float:
    """``max|a|`` over the array; NaN if any entry is NaN."""
    return float(np.max(np.abs(np.asarray(a, dtype=float))))


def rel_err(diff: float, scale: float) -> float:
    """``diff / scale`` under the rules above.

    ``0 / 0`` is 0 (two zeros agree); a nonzero difference against a zero
    scale is infinite; anything involving NaN is infinite.
    """
    if np.isnan(diff) or np.isnan(scale):
        return float("inf")
    if scale > 0.0:
        err = diff / scale
        return float("inf") if np.isnan(err) else float(err)
    return 0.0 if diff == 0.0 else float("inf")


def array_rel_err(candidate, reference) -> float:
    """``max|candidate − reference| / max|reference|`` of one array."""
    c = np.asarray(candidate, dtype=float)
    r = np.asarray(reference, dtype=float)
    with np.errstate(invalid="ignore"):      # inf − inf is NaN, handled below
        diff = max_abs(c - r)
    return rel_err(diff, max_abs(r))


def as_error(value: float) -> float:
    """``value`` as an error: NaN becomes infinite."""
    return float("inf") if np.isnan(value) else float(value)
