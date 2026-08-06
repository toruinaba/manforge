"""Yoshida-Uemori kinematic hardening with a projected stagnation-surface update.

An alternative to the stagnation-surface formulation in
:mod:`manforge.models.yu_kinematic`, provided so the two can be compared under
identical conditions: everything except ``_stagnation_update`` is inherited, so
the residual, the NR loop, the analytical Jacobian blocks and the consistent
tangent are shared code, not copies.

Why an alternative is needed
----------------------------
The published formulation determines mu from

    3·Gn = r_n·(r_n + H)·(1+mu)² + 3·h·Fn·(1+mu),
    H = √(r_n² + 6·h·Fn/(1+mu))

by inner Newton iteration.  The radicand ``r_n² + 6·h·Fn`` goes negative
whenever beta recedes far enough from the stagnation centre -- ``Fn`` is
negative there and bounded only by ‖g_xi‖·‖Δβ‖ -- so mu has no real root once
‖Δβ‖ grows past roughly ``r_n``.  Explicit integration never notices: r_n and
Δβ are not required to be consistent within one increment.  Under implicit
integration they are, and the iteration lands on NaN.  Cutting the time
increment does not recover it, because Δβ shrinks with the increment while r_n
does not, so a solver keeps subdividing the same point forever.

This variant instead projects beta onto the stagnation surface directly, which
needs no iteration; mu comes out in closed form and the negative-radicand case
becomes ``mu < 0``, gated off rather than undefined.
"""

from manforge.models.yu_kinematic import (
    YUKinematic1D,
    YUKinematic3D,
    YUKinematicPS,
)


def _stagnation_update_proj(model, r_n, R_n, g_xi, g_stag, d_beta, dlambda):
    """Projected stagnation-surface update -- the single implementation.

    The three dimension classes below delegate here rather than each carrying
    a copy, so a change cannot land on one dimension and miss the others.

    Returns the gated ``(delta_q, delta_r, delta_R)``, matching the contract of
    :meth:`YUKinematic._stagnation_update`: the caller adds these raw, so the
    activity gate must be applied here.
    """
    raise NotImplementedError(
        "projected stagnation-surface update is not implemented yet"
    )


class YUKinematicProj3D(YUKinematic3D):
    """3-D solid; projected stagnation update. Otherwise :class:`YUKinematic3D`."""

    def _stagnation_update(self, r_n, R_n, g_xi, g_stag, d_beta, dlambda):
        return _stagnation_update_proj(self, r_n, R_n, g_xi, g_stag, d_beta, dlambda)


class YUKinematicProjPS(YUKinematicPS):
    """Plane stress (P metric); projected stagnation update."""

    def _stagnation_update(self, r_n, R_n, g_xi, g_stag, d_beta, dlambda):
        return _stagnation_update_proj(self, r_n, R_n, g_xi, g_stag, d_beta, dlambda)


class YUKinematicProj1D(YUKinematic1D):
    """Uniaxial; projected stagnation update.

    Inherits ``update_state`` from ``YUKinematic`` and has no
    ``user_defined_return_mapping``, so only the NR path applies.
    """

    def _stagnation_update(self, r_n, R_n, g_xi, g_stag, d_beta, dlambda):
        return _stagnation_update_proj(self, r_n, R_n, g_xi, g_stag, d_beta, dlambda)
