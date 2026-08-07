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
from manforge.utils.smooth import smooth_heaviside


def _stagnation_update_proj(model, r_n, R_n, g_xi, g_stag, d_beta, dlambda):
    """Projected stagnation-surface update -- the single implementation.

    The three dimension classes below delegate here rather than each carrying
    a copy, so a change cannot land on one dimension and miss the others.

    Returns the gated ``(delta_q, delta_r, delta_R)``, matching the contract of
    :meth:`YUKinematic._stagnation_update`: the caller adds these raw, so the
    activity gate must be applied here.

    ``d_beta`` is unused: the projection needs only where beta ended up
    relative to the surface, not how it got there.  The signature stays shared
    with the published formulation so both are called from the same code.

    The projection puts beta exactly on the updated surface -- moving the
    centre by (1-h)·g_stag along g_xi and growing the radius by h·g_stag leaves
    ‖beta_new - q_new‖ - r_new = 0 for any h.  h only splits the correction
    between translation and expansion.
    """
    # Dead band matches the published formulation so the transition band is not
    # itself a difference between the two: shift by +1e-10 so boundary noise
    # activates.
    Gg = smooth_heaviside(g_stag + 1.0e-10)
    delta_r = Gg * model.h * g_stag
    g_xi_norm = model.vonmises_norm(g_xi)
    delta_q = Gg * (1 - model.h) * g_stag * g_xi / g_xi_norm
    k_eff = model.k_eff
    delta_R = Gg * ((R_n + k_eff * model.Rsat * dlambda) / (1 + k_eff * dlambda) - R_n)
    return delta_q, delta_r, delta_R
    


class YUKinematicProj3D(YUKinematic3D):
    """3-D solid; projected stagnation update. Otherwise :class:`YUKinematic3D`."""

    def _stagnation_update(self, r_n, R_n, g_xi, g_stag, d_beta, dlambda):
        return _stagnation_update_proj(self, r_n, R_n, g_xi, g_stag, d_beta, dlambda)

    def dRtheta_dbeta(self, theta, theta_max, R, R_n, dlambda):
        """Placeholder for the gate-derivative term; currently the inherited block.

        ``a = B + R - Y`` depends on beta through ``R = R_n + Gg·ΔR`` and
        ``Gg = smooth_heaviside(‖beta - q_n‖ - r_n)``, a term the published
        derivation drops because mu came from an inner Newton and ∂mu/∂beta was
        not available in closed form.  Here it is, so the term can be supplied.

        Deferred: the omission costs ~2% on this block, and only while the
        converged state sits inside the 0.01 MPa gate transition band.  This
        override exists to hold the place and the reasoning; measurements are in
        tests/benchmarks/yu_kinematic/test_proj_jacobian.py.
        """
        return super().dRtheta_dbeta(theta, theta_max, R, R_n, dlambda)


class YUKinematicProjPS(YUKinematicPS):
    """Plane stress (P metric); projected stagnation update."""

    def _stagnation_update(self, r_n, R_n, g_xi, g_stag, d_beta, dlambda):
        return _stagnation_update_proj(self, r_n, R_n, g_xi, g_stag, d_beta, dlambda)

    def calc_ft_fb(self, state, dlambda, state_n):
        """Placeholder for the gate-derivative term; currently the inherited block.

        See :meth:`YUKinematicProj3D.dRtheta_dbeta` -- same missing term, same
        reason it is available here and not in the published formulation, same
        ~2% magnitude confined to the gate transition band.
        """
        return super().calc_ft_fb(state, dlambda, state_n)


class YUKinematicProj1D(YUKinematic1D):
    """Uniaxial; projected stagnation update.

    Inherits ``update_state`` from ``YUKinematic`` and has no
    ``user_defined_return_mapping``, so only the NR path applies.
    """

    def _stagnation_update(self, r_n, R_n, g_xi, g_stag, d_beta, dlambda):
        return _stagnation_update_proj(self, r_n, R_n, g_xi, g_stag, d_beta, dlambda)
