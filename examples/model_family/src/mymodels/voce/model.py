"""J2 plasticity with Voce (saturating) isotropic hardening.

Yield function
--------------
    f = σ_vm(σ) − σ_y(p),        σ_y(p) = σ_y0 + Q (1 − exp(−b p))

``p`` is the equivalent plastic strain.  The flow is associative, and
``Δp = Δλ``.

The family, by how far each class goes:

``J2Voce``
    The physics, for any dimension.  This alone is a complete model: the
    framework solves the return mapping and the consistent tangent by
    automatic differentiation.
``J2Voce1D``
    The same, fixed to uniaxial stress — the class to fit test records with.
``J2Voce3D``
    Adds a hand-written return mapping and tangent, the algebra a UMAT is
    written from.  ``fortran/voce_3d.f90`` is that port.
"""

import autograd.numpy as anp

from returnmap.core import (
    SCALAR, SOLID_3D, UNIAXIAL_1D, Explicit, MaterialModel, ParamMeta, ReturnMappingResult,
)


class J2Voce(MaterialModel):
    """J2 plasticity with Voce isotropic hardening — the physics, any dimension."""

    param_names = ["E", "nu", "sigma_y0", "Q", "b"]
    # Optional.  It is what `J2Voce.at_defaults()`, the tuning notebook and a
    # service's input form read.
    param_meta = {
        "E": ParamMeta(doc="Young's modulus", unit="MPa", default=210_000.0,
                       bounds=(1_000.0, 1_000_000.0)),
        "nu": ParamMeta(doc="Poisson's ratio", default=0.3, bounds=(0.0, 0.49)),
        "sigma_y0": ParamMeta(doc="Initial yield stress", unit="MPa", default=250.0,
                              bounds=(1.0, 5_000.0)),
        "Q": ParamMeta(doc="Saturation value of the hardening", unit="MPa", default=150.0,
                       bounds=(0.0, 5_000.0)),
        "b": ParamMeta(doc="Rate at which the hardening saturates", default=20.0,
                       bounds=(0.0, 1_000.0)),
    }

    ep = Explicit(shape=SCALAR, doc="equivalent plastic strain")

    def yield_stress(self, ep):
        return self.sigma_y0 + self.Q * (1.0 - anp.exp(-self.b * ep))

    def hardening_modulus(self, ep):
        """dσ_y/dp."""
        return self.Q * self.b * anp.exp(-self.b * ep)

    def yield_function(self, state):
        return self.vonmises(state["stress"]) - self.yield_stress(state["ep"])

    def update_state(self, new, dlambda, inc):
        return [self.ep(inc.state_n["ep"] + dlambda)]


class J2Voce1D(J2Voce):
    """Uniaxial stress.  Solved by automatic differentiation."""

    dimension = UNIAXIAL_1D


class J2Voce3D(J2Voce):
    """3D solid, with the return mapping and the tangent written out by hand.

    For isotropic J2 the return is radial, ``σ = σ_trial − 3μ Δλ s_trial/q``
    with ``q = σ_vm(σ_trial)``, so the step reduces to one scalar equation:

        g(Δλ) = q − 3μ Δλ − σ_y(p_n + Δλ) = 0

    The identity behind it, ``C n = 3μ s_trial / q``, needs all three direct
    components — which is why this class is fixed to ``SOLID_3D``.
    """

    dimension = SOLID_3D

    #: Newton iterations allowed on g(Δλ) = 0, and where it stops (relative to q).
    MAX_ITER = 25
    TOL = 1.0e-12

    def user_defined_return_mapping(self, inc):
        mu = self.E / (2.0 * (1.0 + self.nu))
        stress_trial = inc.trial["stress"]
        ep_n = inc.state_n["ep"]
        s_trial = self.dev(stress_trial)
        q = self.vonmises(stress_trial)

        # g is convex and decreasing, and g(0) > 0 on a plastic step, so Newton
        # from Δλ = 0 approaches the root from below and never overshoots it.
        dlambda = 0.0
        residual_history = []
        converged = False
        for _ in range(self.MAX_ITER + 1):
            g = q - 3.0 * mu * dlambda - self.yield_stress(ep_n + dlambda)
            residual_history.append(float(abs(g)))
            if abs(g) < self.TOL * q:
                converged = True
                break
            dlambda = dlambda + g / (3.0 * mu + self.hardening_modulus(ep_n + dlambda))

        stress = stress_trial - (3.0 * mu * dlambda / q) * s_trial
        return ReturnMappingResult(
            stress=stress,
            state={"stress": stress, "ep": ep_n + dlambda},
            dlambda=anp.asarray(dlambda),
            n_iterations=len(residual_history) - 1,
            residual_history=residual_history,
            converged=converged,
        )

    def user_defined_tangent(self, new, dlambda, inc):
        """Consistent tangent of the radial return.

            D = I_vol C + θ I_dev C − β s_trial ⊗ s_trial

        with ``θ = 1 − 3μ Δλ / q`` and
        ``β = 9μ² (σ_y − H Δλ) / ((3μ + H) q³)``, where ``σ_y`` and
        ``H = dσ_y/dp`` are taken at the end of the step.  With a constant
        ``H`` this is the tangent of linear hardening.
        """
        mu = self.E / (2.0 * (1.0 + self.nu))
        C = self.elastic_stiffness(inc.state_n)
        s_trial = self.dev(inc.trial["stress"])
        q = self.vonmises(inc.trial["stress"])
        sigma_y = self.yield_stress(new["ep"])
        H = self.hardening_modulus(new["ep"])

        theta = 1.0 - 3.0 * mu * dlambda / q
        beta = 9.0 * mu**2 * (sigma_y - H * dlambda) / ((3.0 * mu + H) * q**3)
        return (
            self.I_vol() @ C + theta * (self.I_dev() @ C)
            - beta * anp.outer(s_trial, s_trial)
        )
