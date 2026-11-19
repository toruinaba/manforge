"""J2 plasticity with isotropic hardening (reference implementation).

Model parameters
----------------
E        : Young's modulus
nu       : Poisson's ratio
sigma_y0 : Initial yield stress
H        : Isotropic hardening modulus (linear)

Internal state
--------------
ep : equivalent plastic strain (scalar, ≥ 0)

Yield function
--------------
f(state) = σ_vm(state["stress"]) - (σ_y0 + H * ep)

where σ_vm is the von Mises equivalent stress.

Flow rule (associative)
-----------------------
dε_p = Δλ · n,  n = df/dσ = (3/2) s / σ_vm  (unit normal in Mandel sense)
"""

import autograd.numpy as anp

from manforge.core.material import MaterialModel, ParamMeta
from manforge.core.state import Explicit, Increment, SCALAR, State, StateUpdate, StateResidual
from manforge.core.dimension import SOLID_3D, PLANE_STRESS, UNIAXIAL_1D
from manforge.core.result import ReturnMappingResult
from manforge.core.material import verified_against_fortran
from manforge._typing import Scalar as ScalarType, Stiffness, StateDict


class J2Isotropic(MaterialModel):
    """J2 plasticity with isotropic hardening — common physics across stress states.

    Pass ``dimension=`` to select the stress state, or use one of the
    pre-built concrete classes (:class:`J2Isotropic3D`, :class:`J2IsotropicPS`,
    :class:`J2Isotropic1D`), which fix it.

    Parameters
    ----------
    dimension : StressDimension, optional
        Defaults to ``SOLID_3D``.
    E : float
        Young's modulus.
    nu : float
        Poisson's ratio.
    sigma_y0 : float
        Initial yield stress.
    H : float
        Isotropic hardening modulus (linear).
    """

    param_names = ["E", "nu", "sigma_y0", "H"]
    param_meta = {
        "E": ParamMeta(doc="Young's modulus", unit="MPa",
                       default=210_000.0, bounds=(1e3, 1e6)),
        "nu": ParamMeta(doc="Poisson's ratio", default=0.3, bounds=(0.0, 0.49)),
        "sigma_y0": ParamMeta(doc="Initial yield stress", unit="MPa",
                              default=250.0, bounds=(1.0, 5e3)),
        "H": ParamMeta(doc="Isotropic hardening modulus (linear)", unit="MPa",
                       default=1000.0, bounds=(0.0, 1e5)),
    }
    ep = Explicit(shape=SCALAR, doc="equivalent plastic strain")

    def yield_function(self, state: "State | StateDict") -> ScalarType:
        """J2 yield function f = σ_vm − (σ_y0 + H · ep)."""
        sigma_y = self.sigma_y0 + self.H * state["ep"]
        return self.vonmises(state["stress"]) - sigma_y

    def update_state(
        self, new: "State | StateDict", dlambda: ScalarType, inc: Increment,
    ) -> list[StateUpdate | StateResidual]:
        """Δep = Δλ (von Mises associative flow)."""
        return [self.ep(inc.state_n["ep"] + dlambda)]


class J2Isotropic3D(J2Isotropic):
    """J2 plasticity with analytical radial return for full-rank stress states.

    The only state, ``ep``, is explicit (``update_state``: Δep = Δλ), so the
    Newton-Raphson unknowns are ``[σ, Δλ]``.

    Provides closed-form ``user_defined_return_mapping`` and
    ``user_defined_tangent`` using the identity ``C @ n_dev = 2μ · n_dev``,
    which holds exactly for SOLID_3D and PLANE_STRAIN but not for statically
    condensed stress states (PLANE_STRESS, UNIAXIAL_1D).  For those, use
    :class:`J2IsotropicPS` or :class:`J2Isotropic1D` with the autodiff
    return-mapping path.

    Parameters
    ----------
    E : float
        Young's modulus.
    nu : float
        Poisson's ratio.
    sigma_y0 : float
        Initial yield stress.
    H : float
        Isotropic hardening modulus (linear).
    """

    dimension = SOLID_3D

    @verified_against_fortran(
        "j2_isotropic_3d_elastic_stiffness",
        test="tests/integration/verification/test_fortran_check.py::test_check_bindings_elastic_stiffness",
    )
    def elastic_stiffness(self, state: "State | StateDict") -> Stiffness:
        return super().elastic_stiffness(state)

    # ------------------------------------------------------------------
    # Analytical solver hooks
    # ------------------------------------------------------------------

    def user_defined_return_mapping(self, inc: Increment) -> ReturnMappingResult:
        """J2 radial return — closed-form plastic correction.

        Notes
        -----
        For isotropic J2 with linear hardening, ``C n_voigt = 3μ s_trial / σ_vm``
        for all Voigt components (the volumetric part of the gradient vanishes),
        so the radial return reduces to a single scalar equation in Δλ.
        This identity holds when ``ndi == ndi_phys`` (SOLID_3D, PLANE_STRAIN).

        Reads σ_trial from ``inc.trial["stress"]`` and ``ep`` from ``inc.state_n``.

        Returns
        -------
        ReturnMappingResult
        """
        mu = self.E / (2.0 * (1.0 + self.nu))
        stress_trial = inc.trial["stress"]
        ep_n = inc.state_n["ep"]

        sigma_y = self.sigma_y0 + self.H * ep_n
        s_trial = self.dev(stress_trial)
        sigma_vm_trial = self.vonmises(stress_trial)
        f_trial = sigma_vm_trial - sigma_y  # > 0 (caller ensures plasticity)

        # Δλ = f_trial / (3μ + H)
        dlambda = f_trial / (3.0 * mu + self.H)

        # Radial return: σ_new = σ_trial − (3μΔλ/σ_vm) s_trial
        stress_new = stress_trial - (3.0 * mu * dlambda / sigma_vm_trial) * s_trial

        return ReturnMappingResult(
            stress=stress_new,
            state={"stress": stress_new, "ep": ep_n + dlambda},
            dlambda=anp.asarray(dlambda),
            n_iterations=1,
            residual_history=[float(f_trial), 0.0],
            converged=True,
        )

    def user_defined_tangent(
        self, new: "State | StateDict", dlambda: ScalarType, inc: Increment,
    ) -> Stiffness:
        """J2 algorithmic consistent tangent — closed-form (de Souza Neto).

            D^ep = I_vol C + θ I_dev C − β (s_trial ⊗ s_trial)

        where θ = 1 − 3μΔλ/σ_vm_trial and β = 9μ²σ_y / ((3μ+H) σ_vm_trial³).
        ``s_trial`` is the deviator of ``inc.trial["stress"]``, ``ep`` is read
        from ``inc.state_n``, and C is the elastic stiffness at ``inc.state_n``.

        Returns
        -------
        anp.ndarray, shape (ntens, ntens)
        """
        mu = self.E / (2.0 * (1.0 + self.nu))
        C = self.elastic_stiffness(inc.state_n)
        ep_n = inc.state_n["ep"]

        sigma_y = self.sigma_y0 + self.H * ep_n
        sigma_vm_trial = sigma_y + (3.0 * mu + self.H) * dlambda
        theta = 1.0 - 3.0 * mu * dlambda / sigma_vm_trial

        s_trial = self.dev(inc.trial["stress"])

        beta = 9.0 * mu ** 2 * sigma_y / ((3.0 * mu + self.H) * sigma_vm_trial ** 3)

        I_vol = self.I_vol()
        I_dev = self.I_dev()

        return I_vol @ C + theta * (I_dev @ C) - beta * anp.outer(s_trial, s_trial)


class J2IsotropicPS(J2Isotropic):
    """J2 plasticity with isotropic hardening for plane-stress elements.

    The only state, ``ep``, is explicit (``update_state``: Δep = Δλ), so the
    Newton-Raphson unknowns are ``[σ, Δλ]``.  There is no closed form here:
    the return mapping is the autodiff one.

    Parameters
    ----------
    E : float
        Young's modulus.
    nu : float
        Poisson's ratio.
    sigma_y0 : float
        Initial yield stress.
    H : float
        Isotropic hardening modulus (linear).
    """

    dimension = PLANE_STRESS


class J2Isotropic1D(J2Isotropic):
    """J2 plasticity with isotropic hardening for uniaxial (1D) elements.

    The only state, ``ep``, is explicit (``update_state``: Δep = Δλ), so the
    Newton-Raphson unknowns are ``[σ, Δλ]``.  Provides closed-form ``user_defined_return_mapping``
    and ``user_defined_tangent`` using the 1D radial return mapping, written in
    the same ``dev`` / ``vonmises`` / ``I_vol`` / ``I_dev`` form as
    :class:`J2Isotropic3D` (with ``E`` replacing ``3μ`` due to static condensation).

    Parameters
    ----------
    E : float
        Young's modulus.
    nu : float
        Poisson's ratio.
    sigma_y0 : float
        Initial yield stress.
    H : float
        Isotropic hardening modulus (linear).
    """

    dimension = UNIAXIAL_1D

    # ------------------------------------------------------------------
    # Analytical solver hooks
    # ------------------------------------------------------------------

    def user_defined_return_mapping(self, inc: Increment) -> ReturnMappingResult:
        """1D J2 radial return — closed-form, 3D-analogous form.

        Notes
        -----
        For 1D, ``C n_voigt = E · n_voigt`` holds (``n_voigt = sign(σ)``),
        so the radial return reduces to a single scalar equation in Δλ.
        This is structurally analogous to the 3D identity
        ``C n_voigt = 3μ s/σ_vm``, with ``3μ`` replaced by ``E`` due to
        static condensation (ndi=1, ndi_phys=3).

        ``n_voigt = (3/2) · dev(σ) / vonmises(σ) = sign(σ)`` for 1D,
        so the update ``σ_new = σ_trial − Δλ · C n_voigt`` is written as
        ``σ_new = σ_trial − 1.5 · E · Δλ / σ_vm · s_trial``, mirroring
        the 3D formula with ``s_trial`` and ``vonmises`` in place of
        raw ``abs`` and ``sign``.

        Reads σ_trial from ``inc.trial["stress"]`` and ``ep`` from ``inc.state_n``.

        Returns
        -------
        ReturnMappingResult
        """
        E = self.elastic_stiffness(inc.state_n)[0, 0]
        stress_trial = inc.trial["stress"]
        ep_n = inc.state_n["ep"]
        sigma_y = self.sigma_y0 + self.H * ep_n
        s_trial = self.dev(stress_trial)
        sigma_vm_trial = self.vonmises(stress_trial)
        f_trial = sigma_vm_trial - sigma_y  # > 0 (caller ensures plasticity)

        # Δλ = f_trial / (E + H)  [1D analogue of f_trial / (3μ + H)]
        dlambda = f_trial / (E + self.H)

        # Radial return: σ_new = σ_trial − Δλ · C n_voigt
        # n_voigt = (3/2) s_trial / σ_vm = sign(σ_trial) for 1D
        stress_new = stress_trial - 1.5 * (E * dlambda / sigma_vm_trial) * s_trial

        return ReturnMappingResult(
            stress=stress_new,
            state={"stress": stress_new, "ep": ep_n + dlambda},
            dlambda=anp.asarray(dlambda),
            n_iterations=1,
            residual_history=[float(f_trial), 0.0],
            converged=True,
        )

    def user_defined_tangent(
        self, new: "State | StateDict", dlambda: ScalarType, inc: Increment,
    ) -> Stiffness:
        """1D consistent elastoplastic tangent for linear isotropic hardening.

        For uniaxial J2 with linear hardening modulus H, the elastoplastic
        tangent reduces exactly to ``D^ep = E·H/(E+H)`` independently of the
        current plastic strain or step size. Returns a (1, 1) ndarray.
        """
        E = self.elastic_stiffness(inc.state_n)[0, 0]
        return anp.array([[E * self.H / (E + self.H)]])
