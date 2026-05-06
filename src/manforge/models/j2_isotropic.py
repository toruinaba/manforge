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

from manforge.core.material import MaterialModel3D, MaterialModelPS, MaterialModel1D
from manforge.core.state import Explicit, NTENS
from manforge.core.dimension import SOLID_3D, PLANE_STRESS, UNIAXIAL_1D, StressState
from manforge.core.result import ReturnMappingResult
from manforge.verification.fortran_registry import verified_against_fortran


class J2Isotropic3D(MaterialModel3D):
    """J2 plasticity with analytical radial return for full-rank stress states.

    Uses the scalar NR path (all non-stress states explicit): implements
    ``update_state`` which returns the updated state directly in closed form (Δep = Δλ).

    Inherits operator methods from :class:`~manforge.core.material.MaterialModel3D`
    (``_dev``, ``_vonmises``, ``isotropic_C``, ``_I_vol``, ``_I_dev``), which
    provide branch-free implementations valid when all direct stress components
    are stored (``ndi == ndi_phys``).

    Provides closed-form ``user_defined_return_mapping`` and
    ``user_defined_tangent`` using the identity ``C @ n_dev = 2μ · n_dev``,
    which holds exactly for SOLID_3D and PLANE_STRAIN but not for statically
    condensed stress states (PLANE_STRESS, UNIAXIAL_1D).  For those, use
    :class:`J2IsotropicPS` or :class:`J2Isotropic1D` with the autodiff
    return-mapping path.

    Parameters
    ----------
    stress_state : StressState, optional
        Must satisfy ``stress_state.ndi == stress_state.ndi_phys``.
        Defaults to ``SOLID_3D``.  The guard is enforced by the parent.
    E : float
        Young's modulus.
    nu : float
        Poisson's ratio.
    sigma_y0 : float
        Initial yield stress.
    H : float
        Isotropic hardening modulus (linear).

    Raises
    ------
    ValueError
        If ``stress_state.ndi != stress_state.ndi_phys`` (raised by
        :class:`~manforge.core.material.MaterialModel3D`).
    """

    param_names = ["E", "nu", "sigma_y0", "H"]
    stress = Explicit(shape=NTENS, doc="Cauchy stress")
    ep = Explicit(shape=(), doc="equivalent plastic strain")

    def __init__(self, stress_state: StressState = SOLID_3D, *,
                 E: float, nu: float, sigma_y0: float, H: float):
        super().__init__(stress_state)
        self.E = E
        self.nu = nu
        self.sigma_y0 = sigma_y0
        self.H = H

    def yield_function(self, state) -> anp.ndarray:
        """J2 yield function f = σ_vm − (σ_y0 + H · ep)."""
        sigma_y = self.sigma_y0 + self.H * state["ep"]
        return self._vonmises(state["stress"]) - sigma_y

    def update_state(self, dlambda, state_n, state_trial) -> list:
        """Δep = Δλ (von Mises associative flow)."""
        sig = self.default_stress_update(dlambda, state_n, state_trial)
        return [self.stress(sig), self.ep(state_n["ep"] + dlambda)]

    # ------------------------------------------------------------------
    # Analytical solver hooks
    # ------------------------------------------------------------------

    def user_defined_return_mapping(self, stress_trial, C, state_n):
        """J2 radial return — closed-form plastic correction.

        Notes
        -----
        For isotropic J2 with linear hardening, ``C n_voigt = 3μ s_trial / σ_vm``
        for all Voigt components (the volumetric part of the gradient vanishes),
        so the radial return reduces to a single scalar equation in Δλ.
        This identity holds when ``ndi == ndi_phys`` (SOLID_3D, PLANE_STRAIN).

        Parameters
        ----------
        stress_trial : anp.ndarray, shape (ntens,)
        C : anp.ndarray, shape (ntens, ntens)
        state_n : dict with key ``ep``

        Returns
        -------
        ReturnMappingResult
        """
        mu = self.E / (2.0 * (1.0 + self.nu))
        ep_n = state_n["ep"]

        sigma_y = self.sigma_y0 + self.H * ep_n
        s_trial = self._dev(stress_trial)
        sigma_vm_trial = self._vonmises(stress_trial)
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
        )

    def user_defined_tangent(self, stress, state, dlambda, C, state_n):
        """J2 algorithmic consistent tangent — closed-form (de Souza Neto).

            D^ep = I_vol C + θ I_dev C − β (s_trial ⊗ s_trial)

        where θ = 1 − 3μΔλ/σ_vm_trial and β = 9μ²σ_y / ((3μ+H) σ_vm_trial³).

        Parameters
        ----------
        stress : anp.ndarray, shape (ntens,)
        state : dict  (unused; ep taken from state_n)
        dlambda : anp.ndarray, scalar
        C : anp.ndarray, shape (ntens, ntens)
        state_n : dict with key ``ep``

        Returns
        -------
        anp.ndarray, shape (ntens, ntens)
        """
        mu = self.E / (2.0 * (1.0 + self.nu))
        ep_n = state_n["ep"]

        sigma_y = self.sigma_y0 + self.H * ep_n
        sigma_vm_trial = sigma_y + (3.0 * mu + self.H) * dlambda
        theta = 1.0 - 3.0 * mu * dlambda / sigma_vm_trial

        # dev(σ_new) = θ · s_trial  →  s_trial = dev(σ_new) / θ
        s_trial = self._dev(stress) / theta
        beta = 9.0 * mu ** 2 * sigma_y / ((3.0 * mu + self.H) * sigma_vm_trial ** 3)

        I_vol = self._I_vol()
        I_dev = self._I_dev()

        return I_vol @ C + theta * (I_dev @ C) - beta * anp.outer(s_trial, s_trial)


class J2IsotropicPS(MaterialModelPS):
    """J2 plasticity with isotropic hardening for plane-stress elements.

    Uses the scalar NR path (all non-stress states explicit): implements
    ``update_state`` (Δep = Δλ). Uses the autodiff return-mapping path
    (``method="auto"``).

    Parameters
    ----------
    stress_state : StressState, optional
        Must satisfy ``stress_state.is_plane_stress``.
        Defaults to ``PLANE_STRESS``.
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
    stress = Explicit(shape=NTENS, doc="Cauchy stress")
    ep = Explicit(shape=(), doc="equivalent plastic strain")

    def __init__(self, stress_state: StressState = PLANE_STRESS, *,
                 E: float, nu: float, sigma_y0: float, H: float):
        super().__init__(stress_state)
        self.E = E
        self.nu = nu
        self.sigma_y0 = sigma_y0
        self.H = H

    def yield_function(self, state) -> anp.ndarray:
        """J2 yield function f = σ_vm − (σ_y0 + H · ep)."""
        sigma_y = self.sigma_y0 + self.H * state["ep"]
        return self._vonmises(state["stress"]) - sigma_y

    def update_state(self, dlambda, state_n, state_trial) -> list:
        """Δep = Δλ (von Mises associative flow)."""
        sig = self.default_stress_update(dlambda, state_n, state_trial)
        return [self.stress(sig), self.ep(state_n["ep"] + dlambda)]


class J2Isotropic1D(MaterialModel1D):
    """J2 plasticity with isotropic hardening for uniaxial (1D) elements.

    Uses the scalar NR path (all non-stress states explicit): implements
    ``update_state`` (Δep = Δλ). Provides closed-form ``user_defined_return_mapping``
    and ``user_defined_tangent`` using the 1D radial return mapping.

    Parameters
    ----------
    stress_state : StressState, optional
        Must have ``ntens == 1``.  Defaults to ``UNIAXIAL_1D``.
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
    stress = Explicit(shape=NTENS, doc="Cauchy stress")
    ep = Explicit(shape=(), doc="equivalent plastic strain")

    def __init__(self, stress_state: StressState = UNIAXIAL_1D, *,
                 E: float, nu: float, sigma_y0: float, H: float):
        super().__init__(stress_state)
        self.E = E
        self.nu = nu
        self.sigma_y0 = sigma_y0
        self.H = H

    def yield_function(self, state) -> anp.ndarray:
        """J2 yield function f = σ_vm − (σ_y0 + H · ep)."""
        sigma_y = self.sigma_y0 + self.H * state["ep"]
        return self._vonmises(state["stress"]) - sigma_y

    def update_state(self, dlambda, state_n, state_trial) -> list:
        """Δep = Δλ (von Mises associative flow)."""
        sig = self.default_stress_update(dlambda, state_n, state_trial)
        return [self.stress(sig), self.ep(state_n["ep"] + dlambda)]

    # ------------------------------------------------------------------
    # Analytical solver hooks
    # ------------------------------------------------------------------

    def user_defined_return_mapping(self, stress_trial, C, state_n):
        """1D J2 radial return — closed-form.

        Δλ = (|σ_trial| − σ_y) / (E + H)
        σ_new = σ_trial − E · Δλ · sign(σ_trial)

        Parameters
        ----------
        stress_trial : anp.ndarray, shape (1,)
        C : anp.ndarray, shape (1, 1)
        state_n : dict with key ``ep``

        Returns
        -------
        ReturnMappingResult
        """
        E = C[0, 0]
        ep_n = state_n["ep"]
        sigma_y = self.sigma_y0 + self.H * ep_n
        sigma_vm_trial = anp.abs(stress_trial[0])
        f_trial = sigma_vm_trial - sigma_y  # > 0 (caller ensures plasticity)
        dlambda = f_trial / (E + self.H)
        n = stress_trial / sigma_vm_trial  # sign(σ_trial) as length-1 array
        stress_new = stress_trial - E * dlambda * n
        return ReturnMappingResult(
            stress=stress_new,
            state={"stress": stress_new, "ep": ep_n + dlambda},
            dlambda=anp.asarray(dlambda),
            n_iterations=1,
            residual_history=[float(f_trial), 0.0],
        )

    def user_defined_tangent(self, stress, state, dlambda, C, state_n):
        """1D consistent tangent D^ep = [[E · H / (E + H)]].

        Parameters
        ----------
        stress : anp.ndarray, shape (1,)
        state : dict  (unused)
        dlambda : anp.ndarray, scalar
        C : anp.ndarray, shape (1, 1)
        state_n : dict  (unused)

        Returns
        -------
        anp.ndarray, shape (1, 1)
        """
        E = C[0, 0]
        return anp.array([[E * self.H / (E + self.H)]])
