"""Result dataclasses for constitutive integration.

Terminology
-----------
*Return mapping*
    The plastic correction algorithm that projects the elastic trial stress
    back onto the yield surface, producing a converged stress σ_{n+1}, updated
    internal state, and plastic multiplier Δλ.

*Stress update*
    The complete constitutive integration procedure for one load increment:
    elastic trial → yield check → return mapping → consistent tangent.
    Corresponds to what ABAQUS calls from a UMAT subroutine.
"""

from dataclasses import dataclass, field

import autograd.numpy as anp


@dataclass
class ReturnMappingResult:
    """Result of a single return-mapping (plastic correction) step.

    Contains only the output of the plastic correction algorithm — tangent
    and trial stress are part of :class:`StressUpdateResult`.

    Attributes
    ----------
    stress : anp.ndarray, shape (ntens,)
        Converged stress σ_{n+1}.
    state : dict
        Converged internal state at step n+1.
    dlambda : anp.ndarray, scalar
        Plastic multiplier increment Δλ.
    n_iterations : int
        Number of Newton updates performed.  For ``numerical_newton`` plastic
        steps this equals the number of NR updates (J2 linear hardening
        converges in exactly 1).  For closed-form ``user_defined`` correctors
        the recommended convention is ``1`` (one closed-form solve).
    residual_history : list[float]
        Residual norms.  The convention for both paths is
        ``len(residual_history) == n_iterations + 1``: initial residual at
        index 0, then one entry per update, with the final entry being the
        converged residual (< ``tol`` for ``numerical_newton``, 0.0 for
        closed-form ``user_defined``).  Closed-form implementations should
        therefore supply ``[float(f_trial), 0.0]``.
    """

    stress: anp.ndarray
    state: dict
    dlambda: anp.ndarray
    n_iterations: int = 0
    residual_history: list = field(default_factory=list)
    converged: bool = True


@dataclass
class StressUpdateResult:
    """Result of a complete stress update (constitutive integration) step.

    Wraps :class:`ReturnMappingResult` and adds the consistent tangent,
    trial stress, and plasticity flag.

    Attributes
    ----------
    return_mapping : ReturnMappingResult or None
        Plastic correction result.  ``None`` for elastic steps (no plastic
        correction was performed).
    ddsdde : anp.ndarray, shape (ntens, ntens)
        Consistent tangent operator dσ_{n+1}/dΔε.
    stress_trial : anp.ndarray, shape (ntens,)
        Elastic trial stress σ_trial = σ_n + C Δε.
    is_plastic : bool or None
        True if the step activated plasticity.  ``None`` when the information
        is not available (e.g. results produced by :class:`FortranIntegrator`
        where the UMAT does not expose a plasticity flag).
    _state_n : dict
        Internal state at step n (stored for the ``state`` convenience
        property on elastic steps).  Not part of the public API.
    """

    return_mapping: "ReturnMappingResult | None"
    ddsdde: anp.ndarray
    stress_trial: "anp.ndarray | None"
    is_plastic: "bool | None"
    _state_n: dict = field(repr=False)

    @property
    def stress(self) -> anp.ndarray:
        """Converged stress σ_{n+1} (trial stress for elastic steps)."""
        if self.return_mapping is None:
            return self.stress_trial
        return self.return_mapping.stress

    @property
    def state(self) -> dict:
        """Converged internal state (unchanged state_n for elastic steps)."""
        if self.return_mapping is None:
            return self._state_n
        return self.return_mapping.state

    @property
    def dlambda(self) -> anp.ndarray:
        """Plastic multiplier increment (0 for elastic steps)."""
        if self.return_mapping is None:
            return anp.array(0.0)
        return self.return_mapping.dlambda

    @property
    def n_iterations(self) -> int:
        """Newton updates performed (0 for elastic or user_defined steps)."""
        if self.return_mapping is None:
            return 0
        return self.return_mapping.n_iterations

    @property
    def residual_history(self) -> list:
        """Residual norms from the framework NR solver (empty for elastic / user_defined)."""
        if self.return_mapping is None:
            return []
        return self.return_mapping.residual_history

    @property
    def converged(self) -> bool:
        """True if the return mapping converged (always True for elastic steps)."""
        if self.return_mapping is None:
            return True
        return self.return_mapping.converged
