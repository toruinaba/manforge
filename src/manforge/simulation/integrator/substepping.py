"""SubSteppingIntegrator — adaptive subdivision of one strain increment."""

from __future__ import annotations

import types

import numpy as np

from manforge.core.dimension import StressDimension
from manforge.core.result import ReturnMappingResult, StressUpdateResult
from manforge._typing import StressVec, StateDict


class SubSteppingIntegrator:
    """Wraps another integrator and retries failed increments in substeps.

    This is the Python counterpart of subdivision *inside* a UMAT call, not of
    ABAQUS' ``PNEWDT``.  ``PNEWDT`` shrinks the global increment, so one
    yielding integration point slows the whole analysis down; here only the
    increment that actually failed is subdivided, and only within the single
    :meth:`stress_update` call that failed.  That is why the subdivision lives
    at the integrator layer rather than in a driver: a driver-level loop is the
    ``PNEWDT`` behaviour, and ``StressDriver``/``MixedDriver`` are still solving
    for Δε in their outer NR when they call the integrator, so at that point
    there is nothing well-defined to subdivide yet.

    Delegating to the wrapped ``stress_update`` (rather than to its return
    mapping) also gives every substep its own ``elastic_check`` and its own
    σ_trial.  An increment that crosses the yield point would otherwise push
    its elastic portion through the plastic corrector.

    The wrapper is opt-in and transparent: the first attempt is always a single
    undivided call to the wrapped integrator, and its result is returned
    unchanged, so any increment that already worked keeps its exact previous
    stress, iteration count, and tangent.

    Parameters
    ----------
    integrator : StressIntegrator
        Integrator to wrap.
    max_substeps : int, optional
        Largest number of substeps to try (default 16).  The increment is
        retried at 1, 2, 4, 8, ... substeps up to this value.
    raise_on_nonconverged : bool, optional
        Behaviour once ``max_substeps`` has been exhausted (default ``True``):
        raise ``RuntimeError``, or return a result with ``converged=False``.

    Notes
    -----
    Failure of the wrapped integrator is detected both from a raised
    ``RuntimeError`` and from a returned ``converged=False``, so the wrapped
    integrator works whichever way *its* ``raise_on_nonconverged`` is set.
    """

    def __init__(
        self,
        integrator,
        *,
        max_substeps: int = 16,
        raise_on_nonconverged: bool = True,
    ) -> None:
        if max_substeps < 1:
            raise ValueError(f"max_substeps must be >= 1, got {max_substeps}")
        self._integrator = integrator
        self._max_substeps = max_substeps
        self._raise_on_nonconverged = raise_on_nonconverged

        # Mirror the wrapped integrator's surface: FortranIntegrator only has
        # elastic_stiffness when a stiffness hook was supplied, and StressDriver
        # probes for it with hasattr.
        if hasattr(integrator, "elastic_stiffness"):
            self.elastic_stiffness = types.MethodType(
                lambda self_, state=None: self_._integrator.elastic_stiffness(state),
                self,
            )

    @property
    def dimension(self) -> StressDimension:
        return self._integrator.dimension

    @property
    def ntens(self) -> int:
        return self._integrator.ntens

    def initial_state(self) -> StateDict:
        return self._integrator.initial_state()

    def _attempt(
        self, strain_inc: StressVec, stress_n: StressVec, state_n: StateDict, m: int
    ) -> "StressUpdateResult | None":
        """Integrate ``strain_inc`` in *m* equal substeps, or None if any failed."""
        deps = np.asarray(strain_inc, dtype=float) / m

        stress_cur = stress_n
        state_cur = state_n
        n_iterations = 0
        residual_history: list[float] = []
        dlambda_total = 0.0
        any_plastic = False
        last: "StressUpdateResult | None" = None

        for _ in range(m):
            # LinAlgError too: a singular NR Jacobian at a large increment is
            # exactly the failure mode subdivision recovers from.
            try:
                res = self._integrator.stress_update(deps, stress_cur, state_cur)
            except (RuntimeError, np.linalg.LinAlgError):
                return None
            if not res.converged:
                return None

            n_iterations += res.n_iterations
            residual_history.extend(res.residual_history)
            dlambda_total = dlambda_total + float(res.dlambda)
            any_plastic = any_plastic or bool(res.is_plastic)
            stress_cur = res.stress
            state_cur = res.state
            last = res

        assert last is not None  # m >= 1, so the loop body always runs

        if m == 1:
            return last

        # A subdivided step reports the state it actually reached, so it always
        # carries a ReturnMappingResult — even when every substep was elastic.
        # The elastic shortcut (return_mapping=None) rebuilds the final state as
        # state_n plus σ_trial, which only holds for an undivided increment.
        rm = ReturnMappingResult(
            stress=stress_cur,
            state=state_cur,
            dlambda=np.array(dlambda_total),
            n_iterations=n_iterations,
            residual_history=residual_history,
            converged=True,
        )
        return StressUpdateResult(
            return_mapping=rm,
            # Approximation: the last substep's consistent tangent, not the
            # chain-rule product over substeps.  dσ/dΔε for the whole increment
            # is therefore slightly wrong, which can mildly slow the outer NR in
            # StressDriver/MixedDriver.  Composing it exactly would require new
            # hand-derived terms per model; that trade was taken deliberately.
            # Thunked so the wrapped result's own lazy tangent stays deferred.
            ddsdde=lambda: last.ddsdde,
            stress_trial=self._big_trial(strain_inc, stress_n, state_n),
            is_plastic=any_plastic,
            _state_n=state_n,
            n_substeps=m,
        )

    def _big_trial(
        self, strain_inc: StressVec, stress_n: StressVec, state_n: StateDict
    ) -> "StressVec | None":
        """σ_trial of the *undivided* increment — what the caller asked about."""
        if not hasattr(self, "elastic_stiffness"):
            return None
        C_n = np.asarray(self.elastic_stiffness(state_n), dtype=float)
        return np.asarray(stress_n, dtype=float) + C_n @ np.asarray(strain_inc, dtype=float)

    def stress_update(
        self, strain_inc: StressVec, stress_n: StressVec, state_n: StateDict
    ) -> StressUpdateResult:
        """Integrate one increment, subdividing it if the wrapped attempt fails."""
        m = 1
        while m <= self._max_substeps:
            result = self._attempt(strain_inc, stress_n, state_n, m)
            if result is not None:
                return result
            m *= 2

        if self._raise_on_nonconverged:
            raise RuntimeError(
                f"SubSteppingIntegrator: {type(self._integrator).__name__} did not "
                f"converge for this increment even at {self._max_substeps} substeps "
                f"(||Δε||_2 = "
                f"{float(np.linalg.norm(np.asarray(strain_inc, dtype=float))):.3e})"
            )

        # No substep count produced a usable state, so σ and the state hold at
        # step n.  The caller must gate on `converged` before using them.
        rm = ReturnMappingResult(
            stress=stress_n,
            state=state_n,
            dlambda=np.array(0.0),
            n_iterations=0,
            residual_history=[],
            converged=False,
        )
        # elastic_stiffness exists only when the wrapped integrator has one
        # (mirrored in __init__); without it there is no meaningful tangent
        # for a failed step, so fall back to identity as a typed placeholder.
        ddsdde = (
            self.elastic_stiffness(state_n)
            if hasattr(self, "elastic_stiffness")
            else np.eye(self.ntens)
        )
        return StressUpdateResult(
            return_mapping=rm,
            ddsdde=ddsdde,
            stress_trial=self._big_trial(strain_inc, stress_n, state_n),
            is_plastic=None,
            _state_n=state_n,
            n_substeps=self._max_substeps,
        )
