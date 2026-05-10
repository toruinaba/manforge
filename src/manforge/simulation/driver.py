"""Simulation drivers: strain-controlled, stress-controlled, and mixed loading.

All drivers share the same interface via :class:`DriverBase`:

* Input  — the driver holds a :class:`~manforge.simulation.integrator.StressIntegrator`
  (passed at construction) and receives a
  :class:`~manforge.simulation.types.FieldHistory` containing the prescribed
  loading history (strain or stress) at each :meth:`run` / :meth:`iter_run` call.
* Output — :class:`~manforge.simulation.types.DriverResult` containing
  stress and strain at every step, plus any explicitly requested state
  variables.

Conventions
-----------
- Stress and strain arrays use the engineering-shear Voigt convention:
    σ, ε = [11, 22, 33, 12, 13, 23]
- *Cumulative* quantities are the input; increments are computed internally.
- Drivers require a :class:`~manforge.simulation.integrator.StressIntegrator`
  at construction time.  Wrap bare ``MaterialModel`` objects with
  :class:`~manforge.simulation.integrator.PythonIntegrator` (auto solver),
  :class:`~manforge.simulation.integrator.PythonNumericalIntegrator`
  (numerical Newton-Raphson), or
  :class:`~manforge.simulation.integrator.PythonAnalyticalIntegrator`
  (analytical / user-defined) before passing to a driver.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterator

import numpy as np

from manforge.simulation.types import DriverResult, DriverStep, FieldHistory, FieldType


def _check_integrator(obj) -> None:
    """Raise TypeError when obj is not a StressIntegrator."""
    if not (hasattr(obj, "stress_update") and callable(obj.stress_update)):
        raise TypeError(
            f"Driver expects a StressIntegrator, got {type(obj).__name__!r}.  "
            "Wrap a bare MaterialModel with PythonIntegrator(model) first."
        )


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------

class DriverBase(ABC):
    """Abstract base for all simulation drivers.

    Parameters
    ----------
    integrator : StressIntegrator
        Constitutive integrator — use
        :class:`~manforge.simulation.integrator.PythonIntegrator`,
        :class:`~manforge.simulation.integrator.PythonNumericalIntegrator`,
        :class:`~manforge.simulation.integrator.PythonAnalyticalIntegrator`, or
        :class:`~manforge.simulation.integrator.FortranIntegrator`.
    """

    def __init__(self, integrator) -> None:
        _check_integrator(integrator)
        self.integrator = integrator

    @abstractmethod
    def iter_run(
        self,
        load: FieldHistory,
    ) -> Iterator[DriverStep]:
        """Yield per-step results as a generator.

        Parameters
        ----------
        load : FieldHistory
            Loading history (same requirements as :meth:`run`).

        Yields
        ------
        DriverStep
            Snapshot after each step: cumulative strain, full
            :class:`~manforge.core.result.StressUpdateResult`, and
            (for :class:`StressDriver`) outer-NR diagnostics.

        Examples
        --------
        Break on plasticity onset::

            driver = StrainDriver(PythonIntegrator(model))
            for step in driver.iter_run(load):
                if step.result.is_plastic:
                    print(f"Plasticity at step {step.i}")
                    break
        """

    def run(
        self,
        load: FieldHistory,
        *,
        collect_state: dict[str, FieldType] | None = None,
        initial_stress=None,
        initial_state=None,
    ) -> DriverResult:
        """Run the loading simulation.

        Parameters
        ----------
        load : FieldHistory
            Loading history.  The ``type`` and shape of ``load.data`` must
            match the driver's expectations (see subclass documentation).
        collect_state : dict[str, FieldType] or None, optional
            Explicitly request state-variable histories to be included in the
            result.  Keys are model state keys (e.g. ``"ep"``); values specify
            the :class:`~manforge.simulation.types.FieldType` to assign to the
            resulting :class:`~manforge.simulation.types.FieldHistory`.

            If ``None`` (default), no state variables are collected and
            ``DriverResult.fields`` contains only ``"Stress"`` and ``"Strain"``.
        initial_stress : array-like or None, optional
            Stress tensor at the start of the loading history.  If ``None``
            (default), the initial stress is zero.
        initial_state : dict or None, optional
            State dict at the start of the loading history.  If ``None``
            (default), the integrator's initial state is used.

        Returns
        -------
        DriverResult
        """
        step_results = []
        strain_rows = []
        for step in self.iter_run(load, initial_stress=initial_stress, initial_state=initial_state):
            step_results.append(step.result)
            strain_rows.append(step.strain)
        strain_out = (
            np.stack(strain_rows)
            if strain_rows
            else np.zeros((0, self.integrator.ntens))
        )
        return DriverResult(
            step_results=step_results,
            strain=strain_out,
            collect_state=collect_state,
        )


# ---------------------------------------------------------------------------
# Strain-driven driver
# ---------------------------------------------------------------------------

class StrainDriver(DriverBase):
    """Strain-controlled loading driver.

    Accepts both uniaxial (1-D) and general multi-component (2-D) strain
    histories in a single class.

    Parameters
    ----------
    integrator : StressIntegrator
        Constitutive integrator.
    """

    def iter_run(
        self,
        load: FieldHistory,
        *,
        initial_stress=None,
        initial_state=None,
    ) -> Iterator[DriverStep]:
        """Yield per-step results for the strain-controlled loading history.

        Parameters
        ----------
        load : FieldHistory
            Must have ``type = FieldType.STRAIN``.  ``load.data`` shape:

            * ``(N,)``       — uniaxial: only ε11 varies, lateral strains zero.
            * ``(N, ntens)`` — general: all components prescribed.
        initial_stress : array-like or None, optional
            Stress tensor at the start of the loading history.  If ``None``
            (default), the initial stress is zero.
        initial_state : dict or None, optional
            State dict at the start of the loading history.  If ``None``
            (default), the integrator's initial state is used.

        Yields
        ------
        DriverStep
        """
        integrator = self.integrator
        data = np.asarray(load.data, dtype=float)
        uniaxial = data.ndim == 1
        ntens = integrator.ntens

        stress_n = (np.zeros(ntens) if initial_stress is None
                    else np.array(initial_stress, dtype=float))
        state_n = (integrator.initial_state() if initial_state is None
                   else {k: np.asarray(v) for k, v in initial_state.items()})

        for i in range(len(data)):
            if uniaxial:
                deps11 = data[i] - (data[i - 1] if i > 0 else 0.0)
                strain_inc = np.zeros(ntens)
                strain_inc[0] = deps11
                strain_cum = np.zeros(ntens)
                strain_cum[0] = data[i]
            else:
                prev = data[i - 1] if i > 0 else np.zeros(ntens)
                strain_inc = np.array(data[i] - prev)
                strain_cum = np.array(data[i])

            rm = integrator.stress_update(strain_inc, stress_n, state_n)
            stress_n = rm.stress
            state_n = rm.state
            yield DriverStep(i=i, strain=strain_cum.copy(), result=rm)


# ---------------------------------------------------------------------------
# Stress-driven driver
# ---------------------------------------------------------------------------

class StressDriver(DriverBase):
    """Stress-controlled loading driver.

    Prescribes a target stress history and solves for the corresponding
    strain increments using Newton-Raphson iteration with the consistent
    tangent (ddsdde).  Useful for simulating uniaxial stress loading in a
    multi-axial model (e.g. σ11 ramping, all other components zero) where
    the lateral strains adjust freely.

    Parameters
    ----------
    integrator : StressIntegrator
        Constitutive integrator.
    max_iter : int, optional
        Maximum Newton-Raphson iterations per step (default 20).
    tol : float, optional
        Convergence tolerance on the infinity norm of the stress residual
        (default 1e-8).
    """

    def __init__(self, integrator, *, max_iter: int = 20, tol: float = 1e-8):
        super().__init__(integrator)
        self.max_iter = max_iter
        self.tol = tol

    def iter_run(
        self,
        load: FieldHistory,
        *,
        raise_on_nonconverged: bool = True,
        initial_stress=None,
        initial_state=None,
    ) -> Iterator[DriverStep]:
        """Yield per-step results for the stress-controlled loading history.

        Parameters
        ----------
        load : FieldHistory
            Must have ``type = FieldType.STRESS`` and
            ``load.data`` shape ``(N, ntens)`` — cumulative target stress
            tensor (Voigt) at each step.
        raise_on_nonconverged : bool, optional
            If ``True`` (default), raise :exc:`RuntimeError` when Newton-Raphson
            does not converge.  If ``False``, yield a :class:`DriverStep` with
            ``converged=False`` instead; internal state is *not* advanced for
            that step, so the caller should ``break`` immediately.
        initial_stress : array-like or None, optional
            Stress tensor at the start of the loading history.  If ``None``
            (default), the initial stress is zero.  Note: ``strain`` in the
            yielded :class:`~manforge.simulation.types.DriverStep` represents
            the cumulative strain increment from this prestressed state, not
            from the undeformed configuration.
        initial_state : dict or None, optional
            State dict at the start of the loading history.  If ``None``
            (default), the integrator's initial state is used.

        Yields
        ------
        DriverStep
            Includes ``n_outer_iter`` and ``residual_inf`` for NR diagnostics.

        Raises
        ------
        RuntimeError
            If NR does not converge and ``raise_on_nonconverged=True``.
        """
        integrator = self.integrator
        stress_history = np.asarray(load.data, dtype=float)
        ntens = integrator.ntens

        stress_n = (np.zeros(ntens) if initial_stress is None
                    else np.array(initial_stress, dtype=float))
        state_n = (integrator.initial_state() if initial_state is None
                   else {k: np.asarray(v) for k, v in initial_state.items()})
        eps_total = np.zeros(ntens)

        for i in range(stress_history.shape[0]):
            sigma_target = np.array(stress_history[i])
            if hasattr(integrator, "elastic_stiffness"):
                C = integrator.elastic_stiffness(state_n)
                S = np.linalg.inv(np.array(C))
            else:
                rm0 = integrator.stress_update(np.zeros(ntens), stress_n, state_n)
                S = np.linalg.inv(np.array(rm0.ddsdde))
            deps = S @ (sigma_target - stress_n)

            converged = False
            residual = np.full(ntens, np.inf)
            rm = None
            k = 0
            for k in range(self.max_iter):
                rm = integrator.stress_update(deps, stress_n, state_n)
                residual = sigma_target - np.array(rm.stress)
                if float(np.max(np.abs(residual))) < self.tol:
                    converged = True
                    break
                deps = deps + np.linalg.solve(np.array(rm.ddsdde), residual)

            residual_inf = float(np.max(np.abs(residual)))

            if not converged:
                if raise_on_nonconverged:
                    raise RuntimeError(
                        f"StressDriver: NR did not converge at step {i} "
                        f"(||residual||_inf = {residual_inf:.3e}, "
                        f"tol = {self.tol:.3e})"
                    )
                yield DriverStep(
                    i=i,
                    strain=eps_total.copy(),
                    result=rm,
                    converged=False,
                    n_outer_iter=k + 1,
                    residual_inf=residual_inf,
                )
                return

            eps_total = eps_total + np.array(deps)
            stress_n = rm.stress
            state_n = rm.state
            yield DriverStep(
                i=i,
                strain=eps_total.copy(),
                result=rm,
                converged=True,
                n_outer_iter=k + 1,
                residual_inf=residual_inf,
            )

    def run(
        self,
        load: FieldHistory,
        *,
        collect_state: dict[str, FieldType] | None = None,
        initial_stress=None,
        initial_state=None,
    ) -> DriverResult:
        """Run the stress-controlled loading history.

        Parameters
        ----------
        load : FieldHistory
            Must have ``type = FieldType.STRESS`` and
            ``load.data`` shape ``(N, ntens)`` — cumulative target stress
            tensor (Voigt) at each step.
        collect_state : dict[str, FieldType] or None, optional
            State variables to include in the result.
        initial_stress : array-like or None, optional
            Stress tensor at the start of the loading history (default zero).
        initial_state : dict or None, optional
            State dict at the start of the loading history (default integrator initial state).

        Returns
        -------
        DriverResult

        Raises
        ------
        RuntimeError
            If Newton-Raphson does not converge within ``max_iter`` iterations
            at any step.
        """
        step_results = []
        strain_rows = []
        for step in self.iter_run(load, raise_on_nonconverged=True,
                                  initial_stress=initial_stress, initial_state=initial_state):
            step_results.append(step.result)
            strain_rows.append(step.strain)
        strain_out = (
            np.stack(strain_rows)
            if strain_rows
            else np.zeros((0, self.integrator.ntens))
        )
        return DriverResult(
            step_results=step_results,
            strain=strain_out,
            collect_state=collect_state,
        )


# ---------------------------------------------------------------------------
# Mixed boundary-condition driver
# ---------------------------------------------------------------------------

def _validate_idx_arg(name: str, seq, ntens: int) -> list[int]:
    """Validate a Voigt index sequence; return list in original order."""
    try:
        lst = list(seq)
    except TypeError:
        raise TypeError(f"{name} must be iterable, got {type(seq).__name__!r}")
    for v in lst:
        if not isinstance(v, (int, np.integer)):
            raise TypeError(f"All elements of {name} must be int, got {type(v).__name__!r}")
    ints = [int(v) for v in lst]
    if len(ints) != len(set(ints)):
        raise ValueError(f"{name} contains duplicate indices: {ints}")
    for v in ints:
        if not (0 <= v < ntens):
            raise ValueError(
                f"{name} index {v} is out of range for ntens={ntens}"
            )
    return ints


class MixedDriver(DriverBase):
    """Mixed strain/stress boundary-condition driver.

    Some Voigt components are strain-controlled (prescribed history) while the
    remaining components are stress-controlled (target stress, default zero).
    The driver solves the free strain components by inner Newton-Raphson on the
    consistent tangent (ddsdde[F, F]).

    Typical use case: 3-D solid (ntens=6) with ε11 ramped and σ22..σ23 = 0,
    recovering a uniaxial-stress test from a 3-D model without writing a 1-D
    wrapper.

    Parameters
    ----------
    integrator : StressIntegrator
        Constitutive integrator.
    prescribed_strain_idx : Sequence[int]
        Voigt indices whose strain is prescribed by the caller (P components).
        Must not be empty.  The order of indices determines the column order
        expected in ``load.data``: ``load.data[:, k]`` is the history for
        ``prescribed_strain_idx[k]``.
    prescribed_stress_idx : Sequence[int] or None, optional
        Voigt indices whose stress is prescribed (F components).  If ``None``
        (default), inferred as the complement of ``prescribed_strain_idx`` in
        ascending index order.  When provided explicitly, the order determines
        the column order of ``prescribed_stress_history``.
    max_iter : int, optional
        Maximum inner Newton-Raphson iterations per step (default 20).
    tol : float, optional
        Convergence tolerance on the L∞ norm of the stress residual on the
        F components (default 1e-8).

    Notes
    -----
    Voigt order is ``[11, 22, 33, 12, 13, 23]`` (engineering shear).

    The union of ``prescribed_strain_idx`` and ``prescribed_stress_idx`` must
    equal ``range(ntens)`` exactly; the intersection must be empty.

    Index order is preserved: ``load.data[:, k]`` maps to
    ``prescribed_strain_idx[k]``, and ``prescribed_stress_history[:, k]``
    maps to ``prescribed_stress_idx[k]``.

    When ``len(prescribed_stress_idx) == 0`` (full strain control), the inner
    NR is skipped and the driver behaves identically to :class:`StrainDriver`.

    If ``D_FF = ddsdde[F, F]`` becomes singular (rare for J2-associative
    models), ``numpy.linalg.LinAlgError`` propagates to the caller.
    """

    def __init__(
        self,
        integrator,
        *,
        prescribed_strain_idx,
        prescribed_stress_idx=None,
        max_iter: int = 20,
        tol: float = 1e-8,
    ) -> None:
        super().__init__(integrator)
        ntens = integrator.ntens

        P = _validate_idx_arg("prescribed_strain_idx", prescribed_strain_idx, ntens)
        if len(P) == 0:
            raise ValueError(
                "prescribed_strain_idx must not be empty; use StressDriver instead"
            )

        if prescribed_stress_idx is None:
            F = sorted(set(range(ntens)) - set(P))
        else:
            F = _validate_idx_arg("prescribed_stress_idx", prescribed_stress_idx, ntens)

        if set(P) | set(F) != set(range(ntens)):
            raise ValueError(
                f"Union of prescribed_strain_idx and prescribed_stress_idx must cover all "
                f"{ntens} components.  Got P={P}, F={F}."
            )
        if set(P) & set(F):
            raise ValueError(
                "prescribed_strain_idx and prescribed_stress_idx must be disjoint.  "
                f"Overlap: {sorted(set(P) & set(F))}"
            )

        self._P = np.array(P, dtype=int)
        self._F = np.array(F, dtype=int)
        self.max_iter = max_iter
        self.tol = tol

    def iter_run(
        self,
        load: FieldHistory,
        *,
        prescribed_stress_history=None,
        raise_on_nonconverged: bool = True,
        initial_stress=None,
        initial_state=None,
    ) -> Iterator[DriverStep]:
        """Yield per-step results for mixed strain/stress boundary conditions.

        Parameters
        ----------
        load : FieldHistory
            Must have ``type = FieldType.STRAIN``.  ``load.data`` shape
            ``(N, len(prescribed_strain_idx))`` — cumulative prescribed strain
            at each step for the P components.
        prescribed_stress_history : array-like of shape (N, len(prescribed_stress_idx)) or None
            Target stress values for the F components at each step.  If ``None``
            (default), all F components are held at zero.
        raise_on_nonconverged : bool, optional
            If ``True`` (default), raise :exc:`RuntimeError` when inner NR does
            not converge.  If ``False``, yield a :class:`DriverStep` with
            ``converged=False`` and immediately stop iteration.
        initial_stress : array-like or None, optional
            Full stress tensor (ntens,) at the start of the history.
        initial_state : dict or None, optional
            State dict at the start of the history.

        Yields
        ------
        DriverStep
            Includes ``n_outer_iter`` and ``residual_inf`` for NR diagnostics.
        """
        integrator = self.integrator
        ntens = integrator.ntens
        P, F = self._P, self._F
        nP, nF = len(P), len(F)

        if load.type != FieldType.STRAIN:
            raise ValueError(
                f"MixedDriver expects load.type=FieldType.STRAIN, got {load.type!r}"
            )
        eps_P_hist = np.asarray(load.data, dtype=float)
        if eps_P_hist.ndim != 2 or eps_P_hist.shape[1] != nP:
            raise ValueError(
                f"load.data must have shape (N, {nP}), got {eps_P_hist.shape}"
            )
        N = eps_P_hist.shape[0]

        if prescribed_stress_history is None:
            if nF == 0:
                sigF_hist = np.zeros((N, 0))
            else:
                sigF_hist = np.zeros((N, nF))
        else:
            if nF == 0:
                raise ValueError(
                    "prescribed_stress_history provided but there are no "
                    "stress-prescribed components (nF=0)"
                )
            sigF_hist = np.asarray(prescribed_stress_history, dtype=float)
            if sigF_hist.shape != (N, nF):
                raise ValueError(
                    f"prescribed_stress_history must have shape ({N}, {nF}), "
                    f"got {sigF_hist.shape}"
                )

        stress_n = (
            np.zeros(ntens) if initial_stress is None
            else np.array(initial_stress, dtype=float)
        )
        state_n = (
            integrator.initial_state() if initial_state is None
            else {k: np.asarray(v) for k, v in initial_state.items()}
        )
        eps_total = np.zeros(ntens)

        for i in range(N):
            deps = np.zeros(ntens)
            deps[P] = eps_P_hist[i] - eps_total[P]

            if nF == 0:
                rm = integrator.stress_update(deps, stress_n, state_n)
                converged = True
                k = 0
                residual_inf = 0.0
            else:
                sigma_F_target = sigF_hist[i]

                # Initial estimate: solve C0[F,F] @ deps[F] = rhs from elastic stiffness
                if hasattr(integrator, "elastic_stiffness"):
                    C0 = np.array(integrator.elastic_stiffness(state_n))
                else:
                    rm0 = integrator.stress_update(np.zeros(ntens), stress_n, state_n)
                    C0 = np.array(rm0.ddsdde)
                C0_FF = C0[np.ix_(F, F)]
                C0_FP = C0[np.ix_(F, P)]
                rhs0 = (sigma_F_target - stress_n[F]) - C0_FP @ deps[P]
                deps[F] = np.linalg.solve(C0_FF, rhs0)

                converged = False
                residual_F = np.full(nF, np.inf)
                rm = None
                k = 0
                for k in range(self.max_iter):
                    rm = integrator.stress_update(deps, stress_n, state_n)
                    residual_F = sigma_F_target - np.array(rm.stress)[F]
                    if float(np.max(np.abs(residual_F))) < self.tol:
                        converged = True
                        break
                    D_FF = np.array(rm.ddsdde)[np.ix_(F, F)]
                    deps[F] = deps[F] + np.linalg.solve(D_FF, residual_F)

                residual_inf = float(np.max(np.abs(residual_F)))

            if not converged:
                if raise_on_nonconverged:
                    raise RuntimeError(
                        f"MixedDriver: NR did not converge at step {i} "
                        f"(||residual_F||_inf = {residual_inf:.3e}, "
                        f"tol = {self.tol:.3e})"
                    )
                yield DriverStep(
                    i=i,
                    strain=eps_total.copy(),
                    result=rm,
                    converged=False,
                    n_outer_iter=k + 1,
                    residual_inf=residual_inf,
                )
                return

            eps_total = eps_total + deps
            stress_n = rm.stress
            state_n = rm.state
            yield DriverStep(
                i=i,
                strain=eps_total.copy(),
                result=rm,
                converged=True,
                n_outer_iter=k + 1,
                residual_inf=residual_inf,
            )

    def run(
        self,
        load: FieldHistory,
        *,
        prescribed_stress_history=None,
        collect_state: dict[str, FieldType] | None = None,
        initial_stress=None,
        initial_state=None,
    ) -> DriverResult:
        """Run the mixed strain/stress loading history.

        Parameters
        ----------
        load : FieldHistory
            Must have ``type = FieldType.STRAIN`` and shape
            ``(N, len(prescribed_strain_idx))``.
        prescribed_stress_history : array-like of shape (N, len(prescribed_stress_idx)) or None
            Target stress for the F components.  Defaults to zero.
        collect_state : dict[str, FieldType] or None, optional
            State variables to include in the result.
        initial_stress : array-like or None, optional
            Stress tensor at the start (default zero).
        initial_state : dict or None, optional
            State dict at the start (default integrator initial state).

        Returns
        -------
        DriverResult

        Raises
        ------
        RuntimeError
            If inner NR does not converge within ``max_iter`` iterations at
            any step.
        """
        step_results = []
        strain_rows = []
        for step in self.iter_run(
            load,
            prescribed_stress_history=prescribed_stress_history,
            raise_on_nonconverged=True,
            initial_stress=initial_stress,
            initial_state=initial_state,
        ):
            step_results.append(step.result)
            strain_rows.append(step.strain)
        strain_out = (
            np.stack(strain_rows)
            if strain_rows
            else np.zeros((0, self.integrator.ntens))
        )
        return DriverResult(
            step_results=step_results,
            strain=strain_out,
            collect_state=collect_state,
        )
