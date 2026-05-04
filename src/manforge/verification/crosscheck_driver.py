"""Multi-step crosscheck drivers for constitutive model validation.

:class:`CrosscheckStrainDriver` and :class:`CrosscheckStressDriver` are
:class:`~manforge.verification.Comparator` subclasses that drive two
integrators through the same loading sequence and compare stress / state /
tangent at every step.  Each class corresponds directly to its simulation
counterpart:

* :class:`~manforge.simulation.StrainDriver` →
  :class:`CrosscheckStrainDriver` (strain-controlled loading)
* :class:`~manforge.simulation.StressDriver` →
  :class:`CrosscheckStressDriver` (stress-controlled loading,
  exposes ``max_iter`` / ``tol`` directly on the constructor)

Strain-controlled example
--------------------------
::

    from manforge.verification import CrosscheckStrainDriver, FortranModule
    from manforge.simulation import PythonNumericalIntegrator, FortranIntegrator
    from manforge.simulation.types import FieldHistory, FieldType
    import numpy as np

    model   = MyModel(E=210_000.0, nu=0.3, sigma_y0=250.0, H=1_000.0)
    fortran = FortranModule("my_model")
    load    = FieldHistory(FieldType.STRAIN, "eps",
                           np.linspace([0]*6, [1e-3,0,0,0,0,0], 20))

    py_int = PythonNumericalIntegrator(model)
    fc_int = FortranIntegrator.from_model(fortran, "my_model_core", model)

    cc = CrosscheckStrainDriver(py_int, fc_int)
    result = cc.run(load)
    assert result.passed
    print(f"max stress rel error = {result.max_stress_rel_err:.2e}")

Stress-controlled example
--------------------------
::

    from manforge.verification import CrosscheckStressDriver

    stress_load = FieldHistory(FieldType.STRESS, "sigma", stress_data)
    cc = CrosscheckStressDriver(py_int, fc_int, max_iter=30, tol=1e-10)
    result = cc.run(stress_load)
    assert result.passed

Single-step comparison
----------------------
Use :class:`CrosscheckStrainDriver` with a one-row
:class:`~manforge.simulation.types.FieldHistory` and the
``initial_stress`` / ``initial_state`` kwargs to compare a prestressed
single step::

    load = FieldHistory(FieldType.STRAIN, "eps", case["strain_inc"][np.newaxis])
    cc = CrosscheckStrainDriver(py_int, fc_int)
    for cr in cc.iter_run(load, initial_stress=case["stress_n"],
                          initial_state=case["state_n"]):
        assert cr.passed
        # Jacobian inspection on failure:
        # jac = compare_jacobians(model, cr.result_a, cr.result_b, cr.state_n)
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from manforge.core.stress_update import StressUpdateResult

from manforge.simulation.driver import StrainDriver, StressDriver
from manforge.simulation.types import FieldType
from manforge.verification.comparator_base import (
    CaseResult,
    ComparisonResult,
    Comparator,
    _case_passed,
    _state_rel_err,
    _stress_rel_err,
    _tangent_rel_err,
)

# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------

@dataclass
class CrosscheckCaseResult(CaseResult):
    """Per-case / per-step result from a crosscheck harness.

    Extends :class:`~manforge.verification.CaseResult` with raw Python- and
    Fortran-side outputs for inspection.

    Attributes
    ----------
    result_a : StressUpdateResult or None
        Raw result from integrator_a.  Pass to
        :func:`~manforge.verification.compare_jacobians` with ``result_b``
        and ``state_n`` to diagnose Jacobian-level differences.
    result_b : StressUpdateResult or None
        Raw result from integrator_b.
    state_n : dict or None
        State dict at the *start* of this step (integrator_a side).
        Required as the ``state_n`` argument to
        :func:`~manforge.verification.compare_jacobians`.
    """

    py_stress: np.ndarray | None = None
    py_state: dict | None = None
    py_ddsdde: np.ndarray | None = None
    py_dlambda: float | None = None
    f_stress: np.ndarray | None = None
    f_state: dict | None = None
    f_ddsdde: np.ndarray | None = None
    result_a: StressUpdateResult | None = None
    result_b: StressUpdateResult | None = None
    state_n: dict | None = None
    a_n_iterations: int = 0
    a_residual_history: list = field(default_factory=list)
    b_n_iterations: int = 0
    b_residual_history: list = field(default_factory=list)


@dataclass
class CrosscheckResult(ComparisonResult):
    """Aggregate result across all cases / steps.

    Extends :class:`~manforge.verification.ComparisonResult` with the full
    per-case list typed as :class:`CrosscheckCaseResult`.
    """

    cases: list[CrosscheckCaseResult] = field(default_factory=list)  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# _CrosscheckDriverBase
# ---------------------------------------------------------------------------

class _CrosscheckDriverBase(Comparator):
    """Shared base for CrosscheckStrainDriver and CrosscheckStressDriver.

    Subclasses implement :meth:`_make_driver` to return the appropriate
    driver type and :meth:`_iter_driver` to call its ``iter_run`` with any
    driver-specific keyword arguments.
    """

    _result_cls = CrosscheckResult
    _expected_field_type: FieldType  # set by each subclass

    def __init__(
        self,
        integrator_a,
        integrator_b,
        *,
        stress_tol: float = 1e-6,
        tangent_tol: float = 1e-5,
        state_tol: float = 1e-6,
    ) -> None:
        self.integrator_a = integrator_a
        self.integrator_b = integrator_b
        self.stress_tol = stress_tol
        self.tangent_tol = tangent_tol
        self.state_tol = state_tol

    def _make_driver(self, integrator):
        raise NotImplementedError

    def _iter_driver(self, driver, load, **kwargs):
        raise NotImplementedError

    def iter_run(
        self,
        load,
        *,
        initial_stress=None,
        initial_state=None,
        **kwargs,
    ) -> Iterator[CrosscheckCaseResult]:
        if load.type != self._expected_field_type:
            complement = (
                "CrosscheckStressDriver"
                if self._expected_field_type == FieldType.STRAIN
                else "CrosscheckStrainDriver"
            )
            raise ValueError(
                f"{type(self).__name__} expects load.type={self._expected_field_type!r}, "
                f"got {load.type!r}. Use {complement} for "
                f"{'stress' if self._expected_field_type == FieldType.STRAIN else 'strain'}"
                "-controlled loading."
            )

        da = self._make_driver(self.integrator_a)
        db = self._make_driver(self.integrator_b)

        # Copy initial_state so both drivers start from independent objects.
        init_state_a = ({k: np.asarray(v) for k, v in initial_state.items()}
                        if initial_state is not None else None)
        init_state_b = ({k: np.asarray(v) for k, v in initial_state.items()}
                        if initial_state is not None else None)
        init_stress_a = (np.array(initial_stress, dtype=float)
                         if initial_stress is not None else None)
        init_stress_b = (np.array(initial_stress, dtype=float)
                         if initial_stress is not None else None)

        # state_n tracks integrator_a's state at the *start* of each step.
        if initial_state is not None:
            step_state_n = {k: np.asarray(v) for k, v in initial_state.items()}
        else:
            step_state_n = self.integrator_a.initial_state()

        for sa, sb in zip(
            self._iter_driver(da, load, initial_stress=init_stress_a,
                              initial_state=init_state_a, **kwargs),
            self._iter_driver(db, load, initial_stress=init_stress_b,
                              initial_state=init_state_b, **kwargs),
        ):
            ra = sa.result
            rb = sb.result

            py_stress  = np.asarray(ra.stress,  dtype=np.float64)
            py_state   = {k: np.asarray(v) for k, v in ra.state.items()}
            py_ddsdde  = np.asarray(ra.ddsdde,  dtype=np.float64)
            py_dlambda = float(ra.dlambda) if ra.dlambda is not None else float("nan")

            f_stress = np.asarray(rb.stress, dtype=np.float64)
            f_state  = {k: np.asarray(v) for k, v in rb.state.items()}

            f_ddsdde = (
                np.asarray(rb.ddsdde, dtype=np.float64) if rb.ddsdde is not None else None
            )
            t_err = _tangent_rel_err(py_ddsdde, f_ddsdde)

            s_err  = _stress_rel_err(f_stress, py_stress)
            st_err = _state_rel_err(f_state, py_state)
            ok     = _case_passed(s_err, st_err, t_err, self.stress_tol, self.state_tol, self.tangent_tol)

            current_state_n = step_state_n
            step_state_n = {k: np.asarray(v) for k, v in ra.state.items()}

            yield CrosscheckCaseResult(
                index=sa.i,
                py_stress=py_stress,
                py_state=py_state,
                py_ddsdde=py_ddsdde,
                py_dlambda=py_dlambda,
                f_stress=f_stress,
                f_state=f_state,
                f_ddsdde=f_ddsdde,
                result_a=ra,
                result_b=rb,
                state_n=current_state_n,
                stress_rel_err=s_err,
                state_rel_err=st_err,
                tangent_rel_err=t_err,
                passed=ok,
                a_n_iterations=ra.n_iterations,
                a_residual_history=list(ra.residual_history),
                a_converged=ra.converged,
                b_n_iterations=rb.n_iterations,
                b_residual_history=list(rb.residual_history),
                b_converged=rb.converged,
            )


# ---------------------------------------------------------------------------
# CrosscheckStrainDriver
# ---------------------------------------------------------------------------

class CrosscheckStrainDriver(_CrosscheckDriverBase):
    """Compare two integrators over a strain-controlled loading history.

    Drives both integrators through the same multi-step sequence using
    :class:`~manforge.simulation.StrainDriver` and compares stress / state /
    tangent at every step.

    Parameters
    ----------
    integrator_a : StressIntegrator
        Reference implementation (typically
        :class:`~manforge.simulation.PythonNumericalIntegrator`).
    integrator_b : StressIntegrator
        Candidate implementation (typically
        :class:`~manforge.simulation.FortranIntegrator`).
    stress_tol : float, optional
        Pass threshold for relative stress error (default 1e-6).
    tangent_tol : float, optional
        Pass threshold for relative tangent error (default 1e-5).
    state_tol : float, optional
        Pass threshold for per state-variable relative error (default 1e-6).

    Examples
    --------
    ::

        cc = CrosscheckStrainDriver(py_int, fc_int)
        result = cc.run(strain_load)
        assert result.passed

        for cr in cc.iter_run(strain_load):
            if not cr.passed:
                print(f"step {cr.index}: stress_err={cr.stress_rel_err:.2e}")
                break
    """

    _expected_field_type = FieldType.STRAIN

    def _make_driver(self, integrator):
        return StrainDriver(integrator)

    def _iter_driver(self, driver, load, **kwargs):
        return driver.iter_run(load, **kwargs)


# ---------------------------------------------------------------------------
# CrosscheckStressDriver
# ---------------------------------------------------------------------------

class CrosscheckStressDriver(_CrosscheckDriverBase):
    """Compare two integrators over a stress-controlled loading history.

    Drives both integrators through the same multi-step sequence using
    :class:`~manforge.simulation.StressDriver` and compares stress / state /
    tangent at every step.

    Parameters
    ----------
    integrator_a : StressIntegrator
        Reference implementation.
    integrator_b : StressIntegrator
        Candidate implementation.
    stress_tol : float, optional
        Pass threshold for relative stress error (default 1e-6).
    tangent_tol : float, optional
        Pass threshold for relative tangent error (default 1e-5).
    state_tol : float, optional
        Pass threshold for per state-variable relative error (default 1e-6).
    max_iter : int, optional
        Maximum Newton iterations for :class:`~manforge.simulation.StressDriver`
        (default 20).
    tol : float, optional
        Convergence tolerance for :class:`~manforge.simulation.StressDriver`
        (default 1e-8).

    Examples
    --------
    ::

        cc = CrosscheckStressDriver(py_int, fc_int, max_iter=30, tol=1e-10)
        result = cc.run(stress_load)
        assert result.passed

        for cr in cc.iter_run(stress_load, raise_on_nonconverged=False):
            print(f"step {cr.index}: stress_err={cr.stress_rel_err:.2e}")
    """

    _expected_field_type = FieldType.STRESS

    def __init__(
        self,
        integrator_a,
        integrator_b,
        *,
        stress_tol: float = 1e-6,
        tangent_tol: float = 1e-5,
        state_tol: float = 1e-6,
        max_iter: int = 20,
        tol: float = 1e-8,
    ) -> None:
        super().__init__(
            integrator_a,
            integrator_b,
            stress_tol=stress_tol,
            tangent_tol=tangent_tol,
            state_tol=state_tol,
        )
        self.max_iter = max_iter
        self.tol = tol

    def _make_driver(self, integrator):
        return StressDriver(integrator, max_iter=self.max_iter, tol=self.tol)

    def _iter_driver(self, driver, load, **kwargs):
        return driver.iter_run(load, **kwargs)

    def iter_run(  # type: ignore[override]
        self,
        load,
        *,
        raise_on_nonconverged: bool = True,
        initial_stress=None,
        initial_state=None,
    ) -> Iterator[CrosscheckCaseResult]:
        """Yield per-step crosscheck results.

        Parameters
        ----------
        load : FieldHistory
            Stress-controlled loading history (``load.type`` must be
            ``FieldType.STRESS``).
        raise_on_nonconverged : bool, optional
            Forwarded to :meth:`~manforge.simulation.StressDriver.iter_run`
            (default ``True``).
        initial_stress : array-like or None, optional
            Stress tensor at the start of the loading history (default zero).
        initial_state : dict or None, optional
            State dict at the start of the loading history (default integrator initial state).

        Yields
        ------
        CrosscheckCaseResult
        """
        yield from super().iter_run(
            load,
            raise_on_nonconverged=raise_on_nonconverged,
            initial_stress=initial_stress,
            initial_state=initial_state,
        )
