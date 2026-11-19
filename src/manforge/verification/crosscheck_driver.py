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
  takes the driver's own NR settings as ``driver_max_iter`` /
  ``driver_tol`` / ``driver_rtol``)

Strain-controlled example
--------------------------
::

    from manforge.verification import CrosscheckStrainDriver
    from manforge.simulation import (
        PythonNumericalIntegrator, FortranIntegrator, FortranModule,
        FieldHistory, FieldType,
    )
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
    cc = CrosscheckStressDriver(py_int, fc_int, driver_max_iter=30, driver_tol=1e-10)
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
        # jac = JacobianChecker(model).compare(cr.result_a, cr.result_b)
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from manforge.core.result import StressUpdateResult
    from manforge.simulation.integrator import StressIntegrator
    from manforge._typing import FloatArray, StateDict

from manforge.simulation.driver import StrainDriver, StressDriver
from manforge.simulation.types import FieldType, FieldHistory
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

    Extends :class:`~manforge.verification.CaseResult` with the raw results of
    both integrators for inspection (``result_a.stress``, ``result_b.ddsdde``, ...).

    Attributes
    ----------
    result_a : StressUpdateResult or None
        Raw result from integrator_a.  Pass to
        :meth:`~manforge.verification.JacobianChecker.compare` with ``result_b``
        to diagnose Jacobian-level differences; a Python result carries the
        increment (``result_a.increment``) the comparison needs.
    result_b : StressUpdateResult or None
        Raw result from integrator_b.
    """

    result_a: StressUpdateResult | None = None
    result_b: StressUpdateResult | None = None
    a_n_iterations: int = 0
    a_residual_history: list[float] = field(default_factory=list)
    b_n_iterations: int = 0
    b_residual_history: list[float] = field(default_factory=list)


@dataclass
class CrosscheckResult(ComparisonResult):
    """Aggregate result across all cases / steps.

    Extends :class:`~manforge.verification.ComparisonResult` with the full
    per-case list typed as :class:`CrosscheckCaseResult`.
    """

    cases: list[CrosscheckCaseResult] = field(default_factory=list)  # type: ignore[assignment]  # narrowed subtype


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
        integrator_a: StressIntegrator,
        integrator_b: StressIntegrator,
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

    def _make_driver(self, integrator: StressIntegrator) -> StrainDriver | StressDriver:
        raise NotImplementedError

    def _iter_driver(self, driver: StrainDriver | StressDriver, load: FieldHistory, **kwargs):  # type: ignore[return]
        raise NotImplementedError

    def iter_run(
        self,
        load: FieldHistory,
        *,
        initial_stress: FloatArray | None = None,
        initial_state: StateDict | None = None,
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

        for sa, sb in zip(
            self._iter_driver(da, load, initial_stress=init_stress_a,
                              initial_state=init_state_a, **kwargs),
            self._iter_driver(db, load, initial_stress=init_stress_b,
                              initial_state=init_state_b, **kwargs),
        ):
            ra = sa.result
            rb = sb.result

            stress_a = np.asarray(ra.stress, dtype=np.float64)
            state_a = {k: np.asarray(v) for k, v in ra.state.items()}
            ddsdde_a = np.asarray(ra.ddsdde, dtype=np.float64)

            stress_b = np.asarray(rb.stress, dtype=np.float64)
            state_b = {k: np.asarray(v) for k, v in rb.state.items()}

            ddsdde_b = (
                np.asarray(rb.ddsdde, dtype=np.float64) if rb.ddsdde is not None else None
            )
            t_err = _tangent_rel_err(ddsdde_a, ddsdde_b)

            s_err  = _stress_rel_err(stress_b, stress_a)
            st_err = _state_rel_err(state_b, state_a)
            ok     = _case_passed(s_err, st_err, t_err, self.stress_tol, self.state_tol, self.tangent_tol)

            yield CrosscheckCaseResult(
                index=sa.i,
                result_a=ra,
                result_b=rb,
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
    driver_max_iter : int, optional
        Maximum Newton iterations for :class:`~manforge.simulation.StressDriver`
        (default 20).
    driver_tol : float, optional
        Absolute convergence tolerance for
        :class:`~manforge.simulation.StressDriver` (default 1e-8).
    driver_rtol : float, optional
        Relative convergence tolerance for
        :class:`~manforge.simulation.StressDriver` (default 1e-10).

    The ``*_tol`` arguments decide whether the two implementations agree; the
    ``driver_*`` ones are the settings of the driver's own outer NR and are
    passed through to it.

    Examples
    --------
    ::

        cc = CrosscheckStressDriver(py_int, fc_int, driver_max_iter=30, driver_tol=1e-10)
        result = cc.run(stress_load)
        assert result.passed

        for cr in cc.iter_run(stress_load):
            print(f"step {cr.index}: stress_err={cr.stress_rel_err:.2e}")

    Non-convergence follows each integrator's own ``raise_on_nonconverged``,
    exactly as in :class:`~manforge.simulation.StressDriver`.
    """

    _expected_field_type = FieldType.STRESS

    def __init__(
        self,
        integrator_a: StressIntegrator,
        integrator_b: StressIntegrator,
        *,
        stress_tol: float = 1e-6,
        tangent_tol: float = 1e-5,
        state_tol: float = 1e-6,
        driver_max_iter: int = 20,
        driver_tol: float = 1e-8,
        driver_rtol: float = 1e-10,
    ) -> None:
        super().__init__(
            integrator_a,
            integrator_b,
            stress_tol=stress_tol,
            tangent_tol=tangent_tol,
            state_tol=state_tol,
        )
        self.driver_max_iter = driver_max_iter
        self.driver_tol = driver_tol
        self.driver_rtol = driver_rtol

    def _make_driver(self, integrator):
        return StressDriver(
            integrator, max_iter=self.driver_max_iter,
            tol=self.driver_tol, rtol=self.driver_rtol,
        )

    def _iter_driver(self, driver, load, **kwargs):
        return driver.iter_run(load, **kwargs)
