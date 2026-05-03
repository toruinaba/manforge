"""Multi-step crosscheck: Python model vs Fortran UMAT.

:class:`ReturnMappingCrosscheck` and :class:`StressUpdateCrosscheck` are
:class:`~manforge.verification.Comparator` subclasses that compare a Python
constitutive model against a compiled Fortran UMAT.

Both hold the static configuration in ``__init__`` and accept the dynamic
inputs (model, test_cases, driver+load) in ``iter_run`` / ``run``.

Phase 5 workflow — StressUpdateCrosscheck
-----------------------------------------
::

    from manforge.verification import ReturnMappingCrosscheck, StressUpdateCrosscheck, FortranUMAT
    from manforge.simulation import StrainDriver, PythonIntegrator, FortranIntegrator
    from manforge.simulation.types import FieldHistory, FieldType
    import numpy as np

    model   = MyModel(E=210_000.0, nu=0.3, sigma_y0=250.0, H=1_000.0)
    fortran = FortranUMAT("my_model")
    load    = FieldHistory(FieldType.STRAIN, "eps",
                           np.linspace([0]*6, [1e-3,0,0,0,0,0], 20))

    py_int = PythonIntegrator(model, method="numerical_newton")
    fc_int = FortranIntegrator(
        fortran, "my_model_core",
        param_fn=lambda: (model.E, model.nu, model.sigma_y0, model.H),
        state_names=model.state_names,
        initial_state=model.initial_state,
        elastic_stiffness=model.elastic_stiffness,
    )

    cc = StressUpdateCrosscheck(py_int, fc_int)
    result = cc.run(StrainDriver(), load)
    assert result.passed
    print(f"max stress rel error = {result.max_stress_rel_err:.2e}")

ReturnMappingCrosscheck — unchanged from Phase 4
-------------------------------------------------
``ReturnMappingCrosscheck`` still accepts ``fortran`` + hook kwargs directly
because it operates at single-step granularity (no Driver loop), so the
Integrator abstraction adds no value there.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np

from manforge.core.stress_update import stress_update
from manforge.verification.comparator import (
    CaseResult,
    ComparisonResult,
    Comparator,
    _case_passed,
    _state_rel_err,
    _stress_rel_err,
    _tangent_rel_err,
)
from manforge.simulation.integrator import PythonIntegrator


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------

@dataclass
class CrosscheckCaseResult(CaseResult):
    """Per-case / per-step result from a crosscheck harness.

    Extends :class:`~manforge.verification.CaseResult` with raw Python- and
    Fortran-side outputs for inspection.
    """

    py_stress: np.ndarray | None = None
    py_state: dict | None = None
    py_ddsdde: np.ndarray | None = None
    py_dlambda: float | None = None
    f_stress: np.ndarray | None = None
    f_state: dict | None = None
    f_ddsdde: np.ndarray | None = None
    # P2: inner-NR trajectory (a = integrator_a / Python side,
    #     b = integrator_b / Fortran side).  Fortran UMAT default is
    #     a neutral (0 / []) pair — matches FortranIntegrator.
    # a_converged / b_converged live in base CaseResult (P3).
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
# Default hooks
# ---------------------------------------------------------------------------

def _default_state_to_args(
    state: dict[str, Any], state_names: list[str]
) -> tuple:
    """Pack state dict → positional args in state_names order."""
    out = []
    for name in state_names:
        v = state[name]
        if np.ndim(v) == 0:
            out.append(float(v))
        else:
            out.append(np.asarray(v, dtype=np.float64))
    return tuple(out)


def _default_parse_umat_return(
    ret: tuple,
    state_names: list[str],
    initial_state: dict[str, Any],
) -> tuple[np.ndarray, dict[str, Any]]:
    """Unpack f2py return → (stress, state_dict).

    Expects: (stress_out, state[0]_out, state[1]_out, ..., <trailing>)
    Trailing elements (e.g. ddsdde) are discarded.
    """
    stress = np.asarray(ret[0], dtype=np.float64)
    state_out: dict[str, Any] = {}
    for i, name in enumerate(state_names, start=1):
        ref = initial_state[name]
        v = ret[i]
        if np.ndim(ref) == 0:
            state_out[name] = float(v)
        else:
            state_out[name] = np.asarray(v, dtype=np.float64).reshape(
                np.asarray(ref).shape
            )
    return stress, state_out


def _default_parse_umat_ddsdde(
    ret: tuple,
    n_state: int,
) -> np.ndarray:
    """Scan trailing returns for a 2-D array (the ddsdde output).

    Trailing elements start after (stress + n_state state variables).
    Raises ValueError if no 2-D array is found.
    """
    trailing = ret[1 + n_state:]
    for v in trailing:
        arr = np.asarray(v)
        if arr.ndim == 2:
            return arr
    raise ValueError(
        f"Could not locate ddsdde (2D array) in UMAT return: "
        f"trailing shapes = {[np.asarray(v).shape for v in trailing]}"
    )


# ---------------------------------------------------------------------------
# ReturnMappingCrosscheck
# ---------------------------------------------------------------------------

class ReturnMappingCrosscheck(Comparator):
    """Compare Python ``return_mapping`` and Fortran UMAT on independent cases.

    Each case is a single increment; no state is carried between cases.

    Parameters
    ----------
    fortran : FortranUMAT
        Wrapping the compiled f2py module.
    umat_subroutine : str
        Name of the f2py-callable core-logic subroutine.
    param_fn : callable
        ``param_fn(model) -> tuple`` — material parameters in UMAT order.
    method : {"auto", "numerical_newton", "user_defined"}
        Python-side solver strategy.  Required; no default.
    state_to_args : callable, optional
        ``state_to_args(state_dict) -> tuple`` — packs state into positional
        args.  Defaults to ``state_names``-order.
    parse_umat_return : callable, optional
        ``parse_umat_return(ret) -> (stress_ndarray, state_dict)``.
    stress_tol : float, optional
        Pass threshold for relative stress error (default 1e-6).
    state_tol : float, optional
        Pass threshold for per state-variable relative error (default 1e-6).

    Examples
    --------
    ::

        cc = ReturnMappingCrosscheck(
            fortran, umat_subroutine="j2_core",
            param_fn=lambda m: (m.E, m.nu, m.sigma_y0, m.H),
            method="numerical_newton",
        )
        result = cc.run(model, generate_single_step_cases(model))
        assert result.passed

        for cr in cc.iter_run(model, test_cases):
            if not cr.passed:
                print(f"case {cr.index} stress_err={cr.stress_rel_err:.2e}")
                break
    """

    _result_cls = CrosscheckResult

    def __init__(
        self,
        fortran,
        *,
        umat_subroutine: str,
        param_fn: Callable[[Any], tuple],
        method: str,
        state_to_args: Callable[[dict], tuple] | None = None,
        parse_umat_return: Callable[[tuple], tuple[np.ndarray, dict]] | None = None,
        stress_tol: float = 1e-6,
        state_tol: float = 1e-6,
    ) -> None:
        self.fortran = fortran
        self.umat_subroutine = umat_subroutine
        self.param_fn = param_fn
        self.method = method
        self.state_to_args = state_to_args
        self.parse_umat_return = parse_umat_return
        self.stress_tol = stress_tol
        self.state_tol = state_tol

    def iter_run(
        self,
        model,
        test_cases: list[dict],
    ) -> Iterator[CrosscheckCaseResult]:
        """Yield per-case crosscheck results.

        Parameters
        ----------
        model : MaterialModel
        test_cases : list[dict]
            Each dict must have ``"strain_inc"``, ``"stress_n"``, ``"state_n"``.
            Compatible with :func:`~manforge.verification.generate_single_step_cases`.

        Yields
        ------
        CrosscheckCaseResult
        """
        state_names: list[str] = list(model.state_names)
        initial_state: dict = model.initial_state()

        _s2a = self.state_to_args if self.state_to_args is not None else (
            lambda s: _default_state_to_args(s, state_names)
        )
        _pur = self.parse_umat_return if self.parse_umat_return is not None else (
            lambda ret: _default_parse_umat_return(ret, state_names, initial_state)
        )

        for idx, case in enumerate(test_cases):
            strain_inc = np.asarray(case["strain_inc"], dtype=np.float64)
            stress_n   = np.asarray(case["stress_n"],   dtype=np.float64)
            state_n    = case["state_n"]

            py_su = stress_update(model, strain_inc, stress_n, state_n, method=self.method)
            py_stress  = np.asarray(py_su.stress,  dtype=np.float64)
            py_state   = {k: np.asarray(v) for k, v in py_su.state.items()}
            py_dlambda = float(py_su.dlambda)

            state_tup = _s2a(state_n)
            ret = self.fortran.call(
                self.umat_subroutine,
                *self.param_fn(model),
                stress_n,
                *state_tup,
                strain_inc,
            )
            f_stress, f_state = _pur(ret)
            f_stress = np.asarray(f_stress, dtype=np.float64)

            s_err  = _stress_rel_err(f_stress, py_stress)
            st_err = _state_rel_err(f_state, py_state)
            ok     = _case_passed(s_err, st_err, None, self.stress_tol, self.state_tol, 0.0)

            yield CrosscheckCaseResult(
                index=idx,
                py_stress=py_stress,
                py_state=py_state,
                py_dlambda=py_dlambda,
                f_stress=f_stress,
                f_state=f_state,
                stress_rel_err=s_err,
                state_rel_err=st_err,
                passed=ok,
                a_n_iterations=py_su.n_iterations,
                a_residual_history=list(py_su.residual_history),
                a_converged=py_su.converged,
            )


# ---------------------------------------------------------------------------
# StressUpdateCrosscheck
# ---------------------------------------------------------------------------

class StressUpdateCrosscheck(Comparator):
    """Compare two :class:`~manforge.simulation.integrator.StressIntegrator` implementations
    over a loading history.

    Drives both integrators through the same multi-step sequence via a Driver,
    then compares stress / state / tangent at every step.

    Parameters
    ----------
    integrator_a : StressIntegrator
        Reference implementation (typically a
        :class:`~manforge.simulation.integrator.PythonIntegrator`).
    integrator_b : StressIntegrator
        Candidate implementation (typically a
        :class:`~manforge.simulation.integrator.FortranIntegrator`).
    stress_tol : float, optional
        Pass threshold for relative stress error (default 1e-6).
    tangent_tol : float, optional
        Pass threshold for relative tangent error (default 1e-5).
    state_tol : float, optional
        Pass threshold for per state-variable relative error (default 1e-6).

    Examples
    --------
    ::

        from manforge.simulation import PythonIntegrator, FortranIntegrator, StrainDriver

        py_int = PythonIntegrator(model, method="numerical_newton")
        fc_int = FortranIntegrator(
            fortran, "j2_isotropic_3d",
            param_fn=lambda: (model.E, model.nu, model.sigma_y0, model.H),
            state_names=model.state_names,
            initial_state=model.initial_state,
            elastic_stiffness=model.elastic_stiffness,
        )

        cc = StressUpdateCrosscheck(py_int, fc_int)
        result = cc.run(StrainDriver(), load)
        assert result.passed

        for cr in cc.iter_run(StrainDriver(), load):
            if not cr.passed:
                print(f"step {cr.index}: stress_err={cr.stress_rel_err:.2e}")
                break
    """

    _result_cls = CrosscheckResult

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

    def iter_run(
        self,
        driver,
        load,
    ) -> Iterator[CrosscheckCaseResult]:
        """Yield per-step crosscheck results.

        Parameters
        ----------
        driver
            An instantiated driver, e.g. ``StrainDriver()`` or ``StressDriver()``.
        load : FieldHistory

        Yields
        ------
        CrosscheckCaseResult
        """
        for sa, sb in zip(
            driver.iter_run(self.integrator_a, load),
            driver.iter_run(self.integrator_b, load),
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

            yield CrosscheckCaseResult(
                index=sa.i,
                py_stress=py_stress,
                py_state=py_state,
                py_ddsdde=py_ddsdde,
                py_dlambda=py_dlambda,
                f_stress=f_stress,
                f_state=f_state,
                f_ddsdde=f_ddsdde,
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

