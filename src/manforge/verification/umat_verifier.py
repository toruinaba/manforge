"""High-level Fortran UMAT verifier.

Provides :class:`UMATVerifier`, the primary interface for verifying that a
compiled Fortran UMAT subroutine produces the same results as a Python
:class:`~manforge.core.material.MaterialModel` reference implementation.

Usage
-----
::

    import manforge
    from manforge.models.j2_isotropic import J2Isotropic3D
    from manforge.verification import UMATVerifier

    model    = J2Isotropic3D()
    params   = {"E": 210_000.0, "nu": 0.3, "sigma_y0": 250.0, "H": 1_000.0}
    verifier = UMATVerifier(model, "manforge_umat")
    result   = verifier.run(params)

    print(result.passed)
    print(result.summary())

The verifier performs two phases:

1. **Single-step comparison** -- auto-generated test cases (elastic, plastic
   uniaxial/multiaxial, shear, pre-stressed) compared at single-increment
   level via :func:`~manforge.verification.compare.compare_solvers`.

2. **Multi-step comparison** -- a full tension-unload-compression strain
   history run through both Python and Fortran, with each solver propagating
   its own stress and state independently.  Detects errors that accumulate
   over multiple increments.

The compiled Fortran module must be available on ``sys.path`` before calling
``UMATVerifier`` (e.g., after ``make fortran-build-umat``).
"""

from dataclasses import dataclass, field

import numpy as np
import jax.numpy as jnp

from manforge.core.return_mapping import return_mapping
from manforge.verification.compare import compare_solvers, SolverComparisonResult
from manforge.verification.fortran_bridge import FortranUMAT
from manforge.verification.test_cases import (
    estimate_yield_strain,
    generate_single_step_cases,
    generate_strain_history,
)


@dataclass
class StepResult:
    """Comparison result for a single step in the multi-step phase.

    Attributes
    ----------
    step_index : int
        Zero-based index into the strain history.
    strain_inc : np.ndarray
        Strain increment applied at this step, shape (ntens,).
    stress_rel_err : float
        Maximum element-wise relative error in stress.
    tangent_rel_err : float
        Maximum element-wise relative error in DDSDDE.
    state_rel_err : float
        Maximum relative error across all state variables.
    passed : bool
        ``True`` if all errors are within their respective tolerances.
    """

    step_index: int
    strain_inc: np.ndarray
    stress_rel_err: float
    tangent_rel_err: float
    state_rel_err: float
    passed: bool


@dataclass
class VerificationResult:
    """Combined result of single-step and multi-step UMAT verification.

    Attributes
    ----------
    passed : bool
        ``True`` only if both the single-step and multi-step phases pass.
    single_step : SolverComparisonResult
        Result of the single-step comparison phase (from
        :func:`~manforge.verification.compare.compare_solvers`).
    multi_step_passed : bool
        Whether all multi-step increments are within tolerance.
    multi_step_max_stress_err : float
        Peak relative stress error across all multi-step increments.
    multi_step_max_tangent_err : float
        Peak relative tangent error across all multi-step increments.
    multi_step_max_state_err : float
        Peak relative state variable error across all multi-step increments.
    multi_step_steps : list[StepResult]
        Per-step detail records.
    n_multi_steps : int
        Number of increments in the multi-step comparison.
    """

    passed: bool
    single_step: SolverComparisonResult
    multi_step_passed: bool
    multi_step_max_stress_err: float
    multi_step_max_tangent_err: float
    multi_step_max_state_err: float
    multi_step_steps: list = field(default_factory=list)
    n_multi_steps: int = 0

    def summary(self) -> str:
        """Return a human-readable verification summary."""
        lines = []
        sep = "=" * 60
        lines.append(sep)
        lines.append("  UMAT Verification Summary")
        lines.append(sep)
        lines.append(f"  Overall: [{'PASS' if self.passed else 'FAIL'}]")
        lines.append("")

        # Single-step
        ss = self.single_step
        lines.append(f"  Single-step comparison: [{'PASS' if ss.passed else 'FAIL'}]")
        lines.append(f"    Cases         : {ss.n_passed}/{ss.n_cases} passed")
        lines.append(f"    Max stress  err: {ss.max_stress_rel_err:.2e}")
        lines.append(f"    Max tangent err: {ss.max_tangent_rel_err:.2e}")
        if not ss.passed:
            for d in ss.details:
                if not d["passed"]:
                    lines.append(
                        f"    [FAIL] case {d['case_index']}: "
                        f"stress={d['stress_rel_err']:.2e}, "
                        f"tangent={d['tangent_rel_err']:.2e}"
                    )
        lines.append("")

        # Multi-step
        lines.append(
            f"  Multi-step comparison: [{'PASS' if self.multi_step_passed else 'FAIL'}]"
        )
        lines.append(f"    Steps         : {self.n_multi_steps}")
        lines.append(f"    Max stress  err: {self.multi_step_max_stress_err:.2e}")
        lines.append(f"    Max tangent err: {self.multi_step_max_tangent_err:.2e}")
        lines.append(f"    Max state   err: {self.multi_step_max_state_err:.2e}")
        if not self.multi_step_passed:
            failed = [s for s in self.multi_step_steps if not s.passed]
            lines.append(f"    Failed steps  : {len(failed)}/{self.n_multi_steps}")
            for s in failed[:5]:
                lines.append(
                    f"    [FAIL] step {s.step_index}: "
                    f"stress={s.stress_rel_err:.2e}, "
                    f"tangent={s.tangent_rel_err:.2e}, "
                    f"state={s.state_rel_err:.2e}"
                )
            if len(failed) > 5:
                lines.append(f"    ... and {len(failed) - 5} more")

        lines.append(sep)
        return "\n".join(lines)


class UMATVerifier:
    """Convenience utility: auto-generates test cases and runs batch comparison.

    For a single-routine check or component-level debugging, use
    :class:`~manforge.verification.fortran_bridge.FortranUMAT` directly with
    explicit ``np.testing.assert_allclose`` calls — that approach is more
    transparent and easier to trust.

    This class is useful when you want to run a comprehensive set of
    auto-generated single-step and multi-step cases in one call.

    Parameters
    ----------
    model : MaterialModel
        Python reference model instance.
    module_name : str
        Name of the f2py-compiled Python module (must be importable from
        ``sys.path``, e.g. after ``make fortran-build-umat``).
    subroutine : str, optional
        Name of the Fortran ``*_run`` subroutine.  Auto-detected when
        the module exposes exactly one ``*_run`` function.
    """

    def __init__(self, model, module_name: str, subroutine: str | None = None):
        self._model   = model
        self._fortran = FortranUMAT(module_name, subroutine)

    def run(
        self,
        params: dict,
        strain_history=None,
        strain_scale: float | None = None,
        stress_tol: float = 1e-6,
        tangent_tol: float = 1e-5,
        state_tol: float = 1e-6,
        denom_offset: float = 1.0,
    ) -> VerificationResult:
        """Run the full verification suite.

        Parameters
        ----------
        params : dict
            Material parameters.
        strain_history : array-like, shape (N,) or (N, ntens), optional
            Custom cumulative strain history for the multi-step phase.
            A 1-D array is treated as uniaxial (applied to component 0).
            If *None*, a default tension-unload-compression cycle is used.
        strain_scale : float, optional
            Characteristic yield strain override.  If *None*, estimated
            automatically via
            :func:`~manforge.verification.test_cases.estimate_yield_strain`.
        stress_tol : float
            Relative tolerance for stress comparison (default 1e-6).
        tangent_tol : float
            Relative tolerance for tangent comparison (default 1e-5).
        state_tol : float
            Relative tolerance for state variable comparison (default 1e-6).
        denom_offset : float
            Additive denominator offset for relative-error computation
            (default 1.0, appropriate for MPa units).

        Returns
        -------
        VerificationResult
        """
        # Estimate characteristic yield strain
        if strain_scale is None:
            eps_y = estimate_yield_strain(self._model, params)
        else:
            eps_y = float(strain_scale)

        # ------------------------------------------------------------------
        # Phase 1: single-step comparison
        # ------------------------------------------------------------------
        test_cases   = generate_single_step_cases(self._model, params, eps_y)
        py_solver    = self._make_python_solver()
        single_result = compare_solvers(
            py_solver,
            self._call_fortran,
            test_cases,
            stress_tol=stress_tol,
            tangent_tol=tangent_tol,
            denom_offset=denom_offset,
        )

        # ------------------------------------------------------------------
        # Phase 2: multi-step comparison
        # ------------------------------------------------------------------
        if strain_history is None:
            history = generate_strain_history(self._model, params, eps_y)
        else:
            history = np.asarray(strain_history, dtype=float)
            if history.ndim == 1:
                ntens    = self._model.ntens
                full     = np.zeros((len(history), ntens))
                full[:, 0] = history
                history  = full

        ms_passed, ms_stress, ms_tangent, ms_state, ms_steps = (
            self._run_multi_step(
                params, history,
                stress_tol, tangent_tol, state_tol, denom_offset,
            )
        )

        return VerificationResult(
            passed=single_result.passed and ms_passed,
            single_step=single_result,
            multi_step_passed=ms_passed,
            multi_step_max_stress_err=ms_stress,
            multi_step_max_tangent_err=ms_tangent,
            multi_step_max_state_err=ms_state,
            multi_step_steps=ms_steps,
            n_multi_steps=len(ms_steps),
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _call_fortran(self, strain_inc, stress_n, state_n: dict, params: dict):
        """Marshal Python dicts to Fortran positional args and call _run.

        This is the only place in UMATVerifier that knows about param_names
        and state_names.  FortranUMAT itself has no model knowledge.
        """
        param_args = [float(params[n]) for n in self._model.param_names]
        stress_i   = np.asarray(stress_n,   dtype=np.float64)
        state_args = [float(state_n[n]) for n in self._model.state_names]
        dstran     = np.asarray(strain_inc, dtype=np.float64)

        result = self._fortran.run(*param_args, stress_i, *state_args, dstran)

        stress_out = np.array(result[0])
        n_states   = len(self._model.state_names)
        state_new  = {
            name: float(result[1 + i])
            for i, name in enumerate(self._model.state_names)
        }
        ddsdde = np.array(result[1 + n_states])
        return stress_out, state_new, ddsdde

    def _make_python_solver(self):
        """Return a solver callable backed by Python return_mapping."""
        model = self._model

        def _solve(strain_inc, stress_n, state_n, params):
            return return_mapping(
                model,
                jnp.array(strain_inc),
                jnp.array(stress_n),
                state_n,
                params,
            )

        return _solve

    def _run_multi_step(
        self,
        params: dict,
        strain_history: np.ndarray,
        stress_tol: float,
        tangent_tol: float,
        state_tol: float,
        denom_offset: float,
    ):
        """Step-by-step comparison with independent state propagation.

        Each solver maintains its own stress and state.  Per-step drift in
        state variables accumulates across increments, which is exactly what
        this phase is designed to detect.
        """
        model = self._model
        ntens = model.ntens

        # Python state (JAX types)
        stress_py = jnp.zeros(ntens)
        state_py  = model.initial_state()

        # Fortran state (plain floats / numpy)
        stress_f90 = np.zeros(ntens)
        state_f90  = {k: float(v) for k, v in model.initial_state().items()}

        eps_prev = np.zeros(ntens)
        steps    = []
        max_stress_err  = 0.0
        max_tangent_err = 0.0
        max_state_err   = 0.0
        all_passed      = True

        for i in range(len(strain_history)):
            strain_inc = strain_history[i] - eps_prev
            eps_prev   = strain_history[i].copy()

            # Python increment
            stress_py, state_py, ddsdde_py = return_mapping(
                model, jnp.array(strain_inc), stress_py, state_py, params
            )

            # Fortran increment
            stress_f90, state_f90, ddsdde_f90 = self._call_fortran(
                strain_inc, stress_f90, state_f90, params
            )

            # Stress error
            s_py   = np.asarray(stress_py,  dtype=float)
            s_f90  = np.asarray(stress_f90, dtype=float)
            s_denom = np.abs(s_py) + denom_offset
            stress_err = float(np.max(np.abs(s_f90 - s_py) / s_denom))

            # Tangent error
            t_py   = np.asarray(ddsdde_py,  dtype=float)
            t_f90  = np.asarray(ddsdde_f90, dtype=float)
            t_denom = np.abs(t_py) + denom_offset
            tangent_err = float(np.max(np.abs(t_f90 - t_py) / t_denom))

            # State error (max over all state variables)
            state_err = 0.0
            for name in model.state_names:
                py_val  = float(state_py[name])
                f90_val = float(state_f90[name])
                denom   = abs(py_val) + denom_offset
                state_err = max(state_err, abs(f90_val - py_val) / denom)

            step_passed = (
                stress_err  <= stress_tol
                and tangent_err <= tangent_tol
                and state_err   <= state_tol
            )

            steps.append(StepResult(
                step_index=i,
                strain_inc=np.array(strain_inc),
                stress_rel_err=stress_err,
                tangent_rel_err=tangent_err,
                state_rel_err=state_err,
                passed=step_passed,
            ))

            max_stress_err  = max(max_stress_err,  stress_err)
            max_tangent_err = max(max_tangent_err, tangent_err)
            max_state_err   = max(max_state_err,   state_err)
            if not step_passed:
                all_passed = False

        return all_passed, max_stress_err, max_tangent_err, max_state_err, steps
