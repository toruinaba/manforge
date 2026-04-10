"""Fortran UMAT bridge for cross-validation.

Provides :class:`FortranUMAT` for calling compiled Fortran UMAT subroutines
and :func:`compare_with_fortran` for systematic comparison against the
Python implementation.
"""

import importlib
from dataclasses import dataclass, field

import numpy as np

from manforge.core.return_mapping import return_mapping
from manforge.verification.compare import compare_solvers


@dataclass
class UMATComparisonResult:
    """Result of a Python-vs-Fortran UMAT comparison run.

    Attributes
    ----------
    passed : bool
        ``True`` if all test cases pass within tolerance.
    n_cases : int
        Number of test cases evaluated.
    n_passed : int
        Number of test cases that passed.
    max_stress_rel_err : float
        Maximum relative error in stress across all cases.
    max_tangent_rel_err : float
        Maximum relative error in tangent (DDSDDE) across all cases.
    details : list[dict]
        Per-case detail records with keys:
        ``"case_index"``, ``"stress_rel_err"``, ``"tangent_rel_err"``,
        ``"passed"``.
    """

    passed: bool
    n_cases: int
    n_passed: int
    max_stress_rel_err: float
    max_tangent_rel_err: float
    details: list = field(default_factory=list)


class FortranUMAT:
    """Bridge to a compiled Fortran UMAT subroutine via f2py.

    Wraps the f2py-compiled ``manforge_umat`` module so that the Fortran J2
    UMAT can be called with the same Python-level interface as
    :func:`~manforge.core.return_mapping.return_mapping`.

    The module must expose ``umat_j2_run(E, nu, sigma_y0, H,
    stress_in, ep_in, dstran)``.

    Parameters
    ----------
    module_name : str
        Name of the f2py-compiled Python module (must be importable from
        ``fortran/`` after ``make fortran-build-umat``).
    model : MaterialModel, optional
        Model instance used to determine ``param_names`` and ``state_names``
        ordering when marshalling dicts to flat arrays.
        If *None*, the default ordering [E, nu, sigma_y0, H] / [ep] is used.
    """

    def __init__(self, module_name: str = "manforge_umat", model=None):
        self._mod = importlib.import_module(module_name)
        self._model = model

    def call(
        self,
        strain_inc,
        stress_n,
        state_n: dict,
        params: dict,
        *,
        dtime: float = 1.0,
    ):
        """Call the Fortran UMAT with Python-compatible arguments.

        Parameters
        ----------
        strain_inc : array-like, shape (6,)
            Strain increment Δε (engineering shear convention).
        stress_n : array-like, shape (6,)
            Stress at the beginning of the increment.
        state_n : dict
            Internal state variables.  Must contain key ``"ep"``.
        params : dict
            Material parameters.  Must contain keys ``E``, ``nu``,
            ``sigma_y0``, ``H``.
        dtime : float
            Unused (kept for interface symmetry with return_mapping).

        Returns
        -------
        stress_new : np.ndarray, shape (6,)
        state_new : dict
        ddsdde : np.ndarray, shape (6, 6)
        """
        dstran   = np.asarray(strain_inc, dtype=np.float64)
        stress_i = np.asarray(stress_n,   dtype=np.float64)
        ep_i     = float(state_n["ep"])

        E        = float(params["E"])
        nu       = float(params["nu"])
        sigma_y0 = float(params["sigma_y0"])
        H        = float(params["H"])

        stress_out, ep_out, ddsdde = self._mod.umat_j2_run(
            E, nu, sigma_y0, H,
            stress_i, ep_i, dstran,
        )

        return (
            np.array(stress_out),
            {"ep": float(ep_out)},
            np.array(ddsdde),
        )


def compare_with_fortran(
    model,
    fortran_umat: FortranUMAT,
    test_cases: list,
    stress_tol: float = 1e-6,
    tangent_tol: float = 1e-5,
) -> UMATComparisonResult:
    """Compare Python model results with Fortran UMAT for a set of test cases.

    For each test case, calls both :func:`~manforge.core.return_mapping.return_mapping`
    and :meth:`FortranUMAT.call` with identical inputs, then computes the
    relative error in stress and tangent (DDSDDE).

    Parameters
    ----------
    model : MaterialModel
        Python constitutive model.
    fortran_umat : FortranUMAT
        Compiled Fortran UMAT bridge.
    test_cases : list[dict]
        Each dict must contain:

        - ``"strain_inc"`` : array-like, shape (ntens,)
        - ``"stress_n"``   : array-like, shape (ntens,)
        - ``"state_n"``    : dict
        - ``"params"``     : dict
    stress_tol : float
        Relative tolerance for stress comparison (default 1e-6).
    tangent_tol : float
        Relative tolerance for tangent comparison (default 1e-5).

    Returns
    -------
    UMATComparisonResult
    """
    import jax.numpy as jnp

    def _python_solver(strain_inc, stress_n, state_n, params):
        return return_mapping(model, jnp.array(strain_inc), jnp.array(stress_n), state_n, params)

    result = compare_solvers(
        solver_a=_python_solver,
        solver_b=fortran_umat.call,
        test_cases=test_cases,
        stress_tol=stress_tol,
        tangent_tol=tangent_tol,
    )

    return UMATComparisonResult(
        passed=result.passed,
        n_cases=result.n_cases,
        n_passed=result.n_passed,
        max_stress_rel_err=result.max_stress_rel_err,
        max_tangent_rel_err=result.max_tangent_rel_err,
        details=result.details,
    )
