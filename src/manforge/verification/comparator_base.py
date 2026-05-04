"""Comparator ABC — shared base for crosscheck harnesses.

A Comparator holds the static configuration of *what to compare* (ref/cand
implementations, tolerances) in ``__init__`` and accepts *what to apply it to*
(driver+load) as arguments to ``iter_run`` / ``run``.

Concrete subclasses:

* ``CrosscheckStrainDriver``  — two integrators over a strain-controlled history
* ``CrosscheckStressDriver``  — two integrators over a stress-controlled history
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any

import numpy as np

# Stress rel-err denominator: additive offset avoids 0/0 on unloading to zero stress.
_STRESS_NORM = 1.0
# State/tangent/Jacobian rel-err denominator: pure relative (values don't go to zero).
_EPS = 1e-300


def _stress_rel_err(ref: np.ndarray, cand: np.ndarray) -> float:
    """Max component-wise stress relative error with additive denominator."""
    return float(np.max(np.abs(cand - ref) / (np.abs(ref) + _STRESS_NORM)))


def _state_rel_err(
    ref_dict: dict[str, Any],
    cand_dict: dict[str, Any],
) -> dict[str, float]:
    """Per state-variable relative error."""
    errs: dict[str, float] = {}
    for key in ref_dict:
        va = np.asarray(ref_dict[key], dtype=float)
        vb = np.asarray(cand_dict.get(key, np.zeros_like(va)), dtype=float)
        errs[key] = float(np.max(np.abs(vb - va) / (np.abs(va) + _EPS)))
    return errs


def _tangent_rel_err(ref: Any, cand: Any) -> float | None:
    """Max tangent relative error; returns None if either arg is None."""
    if ref is None or cand is None:
        return None
    ra = np.asarray(ref, dtype=float)
    ca = np.asarray(cand, dtype=float)
    return float(np.max(np.abs(ca - ra) / (np.abs(ra) + _EPS)))


def _array_rel_err(ref: np.ndarray, cand: np.ndarray) -> float:
    """Generic array relative error for trial stress, Jacobian blocks, etc."""
    return float(np.max(np.abs(cand - ref) / (np.abs(ref) + _EPS)))


def _case_passed(
    s_err: float,
    st_errs: dict[str, float],
    t_err: float | None,
    stress_tol: float,
    state_tol: float,
    tangent_tol: float,
) -> bool:
    ok = s_err <= stress_tol
    ok = ok and all(v <= state_tol for v in st_errs.values())
    if t_err is not None:
        ok = ok and t_err <= tangent_tol
    return ok


@dataclass
class CaseResult:
    """Per-case comparison result (base form).

    Subclasses add raw output fields (e.g. ``py_stress`` / ``f_stress``,
    ``result_a`` / ``result_b``) via dataclass inheritance.
    """

    index: int
    stress_rel_err: float | None = None
    state_rel_err: dict[str, float] = field(default_factory=dict)
    tangent_rel_err: float | None = None
    passed: bool = False
    # P3: convergence status for both integrators.  False only when
    # PythonIntegrator(raise_on_nonconverged=False) hits max_iter.
    a_converged: bool = True
    b_converged: bool = True


@dataclass
class ComparisonResult:
    """Aggregate result across all cases."""

    passed: bool
    n_cases: int
    n_passed: int
    max_stress_rel_err: float | None = None
    max_state_rel_err: dict[str, float] = field(default_factory=dict)
    max_tangent_rel_err: float | None = None
    cases: list[CaseResult] = field(default_factory=list)
    # P3: count of non-converged steps per side (default 0 → all converged).
    n_a_nonconverged: int = 0
    n_b_nonconverged: int = 0


class Comparator(ABC):
    """Abstract base for all comparison harnesses.

    Subclasses fix the comparison configuration in ``__init__`` and expose
    ``iter_run`` (generator, yields per-case :class:`CaseResult`) and
    ``run`` (consumes ``iter_run``, returns :class:`ComparisonResult`).

    The ``run`` implementation here is **shared** across all concrete
    subclasses — they only need to implement ``iter_run``.
    """

    _result_cls: type[ComparisonResult] = ComparisonResult

    def _aggregate_extra(self, cases: list[CaseResult]) -> dict:
        """Extra kwargs to pass to ``_result_cls``. Override in subclasses."""
        return {}

    def run(self, *args, **kwargs) -> ComparisonResult:
        """Run over all cases, collect results, return aggregate."""
        cases: list[CaseResult] = []
        max_s = 0.0
        max_t: float | None = None
        max_st: dict[str, float] = {}
        n_passed = 0
        n_a_nc = 0
        n_b_nc = 0

        for cr in self.iter_run(*args, **kwargs):
            cases.append(cr)
            max_s = max(max_s, cr.stress_rel_err or 0.0)
            if cr.tangent_rel_err is not None:
                max_t = max(max_t or 0.0, cr.tangent_rel_err)
            for k, v in cr.state_rel_err.items():
                max_st[k] = max(max_st.get(k, 0.0), v)
            if cr.passed:
                n_passed += 1
            if not cr.a_converged:
                n_a_nc += 1
            if not cr.b_converged:
                n_b_nc += 1

        return self._result_cls(
            passed=n_passed == len(cases),
            n_cases=len(cases),
            n_passed=n_passed,
            max_stress_rel_err=max_s,
            max_state_rel_err=max_st,
            max_tangent_rel_err=max_t,
            cases=cases,
            n_a_nonconverged=n_a_nc,
            n_b_nonconverged=n_b_nc,
            **self._aggregate_extra(cases),
        )

    @abstractmethod
    def iter_run(self, *args, **kwargs) -> Iterator[CaseResult]:
        """Yield one :class:`CaseResult` per case/step."""
