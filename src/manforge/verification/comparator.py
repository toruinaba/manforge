"""Comparator ABC — shared base for compare_solvers and crosscheck harnesses.

A Comparator holds the static configuration of *what to compare* (ref/cand
implementations, tolerances) in ``__init__`` and accepts *what to apply it to*
(model, test_cases, driver+load) as arguments to ``iter_run`` / ``run``.

Concrete subclasses:

* ``SolverComparison``       — two Python solvers vs. shared test_cases
* ``ReturnMappingCrosscheck`` — Python return_mapping vs. Fortran UMAT
* ``StressUpdateCrosscheck``  — Python stress_update vs. Fortran UMAT (multi-step)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterator
from dataclasses import dataclass, field


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

        return ComparisonResult(
            passed=n_passed == len(cases),
            n_cases=len(cases),
            n_passed=n_passed,
            max_stress_rel_err=max_s,
            max_state_rel_err=max_st,
            max_tangent_rel_err=max_t,
            cases=cases,
            n_a_nonconverged=n_a_nc,
            n_b_nonconverged=n_b_nc,
        )

    @abstractmethod
    def iter_run(self, *args, **kwargs) -> Iterator[CaseResult]:
        """Yield one :class:`CaseResult` per case/step."""
