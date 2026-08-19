"""Parameter space description for fitting.

:class:`ParameterSpace` separates the *optimisation variables* from the
*model parameters*.  The two coincide in the simple case, but correlated
parameters are better optimised in a reparametrised space — e.g. the YU
model requires ``Y < B``, which is expressible as a ratio variable with
bounds ``(1.0, ...)`` instead of an inequality constraint.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field

import numpy as np
from typing_extensions import TypeAlias

from manforge._typing import FloatArray

Bound: TypeAlias = tuple[float | None, float | None] | None
FitConfig: TypeAlias = "dict[str, tuple[float, Bound]]"
DerivedFn: TypeAlias = Callable[[dict[str, float]], float]


def _norm_bound(b: Bound) -> tuple[float, float]:
    if b is None:
        return (-np.inf, np.inf)
    lo = b[0] if b[0] is not None else -np.inf
    hi = b[1] if b[1] is not None else np.inf
    return (lo, hi)


@dataclass(frozen=True)
class ParameterSpace:
    """Optimisation variables and how they map onto model parameters.

    Parameters
    ----------
    free : dict
        Optimisation variables, ``{name: (initial_value, bounds)}`` where
        ``bounds`` is ``(lower, upper)``, ``None`` for unbounded, and either
        entry may be ``None`` for a one-sided bound.  A free variable need
        not be a model parameter — it may exist only to feed ``derived``.
    fixed : dict
        Values held constant during optimisation.
    derived : dict
        ``{name: fn}`` where ``fn(values) -> float`` computes a value from
        the already-resolved ones.  Evaluated in declaration order, so a
        derived entry may depend on earlier derived entries.

    Examples
    --------
    Enforce ``Y < B`` by optimising the ratio rather than ``B`` itself::

        ParameterSpace(
            free={"Y": (200.0, (50.0, 600.0)), "B_ratio": (1.5, (1.0, 5.0))},
            fixed={"E": 210_000.0, "nu": 0.3},
            derived={"B": lambda p: p["Y"] * p["B_ratio"]},
        )
    """

    free: FitConfig
    fixed: dict[str, float] = field(default_factory=dict)
    derived: dict[str, DerivedFn] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.free:
            raise ValueError("ParameterSpace requires at least one free variable")
        groups = {"free": self.free, "fixed": self.fixed, "derived": self.derived}
        seen: dict[str, str] = {}
        for group, names in groups.items():
            for name in names:
                if name in seen:
                    raise ValueError(
                        f"parameter {name!r} declared in both "
                        f"{seen[name]!r} and {group!r}"
                    )
                seen[name] = group

    @classmethod
    def coerce(
        cls,
        fit_config: "ParameterSpace | FitConfig",
        fixed_params: Mapping[str, float] | None = None,
    ) -> "ParameterSpace":
        """Accept either a :class:`ParameterSpace` or the plain dict form."""
        if isinstance(fit_config, ParameterSpace):
            if fixed_params:
                raise ValueError(
                    "fixed_params cannot be combined with a ParameterSpace — "
                    "declare the constants in ParameterSpace(fixed=...)"
                )
            return fit_config
        return cls(free=dict(fit_config), fixed=dict(fixed_params or {}))

    @property
    def free_names(self) -> tuple[str, ...]:
        return tuple(self.free)

    @property
    def x0(self) -> FloatArray:
        return np.array([self.free[n][0] for n in self.free], dtype=float)

    @property
    def bounds(self) -> list[tuple[float, float]]:
        """Free-variable bounds with ``None`` replaced by ``±inf``."""
        return [_norm_bound(self.free[n][1]) for n in self.free]

    def resolve(self, x: Sequence[float] | FloatArray) -> dict[str, float]:
        """Expand an optimisation vector into fixed + free + derived values."""
        xa = np.asarray(x, dtype=float).ravel()
        if xa.size != len(self.free):
            raise ValueError(
                f"expected {len(self.free)} free values, got {xa.size}"
            )
        values = dict(self.fixed)
        values.update(zip(self.free_names, (float(v) for v in xa)))
        for name, fn in self.derived.items():
            values[name] = float(fn(values))
        return values

    def model_params(
        self, x: Sequence[float] | FloatArray, param_names: Iterable[str]
    ) -> dict[str, float]:
        """Resolve ``x`` and select exactly the model's parameters."""
        values = self.resolve(x)
        names = list(param_names)
        missing = [n for n in names if n not in values]
        if missing:
            raise ValueError(
                f"ParameterSpace supplies no value for model parameter(s) "
                f"{missing}; declare them as free, fixed, or derived"
            )
        return {n: values[n] for n in names}
