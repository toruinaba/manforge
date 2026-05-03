"""StressIntegrator protocol and adapter implementations.

:class:`StressIntegrator` is a structural Protocol that captures the minimal
surface a :class:`~manforge.simulation.driver.DriverBase` needs from any
constitutive implementation:

* ``stress_state`` / ``ntens``  — array-shape information
* ``initial_state()``           — initial internal state dict
* ``elastic_stiffness()``       — (ntens, ntens) stiffness tensor
* ``stress_update(Δε, σ_n, s_n) → StressUpdateResult``  — one step

Four adapters implement this protocol:

:class:`PythonIntegrator`
    Wraps a :class:`~manforge.core.material.MaterialModel` and uses the
    ``"auto"`` solver strategy (user-defined hook if present, else
    numerical Newton-Raphson).

:class:`PythonNumericalIntegrator`
    Same as ``PythonIntegrator`` but always uses ``"numerical_newton"``.

:class:`PythonAnalyticalIntegrator`
    Same as ``PythonIntegrator`` but always uses ``"user_defined"``
    (requires ``model.user_defined_return_mapping`` to be implemented).

:class:`FortranIntegrator`
    Wraps a :class:`~manforge.verification.FortranUMAT` and the four hook
    functions that map between Python state dicts and Fortran argument lists.

Drivers require a :class:`StressIntegrator` — bare ``MaterialModel`` objects
are not accepted directly.  Wrap them with one of the Python integrators above.
"""

from __future__ import annotations

import math
from typing import Any, Callable

import numpy as np

from manforge.core.stress_update import (
    ReturnMappingResult,
    StressUpdateResult,
    stress_update as _core_stress_update,
)
from manforge.core.stress_state import StressState, SOLID_3D


# ---------------------------------------------------------------------------
# Protocol
# ---------------------------------------------------------------------------

class StressIntegrator:
    """Minimal interface required by Driver for any constitutive implementation.

    This is a duck-typed structural interface rather than an ABC — any object
    that provides the four methods/attributes below will work.  Use
    :class:`PythonIntegrator` or :class:`FortranIntegrator` as ready-made
    adapters.
    """

    stress_state: StressState

    @property
    def ntens(self) -> int:
        return self.stress_state.ntens

    def initial_state(self) -> dict:
        raise NotImplementedError

    def elastic_stiffness(self) -> np.ndarray:
        raise NotImplementedError

    def stress_update(
        self, strain_inc, stress_n, state_n
    ) -> StressUpdateResult:
        raise NotImplementedError


# ---------------------------------------------------------------------------
# PythonIntegrator
# ---------------------------------------------------------------------------

class _PythonIntegratorBase:
    """Shared implementation for Python-side integrators.

    Subclasses set the ``_method`` class variable to select the solver strategy.
    """

    _method: str = "auto"

    def __init__(self, model, *, raise_on_nonconverged: bool = True) -> None:
        self._model = model
        self._raise_on_nonconverged = raise_on_nonconverged

    @property
    def stress_state(self) -> StressState:
        return self._model.stress_state

    @property
    def ntens(self) -> int:
        return self._model.ntens

    def initial_state(self) -> dict:
        return self._model.initial_state()

    def elastic_stiffness(self) -> np.ndarray:
        return self._model.elastic_stiffness()

    def stress_update(self, strain_inc, stress_n, state_n) -> StressUpdateResult:
        return _core_stress_update(
            self._model, strain_inc, stress_n, state_n,
            method=self._method,
            raise_on_nonconverged=self._raise_on_nonconverged,
        )


class PythonIntegrator(_PythonIntegratorBase):
    """Wraps a MaterialModel using the ``"auto"`` solver strategy.

    Uses ``model.user_defined_return_mapping`` when available, otherwise
    falls back to numerical Newton-Raphson.

    Parameters
    ----------
    model : MaterialModel
    raise_on_nonconverged : bool, optional
        Raise ``RuntimeError`` if NR does not converge (default ``True``).
    """

    _method = "auto"


class PythonNumericalIntegrator(_PythonIntegratorBase):
    """Wraps a MaterialModel always using numerical Newton-Raphson.

    Parameters
    ----------
    model : MaterialModel
    raise_on_nonconverged : bool, optional
        Raise ``RuntimeError`` if NR does not converge (default ``True``).
    """

    _method = "numerical_newton"


class PythonAnalyticalIntegrator(_PythonIntegratorBase):
    """Wraps a MaterialModel always using the user-defined analytical solver.

    Requires ``model.user_defined_return_mapping`` to be implemented.

    Parameters
    ----------
    model : MaterialModel
    raise_on_nonconverged : bool, optional
        Raise ``RuntimeError`` if NR does not converge (default ``True``).
    """

    _method = "user_defined"


# ---------------------------------------------------------------------------
# FortranIntegrator
# ---------------------------------------------------------------------------

_EPS_INTEGRATOR = 1e-300

# sentinel used for fields that Fortran UMATs do not expose
_NAN = math.nan


def _resolve_callable_or_value(x):
    """Return x() if callable, else x as-is (supports both lambda and ndarray)."""
    return x() if callable(x) else x


class FortranIntegrator:
    """Wraps a compiled Fortran UMAT for use with Driver.

    Parameters
    ----------
    fortran : FortranUMAT
        f2py module wrapper (from :class:`~manforge.verification.FortranUMAT`).
    subroutine : str
        Name of the f2py-callable core-logic subroutine.
    stress_state : StressState, optional
        Dimensionality descriptor.  Defaults to :data:`SOLID_3D` (ntens=6).
    initial_state : callable or dict
        Either a no-arg callable returning a state dict (e.g.
        ``model.initial_state``) or the dict value directly.
    elastic_stiffness : callable or ndarray
        Either a no-arg callable returning the (ntens, ntens) stiffness matrix
        (e.g. ``model.elastic_stiffness``) or the array value directly.
    param_fn : callable
        ``param_fn() -> tuple`` — material parameters in UMAT positional order.
        Note: unlike ``ReturnMappingCrosscheck`` / ``StressUpdateCrosscheck``,
        this takes *no* ``model`` argument — parameters are fully captured at
        construction time.
    state_names : list[str]
        Ordered list of state-variable names, used by the default hooks.
    state_to_args : callable, optional
        ``state_to_args(state_dict) -> tuple`` packs the state dict into
        positional UMAT args.  Defaults to ``state_names``-order packing.
    parse_umat_return : callable, optional
        ``parse_umat_return(ret) -> (stress_ndarray, state_dict)`` unpacks the
        f2py return tuple.  Defaults to scanning ``state_names`` order.
    parse_umat_ddsdde : callable, optional
        ``parse_umat_ddsdde(ret) -> ndarray`` extracts the consistent tangent
        from the f2py return tuple.  When ``None``, the tangent is read from
        the first 2-D array found in the trailing returns (same as the default
        hook used by :class:`~manforge.verification.StressUpdateCrosscheck`).

    Notes
    -----
    Fortran UMATs do not expose a per-step ``is_plastic`` flag or ``dlambda``.
    The resulting :class:`~manforge.core.stress_update.StressUpdateResult` will
    have ``is_plastic=None`` and ``dlambda=nan``.  Driver loops that gate on
    ``step.result.is_plastic`` should guard with ``if result.is_plastic is True``.
    """

    def __init__(
        self,
        fortran,
        subroutine: str,
        *,
        stress_state: StressState = SOLID_3D,
        initial_state,
        elastic_stiffness,
        param_fn: Callable[[], tuple],
        state_names: list[str],
        state_to_args: Callable[[dict], tuple] | None = None,
        parse_umat_return: Callable[[tuple], tuple[np.ndarray, dict]] | None = None,
        parse_umat_ddsdde: Callable[[tuple], np.ndarray] | None = None,
    ) -> None:
        self._fortran = fortran
        self._subroutine = subroutine
        self.stress_state = stress_state
        self._initial_state = initial_state
        self._elastic_stiffness = elastic_stiffness
        self._param_fn = param_fn
        self._state_names = list(state_names)

        n_state = len(self._state_names)
        self._s2a: Callable[[dict], tuple] = (
            state_to_args
            if state_to_args is not None
            else lambda s: _default_state_to_args(s, self._state_names)
        )
        self._pur: Callable[[tuple], tuple[np.ndarray, dict]] = (
            parse_umat_return
            if parse_umat_return is not None
            else lambda ret: _default_parse_umat_return(
                ret, self._state_names, self.initial_state()
            )
        )
        self._pudd: Callable[[tuple], np.ndarray] = (
            parse_umat_ddsdde
            if parse_umat_ddsdde is not None
            else lambda ret: _default_parse_umat_ddsdde(ret, n_state)
        )

    @property
    def ntens(self) -> int:
        return self.stress_state.ntens

    def initial_state(self) -> dict:
        return _resolve_callable_or_value(self._initial_state)

    def elastic_stiffness(self) -> np.ndarray:
        return np.asarray(_resolve_callable_or_value(self._elastic_stiffness), dtype=np.float64)

    def stress_update(self, strain_inc, stress_n, state_n) -> StressUpdateResult:
        strain_inc = np.asarray(strain_inc, dtype=np.float64)
        stress_n   = np.asarray(stress_n,   dtype=np.float64)

        state_tup = self._s2a(state_n)
        ret = self._fortran.call(
            self._subroutine,
            *self._param_fn(),
            stress_n,
            *state_tup,
            strain_inc,
        )

        f_stress, f_state = self._pur(ret)
        f_stress = np.asarray(f_stress, dtype=np.float64)

        try:
            ddsdde = np.asarray(self._pudd(ret), dtype=np.float64)
        except (ValueError, IndexError):
            ddsdde = self.elastic_stiffness()

        C = self.elastic_stiffness()
        stress_trial = stress_n + C @ strain_inc

        rm = ReturnMappingResult(
            stress=f_stress,
            state=f_state,
            dlambda=np.array(_NAN),
            n_iterations=0,
            residual_history=[],
            converged=True,
        )
        return StressUpdateResult(
            return_mapping=rm,
            ddsdde=ddsdde,
            stress_trial=stress_trial,
            is_plastic=None,
            _state_n=state_n,
        )


# ---------------------------------------------------------------------------
# Default hook helpers (mirrors those in umat_crosscheck.py)
# ---------------------------------------------------------------------------

def _default_state_to_args(state: dict[str, Any], state_names: list[str]) -> tuple:
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


def _default_parse_umat_ddsdde(ret: tuple, n_state: int) -> np.ndarray:
    trailing = ret[1 + n_state:]
    for v in trailing:
        arr = np.asarray(v)
        if arr.ndim == 2:
            return arr
    raise ValueError(
        f"Could not locate ddsdde (2D array) in UMAT return: "
        f"trailing shapes = {[np.asarray(v).shape for v in trailing]}"
    )
