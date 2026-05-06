"""StressIntegrator protocol and adapter implementations.

:class:`StressIntegrator` is a structural Protocol that captures the minimal
surface a :class:`~manforge.simulation.driver.DriverBase` needs from any
constitutive implementation:

* ``dimension`` / ``ntens``  — array-shape information
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
    Wraps a :class:`~manforge.verification.FortranModule` and the four hook
    functions that map between Python state dicts and Fortran argument lists.

Drivers require a :class:`StressIntegrator` — bare ``MaterialModel`` objects
are not accepted directly.  Wrap them with one of the Python integrators above.
"""

from __future__ import annotations

import math
from typing import Any, Callable

import autograd
import autograd.numpy as anp
import numpy as np

from manforge.core.result import ReturnMappingResult, StressUpdateResult
from manforge.core.dimension import StressDimension, SOLID_3D
from manforge.core.residual import build_residual, _flatten_state, _wrap_state
from manforge.core.state import _state_with_stress


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

    dimension: StressDimension

    @property
    def ntens(self) -> int:
        return self.dimension.ntens

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

    def __init__(
        self,
        model,
        *,
        max_iter: int = 50,
        tol: float = 1e-10,
        raise_on_nonconverged: bool = True,
    ) -> None:
        self._model = model
        self._max_iter = max_iter
        self._tol = tol
        self._raise_on_nonconverged = raise_on_nonconverged

    @property
    def dimension(self) -> StressDimension:
        return self._model.dimension

    @property
    def ntens(self) -> int:
        return self._model.ntens

    def initial_state(self) -> dict:
        return self._model.initial_state()

    def elastic_stiffness(self, state=None) -> np.ndarray:
        return self._model.elastic_stiffness(state)

    def _try_user_return_mapping(self, stress_trial, C, state_n):
        """Attempt user_defined_return_mapping; return ReturnMappingResult or None."""
        if self._method == "numerical_newton":
            return None
        rm = self._model.user_defined_return_mapping(stress_trial, C, state_n)
        if rm is not None:
            return rm
        if self._method == "user_defined":
            raise NotImplementedError(
                f"{type(self._model).__name__} does not implement "
                "user_defined_return_mapping; cannot use PythonAnalyticalIntegrator."
            )
        return None

    def _try_user_tangent(self, rm, stress_n, state_n, C):
        """Attempt user_defined_tangent; return ddsdde array or None."""
        if self._method == "numerical_newton":
            return None
        ddsdde = self._model.user_defined_tangent(
            rm.stress, rm.state, rm.dlambda, C, state_n
        )
        if ddsdde is not None:
            return ddsdde
        if self._method == "user_defined":
            raise NotImplementedError(
                f"{type(self._model).__name__} does not implement "
                "user_defined_tangent; cannot use PythonAnalyticalIntegrator."
            )
        return None

    def _numerical_newton(self, stress_trial, state_n):
        """Unified NR return mapping.

        Uses self._model / self._max_iter / self._tol / self._raise_on_nonconverged.
        Unknown vector: x = [σ (ntens), Δλ (1), q_implicit_non_stress (n_imp)].
        """
        from manforge.core.residual import _call_update_state
        model = self._model
        ntens = model.ntens
        implicit_keys_non_stress = sorted(
            k for k in model.implicit_state_names if k != "stress"
        )
        explicit_keys_non_stress = set(
            k for k in model.state_names
            if k != "stress" and k not in model.implicit_state_names
        )
        do_implicit_stress = model.state_fields["stress"].kind == "implicit"

        residual_fn, n_unknown, unflatten_implicit = build_residual(
            model, stress_trial, state_n
        )
        n_implicit = n_unknown - ntens - 1

        implicit_state_n = {k: state_n[k] for k in implicit_keys_non_stress}
        flat_impl_n, _ = _flatten_state(implicit_state_n)
        x = anp.concatenate(
            [anp.array(stress_trial), anp.array([0.0]), flat_impl_n]
        )

        residual_history = []
        n_iterations = 0
        converged = False
        for _ in range(self._max_iter):
            R = residual_fn(x)
            norm = float(np.linalg.norm(np.array(R)))
            residual_history.append(norm)
            if norm < self._tol:
                converged = True
                break
            J = autograd.jacobian(residual_fn)(x)
            dx = np.linalg.solve(np.array(J), np.array(R))
            x = x - anp.array(dx)
            n_iterations += 1

        if not converged and self._raise_on_nonconverged:
            raise RuntimeError(
                f"_numerical_newton: NR did not converge in {self._max_iter} "
                f"iterations (||R||_2 = "
                f"{float(np.linalg.norm(np.array(residual_fn(x)))):.3e}, "
                f"tol = {self._tol:.3e})"
            )

        stress = x[:ntens]
        dlambda_val = x[ntens]
        q_imp = unflatten_implicit(x[ntens + 1:]) if n_implicit > 0 else {}
        model_name = type(model).__name__

        if explicit_keys_non_stress:
            state_trial_dict = dict(state_n)
            state_trial_dict["stress"] = stress
            q_exp = _call_update_state(
                model, dlambda_val, state_n, state_trial_dict,
                explicit_keys_non_stress, model_name,
                require_stress=(not do_implicit_stress),
            )
            if not do_implicit_stress and "stress" in q_exp:
                stress = q_exp["stress"]
            state_new = {"stress": stress, **q_imp,
                         **{k: v for k, v in q_exp.items() if k != "stress"}}
        else:
            state_new = {"stress": stress, **q_imp}

        return stress, state_new, anp.array(dlambda_val), n_iterations, residual_history, converged

    def _consistent_tangent(self, rm, stress_n, state_n):
        """Consistent (algorithmic) tangent dσ_{n+1}/dΔε via implicit differentiation."""
        model = self._model
        ntens = model.ntens
        stress = rm.stress
        state = rm.state
        dlambda = rm.dlambda

        state_with_stress_dict = dict(state)
        state_with_stress_dict["stress"] = stress
        state_full = _wrap_state(state_with_stress_dict, model)
        C_n = model.elastic_stiffness(state_n)
        C_conv = model.elastic_stiffness(state_full)
        n_conv = autograd.grad(
            lambda s: model.yield_function(_state_with_stress(state_full, s))
        )(anp.array(stress))
        stress_trial = anp.array(stress) + float(dlambda) * (C_conv @ n_conv)

        state_n_full = dict(state_n)
        if "stress" not in state_n_full:
            state_n_full["stress"] = anp.zeros(ntens)
        implicit_keys_non_stress = sorted(
            k for k in model.implicit_state_names if k != "stress"
        )
        implicit_state = {k: state[k] for k in implicit_keys_non_stress}
        flat_impl, _ = _flatten_state(implicit_state)

        residual_fn, n_unknown, _ = build_residual(model, stress_trial, state_n_full)
        x_conv = anp.concatenate(
            [anp.array(stress), anp.array([float(dlambda)]), flat_impl]
        )
        A = autograd.jacobian(residual_fn)(x_conv)
        rhs = np.vstack(
            [np.array(C_n), np.zeros((n_unknown - ntens, ntens))]
        )
        dxde = np.linalg.solve(np.array(A), rhs)
        return anp.array(dxde[:ntens, :])

    def return_mapping(self, stress_trial, state_n) -> ReturnMappingResult:
        """Perform the plastic correction (return mapping) for one load increment.

        Projects the elastic trial stress back onto the yield surface.
        Does NOT compute the consistent tangent — call :meth:`stress_update`
        for the complete constitutive integration including the tangent.

        Parameters
        ----------
        stress_trial : array-like, shape (ntens,)
            Elastic trial stress σ_trial = σ_n + C Δε.
        state_n : dict
            Internal state at the beginning of the increment.
        """
        C_n = self._model.elastic_stiffness(state_n)
        rm = self._try_user_return_mapping(stress_trial, C_n, state_n)
        if rm is not None:
            return rm

        stress, state_new, dlambda, n_iter, res_hist, converged = self._numerical_newton(
            stress_trial, state_n,
        )
        return ReturnMappingResult(
            stress=stress,
            state=state_new,
            dlambda=dlambda,
            n_iterations=n_iter,
            residual_history=res_hist,
            converged=converged,
        )

    def stress_update(self, strain_inc, stress_n, state_n) -> StressUpdateResult:
        """Perform a complete constitutive stress update for one load increment.

        Executes: elastic trial → yield check → return mapping → consistent tangent.
        """
        C_n = self._model.elastic_stiffness(state_n)
        stress_trial = stress_n + C_n @ strain_inc

        # Yield check: build a state that includes stress for the new API
        from manforge.core.state import _state_with_stress as _swst
        state_trial = _swst(state_n, stress_trial)
        f_trial = self._model.yield_function(state_trial)
        if f_trial <= 0.0:
            return StressUpdateResult(
                return_mapping=None,
                ddsdde=C_n,
                stress_trial=stress_trial,
                is_plastic=False,
                _state_n=state_n,
            )

        rm = self.return_mapping(stress_trial, state_n)

        ddsdde = self._try_user_tangent(rm, stress_n, state_n, C_n)
        if ddsdde is None:
            ddsdde = self._consistent_tangent(rm, stress_n, state_n)

        return StressUpdateResult(
            return_mapping=rm,
            ddsdde=ddsdde,
            stress_trial=stress_trial,
            is_plastic=True,
            _state_n=state_n,
        )


class PythonIntegrator(_PythonIntegratorBase):
    """Wraps a MaterialModel using the ``"auto"`` solver strategy.

    Uses ``model.user_defined_return_mapping`` when available, otherwise
    falls back to numerical Newton-Raphson.

    Parameters
    ----------
    model : MaterialModel
    max_iter : int, optional
        Maximum Newton-Raphson iterations (default 50).
    tol : float, optional
        NR convergence tolerance (default 1e-10).
    raise_on_nonconverged : bool, optional
        Raise ``RuntimeError`` if NR does not converge (default ``True``).
    """

    _method = "auto"


class PythonNumericalIntegrator(_PythonIntegratorBase):
    """Wraps a MaterialModel always using numerical Newton-Raphson.

    Parameters
    ----------
    model : MaterialModel
    max_iter : int, optional
        Maximum Newton-Raphson iterations (default 50).
    tol : float, optional
        NR convergence tolerance (default 1e-10).
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
    max_iter : int, optional
        Maximum Newton-Raphson iterations (default 50, ignored for analytical solver).
    tol : float, optional
        NR convergence tolerance (default 1e-10, ignored for analytical solver).
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
    fortran : FortranModule
        f2py module wrapper (from :class:`~manforge.verification.FortranModule`).
    subroutine : str
        Name of the f2py-callable core-logic subroutine.
    dimension : StressDimension, optional
        Dimensionality descriptor.  Defaults to :data:`SOLID_3D` (ntens=6).
    initial_state : callable or dict
        Either a no-arg callable returning a state dict (e.g.
        ``model.initial_state``) or the dict value directly.
    param_fn : callable
        ``param_fn() -> tuple`` — material parameters in UMAT positional order.
        Note: unlike ``CrosscheckStrainDriver`` / ``CrosscheckStressDriver``,
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
        hook used by :class:`~manforge.verification.CrosscheckStrainDriver` /
        :class:`~manforge.verification.CrosscheckStressDriver`).

    Notes
    -----
    Fortran UMATs do not expose a per-step ``is_plastic`` flag or ``dlambda``.
    The resulting :class:`~manforge.core.result.StressUpdateResult` will
    have ``is_plastic=None``, ``dlambda=nan``, and ``stress_trial=None``.
    Driver loops that gate on ``step.result.is_plastic`` should guard with
    ``if result.is_plastic is True``.  ``stress_trial`` is not reconstructed on
    the Python side — the UMAT is treated as a closed verification target.
    ``ddsdde`` is taken entirely from the UMAT output; if ``parse_umat_ddsdde``
    raises, the exception propagates (no Python fallback).
    """

    def __init__(
        self,
        fortran,
        subroutine: str,
        *,
        dimension: StressDimension = SOLID_3D,
        initial_state,
        param_fn: Callable[[], tuple],
        state_names: list[str],
        state_to_args: Callable[[dict], tuple] | None = None,
        parse_umat_return: Callable[[tuple], tuple[np.ndarray, dict]] | None = None,
        parse_umat_ddsdde: Callable[[tuple], np.ndarray] | None = None,
    ) -> None:
        self._fortran = fortran
        self._subroutine = subroutine
        self.dimension = dimension
        self._initial_state = initial_state
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

    @classmethod
    def from_model(
        cls,
        fortran,
        subroutine: str,
        model,
        *,
        dimension: StressDimension | None = None,
        param_fn: Callable[[], tuple] | None = None,
        state_names: list[str] | None = None,
        initial_state=None,
        state_to_args: Callable[[dict], tuple] | None = None,
        parse_umat_return: Callable[[tuple], tuple[np.ndarray, dict]] | None = None,
        parse_umat_ddsdde: Callable[[tuple], np.ndarray] | None = None,
    ) -> "FortranIntegrator":
        """Build a FortranIntegrator from a MaterialModel.

        Fills ``param_fn``, ``state_names``, ``initial_state``, and
        ``dimension`` from *model* attributes.  Any kwarg passed explicitly
        overrides the auto-derived value.

        ``param_fn`` is auto-generated as
        ``lambda: tuple(getattr(model, n) for n in model.param_names)``,
        matching the convention that Fortran UMAT argument order equals
        ``model.param_names`` order.  Pass an explicit ``param_fn`` when the
        Fortran subroutine uses a different argument order.

        Parameters
        ----------
        fortran : FortranModule
        subroutine : str
            f2py-callable subroutine name.
        model : MaterialModel
            Must have ``param_names``, ``state_names``, ``initial_state``,
            and ``dimension`` attributes.

        Examples
        --------
        ::

            model = J2Isotropic3D(E=210e3, nu=0.3, sigma_y0=250.0, H=1e3)
            fortran = FortranModule("j2_isotropic_3d")
            fc_int = FortranIntegrator.from_model(fortran, "j2_isotropic_3d", model)

            # Override param_fn when Fortran argument order differs:
            fc_int = FortranIntegrator.from_model(
                fortran, "j2_isotropic_3d", model,
                param_fn=lambda: (model.E, model.nu, model.sigma_y0, model.H),
            )
        """
        names = list(model.param_names)
        _param_fn = param_fn if param_fn is not None else (
            lambda: tuple(getattr(model, n) for n in names)
        )
        _state_names = state_names if state_names is not None else list(model.state_names)
        _initial_state = initial_state if initial_state is not None else model.initial_state
        _dimension = (
            dimension if dimension is not None
            else getattr(model, "dimension", SOLID_3D)
        )
        return cls(
            fortran,
            subroutine,
            dimension=_dimension,
            param_fn=_param_fn,
            state_names=_state_names,
            initial_state=_initial_state,
            state_to_args=state_to_args,
            parse_umat_return=parse_umat_return,
            parse_umat_ddsdde=parse_umat_ddsdde,
        )

    @property
    def ntens(self) -> int:
        return self.dimension.ntens

    def initial_state(self) -> dict:
        return _resolve_callable_or_value(self._initial_state)

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
        ddsdde = np.asarray(self._pudd(ret), dtype=np.float64)

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
            stress_trial=None,
            is_plastic=None,
            _state_n=state_n,
        )


# ---------------------------------------------------------------------------
# Default hook helpers (mirrors those in crosscheck_driver.py)
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
