"""FortranIntegrator and default UMAT hook helpers."""

from __future__ import annotations

import math
from typing import Any, Callable

import numpy as np

from manforge.core.result import ReturnMappingResult, StressUpdateResult
from manforge.core.dimension import StressDimension, SOLID_3D


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
    elastic_stiffness_fn : callable, optional
        ``elastic_stiffness_fn(state) -> ndarray`` — returns the elastic
        stiffness matrix.  When provided, the instance exposes an
        ``elastic_stiffness`` method that :class:`~manforge.simulation.StressDriver`
        can use to compute the initial strain increment without probing the UMAT.
        When ``None`` (default), ``StressDriver`` falls back to a zero-strain
        UMAT probe.  :meth:`from_model` fills this automatically from
        ``model.elastic_stiffness`` when the model provides it.
    state_to_args : callable, optional
        ``state_to_args(state_dict) -> tuple`` packs the state dict into
        positional UMAT args.  Defaults to ``state_names``-order packing.
    parse_umat_return : callable, optional
        ``parse_umat_return(ret) -> (stress_ndarray, state_dict)`` unpacks the
        f2py return tuple.  Defaults to scanning ``state_names`` order.
    parse_umat_ddsdde : callable, optional
        ``parse_umat_ddsdde(ret) -> ndarray`` extracts the consistent tangent
        from the f2py return tuple.  When ``None``, the tangent is read from
        the first 2-D array found in the trailing returns.  If extraction fails
        and ``elastic_stiffness_fn`` is provided, the elastic stiffness is used
        as a fallback; otherwise the exception propagates.

    Notes
    -----
    Fortran UMATs do not expose a per-step ``is_plastic`` flag or ``dlambda``.
    The resulting :class:`~manforge.core.result.StressUpdateResult` will
    have ``is_plastic=None``, ``dlambda=nan``, and ``stress_trial=None``.
    Driver loops that gate on ``step.result.is_plastic`` should guard with
    ``if result.is_plastic is True``.  ``stress_trial`` is not reconstructed on
    the Python side — the UMAT is treated as a closed verification target.
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
        elastic_stiffness_fn: Callable | None = None,
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

        if elastic_stiffness_fn is not None:
            # Bind as an instance method so hasattr(self, "elastic_stiffness")
            # returns True only when a callable was actually provided.
            import types
            self.elastic_stiffness = types.MethodType(
                lambda self_, state=None: np.asarray(
                    elastic_stiffness_fn(state), dtype=np.float64
                ),
                self,
            )

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
        elastic_stiffness_fn: Callable | None = None,
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
            and ``dimension`` attributes.  Auto-derived ``state_names``
            excludes ``"stress"`` because ``stress_update`` always passes
            ``stress_n`` as a dedicated positional argument — not as a state
            slot.  Pass ``state_names=`` explicitly only if your UMAT expects
            an additional stress array inside the state argument list.
            If *model* has an ``elastic_stiffness`` method it is wired up
            automatically so that :class:`~manforge.simulation.StressDriver`
            can avoid probing the UMAT with a zero-strain call.

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
        # Exclude "stress" from the auto-derived list: Fortran UMATs receive
        # stress as a dedicated positional argument (stress_in), not as a
        # state-array slot.  Pass state_names= explicitly to override.
        _state_names = (
            state_names
            if state_names is not None
            else [n for n in model.state_names if n != "stress"]
        )
        _initial_state = initial_state if initial_state is not None else model.initial_state
        _dimension = (
            dimension if dimension is not None
            else getattr(model, "dimension", SOLID_3D)
        )
        if elastic_stiffness_fn is None and hasattr(model, "elastic_stiffness"):
            elastic_stiffness_fn = model.elastic_stiffness
        return cls(
            fortran,
            subroutine,
            dimension=_dimension,
            param_fn=_param_fn,
            state_names=_state_names,
            initial_state=_initial_state,
            elastic_stiffness_fn=elastic_stiffness_fn,
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
        try:
            ddsdde = np.asarray(self._pudd(ret), dtype=np.float64)
        except (ValueError, IndexError):
            # UMAT does not return ddsdde (e.g. mock subroutines without tangent).
            # Fall back to elastic stiffness when available, otherwise re-raise.
            if not hasattr(self, "elastic_stiffness"):
                raise
            ddsdde = np.asarray(self.elastic_stiffness(state_n), dtype=np.float64)

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
