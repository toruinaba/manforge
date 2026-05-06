"""StressIntegrator protocol and _PythonIntegratorBase implementation."""

from __future__ import annotations

import autograd
import autograd.numpy as anp
import numpy as np

from manforge.core.result import ReturnMappingResult, StressUpdateResult
from manforge.core.dimension import StressDimension
from manforge.simulation._residual import build_residual, _flatten_state, _wrap_state
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
# _PythonIntegratorBase
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
        from manforge.simulation._residual import _call_update_state
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
