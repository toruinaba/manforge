"""StressIntegrator protocol and _PythonIntegratorBase implementation."""

from __future__ import annotations

import autograd
import autograd.numpy as anp
import numpy as np

from manforge.core.result import ReturnMappingResult, StressUpdateResult
from manforge.core.dimension import StressDimension
from manforge.simulation._residual import build_residual, build_state_from_x, _wrap_state
from manforge._typing import FloatArray, Stiffness, StressVec, StateDict


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

    def initial_state(self) -> StateDict:
        raise NotImplementedError

    def elastic_stiffness(self) -> Stiffness:
        raise NotImplementedError

    def stress_update(
        self, strain_inc: StressVec, stress_n: StressVec, state_n: StateDict
    ) -> StressUpdateResult:
        raise NotImplementedError


# ---------------------------------------------------------------------------
# _PythonIntegratorBase
# ---------------------------------------------------------------------------

class _PythonIntegratorBase:
    """Shared implementation for Python-side integrators.

    Subclasses set the ``_method`` class variable to select the solver strategy.

    Parameters
    ----------
    nr_jacobian : {"exact", "broyden"}
        How the NR Jacobian is obtained on iterations after the first.
        ``"exact"`` (default) re-derives it by autodiff every iteration —
        robust, but the autodiff Jacobian dominates the step cost for models
        without hand-derived solvers.  ``"broyden"`` computes the exact
        Jacobian once per step and then applies Broyden rank-1 updates,
        recomputing the exact Jacobian whenever the residual norm fails to
        decrease.  Typically a few times faster at the cost of 1–2 extra
        iterations; the *consistent tangent* is unaffected (always exact).
    """

    _method: str = "auto"

    def __init__(
        self,
        model,
        *,
        max_iter: int = 50,
        tol: float = 1e-10,
        raise_on_nonconverged: bool = True,
        nr_jacobian: str = "exact",
    ) -> None:
        if nr_jacobian not in ("exact", "broyden"):
            raise ValueError(
                f"nr_jacobian must be 'exact' or 'broyden', got {nr_jacobian!r}"
            )
        self._model = model
        self._max_iter = max_iter
        self._tol = tol
        self._raise_on_nonconverged = raise_on_nonconverged
        self._nr_jacobian = nr_jacobian

    @property
    def dimension(self) -> StressDimension:
        return self._model.dimension

    @property
    def ntens(self) -> int:
        return self._model.ntens

    def initial_state(self) -> StateDict:
        return self._model.initial_state()

    def elastic_stiffness(self, state: "StateDict | None" = None) -> Stiffness:
        return self._model.elastic_stiffness(state)

    def _try_user_return_mapping(
        self, stress_trial: StressVec, C: Stiffness, state_n: StateDict
    ) -> "ReturnMappingResult | None":
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

    def _try_user_tangent(
        self, rm: ReturnMappingResult, stress_n: StressVec, state_n: StateDict, C: Stiffness,
        stress_trial: "StressVec | None" = None,
        strain_inc: "FloatArray | None" = None,
    ) -> "Stiffness | None":
        """Attempt user_defined_tangent; return ddsdde array or None."""
        if self._method == "numerical_newton":
            return None
        ddsdde = self._model.user_defined_tangent(
            rm.stress, rm.state, rm.dlambda, C, state_n,
            stress_trial=stress_trial, strain_inc=strain_inc,
        )
        if ddsdde is not None:
            return ddsdde
        if self._method == "user_defined":
            raise NotImplementedError(
                f"{type(self._model).__name__} does not implement "
                "user_defined_tangent; cannot use PythonAnalyticalIntegrator."
            )
        return None

    def _numerical_newton(
        self, stress_trial: StressVec, state_n: StateDict, strain_inc: "FloatArray | None" = None
    ) -> tuple:
        """Unified NR return mapping.

        Unknown vector: x = [σ (ntens), Δλ (1), q_implicit_non_stress (declaration order)].
        """
        model = self._model
        residual_fn, layout = build_residual(model, stress_trial, state_n, strain_inc)

        x = layout.pack(
            stress_trial, 0.0,
            {k: state_n[k] for k in layout.implicit_keys}
        )

        # The residual rows carry stress units (R_σ and R_Δλ = f both scale
        # with σ), so a purely absolute tolerance is unit-dependent: at Pa
        # scale (σ ~ 1e8) the default 1e-10 sits below float64 round-off and
        # NR can never satisfy it.  Scale by the trial-stress magnitude —
        # not the initial residual, which is tiny whenever NR starts near the
        # solution — floored at 1 to keep the absolute reading for O(1) units.
        tol_eff = self._tol * max(
            1.0, float(np.max(np.abs(np.asarray(stress_trial, dtype=float))))
        )

        jac_fn = autograd.jacobian(residual_fn)  # type: ignore[call-arg]
        use_broyden = self._nr_jacobian == "broyden"

        residual_history = []
        n_iterations = 0
        converged = False
        J: "np.ndarray | None" = None
        R_prev: "np.ndarray | None" = None
        dx_prev: "np.ndarray | None" = None
        for _ in range(self._max_iter):
            R = np.array(residual_fn(x))
            norm = float(np.linalg.norm(R))
            residual_history.append(norm)
            if norm < tol_eff:
                converged = True
                break

            if J is None or not use_broyden:
                J = np.array(jac_fn(x))
            elif norm >= residual_history[-2]:
                # The Broyden approximation stopped making progress — fall
                # back to the exact Jacobian for this iteration.
                J = np.array(jac_fn(x))
            else:
                # Broyden "good" rank-1 update: J += (ΔR − J Δx) Δxᵀ / (ΔxᵀΔx).
                dR = R - R_prev
                J = J + np.outer(dR - J @ dx_prev, dx_prev) / float(dx_prev @ dx_prev)

            dx = np.linalg.solve(J, R)
            x = x - anp.array(dx)
            R_prev, dx_prev = R, -dx
            n_iterations += 1

        if not converged and self._raise_on_nonconverged:
            raise RuntimeError(
                f"_numerical_newton: NR did not converge in {self._max_iter} "
                f"iterations (||R||_2 = "
                f"{float(np.linalg.norm(np.array(residual_fn(x)))):.3e}, "
                f"tol = {tol_eff:.3e})"
            )

        state_new = build_state_from_x(model, x, state_n, layout, stress_trial, strain_inc)
        sigma, dlambda_val, _ = layout.unpack(np.asarray(x))

        # A small residual only says NR found *a* root of f = 0 and the flow
        # rule.  That system has roots with Δλ < 0, which violate KKT loading /
        # unloading: plastic strain cannot run backwards.  Large Δε can push
        # σ_trial past the admissible branch, and NR then converges tightly
        # (||R|| ~ 1e-14) onto a non-physical root, so the residual norm alone
        # cannot reject it.  Subdividing the increment recovers the admissible
        # branch, which is why this is reported as non-convergence rather than
        # returned as a solution.
        if converged and float(dlambda_val) < 0.0:
            converged = False
            if self._raise_on_nonconverged:
                raise RuntimeError(
                    f"_numerical_newton: NR converged to an inadmissible root "
                    f"(Δλ = {float(dlambda_val):.3e} < 0 violates KKT; "
                    f"||R||_2 = {residual_history[-1]:.3e}).  The strain "
                    f"increment is likely too large for the admissible branch."
                )

        return (
            state_new["stress"],
            state_new,
            anp.array(dlambda_val),
            n_iterations,
            residual_history,
            converged,
        )

    def _consistent_tangent(
        self, rm: ReturnMappingResult, stress_n: StressVec, state_n: StateDict,
        strain_inc: "FloatArray | None" = None,
        stress_trial: "StressVec | None" = None,
    ) -> Stiffness:
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
        if stress_trial is None:
            # Reached only via a bare return_mapping() call, which never had a
            # σ_trial of its own.  Invert the σ row of the converged residual
            # instead of re-deriving the direction from grad(yield_function),
            # which is wrong for any model that overrides flow().
            C_conv = model.elastic_stiffness(state_full)
            n_conv = model.flow(state_full).strain_like
            trial: StressVec = anp.array(stress) + float(dlambda) * (C_conv @ n_conv)
        else:
            trial = stress_trial

        residual_fn, layout = build_residual(model, trial, state_n, strain_inc)

        q_imp = {k: state[k] for k in layout.implicit_keys}
        x_conv = layout.pack(stress, dlambda, q_imp)

        A = autograd.jacobian(residual_fn)(anp.array(x_conv))  # type: ignore[call-arg]
        rhs = np.vstack(
            [np.array(C_n), np.zeros((layout.n_unknown - ntens, ntens))]
        )
        dxde = np.linalg.solve(np.array(A), rhs)
        return anp.array(dxde[:ntens, :])

    def _elastic_state(
        self, state_n: StateDict, stress_trial: StressVec,
        strain_inc: "FloatArray | None" = None,
    ) -> StateDict:
        """Apply ``elastic_update_state`` to state_n; omitted fields stay put."""
        from manforge.core.state import StateUpdate, _validate_state_items

        model = self._model
        returned = model.elastic_update_state(
            _wrap_state(state_n, model),
            stress_trial=stress_trial,
            strain_inc=strain_inc,
        )
        if not returned:
            return state_n
        explicit_keys = {
            k for k, f in model.state_fields.items()
            if f.kind == "explicit" and k != "stress"
        }
        updated = _validate_state_items(
            returned, explicit_keys, StateUpdate,
            "elastic_update_state", type(model).__name__,
            allow_subset=True,
        )
        assert isinstance(updated, dict)
        return {**state_n, **updated}

    def return_mapping(
        self, stress_trial: StressVec, state_n: StateDict,
        strain_inc: "FloatArray | None" = None,
    ) -> ReturnMappingResult:
        """Perform the plastic correction (return mapping) for one load increment.

        Both the ``user_defined`` and ``numerical_newton`` paths honour
        ``raise_on_nonconverged``: a ``converged=False`` result raises
        ``RuntimeError`` unless the integrator was built with
        ``raise_on_nonconverged=False``, in which case the flag is returned as-is.
        """
        C_n = self._model.elastic_stiffness(state_n)
        rm = self._try_user_return_mapping(stress_trial, C_n, state_n)
        if rm is not None:
            if not rm.converged and self._raise_on_nonconverged:
                res = rm.residual_history[-1] if rm.residual_history else float("nan")
                raise RuntimeError(
                    f"{type(self._model).__name__}.user_defined_return_mapping "
                    f"reported converged=False after {rm.n_iterations} iterations "
                    f"(final residual = {res:.3e})"
                )
            return rm

        stress, state_new, dlambda, n_iter, res_hist, converged = self._numerical_newton(
            stress_trial, state_n, strain_inc,
        )
        return ReturnMappingResult(
            stress=stress,
            state=state_new,
            dlambda=dlambda,
            n_iterations=n_iter,
            residual_history=res_hist,
            converged=converged,
        )

    def stress_update(
        self, strain_inc: StressVec, stress_n: StressVec, state_n: StateDict
    ) -> StressUpdateResult:
        """Perform a complete constitutive stress update for one load increment."""
        C_n = self._model.elastic_stiffness(state_n)
        stress_trial = stress_n + C_n @ strain_inc

        is_elastic = self._model.elastic_check(
            _wrap_state(state_n, self._model),
            stress_trial=stress_trial,
            strain_inc=strain_inc,
        )
        if not isinstance(is_elastic, bool):
            raise TypeError(
                f"{type(self._model).__name__}.elastic_check must return a bool "
                f"(it selects a code path and is never differentiated), got "
                f"{type(is_elastic).__name__}"
            )
        if is_elastic:
            return StressUpdateResult(
                return_mapping=None,
                ddsdde=C_n,
                stress_trial=stress_trial,
                is_plastic=False,
                _state_n=self._elastic_state(state_n, stress_trial, strain_inc),
            )

        rm = self.return_mapping(stress_trial, state_n, strain_inc)

        # user_defined tangents stay eager: they are cheap, and _method ==
        # "user_defined" must raise NotImplementedError here, not at first
        # access.  Only the autodiff tangent is deferred — StrainDriver never
        # reads ddsdde, and its Jacobian costs as much as several NR iterations.
        ddsdde = self._try_user_tangent(rm, stress_n, state_n, C_n, stress_trial, strain_inc)
        if ddsdde is None:
            ddsdde = lambda: self._consistent_tangent(
                rm, stress_n, state_n, strain_inc, stress_trial
            )

        return StressUpdateResult(
            return_mapping=rm,
            ddsdde=ddsdde,
            stress_trial=stress_trial,
            is_plastic=True,
            _state_n=state_n,
        )
