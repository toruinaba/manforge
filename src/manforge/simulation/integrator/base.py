"""StressIntegrator protocol and _PythonIntegratorBase implementation."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import autograd
import autograd.numpy as anp
import numpy as np

from manforge.core.result import ReturnMappingResult, StressUpdateResult
from manforge.core.dimension import StressDimension
from manforge.core.state import Increment
from manforge.simulation._residual import build_residual, build_state_from_x, _wrap_state
from manforge._typing import Stiffness, StressVec, StateDict


# ---------------------------------------------------------------------------
# Protocol
# ---------------------------------------------------------------------------

@runtime_checkable
class StressIntegrator(Protocol):
    """Contract every constitutive integrator fulfils for the drivers.

    Structural: an integrator does not inherit from this class, it only has to
    provide every member below.  Drivers check it with ``isinstance`` at
    construction, so a missing member fails there rather than deep inside a
    run.  :class:`PythonIntegrator`, :class:`FortranIntegrator` and
    :class:`SubSteppingIntegrator` are the ready-made implementations.

    ``raise_on_nonconverged`` is the single switch for non-convergence.  It
    governs the integrator's own return mapping, and the drivers apply the same
    setting to their outer Newton-Raphson loop: ``True`` raises
    ``RuntimeError``, ``False`` reports ``converged=False`` as a value.
    """

    @property
    def dimension(self) -> StressDimension: ...

    @property
    def ntens(self) -> int: ...

    @property
    def raise_on_nonconverged(self) -> bool: ...

    def initial_state(self) -> StateDict: ...

    def increment(
        self, strain_inc: StressVec, stress_n: StressVec, state_n: StateDict
    ) -> "Increment | None":
        """The step context for this increment, trial state included.

        ``None`` from an integrator with no model-side increment to report — a
        Fortran UMAT is a closed target and exposes no trial state.
        """
        ...

    def elastic_tangent(self, stress_n: StressVec, state_n: StateDict) -> Stiffness:
        """Elastic tangent dσ/dΔε at Δε = 0 from the given state.

        Stress- and mixed-controlled drivers start their outer iteration from
        it.  Every integrator must supply one; an implementation without a
        closed form evaluates ``stress_update`` at a zero increment.
        """
        ...

    def stress_update(
        self, strain_inc: StressVec, stress_n: StressVec, state_n: StateDict
    ) -> StressUpdateResult: ...


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

    @property
    def raise_on_nonconverged(self) -> bool:
        return self._raise_on_nonconverged

    def initial_state(self) -> StateDict:
        return self._model.initial_state()

    def elastic_tangent(self, stress_n: StressVec, state_n: StateDict) -> Stiffness:
        """The model's ``elastic_stiffness`` at ``state_n``.

        For a model that overrides ``elastic_predictor`` this is its reference
        stiffness rather than the derivative of the predictor.  The drivers
        only use it as a starting guess, so the outer iteration still
        converges on the true response.
        """
        return self._model.elastic_stiffness(state_n)

    # ------------------------------------------------------------------
    # Increment: the trial state and the step context built around it
    # ------------------------------------------------------------------

    def _has_custom_predictor(self) -> bool:
        from manforge.core.material import MaterialModel
        return type(self._model).elastic_predictor is not MaterialModel.elastic_predictor

    def _predict(self, state_n, strain_inc):
        """``(σ_trial, non-stress trial updates)`` from the model's predictor hook.

        Validated on every call: it is evaluated once per step plus once per
        autodiff trace, so the cost is negligible next to the NR.
        """
        from manforge.core.state import StateResidual, StateUpdate, _validate_state_items

        model = self._model
        name = type(model).__name__
        returned = model.elastic_predictor(state_n, strain_inc=strain_inc)
        if isinstance(returned, list) and any(isinstance(i, StateResidual) for i in returned):
            raise TypeError(
                f"{name}.elastic_predictor returned a StateResidual — the "
                "predictor assigns values, so wrap implicit fields with "
                "self.<field>.update(value), not self.<field>(value)."
            )
        items = _validate_state_items(
            returned, set(model.state_names), StateUpdate, "elastic_predictor", name,
            allow_subset=True,
        )
        assert isinstance(items, dict)
        if "stress" not in items:
            raise ValueError(
                f"{name}.elastic_predictor must return the trial stress "
                "(self.stress.update(sigma_trial))"
            )
        sigma = items.pop("stress")
        return sigma, items

    def increment(
        self, strain_inc: StressVec, stress_n: StressVec, state_n: StateDict
    ) -> Increment:
        """Build the step context: run the elastic predictor, no branch, no NR."""
        model = self._model
        # σ_n is the stress the caller passes, whatever the dict carries.
        state_n_s = _wrap_state({**state_n, "stress": stress_n}, model)
        if self._has_custom_predictor():
            stress_trial, trial_updates = self._predict(state_n_s, strain_inc)
        else:
            # Kept as the literal expression (not routed through the default
            # hook) so the default path stays bit-identical.
            stress_trial = stress_n + model.elastic_stiffness(state_n_s) @ strain_inc
            trial_updates = {}
        trial = _wrap_state(
            {**state_n_s.as_dict(), **trial_updates, "stress": stress_trial}, model
        )
        return Increment(state_n=state_n_s, strain_inc=strain_inc, trial=trial)

    def _perturbed(self, inc: Increment, d) -> Increment:
        """``inc`` with Δε → Δε + d, the trial state following through the predictor.

        Used to differentiate the residual with respect to the load increment.
        """
        model = self._model
        deps_d = anp.array(inc.strain_inc) + d
        if self._has_custom_predictor():
            # Non-linear elastic predictor: σ_trial(Δε + d), not σ_trial + C_n d.
            stress_d, updates_d = self._predict(inc.state_n, deps_d)
            trial_d = _wrap_state(
                {**inc.state_n.as_dict(), **updates_d, "stress": stress_d}, model
            )
        else:
            C_n = anp.array(model.elastic_stiffness(inc.state_n))
            stress_d = anp.array(inc.trial["stress"]) + anp.dot(C_n, d)
            trial_d = inc.trial.with_stress(stress_d)
        return Increment(state_n=inc.state_n, strain_inc=deps_d, trial=trial_d)

    # ------------------------------------------------------------------
    # Hand-written hooks
    # ------------------------------------------------------------------

    def _try_user_return_mapping(self, inc: Increment) -> "ReturnMappingResult | None":
        """Attempt user_defined_return_mapping; return ReturnMappingResult or None."""
        if self._method == "numerical_newton":
            return None
        rm = self._model.user_defined_return_mapping(inc)
        if rm is not None:
            return rm
        if self._method == "user_defined":
            raise NotImplementedError(
                f"{type(self._model).__name__} does not implement "
                "user_defined_return_mapping; cannot use PythonAnalyticalIntegrator."
            )
        return None

    def _try_user_tangent(
        self, rm: ReturnMappingResult, inc: Increment
    ) -> "Stiffness | None":
        """Attempt user_defined_tangent; return ddsdde array or None."""
        if self._method == "numerical_newton":
            return None
        model = self._model
        new = _wrap_state({**rm.state, "stress": rm.stress}, model)
        ddsdde = model.user_defined_tangent(new, rm.dlambda, inc)
        if ddsdde is not None:
            return ddsdde
        if self._method == "user_defined":
            raise NotImplementedError(
                f"{type(model).__name__} does not implement "
                "user_defined_tangent; cannot use PythonAnalyticalIntegrator."
            )
        return None

    def _elastic_updates(self, inc: Increment) -> StateDict:
        """Only the fields ``elastic_update_state`` returned (possibly empty)."""
        from manforge.core.state import StateUpdate, _validate_state_items

        model = self._model
        returned = model.elastic_update_state(inc)
        if not returned:
            return {}
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
        return updated

    def _elastic_tangent(self, inc: Increment):
        """Tangent of an elastic step: an array, or a thunk when it needs autodiff.

        ===========================  ============================  =====================
        integrator                   default predictor             predictor overridden
        ===========================  ============================  =====================
        PythonNumericalIntegrator    elastic_stiffness(state_n)    autodiff of predictor
        PythonAnalyticalIntegrator   elastic_stiffness(state_n)    model.elastic_tangent
        PythonIntegrator             elastic_stiffness(state_n)    model.elastic_tangent
                                                                   if given, else autodiff
        ===========================  ============================  =====================
        """
        model = self._model
        if not self._has_custom_predictor():
            return model.elastic_stiffness(inc.state_n)

        if self._method != "numerical_newton":
            # Eager, like the plastic user_defined tangent: cheap, and
            # _method == "user_defined" must raise here, not at first access.
            ddsdde = model.elastic_tangent(inc)
            if ddsdde is not None:
                return ddsdde
            if self._method == "user_defined":
                raise NotImplementedError(
                    f"{type(model).__name__} overrides elastic_predictor but does "
                    "not implement elastic_tangent; cannot use "
                    "PythonAnalyticalIntegrator."
                )

        # Lazy, like the plastic tangent: StrainDriver never reads it.
        deps0 = anp.array(inc.strain_inc)
        return lambda: anp.array(np.array(
            autograd.jacobian(lambda d: self._predict(inc.state_n, d)[0])(deps0)  # type: ignore[call-arg]
        ))

    # ------------------------------------------------------------------
    # Numerical return mapping and consistent tangent
    # ------------------------------------------------------------------

    def _numerical_newton(self, inc: Increment) -> tuple:
        """Unified NR return mapping.

        Unknown vector: x = [σ (ntens), Δλ (1), q_implicit_non_stress (declaration order)].
        Every unknown starts from its trial value (σ_trial, and for an implicit
        state the value the elastic predictor returned, else its step-n value).
        """
        model = self._model
        residual_fn, layout = build_residual(model, inc)
        stress_trial = inc.trial["stress"]

        x = layout.pack(
            stress_trial, 0.0,
            {k: inc.trial[k] for k in layout.implicit_keys}
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

        state_new = build_state_from_x(model, x, inc, layout)
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

    def _consistent_tangent(self, rm: ReturnMappingResult, inc: Increment) -> Stiffness:
        """Consistent (algorithmic) tangent dσ_{n+1}/dΔε via implicit differentiation."""
        model = self._model
        ntens = model.ntens

        residual_fn, layout = build_residual(model, inc)

        q_imp = {k: rm.state[k] for k in layout.implicit_keys}
        x_conv = layout.pack(rm.stress, rm.dlambda, q_imp)

        x_conv_arr = anp.array(x_conv)
        A = autograd.jacobian(residual_fn)(x_conv_arr)  # type: ignore[call-arg]

        # Implicit differentiation: A·dx/dΔε = −dR/dΔε.  The load increment
        # reaches R through the trial state and through Δε itself, and a user
        # residual may use either in any row with any coefficient (e.g. a
        # damage residual scaled by (1−D_{n+1})/(1−D_n)).  Hard-coding
        # dR/dΔε = [−C_n; 0] would silently assume R_σ = σ − σ_trial + h(x)
        # and give a wrong tangent for every other form, so differentiate R
        # in a perturbation d of the increment instead.
        def residual_of_increment(d):
            return residual_fn(x_conv_arr, inc=self._perturbed(inc, d))

        B = autograd.jacobian(residual_of_increment)(anp.zeros(ntens))  # type: ignore[call-arg]
        dxde = -np.linalg.solve(np.array(A), np.array(B))
        return anp.array(dxde[:ntens, :])

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def return_mapping(self, inc: Increment) -> ReturnMappingResult:
        """Perform the plastic correction (return mapping) for one increment.

        ``inc`` comes from :meth:`increment`.  Both the ``user_defined`` and
        ``numerical_newton`` paths honour ``raise_on_nonconverged``: a
        ``converged=False`` result raises ``RuntimeError`` unless the integrator
        was built with ``raise_on_nonconverged=False``, in which case the flag
        is returned as-is.
        """
        rm = self._try_user_return_mapping(inc)
        if rm is not None:
            if not rm.converged and self._raise_on_nonconverged:
                res = rm.residual_history[-1] if rm.residual_history else float("nan")
                raise RuntimeError(
                    f"{type(self._model).__name__}.user_defined_return_mapping "
                    f"reported converged=False after {rm.n_iterations} iterations "
                    f"(final residual = {res:.3e})"
                )
            return rm

        stress, state_new, dlambda, n_iter, res_hist, converged = self._numerical_newton(inc)
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
        model = self._model
        inc = self.increment(strain_inc, stress_n, state_n)

        is_elastic = model.elastic_check(inc)
        if not isinstance(is_elastic, bool):
            raise TypeError(
                f"{type(model).__name__}.elastic_check must return a bool "
                f"(it selects a code path and is never differentiated), got "
                f"{type(is_elastic).__name__}"
            )
        if is_elastic:
            # The trial state is the converged state; elastic_update_state is
            # the post-branch update and is applied on top of it.
            return StressUpdateResult(
                return_mapping=None,
                ddsdde=self._elastic_tangent(inc),
                is_plastic=False,
                increment=inc,
                _elastic_updates=self._elastic_updates(inc),
            )

        rm = self.return_mapping(inc)

        # user_defined tangents stay eager: they are cheap, and _method ==
        # "user_defined" must raise NotImplementedError here, not at first
        # access.  Only the autodiff tangent is deferred — StrainDriver never
        # reads ddsdde, and its Jacobian costs as much as several NR iterations.
        ddsdde = self._try_user_tangent(rm, inc)
        if ddsdde is None:
            ddsdde = lambda: self._consistent_tangent(rm, inc)  # noqa: E731

        return StressUpdateResult(
            return_mapping=rm,
            ddsdde=ddsdde,
            is_plastic=True,
            increment=inc,
        )
