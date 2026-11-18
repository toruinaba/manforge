"""Abstract base class for constitutive material models."""

from __future__ import annotations

import functools
import inspect
from abc import ABC, abstractmethod
from typing import ClassVar

import autograd.numpy as anp

from manforge.core.state import (
    StateField, State, Increment, DlambdaField, DlambdaResidual, FlowVector,
    StateResidual, StateUpdate,
    collect_state_fields, _make, NTENS, DLAMBDA_FIELD,
)
from manforge.core.dimension import SOLID_3D, StressDimension
from manforge.core.material.fortran_binding import collect_bindings as _collect_bindings
from manforge.core.material.meta import ParamMeta
from manforge._typing import FloatArray, Scalar, Stiffness, StressVec, StateDict
from manforge.core.result import ReturnMappingResult


def _wrap_flow_return_check(cls) -> None:
    """Wrap a subclass's own ``flow`` so an unwrapped return fails at the source.

    Without this, a bare ndarray leaking out of an overridden ``flow`` surfaces
    as ``AttributeError: 'ndarray' object has no attribute 'stress_like'`` at
    whichever use site happens to run first, rather than naming the offending
    method.
    """
    own_flow = cls.__dict__.get("flow")
    if own_flow is None or getattr(own_flow, "_flow_checked", False):
        return

    @functools.wraps(own_flow)
    def checked(self, state, *args, **kwargs):
        n = own_flow(self, state, *args, **kwargs)
        if not isinstance(n, FlowVector):
            raise TypeError(
                f"{type(self).__name__}.flow() must return a FlowVector — wrap the "
                "result in self.strain_flow(...) (engineering shear, Δε_p conjugate) "
                "or self.stress_flow(...) (physical shear, textbook (3/2)s/‖s‖ form), "
                f"got {type(n).__name__}."
            )
        return n

    checked._flow_checked = True  # type: ignore[attr-defined]
    cls.flow = checked


# Hooks the framework calls positionally.  A model written against another
# argument list would not fail loudly: ``update_state(dlambda, state_new, ...)``
# called as ``update_state(new, dlambda, inc)`` just receives its arguments in
# the wrong slots.  The parameter names are therefore checked when the class is
# created.
_HOOK_PARAMS: dict[str, tuple[str, ...]] = {
    "elastic_predictor": ("state_n", "strain_inc"),
    "elastic_tangent": ("inc",),
    "elastic_check": ("inc",),
    "elastic_update_state": ("inc",),
    "update_state": ("new", "dlambda", "inc"),
    "state_residual": ("new", "dlambda", "inc"),
    "user_defined_return_mapping": ("inc",),
    "user_defined_tangent": ("new", "dlambda", "inc"),
}


def _check_hook_signatures(cls) -> None:
    """Reject hooks defined on *cls* with another argument list."""
    for name, expected in _HOOK_PARAMS.items():
        fn = cls.__dict__.get(name)
        if fn is None:
            continue
        got = tuple(inspect.signature(fn).parameters)[1:]   # drop self
        if got != expected:
            raise TypeError(
                f"{cls.__name__}.{name} must take ({', '.join(expected)}), "
                f"got ({', '.join(got)}).  The state at the start of the step, "
                "the strain increment and the trial state are read from the "
                "Increment: inc.state_n, inc.strain_inc, inc.trial."
            )


class MaterialModel(ABC):
    """Abstract base class for constitutive material models.

    Subclasses declare state variables as class-level ``StateField`` attributes
    using :func:`~manforge.core.state.Implicit` and
    :func:`~manforge.core.state.Explicit`::

        from manforge.core.state import Implicit, Explicit, NTENS, SCALAR

        class MyModel(MaterialModel):
            param_names = ["E", "nu", "sigma_y0", "C_k", "gamma"]
            stress = Implicit(shape=NTENS,   doc="Cauchy stress")   # NR unknown
            alpha  = Implicit(shape=NTENS,   doc="backstress tensor")
            ep     = Explicit(shape=SCALAR,  doc="equivalent plastic strain")

    ``state_names`` and ``implicit_state_names`` are derived automatically from
    the field declarations; hand-declaring these lists raises ``TypeError``.

    σ (stress) is always a state field.  If not declared explicitly,
    ``__init_subclass__`` auto-attaches it as ``Explicit(shape=NTENS)``.
    Declare ``stress = Implicit(shape=NTENS)`` to include σ as an NR unknown.

    User methods receive a ``State`` dict-wrapper for all state arguments.
    Access state variables with bracket notation: ``state["stress"]``.

    One stress update, and the hook behind each stage
    -------------------------------------------------
    1. **Trial state** — :meth:`elastic_predictor`.  Default: linear,
       ``σ_trial = σ_n + C(state_n) Δε``, every other field frozen.
    2. **Branch** — :meth:`elastic_check`.  Default: ``f(trial) ≤ 0``.
    3. **Elastic step** — the trial state is the converged state, with
       :meth:`elastic_update_state` applied on top for variables that follow
       σ; the tangent is :meth:`elastic_tangent`.
    4. **Plastic step** — the residual system in ``x = [σ, Δλ, implicit
       states]``, written through :meth:`update_state` (explicit states) and
       :meth:`state_residual` (implicit states), with :meth:`yield_function`
       as the Δλ row and :meth:`flow` as the direction; or
       :meth:`user_defined_return_mapping` in its place.  The tangent comes by
       implicit differentiation, or from :meth:`user_defined_tangent`.

    Every hook after the predictor receives the same
    :class:`~manforge.core.state.Increment` (``inc.state_n``,
    ``inc.strain_inc``, ``inc.trial``).

    Required methods depend on which states are explicit vs implicit:

    - ``update_state(new, dlambda, inc)`` — required when any non-stress state
      is explicit; returns only the explicit-state keys.
    - ``state_residual(new, dlambda, inc)`` — required when any state is
      implicit; returns only the implicit-state keys.

    The framework auto-injects the associative stress default
    (``R_stress = σ − σ_trial + Δλ·C·∂f/∂σ``) when stress is not declared
    ``Implicit``.

    Attributes
    ----------
    param_names : list[str]
        Names of material parameters (keys expected in ``params`` dicts).
    state_fields : dict[str, StateField]
        Ordered mapping of field name → StateField descriptor, collected from
        the MRO by ``__init_subclass__``.  Always contains ``"stress"``.
    state_names : list[str]
        Derived from ``state_fields`` (all field names, in declaration order).
    implicit_state_names : list[str]
        Derived from ``state_fields`` (names where ``kind == "implicit"``).
    dimension : StressDimension
        Dimensionality descriptor (default: ``SOLID_3D``, 6-component 3D).
    ntens : int
        Read-only property; returns ``self.dimension.ntens``.
    """

    param_names: list[str]
    # Optional UI/schema metadata keyed by parameter name.  Keys must be a
    # subset of param_names (validated in __init_subclass__); models without
    # metadata stay valid.
    param_meta: dict[str, "ParamMeta"] = {}
    dimension: StressDimension = SOLID_3D

    def __init__(self, *, dimension: StressDimension = SOLID_3D):
        self.dimension = dimension

    # Derived by __init_subclass__ from StateField descriptors:
    state_fields: dict[str, StateField] = {}
    state_names: list[str] = []
    implicit_state_names: list[str] = []

    # Residual-row label for the Δλ slot.  Override in subclasses to rename the
    # Δλ row in JacobianBlocks.part (e.g. ``dlambda_residual_name = "R_yield"``).
    dlambda_residual_name: str = "dlambda"

    # Framework-provided pseudo-field for the Δλ NR unknown.  Users can
    # optionally return self.dlambda(R_dl) from state_residual to override
    # the default R_dλ = yield_function(state).
    dlambda: ClassVar[DlambdaField] = DLAMBDA_FIELD

    @property
    def params(self) -> dict[str, float]:
        """Material parameters as a dict keyed by :attr:`param_names`."""
        return {name: getattr(self, name) for name in self.param_names}

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        # Checked for every subclass, abstract intermediates included: a hook
        # with a stale argument list is wrong wherever in the hierarchy it sits.
        _check_hook_signatures(cls)
        # Skip intermediate abstract classes that have not yet implemented yield_function.
        if cls.yield_function is MaterialModel.yield_function:
            return
        # param_meta keys must be a subset of param_names (models without
        # metadata remain valid).
        unknown_meta = set(cls.param_meta) - set(cls.param_names)
        if unknown_meta:
            raise ValueError(
                f"{cls.__name__}: param_meta keys {sorted(unknown_meta)} are not "
                f"in param_names {cls.param_names}"
            )
        # Collect StateField descriptors from MRO and derive state_names / implicit_state_names.
        fields = collect_state_fields(cls)
        # Auto-attach "stress" as Explicit if not declared.
        if "stress" not in fields:
            from manforge.core.state import Explicit as _Explicit
            stress_field = _Explicit(shape=NTENS, doc="Cauchy stress")
            object.__setattr__(stress_field, "name", "stress")
            # Insert stress at front of ordered dict.
            fields = {"stress": stress_field, **fields}
        cls.state_fields = fields
        cls.state_names = list(fields.keys())
        cls.implicit_state_names = [k for k, f in fields.items() if f.kind == "implicit"]
        # Validate residual_name uniqueness: residual names must not collide with
        # each other or with state names they do not belong to.
        residual_names_seen: dict = {}  # residual_name → state_name
        all_state_names = set(fields.keys())
        for state_key, f in fields.items():
            rname = f.effective_residual_name
            if rname in residual_names_seen:
                raise ValueError(
                    f"{cls.__name__}: residual_name {rname!r} is used by both "
                    f"{residual_names_seen[rname]!r} and {state_key!r}"
                )
            if rname != state_key and rname in all_state_names:
                raise ValueError(
                    f"{cls.__name__}: residual_name {rname!r} for field {state_key!r} "
                    f"collides with another state name"
                )
            residual_names_seen[rname] = state_key
        # Validate dlambda_residual_name is a non-empty str
        if not isinstance(cls.dlambda_residual_name, str) or not cls.dlambda_residual_name:
            raise ValueError(
                f"{cls.__name__}: dlambda_residual_name must be a non-empty str, "
                f"got {cls.dlambda_residual_name!r}"
            )
        # Also check dlambda_residual_name
        dl_rname = cls.dlambda_residual_name
        if dl_rname in residual_names_seen:
            raise ValueError(
                f"{cls.__name__}: dlambda_residual_name {dl_rname!r} collides with "
                f"residual_name of field {residual_names_seen[dl_rname]!r}"
            )
        if dl_rname in all_state_names:
            raise ValueError(
                f"{cls.__name__}: dlambda_residual_name {dl_rname!r} collides with "
                f"a state name"
            )
        implicit = set(cls.implicit_state_names)
        all_states = set(cls.state_names)
        # Explicit states excluding "stress" (stress default is handled by framework).
        explicit_non_stress = (all_states - implicit) - {"stress"}
        needs_update = bool(explicit_non_stress)
        needs_residual = bool(implicit)
        if needs_update and cls.update_state is MaterialModel.update_state:
            raise TypeError(
                f"{cls.__name__}: explicit states {sorted(explicit_non_stress)} require "
                "update_state() to be implemented"
            )
        if needs_residual and cls.state_residual is MaterialModel.state_residual:
            raise TypeError(
                f"{cls.__name__}: implicit states {sorted(implicit)} require "
                "state_residual() to be implemented"
            )
        _wrap_flow_return_check(cls)
        cls._fortran_bindings = _collect_bindings(cls)

    @property
    def ntens(self) -> int:
        """Number of stress/strain components (derived from dimension)."""
        return self.dimension.ntens

    # ------------------------------------------------------------------
    # Elasticity
    # ------------------------------------------------------------------

    def isotropic_stiffness(self) -> Stiffness:
        """Isotropic elastic stiffness of this model's dimension, from ``E`` and ``nu``.

        The building block for :meth:`elastic_stiffness`.  A model whose
        stiffness changes with the state scales it::

            def elastic_stiffness(self, state):          # damage
                return (1.0 - state["D"]) * self.isotropic_stiffness()

        Returns
        -------
        anp.ndarray, shape (ntens, ntens)
            C with ``σ = C : ε`` (engineering shear convention).
        """
        if not (hasattr(self, "E") and hasattr(self, "nu")):
            raise NotImplementedError(
                f"{type(self).__name__}.isotropic_stiffness requires self.E and "
                "self.nu — override elastic_stiffness for a model without them."
            )
        # A constant of the instance, but evaluated inside every residual call
        # (and every autodiff trace of it) via default_stress_residual.  Cache
        # keyed on (E, nu) so the matrix assembly happens once; the key guards
        # against parameters being reassigned after construction.
        key = (self.E, self.nu)
        cached = getattr(self, "_elastic_C_cache", None)
        if cached is not None and cached[0] == key:
            return cached[1]
        mu = self.E / (2.0 * (1.0 + self.nu))
        lam = self.E * self.nu / ((1.0 + self.nu) * (1.0 - 2.0 * self.nu))
        C = self.isotropic_C(lam, mu)
        self._elastic_C_cache = (key, C)
        return C

    def elastic_stiffness(self, state: "State | StateDict") -> Stiffness:
        """Elastic stiffness at ``state``, in Voigt notation.

        The shorthand for a *linear* elastic law: the default
        :meth:`elastic_predictor` is ``σ_trial = σ_n + elastic_stiffness(state_n) Δε``,
        and :meth:`default_stress_residual` uses it for the plastic corrector.
        The default here is :meth:`isotropic_stiffness`, independent of the
        state.  Override for a stiffness that follows the state (damage, a
        Young's modulus that degrades with plastic strain).

        A model whose elastic law is not linear in Δε overrides
        :meth:`elastic_predictor` instead; this method is then only a
        reference stiffness for the hooks that still ask for one.

        Parameters
        ----------
        state : State or dict
            The state the stiffness is evaluated at.

        Returns
        -------
        anp.ndarray, shape (ntens, ntens)
            Elastic stiffness C (σ = C : ε, engineering shear convention).
        """
        return self.isotropic_stiffness()

    def elastic_predictor(
        self,
        state_n: "State | StateDict",
        *,
        strain_inc: "FloatArray",
    ) -> list[StateUpdate]:
        """Elastic trial state: where the increment lands if it stays elastic.

        The default is the linear predictor ``σ_trial = σ_n + C(state_n) Δε``
        with every internal variable frozen.  Override when the elastic
        response is not linear in Δε, or when internal variables other than σ
        (e.g. an Implicit effective stress) move elastically.  The motivating
        case is unilateral damage (crack closure): with
        ``σ = (1−D) σ̃₊ + (1−hD) σ̃₋`` the elastic law is piecewise linear, and
        an increment whose principal stress changes sign cannot be represented
        by any constant ``C_n``.  The natural predictor is then written in
        effective space::

            def elastic_predictor(self, state_n, *, strain_inc):
                s_eff = state_n["stress_eff"] + self.C0 @ strain_inc
                return [self.stress.update(self.nominal(s_eff, state_n["D"])),
                        self.stress_eff.update(s_eff)]

        The returned trial state becomes ``inc.trial`` and is used everywhere
        the framework needs the elastic predictor:

        - **branch** — :meth:`elastic_check` reads it;
        - **elastic step** — it *is* the converged state
          (:meth:`elastic_update_state` is applied on top), and the tangent is
          :meth:`elastic_tangent`;
        - **plastic step** — implicit unknowns start their NR from the trial
          values, ``update_state`` / ``state_residual`` read ``inc.trial``, and
          the consistent tangent differentiates through this method.

        :meth:`elastic_update_state` stays the separate hook for explicit
        fields that follow σ only once a step is known to be elastic (the
        subloading ratio R).  The two differ in *when* they apply: the trial
        state is the constant of every step, plastic or not, and takes part in
        the branch; ``elastic_update_state`` runs after it.  Returning such a
        follower from here instead would put it on the trial state, where
        ``yield_function(trial)`` vanishes identically and the default branch
        test can no longer tell loading from unloading.

        Parameters
        ----------
        state_n : State or dict
            State at the beginning of the increment (``state_n["stress"]`` is σ_n).
        strain_inc : anp.ndarray, shape (ntens,), keyword-only
            Strain increment Δε (engineering shear).

        Returns
        -------
        list[StateUpdate]
            Must contain the stress.  Use ``self.<field>.update(value)`` — it
            produces a value for implicit fields too, where calling the field
            would produce a residual.  Must be autograd-traceable in
            ``strain_inc``.
        """
        C_n = self.elastic_stiffness(state_n)
        return [StateUpdate(name="stress", value=state_n["stress"] + C_n @ strain_inc)]

    def elastic_tangent(self, inc: Increment) -> "Stiffness | None":
        """Hand-written elastic tangent ``dσ_trial/dΔε`` (optional).

        Only consulted for a model that overrides :meth:`elastic_predictor`.
        With the default predictor the elastic tangent is
        ``elastic_stiffness(inc.state_n)``, which is already a hand-written
        expression.

        A non-linear predictor's tangent is otherwise obtained by autodiff
        only.  That is enough to run, but a UMAT needs the expression itself,
        and this is where it is written down and checked:
        ``PythonAnalyticalIntegrator`` requires it, ``PythonNumericalIntegrator``
        always differentiates the predictor, and comparing the two verifies the
        derivation.

        Returns
        -------
        None
            No hand-written tangent; the framework differentiates the predictor.
        anp.ndarray, shape (ntens, ntens)
            dσ_trial/dΔε at this increment.
        """
        return None

    # ------------------------------------------------------------------
    # Abstract interface — must be implemented by subclasses
    # ------------------------------------------------------------------

    @abstractmethod
    def yield_function(
        self,
        state: "State | StateDict",
    ) -> Scalar:
        """Evaluate the yield function f(state).

        The material is in the elastic domain when f ≤ 0.

        Parameters
        ----------
        state : State or dict
            Current state including ``state["stress"]`` (Voigt notation) and all
            internal state variables.

        Returns
        -------
        anp.ndarray, scalar
            Yield function value.
        """
        raise NotImplementedError(
            f"{type(self).__name__}.yield_function() is not implemented. "
            "Subclasses must override yield_function()."
        )

    def update_state(
        self,
        new: "State | StateDict",
        dlambda: Scalar,
        inc: Increment,
    ) -> list[StateUpdate | StateResidual]:
        """Return updated non-stress explicit state variables after a plastic increment.

        Closed-form (explicit) update rule: given (new, Δλ, inc), returns the
        explicit **non-stress** state keys as a list of :class:`StateUpdate`.
        Required when any non-stress state is explicit.

        Do **not** return ``stress`` from this method — the framework handles
        the stress update via the NR residual system.

        Parameters
        ----------
        new : State or dict
            Current NR iterate.  ``new["stress"]`` = σ_k (current σ NR iterate);
            ``new[implicit_key]`` = current implicit iterate.  Explicit
            non-stress keys carry their ``inc.state_n`` values at the time this
            method is called (not yet updated, and not the trial values).
        dlambda : anp.ndarray, scalar
            Plastic multiplier increment Δλ ≥ 0.
        inc : Increment
            The step: ``inc.state_n`` (state at the beginning of the
            increment), ``inc.strain_inc`` (Δε), ``inc.trial`` (the elastic
            trial state — ``inc.trial["stress"]`` is σ_trial).

        Returns
        -------
        list[StateUpdate]
            Updated explicit non-stress state items (``self.<field>(value)``).

        Raises
        ------
        NotImplementedError
            If not overridden when explicit non-stress states exist.
        """
        raise NotImplementedError(
            f"{type(self).__name__}.update_state() is not implemented. "
            "Models with explicit states (not in implicit_state_names) must "
            "implement update_state()."
        )

    def state_residual(
        self,
        new: "State | StateDict",
        dlambda: Scalar,
        inc: Increment,
    ) -> list[StateResidual | DlambdaResidual]:
        """Residual of the implicit state evolution equations.

        Defines R_h(new, Δλ, inc) = 0 for state variables listed in
        :attr:`implicit_state_names`.  When stress is declared ``Implicit``,
        return ``self.stress(R_stress)`` in the list as well.

        Parameters
        ----------
        new : State or dict
            Current NR iterate at step n+1.  ``new["stress"]`` = σ_k
            (current σ NR iterate); ``new[implicit_key]`` = current
            implicit iterate; ``new[explicit_key]`` = updated value
            already returned by ``update_state`` for this iteration.
        dlambda : anp.ndarray, scalar
            Plastic multiplier increment Δλ.
        inc : Increment
            The step: ``inc.state_n``, ``inc.strain_inc``, ``inc.trial``.
            ``inc.trial["stress"]`` is the fixed elastic predictor σ_trial for
            the associative residual formula::

                R_stress = σ − σ_trial + Δλ·C·∂f/∂σ

            which :meth:`default_stress_residual` computes from it.

        Notes
        -----
        ``inc.trial`` and ``inc.strain_inc`` may enter any residual row with any
        coefficient.  The autodiff consistent tangent differentiates the
        residual with respect to the load increment through both, so it does
        not require the ``R_σ = σ − σ_trial + …`` form — e.g. a weakly coupled
        damage residual ``σ − (1−D_{n+1})/(1−D_n)·σ_trial + …`` gets the exact
        tangent including the ∂D/∂Δε term.  Both arrive as autograd boxes
        during that differentiation, so operate on them with ``autograd.numpy``.

        Returns
        -------
        list[StateResidual]
            Residual items (use ``self.<field>(value)`` syntax).  Zero at convergence.
            Optionally include ``self.dlambda(R_dl)`` to override the Δλ row.
        """
        raise NotImplementedError(
            f"{type(self).__name__}.state_residual() is not implemented. "
            "Models with implicit states must implement state_residual()."
        )

    def elastic_check(self, inc: Increment) -> bool:
        """Return ``True`` when the increment stays elastic (no return mapping).

        This is the elastic/plastic branch.  The default is the classical test
        ``f(trial) ≤ 0`` on the whole trial state, which is correct whenever the
        elastic domain *is* the interior of the surface written in
        :meth:`yield_function` — so classical plasticity models override nothing
        and keep ``yield_function`` as the single source of truth.

        Subloading-surface models are the motivating case for overriding.  Put
        the subloading surface ``‖s‖ − R·F`` in :meth:`yield_function` and the Δλ
        row and the ``grad``-derived :meth:`flow` both come out right; neither
        :meth:`flow` nor the Δλ row needs an override.  What that surface's sign
        cannot express is the branch, which is a *disjunction*::

            elastic  ⟺  R̂(σ_trial) ≤ R_e   (below the elastic limit)
                     or  R̂(σ_trial) ≤ R_n   (unloading)

        Test only the elastic limit and unloading runs the return mapping, since
        R̂ stays above R_e on the way down.  Test neither and the very first step
        is plastic, because R starts at zero and ``f`` is then positive for any
        non-zero stress.  A disjunction is not the zero level set of one smooth
        function, and folding it into :meth:`yield_function` would destroy the
        gradient :meth:`flow` depends on.

        Written against the increment::

            def elastic_check(self, inc):
                R_hat = self.normal_yield_ratio(inc.state_n, inc.trial["stress"])
                return bool(R_hat <= self.R_e) or bool(R_hat <= inc.state_n["R"])

        Unlike :meth:`yield_function`, this hook sees the step-n state and Δε as
        well, so direction tests of the form ``n : Δε > 0`` are expressible here.

        The return value must be a plain ``bool`` — it selects a code path and is
        never differentiated, so a smoothed indicator is neither needed nor
        allowed.

        Parameters
        ----------
        inc : Increment
            The step: ``inc.trial`` (elastic trial state), ``inc.state_n``,
            ``inc.strain_inc``.

        Returns
        -------
        bool
            ``True`` → elastic step (the state advances to ``inc.trial``, then
            :meth:`elastic_update_state`); ``False`` → run the return mapping.
        """
        return bool(self.yield_function(inc.trial) <= 0.0)

    def elastic_update_state(self, inc: Increment) -> list[StateUpdate]:
        """Return explicit state variables that track σ even without plastic flow.

        Called on every step :meth:`elastic_check` sends down the elastic path —
        the initial elastic loading, unloading after yield, and the elastic
        portion of a reload alike.  Δλ is zero for all of them, so any evolution
        law proportional to Δλ contributes nothing and the update is algebraic in
        the trial stress.

        The subloading-surface normal yield ratio R is the motivating case: the
        subloading surface passes through the current stress by construction,
        so R follows σ down during unloading even though no plastic flow occurs.
        Freezing R here would make the next loading step compare against a stale
        R_n and start plastic flow at the wrong point::

            def elastic_update_state(self, inc):
                return [self.R(self.normal_yield_ratio(inc.state_n, inc.trial["stress"]))]

        This is a separate hook rather than ``update_state(dlambda=0)`` because
        Δλ cannot serve as the discriminator: the plastic NR path also starts
        from Δλ = 0, so a model could not tell the two situations apart.  It is
        also separate from :meth:`elastic_predictor`: the value is the result
        of an elastic step, not part of the trial state the branch is decided
        on.

        The update must not feed back into the stress: σ_{n+1} = σ_trial is
        already fixed, and the elastic tangent does not see this hook.  A
        variable whose elastic-step change alters the stress (damage driven by
        elastic energy, creep) needs a solve on the elastic branch, which the
        framework does not provide.

        Return only the fields that need to move — omitted explicit fields keep
        their trial values.  The default returns an empty list.

        Parameters
        ----------
        inc : Increment
            The step.  ``inc.trial["stress"]`` *is* the converged stress of an
            elastic step, so write the algebraic update against it.

        Returns
        -------
        list[StateUpdate]
            Updated explicit state items (``self.<field>(value)``); possibly empty.
        """
        return []

    # ------------------------------------------------------------------
    # Default helpers provided by the framework
    # ------------------------------------------------------------------

    def initial_state(self) -> StateDict:
        """Return zero-initialised state dict (including ``stress``).

        The default implementation uses the ``StateField`` descriptors to
        produce a ``dict`` with the correct shapes.  Override when a non-zero
        initial state is required (e.g. pre-stress).

        Returns
        -------
        dict
            ``{name: zeros(shape) for name in state_names}``  — always includes
            ``"stress"`` (the key added by ``__init_subclass__``).
        """
        return {name: f.initial_value(self) for name, f in self.state_fields.items()}

    # ------------------------------------------------------------------
    # State factory helpers
    # ------------------------------------------------------------------

    def make_state(self, **kwargs) -> "State":
        """Assemble a complete :class:`~manforge.core.state.State` from keyword arguments.

        All field names (including ``stress``) are required; extra or missing
        keys raise ``TypeError`` at the call site (before any autograd trace).

        Returns
        -------
        State
            Immutable dict-wrapper with attribute-style access.
        """
        required = set(self.state_names)
        data = _make(required, f"{type(self).__name__}.make_state", kwargs)
        return State(data, tuple(self.state_names))

    def strain_flow(self, value: FloatArray) -> FlowVector:
        """Tag ``value`` as an engineering-shear (Δε_p conjugate) flow direction."""
        return FlowVector(value, self.dimension.eng_to_phys_strain_factors_np)

    def stress_flow(self, value: FloatArray) -> FlowVector:
        """Tag ``value`` as a physical-shear (textbook ``(3/2) s/‖s‖``) flow direction."""
        return FlowVector(
            self.dimension.to_strain_like(value),
            self.dimension.eng_to_phys_strain_factors_np,
        )

    def flow(self, state: "State | StateDict") -> FlowVector:
        """Plastic flow direction at ``state``.

        The default is associative: ``∂f/∂σ`` via autodiff.  Because σ is stored
        with physical shear, the raw gradient comes out in the engineering-shear
        convention, so it is tagged with :meth:`strain_flow`.

        Override to implement non-associative flow (plastic potential g ≠ f), or
        to supply a hand-derived expression.  The return value **must** be wrapped
        by :meth:`strain_flow` or :meth:`stress_flow` so the shear convention is
        explicit::

            def flow(self, state):
                s_xi = self.dev(state["stress"]) - state["alpha"]
                return self.stress_flow(1.5 * s_xi / self.vonmises_norm(s_xi))

        Consumers pick a convention at the use site: ``.strain_like`` for
        ``C @ n``, ``Δε_p = Δλ·n``, and strain-dimension residuals;
        ``.stress_like`` for backstress evolution laws.

        Returns
        -------
        FlowVector
        """
        import autograd
        from manforge.core.state import _state_with_stress
        n = autograd.grad(  # type: ignore[call-arg]
            lambda s: self.yield_function(_state_with_stress(state, s))
        )(state["stress"])
        return self.strain_flow(n)

    def default_stress_residual(
        self,
        new: "State | StateDict",
        dlambda: Scalar,
        stress_trial: StressVec,
    ) -> StressVec:
        """Default R_stress = σ − σ_trial + Δλ·C·n, with n from :meth:`flow`.

        Call this from :meth:`state_residual`::

            def state_residual(self, new, dlambda, inc):
                R_stress = self.default_stress_residual(new, dlambda, inc.trial["stress"])
                R_alpha = ...
                return [self.stress(R_stress), self.alpha(R_alpha), self.ep(R_ep)]

        This is a helper, not a hook, so it takes the one thing it reads —
        σ_trial — rather than the whole increment.  A hand-written residual that
        mirrors a Fortran routine only has σ_trial to hand and calls it the
        same way.

        For non-associative flow (plastic potential g ≠ f), override
        :meth:`flow` rather than reimplementing this residual — the algebra here
        is unchanged and only the flow direction differs.

        Parameters
        ----------
        new : State or dict
            Proposed state at step n+1 (NR iterate); ``new["stress"]`` is the
            current σ NR unknown.
        dlambda : scalar
            Plastic multiplier increment Δλ.
        stress_trial : anp.ndarray, shape (ntens,)
            Fixed elastic predictor σ_trial (``inc.trial["stress"]``).

        Returns
        -------
        anp.ndarray, shape (ntens,)
        """
        C = self.elastic_stiffness(new)
        n = self.flow(new)
        return new["stress"] - stress_trial + dlambda * (C @ n.strain_like)

    def vonmises(self, stress: StressVec) -> Scalar:
        """Von Mises equivalent stress; delegates to :meth:`StressDimension.vonmises`."""
        return self.dimension.vonmises(stress)

    def inner_product(self, a: StressVec, b: StressVec) -> Scalar:
        """Mandel double contraction A:B; delegates to :meth:`StressDimension.inner_product`."""
        return self.dimension.inner_product(a, b)

    def hydrostatic(self, stress: StressVec) -> "Scalar":
        """Mean normal stress; delegates to :meth:`StressDimension.hydrostatic`."""
        return self.dimension.hydrostatic(stress)

    def dev(self, stress: StressVec) -> StressVec:
        """Deviatoric stress; delegates to :meth:`StressDimension.dev`."""
        return self.dimension.dev(stress)

    def isotropic_C(self, lam: float, mu: float) -> "Stiffness":
        """Isotropic elastic stiffness from Lamé constants; delegates to :meth:`StressDimension.isotropic_C`.

        :meth:`isotropic_stiffness` is the same matrix built from the model's
        own ``E`` and ``nu``.
        """
        return self.dimension.isotropic_C(lam, mu)

    def I_vol(self) -> "Stiffness":
        """Volumetric projection tensor; delegates to :meth:`StressDimension.I_vol`."""
        return self.dimension.I_vol()

    def I_dev(self) -> "Stiffness":
        """Deviatoric projection tensor; delegates to :meth:`StressDimension.I_dev`."""
        return self.dimension.I_dev()

    def vonmises_norm(self, s: StressVec) -> "Scalar":
        """Von Mises norm of a deviatoric tensor; delegates to :meth:`StressDimension.vonmises_norm`."""
        return self.dimension.vonmises_norm(s)

    def _missing_dev_components(self, s: StressVec) -> StressVec:
        """Unstored deviatoric direct components; delegates to :meth:`StressDimension.missing_dev_components`."""
        return self.dimension.missing_dev_components(s)

    def deviatoric_inner_product(self, s: StressVec, t: StressVec) -> Scalar:
        """Double contraction s:t for deviatoric tensors; delegates to :meth:`StressDimension.deviatoric_inner_product`."""
        return self.dimension.deviatoric_inner_product(s, t)

    def strain_norm(self, strain: StressVec) -> Scalar:
        """Equivalent strain ε_eq = √(2/3 ε:ε); delegates to :meth:`StressDimension.strain_norm`."""
        return self.dimension.strain_norm(strain)

    # ------------------------------------------------------------------
    # Hand-written solver hooks
    # ------------------------------------------------------------------

    def user_defined_return_mapping(self, inc: Increment) -> "ReturnMappingResult | None":
        """User-supplied return mapping (optional).

        Override to provide a model-specific plastic correction algorithm.
        The implementation may use any solver internally — closed-form
        radial return, custom Newton-Raphson, etc.  The default returns
        ``None``, causing the integrator's ``stress_update`` method
        to fall back to the framework's generic ``numerical_newton`` solver.

        Parameters
        ----------
        inc : Increment
            The step.  ``inc.trial`` is the elastic trial state
            (``inc.trial["stress"]`` is σ_trial, and any other field the
            predictor moved is there too), ``inc.state_n`` the state at the
            beginning of the increment, ``inc.strain_inc`` is Δε.  ``State`` is
            read-only; start a working copy with ``dict(inc.state_n)``.

        Returns
        -------
        None
            Signals that no user-defined return mapping is available; the
            framework falls back to ``numerical_newton``.
        ReturnMappingResult
            Converged result with fields ``stress``, ``state``, ``dlambda``,
            ``n_iterations`` (default 0 for closed-form solvers), and
            ``residual_history`` (default ``[]`` for closed-form solvers).
            Import via ``from manforge.core import ReturnMappingResult``.
        """
        return None

    def user_defined_tangent(
        self,
        new: "State | StateDict",
        dlambda: Scalar,
        inc: Increment,
    ) -> "Stiffness | None":
        """User-supplied consistent tangent (optional).

        Override to provide a model-specific analytical expression for
        dσ_{n+1}/dΔε.  The default returns ``None``, causing
        the integrator's ``stress_update`` method to fall back
        to the generic autodiff tangent.

        Parameters
        ----------
        new : State or dict
            Converged state at step n+1; ``new["stress"]`` is σ_{n+1}.
        dlambda : anp.ndarray, scalar
            Converged plastic multiplier increment Δλ.
        inc : Increment
            The step: ``inc.trial`` (elastic trial state), ``inc.state_n``,
            ``inc.strain_inc`` (Δε, engineering shear — the ABAQUS DSTRAN).

        Returns
        -------
        None
            Signals that no user-defined tangent is available; the
            framework falls back to autodiff.
        anp.ndarray, shape (ntens, ntens)
            Consistent tangent dσ_{n+1}/dΔε.
        """
        return None
