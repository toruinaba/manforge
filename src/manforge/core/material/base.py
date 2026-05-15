"""Abstract base class for constitutive material models."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import ClassVar

import autograd.numpy as anp

from manforge.core.state import (
    StateField, State, DlambdaField, DlambdaResidual,
    StateResidual, StateUpdate,
    collect_state_fields, _make, NTENS, DLAMBDA_FIELD,
)
from manforge.core.dimension import SOLID_3D, StressDimension
from manforge.core.material.fortran_binding import collect_bindings as _collect_bindings
from manforge._typing import FloatArray, Scalar, Stiffness, StressVec, StateDict
from manforge.core.result import ReturnMappingResult


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
    Declare ``stress = Implicit(shape=NTENS)`` to include σ as an NR unknown
    (replaces the removed ``implicit_stress = True`` flag).

    User methods receive a ``State`` dict-wrapper for all state arguments.
    Access state variables with bracket notation: ``state["stress"]``.
    The ``stress`` argument has been removed from ``yield_function``,
    ``update_state``, and ``state_residual``.

    Required methods depend on which states are explicit vs implicit:

    - ``update_state(dlambda, state_new, state_n, *, stress_trial, strain_inc)`` — required when any
      non-stress state is explicit; returns only the explicit-state keys.
    - ``state_residual(state_new, dlambda, state_n, *, stress_trial, strain_inc)`` — required
      when any state is implicit; returns only the implicit-state keys.

    The framework auto-injects the associative stress default
    (``σ ← σ_trial − Δλ·C·∂f/∂σ`` or ``R_stress = σ − σ_trial + Δλ·C·∂f/∂σ``)
    when the user does not return ``stress`` from ``update_state`` /
    ``state_residual``.

    ``hardening_type`` and ``implicit_stress`` are no longer used and raise
    ``TypeError`` with migration hints if declared.

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
        # Skip intermediate abstract classes that have not yet implemented yield_function.
        if cls.yield_function is MaterialModel.yield_function:
            return
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
        implicit_non_stress = implicit - {"stress"}
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
        cls._fortran_bindings = _collect_bindings(cls)

    @property
    def ntens(self) -> int:
        """Number of stress/strain components (derived from dimension)."""
        return self.dimension.ntens

    # ------------------------------------------------------------------
    # Abstract interface — must be implemented by subclasses
    # ------------------------------------------------------------------

    def elastic_stiffness(self, state: "State | StateDict | None" = None) -> Stiffness:
        """Return the elastic stiffness tensor in Voigt notation.

        Default implementation derives λ, μ from ``self.E`` and ``self.nu``
        and delegates to ``self.isotropic_C`` (polymorphically dispatched to
        the stress-state subclass).  Override to implement state-dependent
        stiffness (e.g. damage plasticity) or for parameter-less models.

        Parameters
        ----------
        state : dict or None
            Current internal state.  Ignored by the default isotropic
            implementation; damage/YU models use it to scale E.

        Returns
        -------
        anp.ndarray, shape (ntens, ntens)
            Elastic stiffness C (σ = C : ε, engineering shear convention).
        """
        if not (hasattr(self, "E") and hasattr(self, "nu")):
            raise NotImplementedError(
                f"{type(self).__name__}.elastic_stiffness: default implementation "
                "requires self.E and self.nu — override for anisotropic or "
                "parameter-less models."
            )
        mu = self.E / (2.0 * (1.0 + self.nu))
        lam = self.E * self.nu / ((1.0 + self.nu) * (1.0 - 2.0 * self.nu))
        return self.isotropic_C(lam, mu)

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
        dlambda: Scalar,
        state_new: "State | StateDict",
        state_n: "State | StateDict",
        *,
        stress_trial: "StressVec | None" = None,
        strain_inc: "FloatArray | None" = None,
    ) -> list[StateUpdate | StateResidual]:
        """Return updated non-stress explicit state variables after a plastic increment.

        Closed-form (explicit) update rule: given (Δλ, state_new, state_n),
        returns the explicit **non-stress** state keys as a list of
        :class:`StateUpdate`.  Required when any non-stress state is explicit.

        Do **not** return ``stress`` from this method — the framework handles
        the stress update via the NR residual system.

        Parameters
        ----------
        dlambda : anp.ndarray, scalar
            Plastic multiplier increment Δλ ≥ 0.
        state_new : State or dict
            Current NR iterate.  ``state_new["stress"]`` = σ_k (current σ NR iterate);
            ``state_new[implicit_key]`` = current implicit iterate.  Explicit
            non-stress keys carry ``state_n`` values at the time this method is
            called (not yet updated).
        state_n : State or dict
            State at the beginning of the increment.
        stress_trial : anp.ndarray, shape (ntens,), keyword-only
            Fixed elastic predictor σ_trial = σ_n + C Δε.  Use this when the
            closed-form update depends on the trial stress (e.g. elastic-strain
            tracking via Δε_e = C⁻¹(σ_trial − σ_n) − Δλ n̂).
        strain_inc : anp.ndarray, shape (ntens,), keyword-only
            Strain increment Δε for the current load step.  Use this when the
            update depends on the total strain increment (e.g. elastic/plastic
            strain decomposition).

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
        state_new: "State | StateDict",
        dlambda: Scalar,
        state_n: "State | StateDict",
        *,
        stress_trial: "StressVec",
        strain_inc: "FloatArray | None" = None,
    ) -> list[StateResidual | DlambdaResidual]:
        """Residual of the implicit state evolution equations.

        Defines R_h(state_new, Δλ, state_n) = 0 for state variables listed in
        :attr:`implicit_state_names`.  When stress is declared ``Implicit``,
        return ``self.stress(R_stress)`` in the list as well.

        Parameters
        ----------
        state_new : State or dict
            Current NR iterate at step n+1.  ``state_new["stress"]`` = σ_k
            (current σ NR iterate); ``state_new[implicit_key]`` = current
            implicit iterate; ``state_new[explicit_key]`` = updated value
            already returned by ``update_state`` for this iteration.
        dlambda : anp.ndarray, scalar
            Plastic multiplier increment Δλ.
        state_n : State or dict
            State at the beginning of the increment.
        stress_trial : anp.ndarray, shape (ntens,), keyword-only
            Fixed elastic predictor σ_trial = σ_n + C Δε.  Use this when you
            need the fixed trial stress for the associative residual formula::

                R_stress = σ − stress_trial + Δλ·C·∂f/∂σ

            Call :meth:`default_stress_residual` which accepts ``stress_trial``
            directly, or use this kwarg to compute R_stress manually.
        strain_inc : anp.ndarray, shape (ntens,), keyword-only
            Strain increment Δε for the current load step.

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

    def default_stress_residual(
        self,
        state_new: "State | StateDict",
        dlambda: Scalar,
        stress_trial: "StressVec | None",
    ) -> StressVec:
        """Associative default R_stress = σ − σ_trial + Δλ·C·∂f/∂σ.

        Call this from :meth:`state_residual` when the model uses associative
        flow (flow direction = ∂f/∂σ)::

            def state_residual(self, state_new, dlambda, state_n, state_trial,
                               *, stress_trial):
                R_stress = self.default_stress_residual(state_new, dlambda, stress_trial)
                R_alpha = ...
                return [self.stress(R_stress), self.alpha(R_alpha), self.ep(R_ep)]

        For non-associative flow (plastic potential g ≠ f), compute R_stress
        from your g-based flow direction and return ``self.stress(R_stress)``
        directly.

        Parameters
        ----------
        state_new : State or dict
            Proposed state at step n+1 (NR iterate); ``state_new["stress"]``
            is the current σ NR unknown.
        dlambda : scalar
            Plastic multiplier increment Δλ.
        stress_trial : anp.ndarray, shape (ntens,)
            Fixed elastic predictor σ_trial = σ_n + C Δε.

        Returns
        -------
        anp.ndarray, shape (ntens,)
        """
        import autograd
        from manforge.core.state import _state_with_stress
        if stress_trial is None:
            raise ValueError("default_stress_residual: stress_trial is required")
        stress = state_new["stress"]
        C = self.elastic_stiffness(state_new)
        n = autograd.grad(  # type: ignore[call-arg]
            lambda s: self.yield_function(_state_with_stress(state_new, s))
        )(stress)
        return stress - stress_trial + dlambda * (C @ n)

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
        """Isotropic elastic stiffness; delegates to :meth:`StressDimension.isotropic_C`."""
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

    def user_defined_return_mapping(
        self,
        stress_trial: StressVec,
        C: Stiffness,
        state_n: StateDict,
    ) -> "ReturnMappingResult | None":
        """User-supplied return mapping (optional).

        Override to provide a model-specific plastic correction algorithm.
        The implementation may use any solver internally — closed-form
        radial return, custom Newton-Raphson, etc.  The default returns
        ``None``, causing the integrator's ``stress_update`` method
        to fall back to the framework's generic ``numerical_newton`` solver.

        Parameters
        ----------
        stress_trial : anp.ndarray, shape (ntens,)
            Elastic trial stress σ_trial = σ_n + C Δε.
        C : anp.ndarray, shape (ntens, ntens)
            Elastic stiffness tensor (already computed by the caller).
        state_n : dict
            Internal state at the beginning of the increment.

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
        stress: StressVec,
        state: "State | StateDict",
        dlambda: Scalar,
        C: Stiffness,
        state_n: StateDict,
    ) -> "Stiffness | None":
        """User-supplied consistent tangent (optional).

        Override to provide a model-specific analytical expression for
        dσ_{n+1}/dΔε.  The default returns ``None``, causing
        the integrator's ``stress_update`` method to fall back
        to the generic autodiff tangent.

        Parameters
        ----------
        stress : anp.ndarray, shape (ntens,)
            Converged stress σ_{n+1}.
        state : dict
            Converged internal state at step n+1.
        dlambda : anp.ndarray, scalar
            Converged plastic multiplier increment Δλ.
        C : anp.ndarray, shape (ntens, ntens)
            Elastic stiffness tensor (already computed by the caller).
        state_n : dict
            Internal state at the beginning of the increment.

        Returns
        -------
        None
            Signals that no user-defined tangent is available; the
            framework falls back to autodiff.
        anp.ndarray, shape (ntens, ntens)
            Consistent tangent dσ_{n+1}/dΔε.
        """
        return None
