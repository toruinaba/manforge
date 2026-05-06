"""Abstract base class for constitutive material models."""

from abc import ABC, abstractmethod

import autograd.numpy as anp

from manforge.core.state import StateField, State, collect_state_fields, _make, NTENS, DLAMBDA_FIELD
from manforge.core.dimension import SOLID_3D, StressDimension
from manforge.utils.smooth import smooth_sqrt


class MaterialModel(ABC):
    """Abstract base class for constitutive material models.

    Subclasses declare state variables as class-level ``StateField`` attributes
    using :func:`~manforge.core.state.Implicit` and
    :func:`~manforge.core.state.Explicit`::

        from manforge.core.state import Implicit, Explicit, NTENS

        class MyModel(MaterialModel3D):
            param_names = ["E", "nu", "sigma_y0", "C_k", "gamma"]
            stress = Implicit(shape=NTENS, doc="Cauchy stress")   # NR unknown
            alpha  = Implicit(shape=NTENS, doc="backstress tensor")
            ep     = Explicit(shape=(),    doc="equivalent plastic strain")

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

    - ``update_state(dlambda, state_n, state_trial)`` — required when any
      non-stress state is explicit; returns only the explicit-state keys.
    - ``state_residual(state_new, dlambda, state_n, state_trial)`` — required
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

    # Derived by __init_subclass__ from StateField descriptors:
    state_fields: dict[str, StateField] = {}
    state_names: list[str] = []
    implicit_state_names: list[str] = []

    # Framework-provided pseudo-field for the Δλ NR unknown.  Users can
    # optionally return self.dlambda(R_dl) from state_residual to override
    # the default R_dλ = yield_function(state).
    dlambda = DLAMBDA_FIELD

    @property
    def params(self) -> dict:
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
        from manforge.verification.fortran_registry import collect_bindings
        cls._fortran_bindings = collect_bindings(cls)

    @property
    def ntens(self) -> int:
        """Number of stress/strain components (derived from dimension)."""
        return self.dimension.ntens

    # ------------------------------------------------------------------
    # Abstract interface — must be implemented by subclasses
    # ------------------------------------------------------------------

    def elastic_stiffness(self, state=None) -> anp.ndarray:
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
        state,
    ) -> anp.ndarray:
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

    def update_state(
        self,
        dlambda: anp.ndarray,
        state_n,
        state_trial,
    ) -> list:
        """Return updated state variables after a plastic increment.

        Closed-form (explicit) update rule: given (Δλ, state_n, state_trial),
        returns explicit state keys as a list of :class:`StateUpdate`.
        Required for all state variables *not* listed in
        :attr:`implicit_state_names`.  The ``stress`` key need only be returned
        when a custom stress update is required; if omitted, the framework
        auto-injects the associative formula.

        Parameters
        ----------
        dlambda : anp.ndarray, scalar
            Plastic multiplier increment Δλ ≥ 0.
        state_n : State or dict
            State at the beginning of the increment.
        state_trial : State or dict
            Trial state (``state_trial["stress"]`` is the elastic trial stress).

        Returns
        -------
        list[StateUpdate]
            Updated explicit-state items (use ``self.<field>(value)`` syntax).

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
        state_new,
        dlambda: anp.ndarray,
        state_n,
        state_trial,
    ) -> list:
        """Residual of the implicit state evolution equations.

        Defines R_h(state_new, Δλ, state_n, state_trial) = 0 for state
        variables listed in :attr:`implicit_state_names`.  The ``stress`` key
        need only be returned when a custom stress residual is required; if
        omitted, the framework auto-injects the associative formula.

        Parameters
        ----------
        state_new : State or dict
            Proposed state values at step n+1 (NR unknowns).
        dlambda : anp.ndarray, scalar
            Plastic multiplier increment Δλ.
        state_n : State or dict
            State at the beginning of the increment.
        state_trial : State or dict
            Trial state (``state_trial["stress"]`` is the elastic trial stress).

        Returns
        -------
        list[StateResidual]
            Residual items (use ``self.<field>(value)`` syntax).  Zero at convergence.
        """
        raise NotImplementedError(
            f"{type(self).__name__}.state_residual() is not implemented. "
            "Models with implicit states must implement state_residual()."
        )

    # ------------------------------------------------------------------
    # Default helpers provided by the framework
    # ------------------------------------------------------------------

    def initial_state(self) -> dict:
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
        state_new,
        dlambda: anp.ndarray,
        state_trial,
    ) -> anp.ndarray:
        """Associative default R_stress = σ − σ_trial + Δλ·C·∂f/∂σ.

        Call this from :meth:`state_residual` when the model uses associative
        flow (flow direction = ∂f/∂σ)::

            def state_residual(self, state_new, dlambda, state_n, state_trial):
                R_stress = self.default_stress_residual(state_new, dlambda, state_trial)
                R_alpha = ...
                return [self.stress(R_stress), self.alpha(R_alpha), self.ep(R_ep)]

        For non-associative flow (plastic potential g ≠ f), compute R_stress
        from your g-based flow direction and return ``self.stress(R_stress)``
        directly.

        Parameters
        ----------
        state_new : State or dict
            Proposed state at step n+1 (NR iterate); ``state_new["stress"]``
            is the current σ unknown.
        dlambda : scalar
            Plastic multiplier increment Δλ.
        state_trial : State or dict
            Trial state; ``state_trial["stress"]`` is the fixed elastic predictor σ_trial.

        Returns
        -------
        anp.ndarray, shape (ntens,)
        """
        import autograd
        from manforge.core.state import _state_with_stress
        stress = state_new["stress"]
        stress_trial = state_trial["stress"]
        C = self.elastic_stiffness(state_new)
        n = autograd.grad(
            lambda s: self.yield_function(_state_with_stress(state_new, s))
        )(stress)
        return stress - stress_trial + dlambda * (C @ n)

    def default_stress_update(
        self,
        dlambda: anp.ndarray,
        state_n,
        state_trial,
    ) -> anp.ndarray:
        """Return the framework-pre-computed associative stress iterate.

        For ``stress = Explicit`` models the framework derives σ_{n+1} =
        σ_trial − Δλ·C·∂f/∂σ before calling :meth:`update_state` and passes
        it as ``state_trial["stress"]``.  This helper returns that value,
        allowing user code to explicitly acknowledge the default associative
        update::

            def update_state(self, dlambda, state_n, state_trial):
                sig = self.default_stress_update(dlambda, state_n, state_trial)
                return [self.stress(sig), self.ep(state_n["ep"] + dlambda)]

        For a non-associative or damage-coupled Explicit stress update, compute
        ``sig`` directly (from your own formula) and return ``self.stress(sig)``
        instead of calling this helper.

        Parameters
        ----------
        dlambda : scalar
            Plastic multiplier increment Δλ (unused; present for API symmetry
            with :meth:`default_stress_residual`).
        state_n : State or dict
            State at the beginning of the increment (unused by default).
        state_trial : State or dict
            Trial state; ``state_trial["stress"]`` is the framework-pre-computed associative σ iterate.

        Returns
        -------
        anp.ndarray, shape (ntens,)
        """
        return state_trial["stress"]

    def _vonmises(self, stress: anp.ndarray) -> anp.ndarray:
        """Von Mises equivalent stress with missing-component correction.

        Computes √(3/2 · (‖s_m‖² + n_missing · p²)) using ``smooth_sqrt``
        so that JAX gradients are well-defined at zero stress.

        ``n_missing = ndi_phys − ndi`` counts the unstored direct stress
        components that are physically zero but contribute −p each to the
        deviatoric norm:

        * ``SOLID_3D``, ``PLANE_STRAIN``: n_missing=0 → √(3/2 s:s)
        * ``PLANE_STRESS``:              n_missing=1 → √(3/2 (s:s + p²))
        * ``UNIAXIAL_1D``:               n_missing=2 → |σ11|

        Uses :meth:`_dev` and :meth:`_hydrostatic` as defined by the
        stress-state base class.

        Parameters
        ----------
        stress : anp.ndarray, shape (ntens,)

        Returns
        -------
        anp.ndarray, scalar
        """
        s = self._dev(stress)
        p = self._hydrostatic(stress)
        s_m = s * self.dimension.mandel_factors_np
        sq_norm = anp.dot(s_m, s_m) + self.dimension.n_missing * p ** 2
        return smooth_sqrt(1.5 * sq_norm)

    def user_defined_return_mapping(
        self,
        stress_trial: anp.ndarray,
        C: anp.ndarray,
        state_n: dict,
    ):
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
        stress: anp.ndarray,
        state: dict,
        dlambda: anp.ndarray,
        C: anp.ndarray,
        state_n: dict,
    ):
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
