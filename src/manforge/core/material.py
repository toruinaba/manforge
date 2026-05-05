"""Abstract base class for constitutive material models."""

from abc import ABC, abstractmethod

import autograd.numpy as anp

from manforge.autodiff.operators import identity_voigt
from manforge.core.state import StateField, State, collect_state_fields, _make, NTENS
from manforge.core.stress_state import SOLID_3D, PLANE_STRESS, UNIAXIAL_1D, StressState
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

    User methods receive a ``State`` object for all state arguments (including
    ``state.stress``).  The ``stress`` argument has been removed from
    ``yield_function``, ``update_state``, and ``state_residual``.

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
    stress_state : StressState
        Dimensionality descriptor (default: ``SOLID_3D``, 6-component 3D).
    ntens : int
        Read-only property; returns ``self.stress_state.ntens``.
    """

    param_names: list[str]
    stress_state: StressState = SOLID_3D

    # Derived by __init_subclass__ from StateField descriptors:
    state_fields: dict[str, StateField] = {}
    state_names: list[str] = []
    implicit_state_names: list[str] = []

    @property
    def params(self) -> dict:
        """Material parameters as a dict keyed by :attr:`param_names`."""
        return {name: getattr(self, name) for name in self.param_names}

    def __init_subclass__(cls, **kwargs):
        import inspect
        super().__init_subclass__(**kwargs)
        # Skip intermediate abstract classes that have not yet implemented yield_function.
        if cls.yield_function is MaterialModel.yield_function:
            return
        # Reject legacy hardening_type attribute.
        if "hardening_type" in cls.__dict__:
            _val = cls.__dict__["hardening_type"]
            _hint = ""
            if _val == "reduced":
                _hint = " — remove hardening_type; implicit_state_names=[] is the default"
            elif _val == "augmented":
                _hint = (
                    " — remove hardening_type; set implicit_state_names=<your state_names>"
                    " (and stress = Implicit(shape=NTENS) if σ should be an NR unknown)"
                )
            raise TypeError(
                f"{cls.__name__}: hardening_type has been removed. "
                f"Use StateField declarations and stress = Implicit(shape=NTENS) instead.{_hint}"
            )
        # Reject old implicit_stress flag.
        if "implicit_stress" in cls.__dict__:
            _val = cls.__dict__["implicit_stress"]
            if isinstance(_val, bool):
                if _val:
                    _hint = (
                        "\n    from manforge.core import Implicit, NTENS\n"
                        "    stress = Implicit(shape=NTENS, doc='Cauchy stress')"
                    )
                else:
                    _hint = "\n    Remove the implicit_stress = False declaration (it is the default)."
                raise TypeError(
                    f"{cls.__name__}: implicit_stress has been removed. "
                    f"Declare stress as a state field instead:{_hint}"
                )
        # Reject old stress_residual override (replaced by state_residual returning stress).
        if "stress_residual" in cls.__dict__:
            raise TypeError(
                f"{cls.__name__}: stress_residual() override has been removed. "
                "Override state_residual() and return self.stress(R_stress) instead:\n"
                "    def state_residual(self, state_new, dlambda, state_n, state_trial):\n"
                "        ...\n"
                "        return [self.stress(R_stress), self.alpha(R_alpha), ...]"
            )
        # Detect renamed methods and guide migration.
        if "hardening_increment" in cls.__dict__:
            raise TypeError(
                f"{cls.__name__}: hardening_increment() has been renamed to "
                "update_state() — rename the method on your subclass"
            )
        if "hardening_residual" in cls.__dict__:
            raise TypeError(
                f"{cls.__name__}: hardening_residual() has been renamed to "
                "state_residual() — rename the method on your subclass"
            )
        # Reject old list-based declarations BEFORE signature checks so that migration
        # error messages remain accurate (list-based errors fire first).
        if "state_names" in cls.__dict__:
            val = cls.__dict__["state_names"]
            if isinstance(val, list) and val:
                raise TypeError(
                    f"{cls.__name__}: list-based state_names has been removed. "
                    "Declare state variables as class attributes instead:\n"
                    "    from manforge.core.state import Implicit, Explicit\n"
                    "    ep    = Explicit(shape=())\n"
                    "    alpha = Implicit(shape=NTENS)"
                )
        if "implicit_state_names" in cls.__dict__:
            val = cls.__dict__["implicit_state_names"]
            if isinstance(val, list) and val:
                raise TypeError(
                    f"{cls.__name__}: list-based implicit_state_names has been removed. "
                    "Declare implicit state variables using Implicit(shape=...) instead:\n"
                    "    from manforge.core.state import Implicit\n"
                    "    alpha = Implicit(shape=NTENS)"
                )
        # Reject old yield_function(self, stress, state) signature.
        if "yield_function" in cls.__dict__:
            sig = inspect.signature(cls.__dict__["yield_function"])
            params = list(sig.parameters.keys())
            if len(params) >= 3 and params[1] == "stress" and params[2] == "state":
                raise TypeError(
                    f"{cls.__name__}: yield_function(self, stress, state) is no longer accepted. "
                    "Update to yield_function(self, state) and access stress via state.stress:\n"
                    "    def yield_function(self, state):\n"
                    "        xi = state.stress - state.alpha\n"
                    "        return self._vonmises(xi) - self.sigma_y0"
                )
        # Reject old update_state(self, dlambda, stress, state) signature.
        if "update_state" in cls.__dict__:
            sig = inspect.signature(cls.__dict__["update_state"])
            params = list(sig.parameters.keys())
            if len(params) >= 4 and params[2] == "stress":
                raise TypeError(
                    f"{cls.__name__}: update_state(self, dlambda, stress, state) is no longer accepted. "
                    "Update to update_state(self, dlambda, state_n, state_trial):\n"
                    "    def update_state(self, dlambda, state_n, state_trial):\n"
                    "        stress = state_trial.stress\n"
                    "        ..."
                )
        # Reject old state_residual(self, state_new, dlambda, stress, state_n) signature.
        if "state_residual" in cls.__dict__:
            sig = inspect.signature(cls.__dict__["state_residual"])
            params = list(sig.parameters.keys())
            if len(params) >= 4 and params[3] == "stress":
                raise TypeError(
                    f"{cls.__name__}: state_residual(self, state_new, dlambda, stress, state_n) "
                    "is no longer accepted. Update to state_residual(self, state_new, dlambda, "
                    "state_n, state_trial):\n"
                    "    def state_residual(self, state_new, dlambda, state_n, state_trial):\n"
                    "        stress = state_new.stress\n"
                    "        ..."
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
        """Number of stress/strain components (derived from stress_state)."""
        return self.stress_state.ntens

    # ------------------------------------------------------------------
    # Abstract interface — must be implemented by subclasses
    # ------------------------------------------------------------------

    def elastic_stiffness(self, state=None) -> anp.ndarray:
        """Return the elastic stiffness tensor in Voigt notation.

        The base-class implementation raises NotImplementedError.  Concrete
        stress-state base classes (MaterialModel3D, MaterialModelPS,
        MaterialModel1D) provide a default that computes C from ``self.E``
        and ``self.nu`` via :meth:`isotropic_C`.  Override this method to
        implement state-dependent elastic stiffness (e.g. damage plasticity).

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
        raise NotImplementedError(
            f"{type(self).__name__}.elastic_stiffness() is not implemented. "
            "Subclass MaterialModel3D, MaterialModelPS, or MaterialModel1D to "
            "get a default isotropic implementation, or override this method."
        )

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
            Current state including ``state.stress`` (Voigt notation) and all
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
            Trial state (``state_trial.stress`` is the elastic trial stress).

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
            Trial state (``state_trial.stress`` is the elastic trial stress).

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

    def isotropic_C(self, lam: float, mu: float) -> anp.ndarray:
        """Build the isotropic elastic stiffness tensor.

        Builds the full 3D 6×6 stiffness first, then extracts/condenses to
        the dimensionality specified by ``self.stress_state``:

        * **3D solid (NTENS=6)**: returns the full 6×6 tensor.
          C = λ δ⊗δ + μ diag([2, 2, 2, 1, 1, 1])
        * **Plane strain / axisymmetric (NTENS=4)**: returns the 4×4
          submatrix for components [11, 22, 33, 12].
        * **Plane stress (NTENS=3)**: applies static condensation to enforce
          σ33 = 0, returning a 3×3 matrix for [11, 22, 12].
        * **1D truss (NTENS=1)**: returns [[E]] where E = μ(3λ+2μ)/(λ+μ).

        Parameters
        ----------
        lam : float
            First Lamé constant λ.
        mu : float
            Shear modulus μ.

        Returns
        -------
        anp.ndarray, shape (ntens, ntens)
        """
        # Build the full 3D 6×6 tensor
        delta_6 = identity_voigt()  # 6-component, no ss
        scale_6 = anp.array([2.0, 2.0, 2.0, 1.0, 1.0, 1.0])
        C6 = lam * anp.outer(delta_6, delta_6) + mu * anp.diag(scale_6)

        ss = self.stress_state

        if ss.ntens == 6:
            return C6

        if ss.ntens == 4:
            # Plane strain / axisymmetric: components [11,22,33,12]
            idx = anp.array([0, 1, 2, 3])
            return C6[anp.ix_(idx, idx)]

        if ss.ntens == 3:
            # Plane stress: condense out sigma_33 (index 2 of the 4×4 plane-
            # strain submatrix) to enforce sigma_33 = 0.
            # Retain indices [0,1,3] of the 6×6 → [s11, s22, s12].
            # Step 1: 4×4 plane-strain sub-block
            idx4 = anp.array([0, 1, 2, 3])
            C4 = C6[anp.ix_(idx4, idx4)]
            # Step 2: static condensation — eliminate row/col 2 (s33)
            # D_ps = D_00 - D_02 * D_22^{-1} * D_20  (Schur complement)
            # Retained dof within C4: [0, 1, 3] → mapped to [0, 1, 2]
            retain = anp.array([0, 1, 3])
            C_rr = C4[anp.ix_(retain, retain)]
            C_rc = C4[retain, 2]          # shape (3,)
            C_cc = C4[2, 2]               # scalar
            return C_rr - anp.outer(C_rc, C_rc) / C_cc

        if ss.ntens == 1:
            # 1D truss (uniaxial stress): C = [[E]]
            E = mu * (3.0 * lam + 2.0 * mu) / (lam + mu)
            return anp.array([[E]])

        raise ValueError(
            f"isotropic_C: unsupported ntens={ss.ntens} for stress_state '{ss.name}'"
        )

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
        if self.state_fields:
            return {name: f.initial_value(self) for name, f in self.state_fields.items()}
        # Fallback for models that have no StateFields (legacy / stateless).
        return {name: anp.array(0.0) for name in self.state_names}

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

    def _default_stress_residual(
        self,
        stress: anp.ndarray,
        dlambda: anp.ndarray,
        state,
        stress_trial: anp.ndarray,
    ) -> anp.ndarray:
        """Internal default associative stress residual R_stress = σ − σ_trial + Δλ·C·∂f/∂σ.

        Used by the framework when the user does not return ``stress`` from
        ``state_residual``.  Not part of the public API.
        """
        import autograd
        from manforge.core.state import _state_with_stress
        C = self.elastic_stiffness(state)
        n = autograd.grad(
            lambda s: self.yield_function(_state_with_stress(state, s))
        )(stress)
        return stress - stress_trial + dlambda * (C @ n)

    def _default_stress_update(
        self,
        stress_trial: anp.ndarray,
        dlambda: anp.ndarray,
        state,
    ) -> anp.ndarray:
        """Internal default associative stress update σ ← σ_trial − Δλ·C·∂f/∂σ.

        Used by the framework when the user does not return ``stress`` from
        ``update_state``.  Not part of the public API.
        """
        import autograd
        from manforge.core.state import _state_with_stress
        C = self.elastic_stiffness(state)
        n = autograd.grad(
            lambda s: self.yield_function(_state_with_stress(state, s))
        )(anp.array(stress_trial))
        return anp.array(stress_trial) - dlambda * (C @ n)

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
        s_m = s * self.stress_state.mandel_factors_np
        sq_norm = anp.dot(s_m, s_m) + self.stress_state.n_missing * p ** 2
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
        ``None``, causing :func:`~manforge.core.stress_update.stress_update`
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
        :func:`~manforge.core.stress_update.stress_update` to fall back
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


class MaterialModel3D(MaterialModel):
    """Stress-state base class for full-rank stress states (ndi == ndi_phys).

    Valid for ``SOLID_3D`` (ntens=6) and ``PLANE_STRAIN`` (ntens=4), and any
    future stress state that stores all direct components (e.g. axisymmetric).
    For these states the three physical direct stresses σ11, σ22, σ33 are all
    stored explicitly, so deviatoric and von Mises operators need no
    missing-component corrections.

    Provides concrete implementations of the operator methods used by concrete
    material models:

    * :meth:`_hydrostatic` — p = (σ11 + σ22 + σ33) / 3
    * :meth:`_dev`         — s = σ − p δ
    * :meth:`isotropic_C`  — submatrix extraction from the full 6×6 tensor
    * :meth:`_I_vol`       — δ⊗δ / 3
    * :meth:`_I_dev`       — I − P_vol

    :meth:`_vonmises` is inherited from :class:`MaterialModel` (uses
    ``smooth_sqrt`` with n_missing=0, equivalent to √(3/2 s:s)).

    Subclasses still must implement the required material methods:
    :meth:`elastic_stiffness`, :meth:`yield_function`, and either
    :meth:`update_state` (reduced) or :meth:`state_residual` (augmented).

    Parameters
    ----------
    stress_state : StressState, optional
        Must satisfy ``stress_state.ndi == stress_state.ndi_phys``.
        Defaults to ``SOLID_3D``.

    Raises
    ------
    ValueError
        If ``stress_state.ndi != stress_state.ndi_phys``.
    """

    def __init__(self, stress_state: StressState = SOLID_3D):
        if stress_state.ndi != stress_state.ndi_phys:
            raise ValueError(
                f"MaterialModel3D requires ndi == ndi_phys "
                f"(e.g. SOLID_3D or PLANE_STRAIN). "
                f"Got '{stress_state.name}' with ndi={stress_state.ndi}, "
                f"ndi_phys={stress_state.ndi_phys}."
            )
        self.stress_state = stress_state

    # ------------------------------------------------------------------
    # Operator methods — concrete for full-rank stress states
    # ------------------------------------------------------------------

    def _hydrostatic(self, stress: anp.ndarray) -> anp.ndarray:
        """Mean normal stress p = (σ11 + σ22 + σ33) / 3.

        All three direct components are stored, so no correction is needed.
        """
        return (stress[0] + stress[1] + stress[2]) / 3.0

    def _dev(self, stress: anp.ndarray) -> anp.ndarray:
        """Deviatoric stress s = σ − p δ."""
        p = self._hydrostatic(stress)
        return stress - p * self.stress_state.identity_np

    def isotropic_C(self, lam: float, mu: float) -> anp.ndarray:
        """Isotropic elastic stiffness via submatrix extraction.

        Builds the full 6×6 tensor, then extracts the ntens×ntens subblock
        for components [11, 22, 33, 12, ...].  No condensation is required
        because all direct stress components are free.

        Parameters
        ----------
        lam : float
            First Lamé constant λ.
        mu : float
            Shear modulus μ.

        Returns
        -------
        anp.ndarray, shape (ntens, ntens)
        """
        delta_6 = anp.array([1.0, 1.0, 1.0, 0.0, 0.0, 0.0])
        scale_6 = anp.array([2.0, 2.0, 2.0, 1.0, 1.0, 1.0])
        C6 = lam * anp.outer(delta_6, delta_6) + mu * anp.diag(scale_6)
        if self.ntens == 6:
            return C6
        # PLANE_STRAIN / AXISYMMETRIC: components [11, 22, 33, 12]
        idx = anp.array([0, 1, 2, 3])
        return C6[anp.ix_(idx, idx)]

    def elastic_stiffness(self, state=None) -> anp.ndarray:
        """Default isotropic elastic stiffness from self.E and self.nu."""
        if not (hasattr(self, "E") and hasattr(self, "nu")):
            raise NotImplementedError(
                f"{type(self).__name__}.elastic_stiffness: default implementation "
                "requires self.E and self.nu — override for anisotropic or "
                "parameter-less models."
            )
        mu = self.E / (2.0 * (1.0 + self.nu))
        lam = self.E * self.nu / ((1.0 + self.nu) * (1.0 - 2.0 * self.nu))
        return self.isotropic_C(lam, mu)

    def _I_vol(self) -> anp.ndarray:
        """Volumetric projection tensor P_vol = δ⊗δ / 3."""
        delta = self.stress_state.identity_np
        return anp.outer(delta, delta) / 3.0

    def _I_dev(self) -> anp.ndarray:
        """Deviatoric projection tensor P_dev = I − P_vol."""
        return anp.eye(self.ntens) - self._I_vol()


class MaterialModelPS(MaterialModel):
    """Stress-state base class for plane-stress elements (PLANE_STRESS).

    For plane stress, σ33 = 0 is enforced by static condensation of the
    elastic stiffness.  Only two direct components (σ11, σ22) are stored,
    so the von Mises computation must account for the physically-present
    but unstored deviatoric contribution of σ33 = 0.

    Provides concrete implementations of the operator methods:

    * :meth:`_hydrostatic` — p = (σ11 + σ22) / 3  (σ33 = 0)
    * :meth:`_dev`         — s = σ − p δ  (stored components only)
    * :meth:`isotropic_C`  — Schur complement (static condensation of σ33)
    * :meth:`_I_vol`       — δ⊗δ / 3  (δ = [1, 1, 0] for ntens=3)
    * :meth:`_I_dev`       — I − P_vol

    :meth:`_vonmises` is inherited from :class:`MaterialModel` (uses
    ``smooth_sqrt`` with n_missing=1, giving √(3/2 (s:s + p²))).

    Parameters
    ----------
    stress_state : StressState, optional
        Must satisfy ``stress_state.is_plane_stress``.
        Defaults to ``PLANE_STRESS``.

    Raises
    ------
    ValueError
        If ``stress_state.is_plane_stress`` is ``False``.
    """

    def __init__(self, stress_state: StressState = PLANE_STRESS):
        if not stress_state.is_plane_stress:
            raise ValueError(
                f"MaterialModelPS requires a plane-stress StressState "
                f"(is_plane_stress=True). "
                f"Got '{stress_state.name}'."
            )
        self.stress_state = stress_state

    # ------------------------------------------------------------------
    # Operator methods — concrete for PLANE_STRESS
    # ------------------------------------------------------------------

    def _hydrostatic(self, stress: anp.ndarray) -> anp.ndarray:
        """Mean normal stress p = (σ11 + σ22) / 3.

        σ33 = 0 is enforced externally; ndi_phys = 3 so we divide by 3.
        """
        return (stress[0] + stress[1]) / 3.0

    def _dev(self, stress: anp.ndarray) -> anp.ndarray:
        """Deviatoric stress of the stored components, s = σ − p δ."""
        p = self._hydrostatic(stress)
        return stress - p * self.stress_state.identity_np  # δ = [1, 1, 0]

    def isotropic_C(self, lam: float, mu: float) -> anp.ndarray:
        """Plane-stress isotropic stiffness via static condensation.

        Starts from the 4×4 plane-strain submatrix and applies the Schur
        complement to enforce σ33 = 0, yielding a 3×3 matrix for
        components [11, 22, 12].

        Parameters
        ----------
        lam : float
        mu : float

        Returns
        -------
        anp.ndarray, shape (3, 3)
        """
        delta_6 = anp.array([1.0, 1.0, 1.0, 0.0, 0.0, 0.0])
        scale_6 = anp.array([2.0, 2.0, 2.0, 1.0, 1.0, 1.0])
        C6 = lam * anp.outer(delta_6, delta_6) + mu * anp.diag(scale_6)
        # 4×4 plane-strain sub-block
        idx4 = anp.array([0, 1, 2, 3])
        C4 = C6[anp.ix_(idx4, idx4)]
        # Schur complement: eliminate σ33 (index 2 of C4)
        retain = anp.array([0, 1, 3])
        C_rr = C4[anp.ix_(retain, retain)]
        C_rc = C4[retain, 2]
        C_cc = C4[2, 2]
        return C_rr - anp.outer(C_rc, C_rc) / C_cc

    def elastic_stiffness(self, state=None) -> anp.ndarray:
        """Default plane-stress isotropic elastic stiffness from self.E and self.nu."""
        if not (hasattr(self, "E") and hasattr(self, "nu")):
            raise NotImplementedError(
                f"{type(self).__name__}.elastic_stiffness: default implementation "
                "requires self.E and self.nu — override for anisotropic or "
                "parameter-less models."
            )
        mu = self.E / (2.0 * (1.0 + self.nu))
        lam = self.E * self.nu / ((1.0 + self.nu) * (1.0 - 2.0 * self.nu))
        return self.isotropic_C(lam, mu)

    def _I_vol(self) -> anp.ndarray:
        """Volumetric projection tensor P_vol = δ⊗δ / 3."""
        delta = self.stress_state.identity_np  # [1, 1, 0]
        return anp.outer(delta, delta) / 3.0

    def _I_dev(self) -> anp.ndarray:
        """Deviatoric projection tensor P_dev = I − P_vol."""
        return anp.eye(self.ntens) - self._I_vol()


class MaterialModel1D(MaterialModel):
    """Stress-state base class for uniaxial (1D) elements (UNIAXIAL_1D).

    Only σ11 is stored; σ22 = σ33 = 0 are enforced by the element
    formulation.  The von Mises computation must account for two missing
    deviatoric components (n_missing = 2).

    Provides concrete implementations of the operator methods:

    * :meth:`_hydrostatic` — p = σ11 / 3  (σ22 = σ33 = 0)
    * :meth:`_dev`         — s = σ − p δ  (stored component only)
    * :meth:`isotropic_C`  — [[E]] where E = μ(3λ + 2μ) / (λ + μ)
    * :meth:`_I_vol`       — [[1/3]]
    * :meth:`_I_dev`       — [[2/3]]

    :meth:`_vonmises` is inherited from :class:`MaterialModel` (uses
    ``smooth_sqrt`` with n_missing=2, giving √(3/2 (s11² + 2p²)) ≈ |σ11|).

    Parameters
    ----------
    stress_state : StressState, optional
        Must have ``ntens == 1``.  Defaults to ``UNIAXIAL_1D``.

    Raises
    ------
    ValueError
        If ``stress_state.ntens != 1``.
    """

    def __init__(self, stress_state: StressState = UNIAXIAL_1D):
        if stress_state.ntens != 1:
            raise ValueError(
                f"MaterialModel1D requires a 1D StressState (ntens=1). "
                f"Got '{stress_state.name}' with ntens={stress_state.ntens}."
            )
        self.stress_state = stress_state

    # ------------------------------------------------------------------
    # Operator methods — concrete for UNIAXIAL_1D
    # ------------------------------------------------------------------

    def _hydrostatic(self, stress: anp.ndarray) -> anp.ndarray:
        """Mean normal stress p = σ11 / 3.

        σ22 = σ33 = 0 are enforced externally; ndi_phys = 3 so we divide by 3.
        """
        return stress[0] / 3.0

    def _dev(self, stress: anp.ndarray) -> anp.ndarray:
        """Deviatoric stress of the stored component, s = σ − p δ."""
        p = self._hydrostatic(stress)
        return stress - p * self.stress_state.identity_np  # δ = [1.0]

    def isotropic_C(self, lam: float, mu: float) -> anp.ndarray:
        """1D elastic stiffness [[E]] where E = μ(3λ + 2μ) / (λ + μ).

        Parameters
        ----------
        lam : float
        mu : float

        Returns
        -------
        anp.ndarray, shape (1, 1)
        """
        E = mu * (3.0 * lam + 2.0 * mu) / (lam + mu)
        return anp.array([[E]])

    def elastic_stiffness(self, state=None) -> anp.ndarray:
        """Default 1D isotropic elastic stiffness [[E]] from self.E and self.nu."""
        if not (hasattr(self, "E") and hasattr(self, "nu")):
            raise NotImplementedError(
                f"{type(self).__name__}.elastic_stiffness: default implementation "
                "requires self.E and self.nu — override for anisotropic or "
                "parameter-less models."
            )
        mu = self.E / (2.0 * (1.0 + self.nu))
        lam = self.E * self.nu / ((1.0 + self.nu) * (1.0 - 2.0 * self.nu))
        return self.isotropic_C(lam, mu)

    def _I_vol(self) -> anp.ndarray:
        """Volumetric projection tensor [[1/3]] for ntens=1."""
        delta = self.stress_state.identity_np  # [1.0]
        return anp.outer(delta, delta) / 3.0

    def _I_dev(self) -> anp.ndarray:
        """Deviatoric projection tensor [[2/3]] for ntens=1."""
        return anp.eye(1) - self._I_vol()
