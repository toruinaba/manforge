"""Stress update procedure (constitutive integration) and return mapping.

Terminology
-----------
*Return mapping*
    The plastic correction algorithm that projects the elastic trial stress
    back onto the yield surface, producing a converged stress σ_{n+1}, updated
    internal state, and plastic multiplier Δλ.  This is the inner loop —
    tangent computation is not part of return mapping.

*Stress update*
    The complete constitutive integration procedure for one load increment:
    elastic trial → yield check → return mapping → consistent tangent.
    Corresponds to what ABAQUS calls from a UMAT subroutine.

Algorithm (stress_update)
-------------------------
1. Elastic trial:  σ_trial = σ_n + C Δε
2. Yield check:    f_trial = f(σ_trial, state_n)
   → elastic if f_trial ≤ 0
3. Return mapping — solver selected by ``method``:

   *numerical_newton*  (generic, default fallback)
     Scalar NR on Δλ using ``jax.grad`` for flow direction and linearisation
     (reduced hardening), or augmented (ntens+1+n_state) vector NR (augmented).

   *user_defined*  (model-supplied, opt-in)
     Calls ``model.user_defined_return_mapping(σ_trial, C, state_n)`` if the
     model provides one.  May use any algorithm internally (closed-form,
     custom NR, etc.).

4. Consistent tangent — path selected by ``method``:

   *autodiff*  (generic, always available)
     Implicit differentiation via ``jax.jacobian``.

   *user_defined*  (model-supplied, opt-in)
     Calls ``model.user_defined_tangent(σ, state, Δλ, C, state_n)``
     if the model provides one.

The ``method`` parameter controls which paths are attempted:

* ``"auto"``              — use user_defined hooks if available, else
                            numerical_newton + autodiff tangent (default).
* ``"numerical_newton"``  — always use the generic NR solver, even if the
                            model provides user_defined hooks.
* ``"user_defined"``      — require model hooks; raise ``NotImplementedError``
                            if absent.

Future solver strategies (``"cutting_plane"``, ``"bisection"``, …) will be
added as additional ``method`` values without changing the API.

Notes
-----
- Python if/break is used for the elastic check and NR convergence.
  jax.jit compatibility is left as a TODO.
- For J2 with linear isotropic hardening the NR converges in a single step
  (the classic radial-return closed form).
"""

from dataclasses import dataclass, field

import jax.numpy as jnp

from manforge.core.solver import _select_nr
from manforge.core.tangent import _select_tangent

_VALID_METHODS = ("auto", "numerical_newton", "user_defined")


def _validate_method(method: str) -> None:
    if method not in _VALID_METHODS:
        raise ValueError(
            f"method must be 'auto', 'numerical_newton', or 'user_defined'; "
            f"got {method!r}"
        )


def _try_user_return_mapping(model, stress_trial, C, state_n, method):
    """Attempt user_defined_return_mapping; return ReturnMappingResult or None.

    Returns None when method='auto' and the model has no return mapping hook.
    Raises NotImplementedError when method='user_defined' and hook is absent.
    """
    if method == "numerical_newton":
        return None
    rm = model.user_defined_return_mapping(stress_trial, C, state_n)
    if rm is not None:
        return rm
    if method == "user_defined":
        raise NotImplementedError(
            f"{type(model).__name__} does not implement user_defined_return_mapping; "
            "cannot use method='user_defined'."
        )
    return None


def _try_user_tangent(model, rm, stress_n, state_n, C, method):
    """Attempt user_defined_tangent; return ddsdde array or None."""
    if method == "numerical_newton":
        return None
    ddsdde = model.user_defined_tangent(rm.stress, rm.state, rm.dlambda, C, state_n)
    if ddsdde is not None:
        return ddsdde
    if method == "user_defined":
        raise NotImplementedError(
            f"{type(model).__name__} does not implement user_defined_tangent; "
            "cannot use method='user_defined'."
        )
    return None


@dataclass
class ReturnMappingResult:
    """Result of a single return-mapping (plastic correction) step.

    Contains only the output of the plastic correction algorithm — tangent
    and trial stress are part of :class:`StressUpdateResult`.

    Attributes
    ----------
    stress : jnp.ndarray, shape (ntens,)
        Converged stress σ_{n+1}.
    state : dict
        Converged internal state at step n+1.
    dlambda : jnp.ndarray, scalar
        Plastic multiplier increment Δλ.
    n_iterations : int
        Number of Newton updates performed by the framework's numerical solver.
        Zero when ``method="user_defined"`` (the model's own corrector is
        responsible for convergence tracking).
    residual_history : list[float]
        Residual norm recorded at the start of each NR iteration by the
        framework's numerical solver.  For ``method="numerical_newton"``
        plastic steps ``len(residual_history) == n_iterations + 1`` (initial
        residual plus one entry after each Newton update; the last entry is the
        converged residual below ``tol``).  Empty when ``method="user_defined"``
        unless the corrector returns a 5-tuple with its own history.
    """

    stress: jnp.ndarray
    state: dict
    dlambda: jnp.ndarray
    n_iterations: int = 0
    residual_history: list = field(default_factory=list)


@dataclass
class StressUpdateResult:
    """Result of a complete stress update (constitutive integration) step.

    Wraps :class:`ReturnMappingResult` and adds the consistent tangent,
    trial stress, and plasticity flag.

    Attributes
    ----------
    return_mapping : ReturnMappingResult or None
        Plastic correction result.  ``None`` for elastic steps (no plastic
        correction was performed).
    ddsdde : jnp.ndarray, shape (ntens, ntens)
        Consistent tangent operator dσ_{n+1}/dΔε.
    stress_trial : jnp.ndarray, shape (ntens,)
        Elastic trial stress σ_trial = σ_n + C Δε.
    is_plastic : bool
        True if the step activated plasticity.
    _state_n : dict
        Internal state at step n (stored for the ``state`` convenience
        property on elastic steps).  Not part of the public API.
    """

    return_mapping: "ReturnMappingResult | None"
    ddsdde: jnp.ndarray
    stress_trial: jnp.ndarray
    is_plastic: bool
    _state_n: dict = field(repr=False)

    @property
    def stress(self) -> jnp.ndarray:
        """Converged stress σ_{n+1} (trial stress for elastic steps)."""
        if self.return_mapping is None:
            return self.stress_trial
        return self.return_mapping.stress

    @property
    def state(self) -> dict:
        """Converged internal state (unchanged state_n for elastic steps)."""
        if self.return_mapping is None:
            return self._state_n
        return self.return_mapping.state

    @property
    def dlambda(self) -> jnp.ndarray:
        """Plastic multiplier increment (0 for elastic steps)."""
        if self.return_mapping is None:
            return jnp.array(0.0)
        return self.return_mapping.dlambda

    @property
    def n_iterations(self) -> int:
        """Newton updates performed (0 for elastic or user_defined steps)."""
        if self.return_mapping is None:
            return 0
        return self.return_mapping.n_iterations

    @property
    def residual_history(self) -> list:
        """Residual norms from the framework NR solver (empty for elastic / user_defined)."""
        if self.return_mapping is None:
            return []
        return self.return_mapping.residual_history


def return_mapping(
    model,
    stress_trial: jnp.ndarray,
    C: jnp.ndarray,
    state_n: dict,
    method: str = "auto",
    max_iter: int = 50,
    tol: float = 1e-10,
) -> ReturnMappingResult:
    """Perform the plastic correction (return mapping) for one load increment.

    Projects the elastic trial stress back onto the yield surface using the
    solver strategy specified by ``method``.  Does NOT compute the consistent
    tangent — call :func:`stress_update` for the complete constitutive
    integration including the tangent.

    Parameters
    ----------
    model : MaterialModel
        Constitutive model instance.
    stress_trial : jnp.ndarray, shape (ntens,)
        Elastic trial stress σ_trial = σ_n + C Δε.  The caller is responsible
        for computing this before calling return_mapping.
    C : jnp.ndarray, shape (ntens, ntens)
        Elastic stiffness tensor.
    state_n : dict
        Internal state at the beginning of the increment.
    method : {"auto", "numerical_newton", "user_defined"}, optional
        Solver strategy.

        * ``"auto"``             — use ``model.user_defined_return_mapping``
                                   if available, else ``"numerical_newton"``
                                   (default).
        * ``"numerical_newton"`` — framework generic NR solver (explicit:
                                   scalar NR on Δλ; implicit: augmented
                                   vector NR).
        * ``"user_defined"``     — require ``model.user_defined_return_mapping``;
                                   raise ``NotImplementedError`` if absent.
    max_iter : int, optional
        Maximum Newton iterations (default 50, ``numerical_newton`` only).
    tol : float, optional
        Convergence tolerance (default 1e-10, ``numerical_newton`` only).

    Returns
    -------
    ReturnMappingResult
        Converged stress, state, plastic multiplier, and NR convergence info.

    Raises
    ------
    RuntimeError
        If the NR iteration does not converge within ``max_iter`` steps.
    NotImplementedError
        If ``method="user_defined"`` and the model does not implement
        ``user_defined_return_mapping``.
    ValueError
        If ``method`` is not one of the recognised values.
    """
    _validate_method(method)

    rm = _try_user_return_mapping(model, stress_trial, C, state_n, method)
    if rm is not None:
        return rm

    # numerical_newton path
    nr = _select_nr(model)
    stress, state_new, dlambda, _n_iter, _res_hist = nr(
        model, stress_trial, C, state_n, max_iter, tol
    )

    return ReturnMappingResult(
        stress=stress,
        state=state_new,
        dlambda=dlambda,
        n_iterations=_n_iter,
        residual_history=_res_hist,
    )


def stress_update(
    model,
    strain_inc: jnp.ndarray,
    stress_n: jnp.ndarray,
    state_n: dict,
    method: str = "auto",
    max_iter: int = 50,
    tol: float = 1e-10,
) -> StressUpdateResult:
    """Perform a complete constitutive stress update for one load increment.

    Executes the full integration procedure: elastic trial → yield check →
    return mapping → consistent tangent.

    Parameters
    ----------
    model : MaterialModel
        Constitutive model instance.
    strain_inc : jnp.ndarray, shape (ntens,)
        Strain increment Δε (engineering shear convention).
    stress_n : jnp.ndarray, shape (ntens,)
        Stress at the beginning of the increment.
    state_n : dict
        Internal state at the beginning of the increment.
    method : {"auto", "numerical_newton", "user_defined"}, optional
        Solver strategy passed to :func:`return_mapping` and used to select
        the tangent computation path.

        * ``"auto"``             — use model hooks if available, else generic
                                   numerical solver + autodiff tangent (default).
        * ``"numerical_newton"`` — always use the framework NR solver and
                                   autodiff tangent.
        * ``"user_defined"``     — require model hooks for both corrector and
                                   tangent; raise ``NotImplementedError`` if
                                   absent.
    max_iter : int, optional
        Maximum Newton iterations (default 50, ``numerical_newton`` only).
    tol : float, optional
        Convergence tolerance (default 1e-10, ``numerical_newton`` only).

    Returns
    -------
    StressUpdateResult
        Complete result including :class:`ReturnMappingResult` (or ``None``
        for elastic steps), consistent tangent, trial stress, and plasticity
        flag.

    Raises
    ------
    RuntimeError
        If the NR iteration does not converge within ``max_iter`` steps.
    NotImplementedError
        If ``method="user_defined"`` and the model does not implement the
        required hooks.
    ValueError
        If ``method`` is not one of the recognised values.
    """
    _validate_method(method)

    # ------------------------------------------------------------------
    # Step 1 — elastic stiffness and trial stress
    # ------------------------------------------------------------------
    C = model.elastic_stiffness()
    stress_trial = stress_n + C @ strain_inc

    # ------------------------------------------------------------------
    # Step 2 — elastic check
    # TODO: replace Python if with jax.lax.cond for jit compatibility
    # ------------------------------------------------------------------
    f_trial = model.yield_function(stress_trial, state_n)

    if f_trial <= 0.0:
        return StressUpdateResult(
            return_mapping=None,
            ddsdde=C,
            stress_trial=stress_trial,
            is_plastic=False,
            _state_n=state_n,
        )

    # ------------------------------------------------------------------
    # Step 3 — return mapping (plastic correction)
    # ------------------------------------------------------------------
    rm = return_mapping(model, stress_trial, C, state_n, method=method,
                        max_iter=max_iter, tol=tol)

    # ------------------------------------------------------------------
    # Step 4 — consistent tangent
    # ------------------------------------------------------------------
    _ddsdde = _try_user_tangent(model, rm, stress_n, state_n, C, method)
    if _ddsdde is not None:
        return StressUpdateResult(
            return_mapping=rm,
            ddsdde=_ddsdde,
            stress_trial=stress_trial,
            is_plastic=True,
            _state_n=state_n,
        )

    # Autodiff tangent
    tangent_fn = _select_tangent(model)
    ddsdde = tangent_fn(model, rm.stress, rm.state, rm.dlambda, stress_n, state_n)

    return StressUpdateResult(
        return_mapping=rm,
        ddsdde=ddsdde,
        stress_trial=stress_trial,
        is_plastic=True,
        _state_n=state_n,
    )
