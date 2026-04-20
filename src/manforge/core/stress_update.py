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
     (explicit hardening), or augmented (ntens+1+n_state) vector NR (implicit).

   *user_defined*  (model-supplied, opt-in)
     Calls ``model.user_defined_corrector(σ_trial, C, state_n)`` if the model
     provides one.  The corrector may use any algorithm internally (closed-form,
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

import jax
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree

from manforge.core.tangent import augmented_consistent_tangent, consistent_tangent


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


def _augmented_nr(model, stress_trial, C, state_n, max_iter, tol):
    """Newton-Raphson solver for the augmented (ntens+1+n_state) residual system.

    Used when ``model.hardening_type == 'implicit'``.

    Parameters
    ----------
    model : MaterialModel
    stress_trial : jnp.ndarray, shape (ntens,)
    C : jnp.ndarray, shape (ntens, ntens)
    state_n : dict
    max_iter : int
    tol : float

    Returns
    -------
    stress : jnp.ndarray, shape (ntens,)
    state_new : dict
    dlambda : jnp.ndarray, scalar
    n_iterations : int
        Number of Newton updates performed.
    residual_history : list[float]
        Residual norm (inf-norm) at the start of each iteration.
    """
    from manforge.core.residual import make_augmented_residual

    ntens = model.ntens
    residual_fn, n_state, unflatten_fn = make_augmented_residual(
        model, stress_trial, C, state_n
    )

    flat_state_n, _ = ravel_pytree(state_n)
    x = jnp.concatenate([stress_trial, jnp.array([0.0]), flat_state_n])

    residual_history = []
    n_iterations = 0
    for _iteration in range(max_iter):
        R = residual_fn(x)
        res_norm = float(jnp.max(jnp.abs(R)))
        residual_history.append(res_norm)
        if res_norm < tol:
            break
        J = jax.jacobian(residual_fn)(x)
        dx = jnp.linalg.solve(J, R)
        x = x - dx
        n_iterations += 1
    else:
        raise RuntimeError(
            f"_augmented_nr: NR did not converge in {max_iter} iterations "
            f"(||R||_inf = {float(jnp.max(jnp.abs(R))):.3e}, tol = {tol:.3e})"
        )

    stress = x[:ntens]
    dlambda = jnp.asarray(x[ntens])
    state_new = unflatten_fn(x[ntens + 1 :])
    return stress, state_new, dlambda, n_iterations, residual_history


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

        * ``"auto"``             — use ``model.user_defined_corrector`` if
                                   available, else ``"numerical_newton"``
                                   (default).
        * ``"numerical_newton"`` — framework generic NR solver (explicit:
                                   scalar NR on Δλ; implicit: augmented
                                   vector NR).
        * ``"user_defined"``     — require ``model.user_defined_corrector``;
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
        ``user_defined_corrector``.
    ValueError
        If ``method`` is not one of the recognised values.
    """
    if method not in ("auto", "numerical_newton", "user_defined"):
        raise ValueError(
            f"method must be 'auto', 'numerical_newton', or 'user_defined'; "
            f"got {method!r}"
        )

    _n_iter = 0
    _res_hist = []

    if method != "numerical_newton":
        _result = model.user_defined_corrector(stress_trial, C, state_n)
        if _result is not None:
            if len(_result) == 5:
                stress, state_new, dlambda, _n_iter, _res_hist = _result
            else:
                stress, state_new, dlambda = _result
            return ReturnMappingResult(
                stress=stress,
                state=state_new,
                dlambda=dlambda,
                n_iterations=_n_iter,
                residual_history=_res_hist,
            )
        if method == "user_defined":
            raise NotImplementedError(
                f"{type(model).__name__} does not implement user_defined_corrector; "
                "cannot use method='user_defined'."
            )

    # numerical_newton path
    if model.hardening_type == "implicit":
        stress, state_new, dlambda, _n_iter, _res_hist = _augmented_nr(
            model, stress_trial, C, state_n, max_iter, tol
        )
    else:
        dlambda = jnp.array(0.0)
        stress = stress_trial
        state_new = state_n

        for _iteration in range(max_iter):
            state_new = model.hardening_increment(dlambda, stress, state_n)
            n = jax.grad(lambda s: model.yield_function(s, state_new))(stress)
            stress = stress_trial - dlambda * (C @ n)
            state_new = model.hardening_increment(dlambda, stress, state_n)
            f = model.yield_function(stress, state_new)
            _res_hist.append(float(jnp.abs(f)))

            if jnp.abs(f) < tol:
                break

            def _f_residual(dl, _stress=stress):
                st = model.hardening_increment(dl, _stress, state_n)
                nn = jax.grad(lambda s: model.yield_function(s, st))(_stress)
                s_upd = stress_trial - dl * (C @ nn)
                return model.yield_function(s_upd, st)

            dfddl = jax.grad(_f_residual)(dlambda)
            dlambda = dlambda - f / dfddl
            _n_iter += 1

        else:
            raise RuntimeError(
                f"return_mapping: NR did not converge in {max_iter} iterations "
                f"(|f| = {float(jnp.abs(f)):.3e}, tol = {tol:.3e})"
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
    if method not in ("auto", "numerical_newton", "user_defined"):
        raise ValueError(
            f"method must be 'auto', 'numerical_newton', or 'user_defined'; "
            f"got {method!r}"
        )

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
    if method != "numerical_newton":
        _ddsdde = model.user_defined_tangent(
            rm.stress, rm.state, rm.dlambda, C, state_n
        )
        if _ddsdde is not None:
            return StressUpdateResult(
                return_mapping=rm,
                ddsdde=_ddsdde,
                stress_trial=stress_trial,
                is_plastic=True,
                _state_n=state_n,
            )
        if method == "user_defined":
            raise NotImplementedError(
                f"{type(model).__name__} does not implement user_defined_tangent; "
                "cannot use method='user_defined'."
            )

    # Autodiff tangent
    if model.hardening_type == "implicit":
        ddsdde = augmented_consistent_tangent(
            model, rm.stress, rm.state, rm.dlambda, stress_n, state_n
        )
    else:
        ddsdde = consistent_tangent(
            model, rm.stress, rm.state, rm.dlambda, stress_n, state_n
        )

    return StressUpdateResult(
        return_mapping=rm,
        ddsdde=ddsdde,
        stress_trial=stress_trial,
        is_plastic=True,
        _state_n=state_n,
    )
