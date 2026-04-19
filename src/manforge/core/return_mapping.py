"""Generic return mapping solver (closest point projection, associative flow).

Algorithm
---------
1. Elastic trial:  σ_trial = σ_n + C Δε
2. Yield check:    f_trial = f(σ_trial, state_n)
   → elastic if f_trial ≤ 0
3. Plastic correction — two paths depending on ``method``:

   *autodiff*  (generic, default fallback)
     Scalar NR on Δλ using ``jax.grad`` for flow direction and linearisation.

   *analytical*  (closed-form, opt-in)
     Calls ``model.plastic_corrector(σ_trial, C, state_n)`` if
     the model provides one.

4. Consistent tangent — two paths depending on ``method``:

   *autodiff*  (generic, default fallback)
     Implicit differentiation of the (ntens+1)×(ntens+1) residual via
     ``jax.jacobian`` (see :mod:`manforge.core.tangent`).

   *analytical*  (closed-form, opt-in)
     Calls ``model.analytical_tangent(σ, state, Δλ, C, state_n)``
     if the model provides one.

The ``method`` parameter controls which paths are attempted:

* ``"auto"``       — use analytical if available, else autodiff.
* ``"autodiff"``   — always use the generic NR + autodiff path.
* ``"analytical"`` — require closed-form; raise ``NotImplementedError``
                     if the model does not implement the hook.

Notes
-----
- Python if/break is used for the elastic check and NR convergence.
  jax.jit compatibility is left as a TODO.
- For J2 with linear isotropic hardening the NR converges in a single step
  (the classic radial-return closed form).
"""

from dataclasses import dataclass

import jax
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree

from manforge.core.tangent import augmented_consistent_tangent, consistent_tangent


@dataclass
class ReturnMappingResult:
    """Result of a single return-mapping increment.

    Attributes
    ----------
    stress : jnp.ndarray, shape (ntens,)
        Converged stress σ_{n+1}.
    state : dict
        Converged internal state at step n+1.
    ddsdde : jnp.ndarray, shape (ntens, ntens)
        Consistent tangent dσ_{n+1}/dΔε.
    dlambda : jnp.ndarray, scalar
        Plastic multiplier increment Δλ (0 for elastic steps).
    stress_trial : jnp.ndarray, shape (ntens,)
        Elastic trial stress σ_trial = σ_n + C Δε.
    is_plastic : bool
        True if the step is plastic (yield function was active).
    """

    stress: jnp.ndarray
    state: dict
    ddsdde: jnp.ndarray
    dlambda: jnp.ndarray
    stress_trial: jnp.ndarray
    is_plastic: bool


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
    """
    from manforge.core.residual import make_augmented_residual

    ntens = model.ntens
    residual_fn, n_state, unflatten_fn = make_augmented_residual(
        model, stress_trial, C, state_n
    )

    flat_state_n, _ = ravel_pytree(state_n)
    x = jnp.concatenate([stress_trial, jnp.array([0.0]), flat_state_n])

    for _iteration in range(max_iter):
        R = residual_fn(x)
        if jnp.max(jnp.abs(R)) < tol:
            break
        J = jax.jacobian(residual_fn)(x)
        dx = jnp.linalg.solve(J, R)
        x = x - dx
    else:
        raise RuntimeError(
            f"_augmented_nr: NR did not converge in {max_iter} iterations "
            f"(||R||_inf = {float(jnp.max(jnp.abs(R))):.3e}, tol = {tol:.3e})"
        )

    stress = x[:ntens]
    dlambda = jnp.asarray(x[ntens])
    state_new = unflatten_fn(x[ntens + 1 :])
    return stress, state_new, dlambda


def return_mapping(
    model,
    strain_inc: jnp.ndarray,
    stress_n: jnp.ndarray,
    state_n: dict,
    method: str = "auto",
    max_iter: int = 50,
    tol: float = 1e-10,
):
    """Perform one load increment via closest point projection.

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
    method : {"auto", "autodiff", "analytical"}, optional
        Selects which plastic-correction and tangent path to use.

        * ``"auto"``       — use model's analytical hooks if available,
                             fall back to generic NR + autodiff otherwise
                             (default).
        * ``"autodiff"``   — always use the generic NR + autodiff path,
                             even if the model provides analytical hooks.
        * ``"analytical"`` — require the model's analytical hooks; raise
                             ``NotImplementedError`` if they are absent.
    max_iter : int, optional
        Maximum Newton-Raphson iterations (default 50, autodiff path only).
    tol : float, optional
        Absolute convergence tolerance on |f| (default 1e-10, autodiff only).

    Returns
    -------
    ReturnMappingResult
        Converged stress, state, consistent tangent, plastic multiplier,
        trial stress, and plasticity flag.

    Raises
    ------
    RuntimeError
        If the NR iteration does not converge within ``max_iter`` steps.
    NotImplementedError
        If ``method="analytical"`` and the model does not implement
        ``plastic_corrector`` or ``analytical_tangent``.
    ValueError
        If ``method`` is not one of the recognised values.
    """
    if method not in ("auto", "autodiff", "analytical"):
        raise ValueError(
            f"method must be 'auto', 'autodiff', or 'analytical'; got {method!r}"
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
        return ReturnMappingResult(
            stress=stress_trial,
            state=state_n,
            ddsdde=C,
            dlambda=jnp.array(0.0),
            stress_trial=stress_trial,
            is_plastic=False,
        )

    # ------------------------------------------------------------------
    # Step 3 — plastic correction
    # ------------------------------------------------------------------
    _plastic_done = False

    if method != "autodiff":
        _result = model.plastic_corrector(stress_trial, C, state_n)
        if _result is not None:
            stress, state_new, dlambda = _result
            _plastic_done = True
        elif method == "analytical":
            raise NotImplementedError(
                f"{type(model).__name__} does not implement plastic_corrector; "
                "cannot use method='analytical'."
            )

    if not _plastic_done:
        if model.hardening_type == "implicit":
            # Augmented vector NR — state variables are independent unknowns
            stress, state_new, dlambda = _augmented_nr(
                model, stress_trial, C, state_n, max_iter, tol
            )
        else:
            # Generic NR + autodiff path (unchanged from original)
            dlambda = jnp.array(0.0)
            stress = stress_trial
            state_new = state_n

            for _iteration in range(max_iter):
                # Update state and flow direction
                state_new = model.hardening_increment(dlambda, stress, state_n)
                n = jax.grad(lambda s: model.yield_function(s, state_new))(stress)

                # Stress correction (radial return for fixed n)
                stress = stress_trial - dlambda * (C @ n)

                # Re-evaluate after stress update
                state_new = model.hardening_increment(dlambda, stress, state_n)
                f = model.yield_function(stress, state_new)

                if jnp.abs(f) < tol:
                    break

                # df/dΔλ  (total derivative, freezing n direction at current stress)
                def _f_residual(dl, _stress=stress):
                    st = model.hardening_increment(dl, _stress, state_n)
                    nn = jax.grad(lambda s: model.yield_function(s, st))(_stress)
                    s_upd = stress_trial - dl * (C @ nn)
                    return model.yield_function(s_upd, st)

                dfddl = jax.grad(_f_residual)(dlambda)

                dlambda = dlambda - f / dfddl

            else:
                raise RuntimeError(
                    f"return_mapping: NR did not converge in {max_iter} iterations "
                    f"(|f| = {float(jnp.abs(f)):.3e}, tol = {tol:.3e})"
                )

    # ------------------------------------------------------------------
    # Step 4 — consistent tangent
    # ------------------------------------------------------------------
    if method != "autodiff":
        _ddsdde = model.analytical_tangent(stress, state_new, dlambda, C, state_n)
        if _ddsdde is not None:
            return ReturnMappingResult(
                stress=stress,
                state=state_new,
                ddsdde=_ddsdde,
                dlambda=dlambda,
                stress_trial=stress_trial,
                is_plastic=True,
            )
        elif method == "analytical":
            raise NotImplementedError(
                f"{type(model).__name__} does not implement analytical_tangent; "
                "cannot use method='analytical'."
            )

    # Autodiff tangent — augmented system for implicit hardening models, else standard
    if model.hardening_type == "implicit":
        ddsdde = augmented_consistent_tangent(
            model, stress, state_new, dlambda, stress_n, state_n
        )
    else:
        ddsdde = consistent_tangent(
            model, stress, state_new, dlambda, stress_n, state_n
        )

    return ReturnMappingResult(
        stress=stress,
        state=state_new,
        ddsdde=ddsdde,
        dlambda=dlambda,
        stress_trial=stress_trial,
        is_plastic=True,
    )
