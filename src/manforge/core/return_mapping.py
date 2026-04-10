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
     Calls ``model.plastic_corrector(σ_trial, C, state_n, params)`` if
     the model provides one.

4. Consistent tangent — two paths depending on ``method``:

   *autodiff*  (generic, default fallback)
     Implicit differentiation of the 7×7 residual via ``jax.jacobian``
     (see :mod:`manforge.core.tangent`).

   *analytical*  (closed-form, opt-in)
     Calls ``model.analytical_tangent(σ, state, Δλ, C, state_n, params)``
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

import jax
import jax.numpy as jnp

from manforge.core.tangent import consistent_tangent


def return_mapping(
    model,
    strain_inc: jnp.ndarray,
    stress_n: jnp.ndarray,
    state_n: dict,
    params: dict,
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
    params : dict
        Material parameters.
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
    stress_new : jnp.ndarray, shape (ntens,)
        Updated stress σ_{n+1}.
    state_new : dict
        Updated internal state.
    ddsdde : jnp.ndarray, shape (ntens, ntens)
        Consistent tangent dσ_{n+1}/dΔε.

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
    C = model.elastic_stiffness(params)
    stress_trial = stress_n + C @ strain_inc

    # ------------------------------------------------------------------
    # Step 2 — elastic check
    # TODO: replace Python if with jax.lax.cond for jit compatibility
    # ------------------------------------------------------------------
    f_trial = model.yield_function(stress_trial, state_n, params)

    if f_trial <= 0.0:
        return stress_trial, state_n, C

    # ------------------------------------------------------------------
    # Step 3 — plastic correction
    # ------------------------------------------------------------------
    _plastic_done = False

    if method != "autodiff":
        _result = model.plastic_corrector(stress_trial, C, state_n, params)
        if _result is not None:
            stress, state_new, dlambda = _result
            _plastic_done = True
        elif method == "analytical":
            raise NotImplementedError(
                f"{type(model).__name__} does not implement plastic_corrector; "
                "cannot use method='analytical'."
            )

    if not _plastic_done:
        # Generic NR + autodiff path (unchanged from original)
        dlambda = jnp.array(0.0)
        stress = stress_trial
        state_new = state_n

        for _iteration in range(max_iter):
            # Update state and flow direction
            state_new = model.hardening_increment(dlambda, stress, state_n, params)
            n = jax.grad(lambda s: model.yield_function(s, state_new, params))(stress)

            # Stress correction (radial return for fixed n)
            stress = stress_trial - dlambda * (C @ n)

            # Re-evaluate after stress update
            state_new = model.hardening_increment(dlambda, stress, state_n, params)
            f = model.yield_function(stress, state_new, params)

            if jnp.abs(f) < tol:
                break

            # df/dΔλ  (total derivative, freezing n direction at current stress)
            #   = −n^T C n  +  h
            # where h = (∂f/∂state)(∂state/∂Δλ)  computed via jax.grad
            def _f_residual(dl, _stress=stress):
                st = model.hardening_increment(dl, _stress, state_n, params)
                nn = jax.grad(lambda s: model.yield_function(s, st, params))(_stress)
                s_upd = stress_trial - dl * (C @ nn)
                return model.yield_function(s_upd, st, params)

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
        _ddsdde = model.analytical_tangent(
            stress, state_new, dlambda, C, state_n, params
        )
        if _ddsdde is not None:
            return stress, state_new, _ddsdde
        elif method == "analytical":
            raise NotImplementedError(
                f"{type(model).__name__} does not implement analytical_tangent; "
                "cannot use method='analytical'."
            )

    # Generic autodiff tangent (unchanged from original)
    ddsdde = consistent_tangent(
        model, stress, state_new, dlambda, stress_n, state_n, params
    )

    return stress, state_new, ddsdde
