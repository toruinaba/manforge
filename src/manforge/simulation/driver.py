"""Strain-driven and stress-driven loading drivers.

Strain-driven drivers loop over a prescribed strain history, call
``return_mapping`` at each step, and accumulate the resulting stress history.

Stress-driven drivers prescribe a target stress history and solve for the
corresponding strain increments using Newton-Raphson iteration with the
consistent tangent (ddsdde).

Conventions
-----------
- All strain/stress arrays use the engineering-shear Voigt convention:
    ε, σ = [11, 22, 33, 12, 13, 23]
- *Cumulative* quantities are the input; increments are computed internally.
"""

import numpy as np
import jax.numpy as jnp

from manforge.core.return_mapping import return_mapping


class UniaxialDriver:
    """Uniaxial strain-driven loading (ε11 prescribed, lateral strains zero).

    Parameters
    ----------
    (none — stateless, all inputs passed to :meth:`run`)
    """

    def run(self, model, strain_history, params):
        """Run the uniaxial loading history.

        Parameters
        ----------
        model : MaterialModel
            Constitutive model instance.
        strain_history : array-like, shape (N,)
            Cumulative axial strain ε11 at each step.
            Increments are computed as Δε_i = ε_i − ε_{i-1}  (ε_0 = 0).
        params : dict
            Material parameters.

        Returns
        -------
        stress_history : np.ndarray, shape (N,)
            Axial stress σ11 at each step.
        """
        strain_history = np.asarray(strain_history, dtype=float)
        N = len(strain_history)
        ntens = model.ntens

        stress_n = jnp.zeros(ntens)
        state_n = model.initial_state()
        eps_prev = 0.0

        stress_out = np.zeros(N)

        for i, eps_i in enumerate(strain_history):
            deps11 = eps_i - eps_prev
            strain_inc = jnp.zeros(ntens).at[0].set(deps11)

            stress_n, state_n, _ = return_mapping(
                model, strain_inc, stress_n, state_n, params
            )
            stress_out[i] = float(stress_n[0])
            eps_prev = eps_i

        return stress_out


class GeneralDriver:
    """General multi-component strain-driven loading.

    Parameters
    ----------
    (none — stateless)
    """

    def run(self, model, strain_history, params):
        """Run a general strain history.

        Parameters
        ----------
        model : MaterialModel
        strain_history : array-like, shape (N, ntens)
            Cumulative strain tensor (Voigt, engineering shear) at each step.
        params : dict

        Returns
        -------
        stress_history : np.ndarray, shape (N, ntens)
            Full stress tensor at each step.
        """
        strain_history = np.asarray(strain_history, dtype=float)
        N = strain_history.shape[0]
        ntens = model.ntens

        stress_n = jnp.zeros(ntens)
        state_n = model.initial_state()
        eps_prev = np.zeros(ntens)

        stress_out = np.zeros((N, ntens))

        for i in range(N):
            strain_inc = jnp.array(strain_history[i] - eps_prev)

            stress_n, state_n, _ = return_mapping(
                model, strain_inc, stress_n, state_n, params
            )
            stress_out[i] = np.array(stress_n)
            eps_prev = strain_history[i]

        return stress_out


class StressDriver:
    """Stress-controlled loading driver.

    Prescribes a target stress history and solves for the corresponding strain
    increments using Newton-Raphson iteration with the consistent tangent
    (ddsdde).  Useful for simulating uniaxial stress loading in a multi-axial
    model (e.g., σ11 ramping, all other components zero) where the lateral
    strains adjust freely.

    Parameters
    ----------
    max_iter : int, optional
        Maximum Newton-Raphson iterations per step (default 20).
    tol : float, optional
        Convergence tolerance on the infinity norm of the stress residual
        (default 1e-8).
    """

    def __init__(self, max_iter: int = 20, tol: float = 1e-8):
        self.max_iter = max_iter
        self.tol = tol

    def run(self, model, stress_history, params):
        """Run the stress-controlled loading history.

        Parameters
        ----------
        model : MaterialModel
            Constitutive model instance.
        stress_history : array-like, shape (N, ntens)
            Cumulative target stress tensor (Voigt) at each step.
        params : dict
            Material parameters.

        Returns
        -------
        dict with keys:

        * ``"stress"`` : np.ndarray, shape (N, ntens) — converged stress.
        * ``"strain"`` : np.ndarray, shape (N, ntens) — accumulated strain.

        Raises
        ------
        RuntimeError
            If Newton-Raphson does not converge within ``max_iter`` iterations
            at any step.
        """
        stress_history = np.asarray(stress_history, dtype=float)
        N = stress_history.shape[0]
        ntens = model.ntens

        stress_n = jnp.zeros(ntens)
        state_n = model.initial_state()
        eps_total = np.zeros(ntens)

        # Elastic compliance for the initial strain-increment guess
        C = model.elastic_stiffness(params)
        S = jnp.linalg.inv(C)

        stress_out = np.zeros((N, ntens))
        strain_out = np.zeros((N, ntens))

        for i in range(N):
            sigma_target = jnp.array(stress_history[i])

            # Initial guess: elastic compliance applied to stress increment
            deps = S @ (sigma_target - stress_n)

            converged = False
            residual = jnp.full(ntens, jnp.inf)
            for _ in range(self.max_iter):
                stress_new, state_new, ddsdde = return_mapping(
                    model, deps, stress_n, state_n, params
                )
                residual = sigma_target - stress_new
                if float(jnp.max(jnp.abs(residual))) < self.tol:
                    converged = True
                    break
                deps = deps + jnp.linalg.solve(ddsdde, residual)

            if not converged:
                raise RuntimeError(
                    f"StressDriver: NR did not converge at step {i} "
                    f"(||residual||_inf = {float(jnp.max(jnp.abs(residual))):.3e}, "
                    f"tol = {self.tol:.3e})"
                )

            stress_n = stress_new
            state_n = state_new
            eps_total = eps_total + np.array(deps)
            stress_out[i] = np.array(stress_new)
            strain_out[i] = eps_total.copy()

        return {"stress": stress_out, "strain": strain_out}


class BiaxialDriver:
    """Biaxial loading (placeholder).

    .. note::
        Not yet implemented.  For stress-controlled or mixed boundary
        conditions, use :class:`StressDriver` (pure stress control) or
        :class:`GeneralDriver` (full strain prescription) instead.
    """

    def run(self, model, strain_history, params):
        raise NotImplementedError(
            "BiaxialDriver is not yet implemented. "
            "Use StressDriver for stress-controlled loading or "
            "GeneralDriver for full strain prescription instead."
        )
