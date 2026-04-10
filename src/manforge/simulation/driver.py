"""Strain-driven loading drivers.

A driver loops over a prescribed strain history, calls ``return_mapping`` at
each step, and accumulates the resulting stress (and state) history.

Conventions
-----------
- All strain arrays use the engineering-shear Voigt convention:
    ε = [ε11, ε22, ε33, γ12, γ13, γ23]
- *Cumulative* total strain is the input; increments are computed internally.
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
    """General 6-component strain-driven loading.

    Parameters
    ----------
    (none — stateless)
    """

    def run(self, model, strain_history_6, params):
        """Run a general strain history.

        Parameters
        ----------
        model : MaterialModel
        strain_history_6 : array-like, shape (N, 6)
            Cumulative strain tensor (Voigt, engineering shear) at each step.
        params : dict

        Returns
        -------
        stress_history : np.ndarray, shape (N, 6)
            Full stress tensor at each step.
        """
        strain_history_6 = np.asarray(strain_history_6, dtype=float)
        N = strain_history_6.shape[0]
        ntens = model.ntens

        stress_n = jnp.zeros(ntens)
        state_n = model.initial_state()
        eps_prev = np.zeros(ntens)

        stress_out = np.zeros((N, ntens))

        for i in range(N):
            strain_inc = jnp.array(strain_history_6[i] - eps_prev)

            stress_n, state_n, _ = return_mapping(
                model, strain_inc, stress_n, state_n, params
            )
            stress_out[i] = np.array(stress_n)
            eps_prev = strain_history_6[i]

        return stress_out


class BiaxialDriver:
    """Biaxial strain-driven loading (placeholder).

    .. note::
        Not yet implemented.  Biaxial loading requires specifying which
        components are strain-driven and which are stress-controlled
        (mixed boundary conditions).  Use :class:`GeneralDriver` for full
        6-component prescriptions in the meantime.
    """

    def run(self, model, strain_history, params):
        raise NotImplementedError(
            "BiaxialDriver is not yet implemented. "
            "Use GeneralDriver with a (N, 6) history instead."
        )
