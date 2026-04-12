"""Fortran UMAT bridge for cross-validation.

Provides :class:`FortranUMAT` for calling compiled Fortran UMAT subroutines
with the same Python-level interface as
:func:`~manforge.core.return_mapping.return_mapping`.

Calling convention for Fortran ``_run`` subroutines
-----------------------------------------------------
Each compiled ``*_run`` subroutine must follow::

    fn(*params, stress_in, *states, dstran)
        -> (stress_out, *state_scalars, ddsdde)

where parameters and state variables are ordered according to
``model.param_names`` and ``model.state_names`` respectively.
``umat_j2_run(E, nu, sigma_y0, H, stress_in, ep_in, dstran)`` conforms to
this convention.

Comparison workflow
-------------------
For the typical use case (verifying a Fortran UMAT against a Python model),
use :class:`~manforge.verification.umat_verifier.UMATVerifier`::

    verifier = UMATVerifier(model, "manforge_umat")
    result   = verifier.run(params)

For lower-level access, use :func:`~manforge.verification.compare.compare_solvers`
directly, passing :meth:`FortranUMAT.call` as one of the solvers.
"""

import importlib

import numpy as np


class FortranUMAT:
    """Bridge to a compiled Fortran UMAT subroutine via f2py.

    Wraps an f2py-compiled module so that the Fortran UMAT can be called
    with the same Python-level interface as
    :func:`~manforge.core.return_mapping.return_mapping`.

    Parameter and state variable ordering is derived automatically from
    ``model.param_names`` and ``model.state_names``, so the bridge works
    with any model whose Fortran ``_run`` subroutine follows the calling
    convention described in the module docstring.

    Parameters
    ----------
    module_name : str
        Name of the f2py-compiled Python module (must be importable, e.g.
        after ``make fortran-build-umat``).
    model : MaterialModel
        Model instance that provides ``param_names`` and ``state_names``
        for dict-to-positional-argument marshalling.
    subroutine : str, optional
        Name of the Fortran subroutine to call.  If *None* (default), the
        module is scanned for subroutines whose names end with ``"_run"``.
        Exactly one such subroutine must exist; otherwise pass this argument
        explicitly.
    """

    def __init__(self, module_name: str, model, subroutine: str | None = None):
        self._mod = importlib.import_module(module_name)
        self._model = model
        if subroutine is None:
            candidates = [n for n in dir(self._mod) if n.endswith("_run")]
            if len(candidates) == 1:
                subroutine = candidates[0]
            else:
                raise ValueError(
                    f"Module '{module_name}' has {len(candidates)} *_run subroutines: "
                    f"{candidates}. Specify one via the 'subroutine' argument."
                )
        self._fn = getattr(self._mod, subroutine)

    def call(
        self,
        strain_inc,
        stress_n,
        state_n: dict,
        params: dict,
        *,
        dtime: float = 1.0,
    ):
        """Call the Fortran UMAT with Python-compatible arguments.

        Parameters
        ----------
        strain_inc : array-like, shape (ntens,)
            Strain increment Δε.
        stress_n : array-like, shape (ntens,)
            Stress at the beginning of the increment.
        state_n : dict
            Internal state variables, keyed by ``model.state_names``.
        params : dict
            Material parameters, keyed by ``model.param_names``.
        dtime : float
            Unused; kept for interface symmetry with return_mapping.

        Returns
        -------
        stress_new : np.ndarray, shape (ntens,)
        state_new : dict
        ddsdde : np.ndarray, shape (ntens, ntens)
        """
        dstran   = np.asarray(strain_inc, dtype=np.float64)
        stress_i = np.asarray(stress_n,   dtype=np.float64)

        param_args = [float(params[name]) for name in self._model.param_names]
        state_args = [float(state_n[name]) for name in self._model.state_names]

        result = self._fn(*param_args, stress_i, *state_args, dstran)

        stress_out = np.array(result[0])
        n_states = len(self._model.state_names)
        state_new = {
            name: float(result[1 + i])
            for i, name in enumerate(self._model.state_names)
        }
        ddsdde = np.array(result[1 + n_states])

        return stress_out, state_new, ddsdde

