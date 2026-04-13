"""Fortran UMAT bridge for cross-validation.

Provides :class:`FortranUMAT`, a thin Pythonic wrapper around an f2py-compiled
Fortran module.  Its sole responsibility is type conversion and subroutine
discovery — it has no knowledge of the Python-side ``MaterialModel``.

Primary interface
-----------------
Call any Fortran subroutine with automatic float64 conversion::

    from manforge.verification import FortranUMAT
    import numpy as np

    fortran = FortranUMAT("manforge_umat")

    # Call the auto-detected *_run subroutine
    stress_f, ep_f, ddsdde_f = fortran.run(
        210_000.0, 0.3, 250.0, 1_000.0,          # E, nu, sigma_y0, H
        np.zeros(6), 0.0,                          # stress_in, ep_in
        np.array([2e-3, 0, 0, 0, 0, 0]),           # dstran
    )

    # Call any other subroutine by name (same pattern)
    C_f = fortran.call("umat_j2_elastic_stiffness", 210_000.0, 0.3)

Compare results explicitly against the Python reference::

    from manforge.core.return_mapping import return_mapping
    import jax.numpy as jnp

    stress_py, state_py, ddsdde_py = return_mapping(
        model, jnp.array(dstran), jnp.zeros(6), model.initial_state(), params
    )
    np.testing.assert_allclose(np.array(stress_py), stress_f, rtol=1e-6)

Calling convention for ``*_run`` subroutines
---------------------------------------------
Each compiled ``*_run`` subroutine must follow::

    fn(*params, stress_in, *states, dstran)
        -> (stress_out, *state_scalars, ddsdde)

where parameter and state ordering matches the model's ``param_names`` and
``state_names``.  ``umat_j2_run(E, nu, sigma_y0, H, stress_in, ep_in, dstran)``
conforms to this convention.

For automated multi-case comparison, use
:class:`~manforge.verification.umat_verifier.UMATVerifier` as a convenience
utility.
"""

import importlib

import numpy as np


def _ensure_float64(args):
    """Convert args to float64: scalars to float, arrays to np.float64."""
    out = []
    for a in args:
        if hasattr(a, "__len__") or isinstance(a, np.ndarray):
            out.append(np.asarray(a, dtype=np.float64))
        else:
            out.append(float(a))
    return out


class FortranUMAT:
    """Thin Pythonic wrapper around an f2py-compiled Fortran UMAT module.

    Handles float64 type conversion and auto-detects the ``*_run`` subroutine.
    Has no knowledge of the Python-side ``MaterialModel``.

    Parameters
    ----------
    module_name : str
        Name of the f2py-compiled Python module (must be importable, e.g.
        after ``make fortran-build-umat``).
    subroutine : str, optional
        Name of the Fortran ``*_run`` subroutine used by :meth:`run`.
        If *None* (default), the module is scanned for subroutines whose
        names end with ``"_run"``.  Exactly one such subroutine must exist;
        otherwise pass this argument explicitly.
    """

    def __init__(self, module_name: str, subroutine: str | None = None):
        self._mod = importlib.import_module(module_name)
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

    @property
    def module(self):
        """The raw f2py module.

        Provides direct access to every subroutine in the compiled module::

            C = fortran.module.umat_j2_elastic_stiffness(210_000.0, 0.3)
        """
        return self._mod

    def run(self, *args):
        """Call the auto-detected ``*_run`` subroutine.

        Arguments are passed in Fortran order (same as the f2py function).
        Scalars are converted to ``float``, arrays to ``np.float64``.
        Returns the raw f2py output tuple (numpy arrays).

        Example
        -------
        ::

            stress_f, ep_f, ddsdde_f = fortran.run(
                E, nu, sigma_y0, H,       # material params
                stress_in, ep_in,         # state
                dstran,                   # strain increment
            )
        """
        return self._fn(*_ensure_float64(args))

    def call(self, name: str, *args):
        """Call any subroutine in the module by name.

        Applies the same float64 conversion as :meth:`run`.

        Parameters
        ----------
        name : str
            Exact subroutine name as it appears in the f2py module.
        *args
            Positional arguments forwarded to the subroutine.

        Example
        -------
        ::

            C = fortran.call("umat_j2_elastic_stiffness", E, nu)
        """
        fn = getattr(self._mod, name)
        return fn(*_ensure_float64(args))
