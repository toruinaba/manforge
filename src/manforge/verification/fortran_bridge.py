"""Fortran UMAT bridge for cross-validation.

Provides :class:`FortranUMAT`, a thin wrapper around an f2py-compiled Fortran
module.  Its sole responsibility is float64 type conversion — it has no
knowledge of the Python-side ``MaterialModel`` and no auto-detection logic.

Usage
-----
::

    from manforge.verification import FortranUMAT
    import numpy as np

    fortran = FortranUMAT("j2_isotropic_3d")

    # Call any subroutine by name — same pattern for _run and sub-components
    stress_f, ep_f, ddsdde_f = fortran.call(
        "j2_isotropic_3d",
        210_000.0, 0.3, 250.0, 1_000.0,          # E, nu, sigma_y0, H
        np.zeros(6), 0.0,                          # stress_in, ep_in
        np.array([2e-3, 0, 0, 0, 0, 0]),           # dstran
    )

    # Component-level check — same interface
    C_f = fortran.call("j2_isotropic_3d_elastic_stiffness", 210_000.0, 0.3)

Compare results explicitly against the Python reference::

    from manforge.simulation.integrator import PythonNumericalIntegrator
    import numpy as np

    integrator = PythonNumericalIntegrator(model)
    result = integrator.stress_update(np.array(dstran), np.zeros(6), model.initial_state())
    stress_py, ddsdde_py = result.stress, result.ddsdde
    np.testing.assert_allclose(np.array(stress_py), stress_f, rtol=1e-6)
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
    """Thin wrapper around an f2py-compiled Fortran module.

    Handles float64 type conversion only.  Has no knowledge of the Python-side
    ``MaterialModel`` and performs no subroutine auto-detection.

    Parameters
    ----------
    module_name : str
        Name of the f2py-compiled Python module — **not a file path**.
        This is the importable module name produced by
        ``uv run manforge build <source>.f90 --name <module_name>``.
        Run ``uv run manforge list`` to see which modules are available.

    Raises
    ------
    ModuleNotFoundError
        If *module_name* cannot be imported.  Build the module first with
        ``uv run manforge build fortran/<source>.f90 --name <module_name>``,
        or run ``uv run manforge list`` to see already-compiled modules.
    """

    def __init__(self, module_name: str):
        try:
            self._mod = importlib.import_module(module_name)
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                f"f2py module {module_name!r} is not importable.\n"
                f"Build it with:  uv run manforge build fortran/<source>.f90 --name {module_name}\n"
                f"Or run:         uv run manforge list   (to see compiled modules)"
            ) from exc

    @property
    def module(self):
        """The raw f2py module object."""
        return self._mod

    def call(self, name: str, *args):
        """Call a subroutine in the module by name.

        Scalars are converted to ``float``; arrays to ``np.float64``.
        Returns the raw f2py output (numpy arrays or tuple thereof).

        Parameters
        ----------
        name : str
            Exact subroutine name as it appears in the f2py module.
        *args
            Positional arguments forwarded to the subroutine.

        Example
        -------
        ::

            stress_f, ep_f, ddsdde_f = fortran.call(
                "j2_isotropic_3d",
                E, nu, sigma_y0, H, stress_in, ep_in, dstran,
            )
            C_f = fortran.call("j2_isotropic_3d_elastic_stiffness", E, nu)
        """
        fn = getattr(self._mod, name)
        return fn(*_ensure_float64(args))
