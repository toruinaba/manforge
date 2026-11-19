"""manforge — validate hand-written Fortran UMAT constitutive models with autodiff.

Import from the seven public packages; each name has one home:

=========================  ====================================================
``manforge.core``          writing a model: :class:`~manforge.core.MaterialModel`,
                           state declarations, dimensions, results
``manforge.models``        reference models (J2, damage, AF, OW, YU)
``manforge.simulation``    integrators, drivers, load histories
``manforge.fitting``       parameter fitting
``manforge.verification``  checks against autodiff, finite differences, Fortran
``manforge.service``       plain-I/O contract layer for a web API
``manforge.utils``         smooth and spectral helpers for model code
=========================  ====================================================

The submodules below them (``manforge.core.state``, ``manforge.simulation.driver``,
...) are where the code lives, not part of the documented interface.
"""

from importlib.metadata import version as _get_version
try:
    __version__ = _get_version("manforge")
except Exception:
    __version__ = "0.2.0"
