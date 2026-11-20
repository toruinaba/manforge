"""returnmap — validate hand-written Fortran UMAT constitutive models with autodiff.

Import from the seven public packages; each name has one home:

==========================  ====================================================
``returnmap.core``          writing a model: :class:`~returnmap.core.MaterialModel`,
                            state declarations, dimensions, results
``returnmap.models``        reference models (J2, damage, AF, OW, YU)
``returnmap.simulation``    integrators, drivers, load histories
``returnmap.fitting``       parameter fitting
``returnmap.verification``  checks against autodiff, finite differences, Fortran
``returnmap.service``       plain-I/O contract layer for a web API
``returnmap.utils``         smooth and spectral helpers for model code
==========================  ====================================================

The submodules below them (``returnmap.core.state``, ``returnmap.simulation.driver``,
...) are where the code lives, not part of the documented interface.
"""

from importlib.metadata import version as _get_version
try:
    __version__ = _get_version("returnmap")
except Exception:
    __version__ = "0.3.1"
