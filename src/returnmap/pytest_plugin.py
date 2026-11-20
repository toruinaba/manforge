"""pytest plugin: what the tests of a model need, wherever they are run from.

Loaded automatically wherever returnmap is installed (it is registered as a
``pytest11`` entry point), so a repository of models built on returnmap — and
the model tests shipped inside this package, run with
``pytest --pyargs returnmap.models`` — get the same setup without a
``conftest.py`` of their own:

* the markers ``fortran`` and ``slow`` are registered;
* the directory compiled f2py modules are built into, ``fortran/`` under the
  pytest root (where ``returnmap build`` puts them), is on ``sys.path``, so
  ``pytest.importorskip("my_umat")`` and ``FortranModule("my_umat")`` find a
  module that has been built and skip or fail cleanly when it has not.

Nothing else is touched.  In a checkout where the package is on ``PYTHONPATH``
rather than installed, name it in the root ``conftest.py``::

    pytest_plugins = ["returnmap.pytest_plugin"]
"""

from __future__ import annotations

import sys

MARKERS = (
    "slow: measured >1s (fitting, long cyclic loops) -- deselect with -m 'not slow'",
    "fortran: requires compiled Fortran modules (f2py). Skipped if the module "
    "is missing -- deselect with -m 'not fortran'",
)


def pytest_configure(config) -> None:
    for marker in MARKERS:
        config.addinivalue_line("markers", marker)
    build_dir = str(config.rootpath / "fortran")
    if build_dir not in sys.path:
        # Harmless when the directory does not exist or holds no module.
        sys.path.insert(0, build_dir)
