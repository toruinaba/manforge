"""Repository-wide pytest setup.

The tests live in two places — ``tests/`` for the framework, and
``src/returnmap/models/<family>/tests/`` for each model family — so what both
need is set up here, at the root.
"""

import os
import sys

# Compiled f2py modules are built into fortran/; make them importable for
# every test.  Harmless if the directory holds none.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "fortran"))
