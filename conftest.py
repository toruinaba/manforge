"""Repository-wide pytest setup.

The tests live in two places — ``tests/`` for the framework, and
``src/returnmap/models/<family>/tests/`` for each model family.  What both
need (the ``fortran`` / ``slow`` markers, compiled f2py modules on the path)
comes from the plugin the package ships.  An installed returnmap registers it
by itself; naming it here covers a checkout that is on ``PYTHONPATH`` instead
(the Docker image).
"""

pytest_plugins = ["returnmap.pytest_plugin"]
