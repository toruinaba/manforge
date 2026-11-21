"""Only needed where returnmap is on PYTHONPATH rather than installed (as when
this example is run from a returnmap checkout): an installed returnmap
registers its pytest plugin by itself."""

pytest_plugins = ["returnmap.pytest_plugin"]
