"""conftest for tests/fortran/.

Each test file still declares its own pytestmark = pytest.mark.fortran
(for grep-ability) and pytest.importorskip (for auto-skip when binaries
are absent).
"""
