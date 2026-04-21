"""conftest for tests/slow/.

All tests in this directory are automatically marked slow.
"""

import pytest


def pytest_collection_modifyitems(items):
    for item in items:
        if "tests/slow" in str(item.fspath):
            item.add_marker(pytest.mark.slow)
