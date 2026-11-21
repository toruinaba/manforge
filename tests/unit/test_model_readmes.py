"""Each model family documents its formulation next to the model.

``src/returnmap/models/<family>/README.md`` is the formulation page of the
family (included into the guide by ``docs/guide/models/<family>.md``).  It is
prose and formulas, which no test can check — but the parts that restate the
code can be held to it: every parameter and state variable is listed, and the
typical values in the parameter table are the ones in ``param_meta``.
"""

import importlib
import pkgutil
import re
from pathlib import Path

import pytest

import returnmap.models as models

DOCS = Path(__file__).parents[2] / "docs" / "guide" / "models"
FAMILIES = sorted(
    info.name for info in pkgutil.iter_modules(models.__path__) if info.ispkg
)


def _readme(family: str) -> Path:
    return Path(models.__path__[0]) / family / "README.md"


def _classes(family: str):
    """The public model classes the family exports."""
    package = importlib.import_module(f"returnmap.models.{family}")
    return [getattr(package, name) for name in package.__all__]


def _table_rows(text: str) -> dict[str, list[str]]:
    """Rows of the Markdown tables whose first cell is a name in backticks."""
    rows: dict[str, list[str]] = {}
    for line in text.splitlines():
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        match = re.fullmatch(r"`(\w+)`", cells[0]) if cells else None
        if match:
            rows.setdefault(match.group(1), cells)
    return rows


def test_there_is_a_family_to_check():
    assert FAMILIES


@pytest.mark.parametrize("family", FAMILIES)
def test_the_family_has_a_formulation_page(family):
    assert _readme(family).is_file(), f"{family} has no README.md"
    page = DOCS / f"{family}.md"
    assert page.is_file(), f"docs/guide/models/{family}.md does not exist"
    assert f"models/{family}/README.md" in page.read_text(encoding="utf-8")


@pytest.mark.parametrize("family", FAMILIES)
def test_every_class_parameter_and_state_is_listed(family):
    text = _readme(family).read_text(encoding="utf-8")
    rows = _table_rows(text)
    for cls in _classes(family):
        # A class is named in full, or by the pattern of its dimension suffix
        # (`J2IsotropicDamage*`).
        stem = re.sub(r"(3D|PS|1D)$", "", cls.__name__)
        assert f"`{cls.__name__}`" in text or f"`{stem}*`" in text, (
            f"{family}/README.md does not mention {cls.__name__}")
        for name in cls.param_names:
            assert name in rows, f"{family}/README.md: parameter {name} has no table row"
        for name in cls.state_names:
            assert name in rows, f"{family}/README.md: state {name} has no table row"


@pytest.mark.parametrize("family", FAMILIES)
def test_typical_values_are_those_of_param_meta(family):
    rows = _table_rows(_readme(family).read_text(encoding="utf-8"))
    for cls in _classes(family):
        for name, meta in cls.param_meta.items():
            # name | meaning | unit | typical value [| ...]
            cells = rows[name]
            assert float(cells[3]) == meta.default, (
                f"{family}/README.md: {name} is listed with {cells[3]}, "
                f"param_meta says {meta.default}")
            unit = meta.unit or "—"
            assert cells[2] == unit, (
                f"{family}/README.md: {name} is listed in {cells[2]}, "
                f"param_meta says {unit}")
