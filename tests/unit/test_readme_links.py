"""Links from the README files to the published documentation name pages that exist.

The READMEs link to the site built from ``docs/`` (GitHub Pages).  A page
renamed or removed in ``docs/`` would leave such a link dead without any
build noticing — the site is not part of the repository.  Checked here
against the sources, so no network is needed: ``<site>/guide/x.html`` is
built from ``docs/guide/x.md``.
"""

import re
from pathlib import Path

import pytest

REPO = Path(__file__).parents[2]
SITE = "https://toruinaba.github.io/returnmap/"
READMES = [
    REPO / "README.md",
    REPO / "examples" / "model_family" / "README.md",
]


def _site_links(path: Path) -> list[str]:
    return re.findall(re.escape(SITE) + r"([\w/.-]*)", path.read_text(encoding="utf-8"))


@pytest.mark.parametrize("readme", READMES, ids=lambda p: str(p.relative_to(REPO)))
def test_site_links_name_existing_pages(readme):
    links = _site_links(readme)
    assert links, f"{readme.name} has no link to the published documentation"
    for link in links:
        if link == "":
            continue                                   # the front page
        assert link.endswith(".html"), f"{link} is not a page of the site"
        source = REPO / "docs" / (link[: -len(".html")] + ".md")
        assert source.is_file(), (
            f"{readme.relative_to(REPO)} links to {SITE}{link}, "
            f"but docs/{source.relative_to(REPO / 'docs')} does not exist")


def test_the_main_readme_links_to_every_guide_page_group():
    """The page tables are the way in: at least the pages the index lists."""
    links = set(_site_links(REPO / "README.md"))
    for page in ("guide/tutorial.html", "guide/models.html", "guide/adding-a-model.html",
                 "concepts.html", "changelog.html"):
        assert page in links, f"README.md does not link to {page}"


def test_the_main_readme_has_no_link_relative_to_the_repository():
    """README.md is also the project page on PyPI, where a relative link is dead."""
    targets = re.findall(r"\]\(([^)\s]+)\)", (REPO / "README.md").read_text(encoding="utf-8"))
    assert targets, "README.md has no link at all"
    relative = [t for t in targets if not t.startswith(("https://", "#"))]
    assert not relative, f"README.md links relative to the repository: {relative}"
