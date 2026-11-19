"""Sphinx configuration.

Guide pages are Markdown (MyST) and written in Japanese; the API reference is
generated from the English docstrings.  Build with ``make docs``.
"""

from importlib.metadata import version as _version

project = "manforge"
author = "Toru Inaba"
copyright = "2026, Toru Inaba"
release = _version("manforge")
version = release

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "myst_parser",
]

language = "ja"
source_suffix = {".md": "markdown"}
exclude_patterns = ["_build"]

html_theme = "furo"
html_title = f"manforge {release}"

# -- MyST ---------------------------------------------------------------------
myst_enable_extensions = ["colon_fence", "dollarmath"]
myst_heading_anchors = 3

# -- API reference --------------------------------------------------------------
# The reference documents the seven public packages.  Each lists its public
# names in __all__, and automodule follows that, so a name appears exactly
# once, under the path it is meant to be imported from.
autodoc_member_order = "bysource"
autodoc_typehints = "none"
autodoc_default_options = {
    "members": True,
    "imported-members": True,
    "show-inheritance": True,
}
# A hook a model overrides without a docstring of its own is not repeated on
# the model's entry; the contract is documented once, on MaterialModel.
autodoc_inherit_docstrings = False
napoleon_google_docstring = False
napoleon_numpy_docstring = True
napoleon_use_param = False
napoleon_use_rtype = False
napoleon_use_ivar = True

# A cross-reference that does not resolve is an error: `make docs` runs with
# -W, so a renamed or removed name cannot leave a dead link behind.
nitpicky = True
# NumPy-style sections describe types in prose ("array, shape (n,)",
# "dict[row][col]"), which Sphinx reads as class references.  Only two kinds
# of class target are held to the rule: a full manforge path and a bare
# CapitalisedName (a class of this package written without its path).
nitpick_ignore_regex = [
    ("py:class", r"^(?!manforge\.)(?![A-Z]\w*$).+$"),
]


# -- Completeness -----------------------------------------------------------------
PUBLIC_PACKAGES = ("core", "models", "simulation", "fitting", "verification", "service", "utils")


def _check_reference_is_complete(app, env):
    """Fail the build if a public name has no entry in the reference.

    ``__all__`` is the list of public names; a name added there without a
    docstring (or a constant without an entry in docs/reference/) would
    otherwise be exported and silently undocumented.
    """
    import importlib

    from sphinx.errors import ExtensionError

    documented = set(env.domaindata["py"]["objects"])
    missing = [
        f"manforge.{package}.{name}"
        for package in PUBLIC_PACKAGES
        for name in importlib.import_module(f"manforge.{package}").__all__
        if f"manforge.{package}.{name}" not in documented
    ]
    if missing:
        raise ExtensionError(
            "public names missing from the API reference: " + ", ".join(sorted(missing))
        )


def setup(app):
    app.connect("env-check-consistency", _check_reference_is_complete)
