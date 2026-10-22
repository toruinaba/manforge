"""Parameter metadata for UI/schema generation.

:class:`ParamMeta` carries human-facing information about a material
parameter (description, unit, default, plausible range).  It lives on the
model class itself (``param_meta`` class variable) so metadata cannot drift
apart from the model in a separate repository.

``bounds`` is a form hint, not a runtime constraint — the framework never
rejects out-of-range values (deliberately exploring outside the plausible
range is a legitimate research workflow).
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ParamMeta:
    """Metadata for one material parameter.

    Parameters
    ----------
    doc : str
        Human-readable description (form label / tooltip).
    unit : str
        Physical unit, e.g. ``"MPa"``.  Empty string for dimensionless.
    default : float or None
        Suggested initial value for input forms.  ``None`` = required input.
    bounds : tuple[float, float] or None
        Plausible range hint ``(lower, upper)``.  Not enforced at runtime.
    """

    doc: str
    unit: str = ""
    default: float | None = None
    bounds: tuple[float, float] | None = None
