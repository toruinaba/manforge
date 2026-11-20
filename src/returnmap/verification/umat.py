"""Declaring the Fortran UMAT that belongs to a model.

A model that has been ported to a UMAT says so in its family's
``verification.py``, next to the models and loading histories it is verified
on::

    from returnmap.verification import Umat

    UMATS = [
        Umat(YUKinematic3D, module="yu_kinematic_3d",
             sources=["fortran/yu_kinematic_3d.f90"], libraries=["lapack", "blas"]),
    ]

That one declaration is what the tools read:

* ``returnmap build --family <package>`` compiles every UMAT a package of
  models declares — sources, the ABAQUS stubs, the libraries to link;
* :func:`~returnmap.verification.check_model` runs the UMAT through the same
  loading histories as the Python model and compares them step by step, when
  the module has been built.
"""

from __future__ import annotations

import importlib
import importlib.util
import inspect
import pkgutil
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from returnmap.core.material import MaterialModel
    from returnmap.simulation.integrator import FortranIntegrator


@dataclass(frozen=True)
class Umat:
    """The Fortran UMAT of one model class.

    Parameters
    ----------
    model : type[MaterialModel]
        The model class the UMAT is the port of — one dimension of a family
        (``YUKinematic3D``), since a UMAT is written for one.
    module : str
        Name of the f2py module the sources are compiled into.
    sources : sequence of str
        Fortran source files, relative to the directory of the model's
        module (``"fortran/yu_kinematic_3d.f90"`` for a model defined in
        ``.../yu_kinematic/model.py``), so the declaration moves with the
        family directory.
    libraries : sequence of str
        Libraries to link (``["lapack", "blas"]``).
    subroutine : str or None
        The subroutine that does one stress update.  ``None`` (default): the
        one named like the module.
    props : sequence of str or None
        The model's parameter names in the order the UMAT takes them — its
        ``PROPS``.  ``None`` (default): ``param_names``, in that order.  Give
        it when the UMAT orders them differently or takes only some of them
        (a parameter that selects a branch on the Python side, say).
    integrator_options : mapping
        Extra keyword arguments for
        :meth:`FortranIntegrator.from_model
        <returnmap.simulation.FortranIntegrator.from_model>` — typically
        ``parse_umat_failure``, to read the UMAT's own convergence flag.

    Notes
    -----
    The argument convention is that of ``from_model``: the parameters
    (``props``), the stress, the state variables in declaration order, the
    strain increment.
    """

    model: "type[MaterialModel]"
    module: str
    sources: Sequence[str]
    libraries: Sequence[str] = ()
    subroutine: str | None = None
    props: Sequence[str] | None = None
    integrator_options: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.props is not None:
            unknown = [name for name in self.props if name not in self.model.param_names]
            if unknown:
                raise ValueError(
                    f"Umat({self.model.__name__}, module={self.module!r}): props names "
                    f"{unknown}, which are not parameters of the model "
                    f"{list(self.model.param_names)}"
                )

    def source_paths(self) -> list[Path]:
        """The sources as absolute paths."""
        source_file = inspect.getsourcefile(self.model)
        if source_file is None:
            raise ValueError(f"cannot locate the module {self.model.__name__} is defined in")
        base = Path(source_file).parent
        return [base / source for source in self.sources]

    def is_built(self) -> bool:
        """Whether the compiled module can be imported."""
        _build_dir_on_path()
        try:
            return importlib.util.find_spec(self.module) is not None
        except (ImportError, ValueError):
            return False

    def integrator(self, model: "MaterialModel", **options) -> "FortranIntegrator":
        """The UMAT as an integrator, with ``model``'s parameters.

        Raises
        ------
        ModuleNotFoundError
            The module has not been built.
        """
        from returnmap.simulation.integrator import FortranIntegrator, FortranModule

        _build_dir_on_path()
        declared: dict[str, Any] = dict(self.integrator_options)
        if self.props is not None:
            names = tuple(self.props)
            declared["param_fn"] = lambda: tuple(getattr(model, name) for name in names)
        return FortranIntegrator.from_model(
            FortranModule(self.module), self.subroutine or self.module, model,
            **{**declared, **options},
        )


def _build_dir_on_path() -> None:
    """Make modules built by ``returnmap build`` importable (<project>/fortran)."""
    from returnmap.cli import build_dir

    path = str(build_dir())
    if path not in sys.path:
        sys.path.insert(0, path)


def declared_umats(package: str) -> list[Umat]:
    """The UMATs a package of models declares, its subpackages included.

    ``package`` is a dotted name: a family (``"returnmap.models.yu_kinematic"``),
    or a package of families (``"returnmap.models"``).  A UMAT is declared in
    the ``UMATS`` list of a module named ``verification``.
    """
    root = importlib.import_module(package)
    names = [package]
    if hasattr(root, "__path__"):
        names += [
            info.name
            for info in pkgutil.walk_packages(root.__path__, prefix=f"{package}.")
            if info.ispkg and not info.name.endswith(".tests")
        ]
    found: list[Umat] = []
    for name in names:
        if importlib.util.find_spec(f"{name}.verification") is None:
            continue
        module = importlib.import_module(f"{name}.verification")
        for umat in getattr(module, "UMATS", ()):
            if not isinstance(umat, Umat):
                raise TypeError(
                    f"{name}.verification.UMATS holds {umat!r}; "
                    "declare each UMAT with returnmap.verification.Umat"
                )
            found.append(umat)
    return found


def umat_of(model: "MaterialModel") -> Umat | None:
    """The UMAT declared for exactly this model's class, or ``None``."""
    package = type(model).__module__.rpartition(".")[0]
    if not package or importlib.util.find_spec(f"{package}.verification") is None:
        return None
    module = importlib.import_module(f"{package}.verification")
    for umat in getattr(module, "UMATS", ()):
        if umat.model is type(model):
            return umat
    return None
