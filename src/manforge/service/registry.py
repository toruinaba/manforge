"""Explicit name → model-class registry for service consumers.

No global singleton: each consumer (e.g. a web API deployment) builds its own
instance and registers exactly the models it intends to expose, so the public
surface is auditable at the registration site.  :func:`default_registry`
provides the reference models for local use, demos, and tests.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from manforge.core.material import MaterialModel, ParamMeta


@dataclass(frozen=True)
class ModelDescriptor:
    """Serializable description of a registered model (``GET /models`` payload).

    Parameters
    ----------
    name : str
        Registry key.
    class_name : str
        Python class name of the model.
    param_names : tuple[str, ...]
        Constructor parameter names, in declaration order.
    param_meta : dict[str, ParamMeta]
        Metadata for parameters that declare it (may be empty).
    ntens : int
        Number of stress/strain components of the registered class's
        default dimension.
    doc : str
        First line of the model class docstring.
    """

    name: str
    class_name: str
    param_names: tuple[str, ...]
    param_meta: dict[str, ParamMeta] = field(default_factory=dict)
    ntens: int = 1
    doc: str = ""


class ModelRegistry:
    """Explicit mapping of public model names to concrete model classes."""

    def __init__(self) -> None:
        self._models: dict[str, type[MaterialModel]] = {}

    def register(self, name: str, model_cls: type[MaterialModel]) -> None:
        """Register ``model_cls`` under ``name``.

        Raises
        ------
        ValueError
            If ``name`` is already registered or ``model_cls`` is not a
            concrete ``MaterialModel`` subclass.
        """
        if not (isinstance(model_cls, type) and issubclass(model_cls, MaterialModel)):
            raise ValueError(
                f"model_cls must be a MaterialModel subclass, got {model_cls!r}"
            )
        if name in self._models:
            raise ValueError(
                f"model name {name!r} is already registered "
                f"(to {self._models[name].__name__})"
            )
        self._models[name] = model_cls

    def get(self, name: str) -> type[MaterialModel]:
        """Return the class registered under ``name``.

        Raises
        ------
        KeyError
            Unknown name; the message lists registered names.
        """
        try:
            return self._models[name]
        except KeyError:
            raise KeyError(
                f"unknown model {name!r}; registered: {sorted(self._models) or '(none)'}"
            ) from None

    def names(self) -> list[str]:
        """Registered model names, sorted."""
        return sorted(self._models)

    def describe(self, name: str) -> ModelDescriptor:
        """Return the serializable descriptor for ``name``."""
        cls = self.get(name)
        doc = (cls.__doc__ or "").strip().splitlines()
        return ModelDescriptor(
            name=name,
            class_name=cls.__name__,
            param_names=tuple(cls.param_names),
            param_meta=dict(cls.param_meta),
            ntens=cls.dimension.ntens,
            doc=doc[0] if doc else "",
        )


def default_registry() -> ModelRegistry:
    """Registry pre-loaded with the uniaxial reference models."""
    from manforge.models import J2Isotropic1D, AFKinematic1D, OWKinematic1D

    reg = ModelRegistry()
    reg.register("j2_isotropic", J2Isotropic1D)
    reg.register("af_kinematic", AFKinematic1D)
    reg.register("ow_kinematic", OWKinematic1D)
    return reg
