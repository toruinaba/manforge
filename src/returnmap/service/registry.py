"""Explicit name → model registry for service consumers.

No global singleton: each consumer (e.g. a web API deployment) builds its own
instance and registers exactly the models it intends to expose, so the public
surface is auditable at the registration site.  :func:`default_registry`
provides the reference models for local use, demos, and tests.

What is registered is a model *instance* — a template, in the sense
:func:`~returnmap.fitting.fit_params` uses the word.  A class alone cannot say
which dimension a dimension-free family runs in, nor carry a setting that is
not a parameter (the ``c`` of
:class:`~returnmap.models.J2IsotropicDamageUnilateral`); an instance fixes
both, and :func:`~returnmap.service.run_simulation` puts each request's
parameters into it with
:meth:`~returnmap.core.MaterialModel.with_params`.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from returnmap.core.material import MaterialModel, ParamMeta


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
        Parameter names, in declaration order.
    param_meta : dict[str, ParamMeta]
        Metadata for parameters that declare it (may be empty).
    ntens : int
        Number of stress/strain components of the registered model — the
        width of the ``load`` rows it takes.
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
    """Explicit mapping of public model names to template models."""

    def __init__(self) -> None:
        self._models: dict[str, MaterialModel] = {}

    def register(self, name: str, model: MaterialModel) -> None:
        """Register the template ``model`` under ``name``.

        The template supplies the class, the dimension and every setting
        outside ``param_names``.  Its parameter values are placeholders:
        :func:`~returnmap.service.run_simulation` replaces all of them on
        every run.  One class can be registered more than once, under
        different names, at different dimensions or settings.

        Raises
        ------
        ValueError
            If ``name`` is already registered or ``model`` is not a
            ``MaterialModel`` instance (a model class is not one — construct
            it).
        """
        if not isinstance(model, MaterialModel):
            raise ValueError(
                "model must be a MaterialModel instance (a template to put "
                f"parameters into), got {model!r}"
                + (
                    "; construct the class, e.g. with the defaults of its param_meta"
                    if isinstance(model, type) and issubclass(model, MaterialModel)
                    else ""
                )
            )
        if name in self._models:
            raise ValueError(
                f"model name {name!r} is already registered "
                f"(to {type(self._models[name]).__name__})"
            )
        self._models[name] = model

    def get(self, name: str) -> MaterialModel:
        """Return the template registered under ``name``.

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
        model = self.get(name)
        cls = type(model)
        doc = (cls.__doc__ or "").strip().splitlines()
        return ModelDescriptor(
            name=name,
            class_name=cls.__name__,
            param_names=tuple(model.param_names),
            param_meta=dict(model.param_meta),
            ntens=model.ntens,
            doc=doc[0] if doc else "",
        )


def default_registry() -> ModelRegistry:
    """Registry pre-loaded with the uniaxial reference models.

    Each template holds the ``default`` values of its class's ``param_meta``.
    """
    from returnmap.models import J2Isotropic1D, AFKinematic1D, OWKinematic1D

    reg = ModelRegistry()
    reg.register("j2_isotropic", J2Isotropic1D.at_defaults())
    reg.register("af_kinematic", AFKinematic1D.at_defaults())
    reg.register("ow_kinematic", OWKinematic1D.at_defaults())
    return reg
