"""Tests for ModelRegistry / ModelDescriptor / default_registry."""

import pytest

from manforge.core.material import ParamMeta
from manforge.models import J2Isotropic1D, J2Isotropic3D
from manforge.service import ModelRegistry, ModelDescriptor, default_registry


class TestRegister:
    def test_register_and_get(self):
        reg = ModelRegistry()
        reg.register("j2", J2Isotropic1D)
        assert reg.get("j2") is J2Isotropic1D

    def test_duplicate_name_raises(self):
        reg = ModelRegistry()
        reg.register("j2", J2Isotropic1D)
        with pytest.raises(ValueError, match="already registered"):
            reg.register("j2", J2Isotropic3D)

    def test_non_model_class_raises(self):
        reg = ModelRegistry()
        with pytest.raises(ValueError, match="MaterialModel subclass"):
            reg.register("x", dict)

    def test_instance_rejected(self):
        reg = ModelRegistry()
        model = J2Isotropic1D(E=210e3, nu=0.3, sigma_y0=250.0, H=1e3)
        with pytest.raises(ValueError, match="MaterialModel subclass"):
            reg.register("x", model)


class TestGet:
    def test_unknown_name_lists_registered(self):
        reg = ModelRegistry()
        reg.register("j2", J2Isotropic1D)
        with pytest.raises(KeyError, match=r"unknown model 'nope'.*j2"):
            reg.get("nope")

    def test_names_sorted(self):
        reg = ModelRegistry()
        reg.register("b", J2Isotropic1D)
        reg.register("a", J2Isotropic3D)
        assert reg.names() == ["a", "b"]


class TestDescribe:
    def test_descriptor_contents(self):
        reg = ModelRegistry()
        reg.register("j2", J2Isotropic1D)
        d = reg.describe("j2")
        assert isinstance(d, ModelDescriptor)
        assert d.name == "j2"
        assert d.class_name == "J2Isotropic1D"
        assert d.param_names == ("E", "nu", "sigma_y0", "H")
        assert d.ntens == 1
        assert d.doc  # first docstring line, non-empty

    def test_descriptor_param_meta(self):
        reg = ModelRegistry()
        reg.register("j2", J2Isotropic1D)
        d = reg.describe("j2")
        assert set(d.param_meta) == set(d.param_names)
        assert isinstance(d.param_meta["E"], ParamMeta)
        assert d.param_meta["E"].unit == "MPa"

    def test_ntens_reflects_registered_class(self):
        reg = ModelRegistry()
        reg.register("j2_3d", J2Isotropic3D)
        assert reg.describe("j2_3d").ntens == 6

    def test_model_without_meta_describes_empty(self):
        from manforge.core.material import MaterialModel
        from manforge.core.state import Explicit, SCALAR
        from manforge.core.dimension import UNIAXIAL_1D

        class Bare(MaterialModel):
            param_names = ["E", "nu", "sigma_y0"]
            ep = Explicit(shape=SCALAR, doc="ep")

            def __init__(self, *, E, nu, sigma_y0):
                super().__init__(dimension=UNIAXIAL_1D)
                self.E, self.nu, self.sigma_y0 = E, nu, sigma_y0

            def yield_function(self, state):
                return abs(state["stress"][0]) - self.sigma_y0

            def update_state(self, dlambda, state_new, state_n, *,
                             stress_trial=None, strain_inc=None):
                return [self.ep(state_n["ep"] + dlambda)]

        reg = ModelRegistry()
        reg.register("bare", Bare)
        assert reg.describe("bare").param_meta == {}


class TestDefaultRegistry:
    def test_reference_models_present(self):
        reg = default_registry()
        assert reg.names() == ["af_kinematic", "j2_isotropic", "ow_kinematic"]

    def test_all_uniaxial(self):
        reg = default_registry()
        for name in reg.names():
            assert reg.describe(name).ntens == 1

    def test_all_have_complete_meta(self):
        reg = default_registry()
        for name in reg.names():
            d = reg.describe(name)
            assert set(d.param_meta) == set(d.param_names), name

    def test_independent_instances(self):
        a, b = default_registry(), default_registry()
        a.register("extra", J2Isotropic3D)
        assert "extra" not in b.names()
