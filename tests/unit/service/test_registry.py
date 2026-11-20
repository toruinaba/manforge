"""Tests for ModelRegistry / ModelDescriptor / default_registry."""

import pytest

from returnmap.core import PLANE_STRESS, ParamMeta
from returnmap.models import J2Isotropic, J2Isotropic1D, J2Isotropic3D
from returnmap.service import ModelRegistry, ModelDescriptor, default_registry

J2 = dict(E=210e3, nu=0.3, sigma_y0=250.0, H=1e3)
J2_1D = J2Isotropic1D(**J2)
J2_3D = J2Isotropic3D(**J2)


class TestRegister:
    def test_register_and_get(self):
        reg = ModelRegistry()
        reg.register("j2", J2_1D)
        assert reg.get("j2") is J2_1D

    def test_duplicate_name_raises(self):
        reg = ModelRegistry()
        reg.register("j2", J2_1D)
        with pytest.raises(ValueError, match="already registered.*J2Isotropic1D"):
            reg.register("j2", J2_3D)

    def test_non_model_raises(self):
        reg = ModelRegistry()
        with pytest.raises(ValueError, match="MaterialModel instance"):
            reg.register("x", dict)

    def test_class_rejected(self):
        """A class cannot say which dimension or settings to run with."""
        reg = ModelRegistry()
        with pytest.raises(ValueError, match="MaterialModel instance.*construct the class"):
            reg.register("x", J2Isotropic1D)

    def test_one_class_under_two_names(self):
        reg = ModelRegistry()
        reg.register("j2_3d", J2Isotropic(**J2))
        reg.register("j2_ps", J2Isotropic(dimension=PLANE_STRESS, **J2))
        assert reg.describe("j2_3d").ntens == 6
        assert reg.describe("j2_ps").ntens == 3


class TestGet:
    def test_unknown_name_lists_registered(self):
        reg = ModelRegistry()
        reg.register("j2", J2_1D)
        with pytest.raises(KeyError, match=r"unknown model 'nope'.*j2"):
            reg.get("nope")

    def test_names_sorted(self):
        reg = ModelRegistry()
        reg.register("b", J2_1D)
        reg.register("a", J2_3D)
        assert reg.names() == ["a", "b"]


class TestDescribe:
    def test_descriptor_contents(self):
        reg = ModelRegistry()
        reg.register("j2", J2_1D)
        d = reg.describe("j2")
        assert isinstance(d, ModelDescriptor)
        assert d.name == "j2"
        assert d.class_name == "J2Isotropic1D"
        assert d.param_names == ("E", "nu", "sigma_y0", "H")
        assert d.ntens == 1
        assert d.doc  # first docstring line, non-empty

    def test_descriptor_param_meta(self):
        reg = ModelRegistry()
        reg.register("j2", J2_1D)
        d = reg.describe("j2")
        assert set(d.param_meta) == set(d.param_names)
        assert isinstance(d.param_meta["E"], ParamMeta)
        assert d.param_meta["E"].unit == "MPa"

    def test_ntens_reflects_registered_model(self):
        reg = ModelRegistry()
        reg.register("j2_3d", J2_3D)
        assert reg.describe("j2_3d").ntens == 6

    def test_model_without_meta_describes_empty(self):
        from returnmap.core import MaterialModel, Explicit, SCALAR, UNIAXIAL_1D

        class Bare(MaterialModel):
            param_names = ["E", "nu", "sigma_y0"]
            ep = Explicit(shape=SCALAR, doc="ep")

            dimension = UNIAXIAL_1D

            def yield_function(self, state):
                return abs(state["stress"][0]) - self.sigma_y0

            def update_state(self, new, dlambda, inc):
                return [self.ep(inc.state_n["ep"] + dlambda)]

        reg = ModelRegistry()
        reg.register("bare", Bare(E=210e3, nu=0.3, sigma_y0=250.0))
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

    def test_templates_hold_the_meta_defaults(self):
        reg = default_registry()
        for name in reg.names():
            model = reg.get(name)
            assert model.params == {
                k: m.default for k, m in reg.describe(name).param_meta.items()
            }, name

    def test_independent_instances(self):
        a, b = default_registry(), default_registry()
        a.register("extra", J2_3D)
        assert "extra" not in b.names()
