"""Integration tests: YUKinematic3D Fortran binding registry.

Checks that @verified_against_fortran decorators are correctly registered
on YUKinematic3D._fortran_bindings.  Does not require the compiled .so.
"""

import pytest


def test_all_bindings_registered():
    from manforge.models import YUKinematic3D
    bindings = YUKinematic3D._fortran_bindings
    expected = {
        "elastic_stiffness":    "yu_kinematic_3d_elastic_stiffness",
        "calc_norm_n_flow":     "yu_calc_norm_n_flow",
        "_prepare_Rstress":     "yu_prepare_rstress",
        "_prepare_Rtheta":      "yu_prepare_rtheta",
        "calc_residual":        "yu_calc_residual",
        "calc_jacobian":        "yu_calc_jacobian",
        "dRstress_dstress":     "yu_drs_dstress",
        "dRstress_dbeta":       "yu_drs_dbeta",
        "dRstress_dtheta":      "yu_drs_dtheta",
        "dRstress_dlambda":     "yu_drs_dlambda",
        "dRbeta_dstress":       "yu_drb_dstress",
        "dRbeta_dbeta":         "yu_drb_dbeta",
        "dRbeta_dtheta":        "yu_drb_dtheta",
        "dRbeta_dlambda":       "yu_drb_dlambda",
        "dRtheta_dstress":      "yu_drt_dstress",
        "dRtheta_dbeta":        "yu_drt_dbeta",
        "dRtheta_dtheta":       "yu_drt_dtheta",
        "dRtheta_dlambda":      "yu_drt_dlambda",
        "dRyield_dstress":          "yu_drl_dstress",
        "dRyield_dbeta":            "yu_drl_dbeta",
        "dRyield_dtheta":           "yu_drl_dtheta",
        "dRyield_dlambda":          "yu_drl_dlambda",
        "user_defined_return_mapping": "yu_kinematic_3d",
        "calc_ddsdde":              "yu_calc_ddsdde",
    }
    for method, subroutine in expected.items():
        assert method in bindings, f"{method} not in _fortran_bindings"
        assert bindings[method].subroutine == subroutine, (
            f"{method}: expected subroutine={subroutine!r}, got {bindings[method].subroutine!r}"
        )
