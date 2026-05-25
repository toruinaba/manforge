from copy import deepcopy
import numpy as np
import autograd.numpy as anp
from manforge.utils.smooth import smooth_sqrt, smooth_max, smooth_heaviside
from manforge.core.material import MaterialModel, verified_against_fortran
from manforge.core.result import ReturnMappingResult
from manforge.core.state import Explicit, Implicit, NTENS, SCALAR
from manforge.core.dimension import SOLID_3D, PLANE_STRESS, UNIAXIAL_1D, StressDimension

class YUKinematic(MaterialModel):
    """Yoshida-Uemori two-surface + stagnation-surface kinematic hardening model."""

    param_names = ["E", "nu", "Y", "B", "C_1", "C_2", "Rsat", "k", "b", "h", "Ea", "xi"]
    stress = Implicit(shape=NTENS, doc="cauchy stress")
    theta = Implicit(shape=NTENS, doc="relative backstress tensor of yield surface(deviatoric)")
    beta = Implicit(shape=NTENS, doc="relative backstress tensor of boundary surface(deviatoric)")
    R = Explicit(shape=SCALAR, doc="Radius increment of boundary surface")
    q = Explicit(shape=NTENS, doc="center of stagnation surface")
    r = Explicit(shape=SCALAR, doc="radius of stagnation surface")
    eps_eq = Explicit(shape=SCALAR, doc="equivalent plastic strain")
    theta_max = Explicit(shape=SCALAR, doc="Theta norm max in history")

    def __init__(self, dimension: StressDimension = SOLID_3D, *,
                 E: float, nu: float, Y: float, C_1: float, C_2: float,
                 B: float, Rsat: float, k: float, b: float,
                 h: float, Ea: float, xi: float):
        super().__init__(dimension=dimension)
        self.E = E
        self.nu = nu
        self.Y = Y
        self.C_1 = C_1
        self.C_2 = C_2
        self.B = B
        self.Rsat = Rsat
        self.k = k
        self.b = b
        self.h = h
        self.Ea = Ea
        self.xi = xi

    @verified_against_fortran(
        "yu_kinematic_3d_elastic_stiffness",
        test="tests/benchmarks/yu_kinematic/test_numerical_vs_fortran.py::TestHelpers::test_elastic_stiffness",
    )
    def elastic_stiffness(self, state):
        eps_eq = state["eps_eq"]
        mu = self.E / (2.0 * (1.0 + self.nu))
        lam = self.E * self.nu / ((1.0 + self.nu) * (1.0 - 2.0 * self.nu))
        return self._calc_E_factor(eps_eq) * self.isotropic_C(lam, mu) # f

    def yield_function(self, state):
        s_xi = self.dev(state["stress"]) - state["theta"] - state["beta"]
        return self.vonmises_norm(s_xi) - self.Y

    def update_state(self, dlambda, state_new, state_n, *, stress_trial=None, strain_inc=None):
        R_n = state_n["R"]
        q_n = state_n["q"]
        r_n = state_n["r"]
        s = 1 / (1 + self.k * dlambda)
        beta_new = state_new["beta"]
        d_beta = beta_new - state_n["beta"]
        theta_new = state_new["theta"]
        theta_norm = self.vonmises_norm(theta_new)
        g_xi = beta_new - q_n
        stag_norm = self.vonmises_norm(g_xi)
        g_stag = stag_norm - r_n
        g_flag = smooth_heaviside(g_stag)
        Gn = self.deviatoric_inner_product(g_xi, g_xi)
        Fn = self.deviatoric_inner_product(g_xi, d_beta)
        mu = 0.0
        for i in range(10):
            H_mu = smooth_sqrt(r_n * r_n + 6 * self.h * Fn / (1 + mu))
            F_mu = 3 * Gn - r_n * (r_n + H_mu) * (1 + mu) * (1 + mu) - 3 * self.h * Fn * (1 + mu)
            if F_mu < 1.0e-16:
                break
            F_mu_prime = 3 * self.h * Fn / H_mu * (r_n - H_mu) - 2 * r_n * (1 + mu) * (r_n + H_mu)
            mu -= F_mu / F_mu_prime
        else:
            raise ValueError("Not converged mu (update_state)")
        delta_q = mu * g_xi / (1 + mu)
        delta_r = 0.5 * (r_n + smooth_sqrt(r_n * r_n + 6 * self.h * Fn / (1 + mu))) - r_n
        delta_R = s * (R_n + self.k * self.Rsat * dlambda) - R_n
        return [
            self.R(R_n + g_flag * delta_R),
            self.q(q_n + g_flag * delta_q),
            self.r(r_n + g_flag * delta_r),
            self.eps_eq(state_n["eps_eq"] + dlambda),
            self.theta_max(smooth_max(state_n["theta_max"], theta_norm))
        ]

    def state_residual(self, state_new, dlambda, state_n, *, stress_trial, strain_inc=None):
        stress_new = state_new["stress"]
        theta_new = state_new["theta"]
        beta_new = state_new["beta"]
        R_new = state_new["R"]
        theta_max = state_n["theta_max"]
        s_xi = self.dev(stress_new) - theta_new - beta_new
        a = self.B + R_new - self.Y
        theta_norm = self.vonmises_norm(theta_new)
        C_k = self.C_1 - (self.C_1 - self.C_2) * smooth_heaviside(theta_max - (self.B - self.Y))
        R_stress = self.default_stress_residual(state_new, dlambda, stress_trial)
        R_theta = theta_new - state_n["theta"] - (C_k * a / self.Y * s_xi - C_k * smooth_sqrt(a / theta_norm) * theta_new) * dlambda 
        R_beta = beta_new - state_n["beta"] - (self.k * self.b / self.Y * s_xi - self.k * beta_new) * dlambda
        return [self.stress(R_stress), self.theta(R_theta), self.beta(R_beta)]

    def _calc_E_factor(self, eps_eq):
        factor = 1.0 - (1.0 - self.Ea / self.E) * (1.0 - anp.exp(-self.xi * eps_eq))
        return factor


class YUKinematic3D(YUKinematic):
    """YUKinematic specialised for 3D solid elements (ntens=6).

    Provides user_defined_return_mapping and user_defined_tangent via
    an explicit analytical Jacobian. These analytical paths assume ntens=6
    and are therefore only valid for SOLID_3D; YUKinematicPS and
    YUKinematic1D rely on the autograd path inherited from YUKinematic.
    """

    I = np.eye(6)
    T = np.diag([1.0, 1.0, 1.0, 2.0, 2.0, 2.0])

    def __init__(self, *, E: float, nu: float, Y: float, C_1: float, C_2: float,
                 B: float, Rsat: float, k: float, b: float, 
                 h: float, Ea: float, xi: float):
        super().__init__(dimension=SOLID_3D, E=E, nu=nu, Y=Y, C_1=C_1, C_2=C_2,
                 B=B, Rsat=Rsat, k=k, b=b, h=h, Ea=Ea, xi=xi)

    @verified_against_fortran(
        "yu_kinematic_3d",
        test="tests/benchmarks/yu_kinematic/test_numerical_vs_fortran.py::TestCrosscheckTrajectory::test_analytical_vs_fortran",
    )
    def user_defined_return_mapping(
            self, stress_trial: anp.ndarray, C: anp.ndarray, state_n: dict
        ):
        iter_rm = 50
        n_iteration = 0
        converged = False
        r_hist = []
        state_new = deepcopy(state_n)
        state_new["stress"] = deepcopy(stress_trial)
        dlambda = 0.0
        for iter in range(iter_rm):
            r_vector = self.calc_residual(state_new, state_n, stress_trial, dlambda)
            r_norm = np.linalg.norm(r_vector)
            r_hist.append(r_norm)
            if anp.abs(r_norm) < 1.0e-10:
                converged = True
                break
            jacobian = self.calc_jacobian(state_new, state_n, stress_trial, dlambda)
            dx = np.linalg.solve(jacobian, r_vector)
            state_new["stress"] -= dx[0:6]
            state_new["theta"] -= dx[7:13]
            state_new["beta"] -= dx[13:]
            dlambda -= dx[6]
            theta_norm = self.vonmises_norm(state_new["theta"])
            s = 1 / (1 + self.k * dlambda)
            d_beta = state_new["beta"] - state_n["beta"]
            g_xi = state_new["beta"] - state_n["q"]
            stag_norm = self.vonmises_norm(g_xi)
            g_stag = stag_norm - state_n["r"]
            g_flag = 1.0 if g_stag > 0.0 else 0.0
            Gn = self.deviatoric_inner_product(g_xi, g_xi)
            Fn = self.deviatoric_inner_product(g_xi, d_beta)
            mu = 0.0
            for i in range(10):
                H_mu = smooth_sqrt(state_n["r"] * state_n["r"] + 6 * self.h * Fn / (1 + mu))
                F_mu = 3 * Gn - state_n["r"] * (state_n["r"] + H_mu) * (1 + mu) * (1 + mu) - 3 * self.h * Fn * (1 + mu)
                if F_mu < 1.0e-16:
                    break
                F_mu_prime = 3 * self.h * Fn / H_mu * (state_n["r"] - H_mu) - 2 * state_n["r"] * (1 + mu) * (state_n["r"] + H_mu)
                mu -= F_mu / F_mu_prime
            else:
                raise ValueError("Not converged mu (user_defined_return_mapping)")
            delta_q = mu * g_xi / (1 + mu)
            delta_r = 0.5 * (state_n["r"] + smooth_sqrt(state_n["r"] * state_n["r"] + 6 * self.h * Fn / (1 + mu))) - state_n["r"]
            delta_R = s * (state_n["R"] + self.k * self.Rsat * dlambda) - state_n["R"]
            state_new["R"] = state_n["R"] + delta_R * g_flag
            state_new["q"] = state_n["q"] + delta_q * g_flag
            state_new["r"] = state_n["r"] + delta_r * g_flag
            state_new["eps_eq"] = state_n["eps_eq"] + dlambda
            n_iteration += 1
        else:
            converged = False
        theta_norm_final = self.vonmises_norm(state_new["theta"])
        state_new["theta_max"] = float(smooth_max(state_n["theta_max"], theta_norm_final))
        return ReturnMappingResult(
            stress=state_new["stress"],
            state=state_new,
            dlambda=dlambda,
            n_iterations=n_iteration,
            residual_history=r_hist,
            converged=converged,
        )

    def user_defined_tangent(self, stress, state, dlambda, C, state_n, stress_trial=None, strain_inc=None):
        ddsdde = self.calc_ddsdde(state, state_n, stress_trial, dlambda)
        return ddsdde

    @verified_against_fortran(
        "yu_calc_norm_n_flow",
        test="tests/benchmarks/yu_kinematic/test_numerical_vs_fortran.py::TestHelpers::test_calc_norm_n_flow",
    )
    def calc_norm_n_flow(self, xi):
        xi_norm = self.vonmises_norm(xi)
        flow = self.T @ xi / xi_norm * 1.5
        return xi_norm, flow

    @verified_against_fortran(
        "yu_calc_residual",
        test="tests/benchmarks/yu_kinematic/test_numerical_vs_fortran.py::TestResidualAndJacobian::test_residual_and_jacobian_match_fortran",
    )
    def calc_residual(self, state_new, state_n, stress_trial, dlambda):
        C = self.elastic_stiffness(state_new)
        C_k = self.C_1 - (self.C_1 - self.C_2) * smooth_heaviside(state_n["theta_max"] - (self.B - self.Y))
        a = self.B + state_new["R"] - self.Y
        dev_stress = self.dev(state_new["stress"])
        xi = dev_stress - state_new["theta"] - state_new["beta"]
        _, flow = self.calc_norm_n_flow(xi)
        theta_norm, _ = self.calc_norm_n_flow(state_new["theta"])
        R_stress = state_new["stress"] - stress_trial + dlambda * C @ flow
        R_theta = state_new["theta"] - state_n["theta"] - (C_k * a / self.Y * xi - C_k * smooth_sqrt(a / theta_norm) * state_new["theta"]) * dlambda 
        R_beta = state_new["beta"] - state_n["beta"] - (self.k * self.b / self.Y * xi - self.k * state_new["beta"]) * dlambda
        R_yield = self.yield_function(state_new)
        r_vector = anp.hstack((R_stress, R_yield, R_theta, R_beta))
        return r_vector

    @verified_against_fortran(
        "yu_prepare_rstress",
        test="tests/benchmarks/yu_kinematic/test_numerical_vs_fortran.py::TestHelpers::test_prepare_rstress",
    )
    def _prepare_Rstress(self, xi):
        xi_norm, flow = self.calc_norm_n_flow(xi)
        dn_dsig = (
            1.5 / xi_norm * (
                self.T @ self.I_dev() - np.outer(flow / np.sqrt(1.5), flow / np.sqrt(1.5))
            )
        )
        return dn_dsig

    @verified_against_fortran(
        "yu_prepare_rtheta",
        test="tests/benchmarks/yu_kinematic/test_numerical_vs_fortran.py::TestHelpers::test_prepare_rtheta",
    )
    def _prepare_Rtheta(self, theta, theta_max, R, R_n, dlambda):
        theta_bar = self.vonmises_norm(theta)
        theta_flow = self.T @ theta / theta_bar * 1.5
        C_k = self.C_1 if self.B - self.Y > theta_max else self.C_2
        s = 1 / (1 + self.k * dlambda)
        a = self.B + R - self.Y
        if R != R_n:
            a_prime = (
                -self.k * s * s * (R_n + self.k * self.Rsat * dlambda)
                + s * self.k * self.Rsat
            )
        else:
            a_prime = 0.0
        return theta_bar, theta_flow, C_k, s, a, a_prime

    @verified_against_fortran(
        "yu_drs_dstress",
        test="tests/benchmarks/yu_kinematic/test_numerical_vs_fortran.py::TestJacobianBlocks::test_dRstress_dstress",
    )
    def dRstress_dstress(self, C, xi, dlambda):
        dn_dsig = self._prepare_Rstress(xi)
        return self.I + dlambda * C @ dn_dsig

    @verified_against_fortran(
        "yu_drs_dbeta",
        test="tests/benchmarks/yu_kinematic/test_numerical_vs_fortran.py::TestJacobianBlocks::test_dRstress_dbeta",
    )
    def dRstress_dbeta(self, C, xi, dlambda):
        dn_dsig = self._prepare_Rstress(xi)
        return - dlambda * C @ dn_dsig

    @verified_against_fortran(
        "yu_drs_dtheta",
        test="tests/benchmarks/yu_kinematic/test_numerical_vs_fortran.py::TestJacobianBlocks::test_dRstress_dtheta",
    )
    def dRstress_dtheta(self, C, xi, dlambda):
        dn_dsig = self._prepare_Rstress(xi)
        return - dlambda * C @ dn_dsig

    @verified_against_fortran(
        "yu_drs_dlambda",
        test="tests/benchmarks/yu_kinematic/test_numerical_vs_fortran.py::TestJacobianBlocks::test_dRstress_dlambda",
    )
    def dRstress_dlambda(self, C, xi, eps_eq, dlambda):
        _, flow = self.calc_norm_n_flow(xi)
        factor = self.Ea / self.E + (1 - self.Ea / self.E) * anp.exp(-self.xi * eps_eq)
        return C @ flow - self.xi * (1 - self.Ea / self.E) * anp.exp(-self.xi * eps_eq) / factor * dlambda * (C @ flow)

    @verified_against_fortran(
        "yu_drb_dstress",
        test="tests/benchmarks/yu_kinematic/test_numerical_vs_fortran.py::TestJacobianBlocks::test_dRbeta_dstress",
    )
    def dRbeta_dstress(self, dlambda):
        return -self.k * self.b * dlambda / self.Y * self.I_dev()

    @verified_against_fortran(
        "yu_drb_dbeta",
        test="tests/benchmarks/yu_kinematic/test_numerical_vs_fortran.py::TestJacobianBlocks::test_dRbeta_dbeta",
    )
    def dRbeta_dbeta(self, dlambda):
        return (1.0 + self.k * self.b / self.Y * dlambda + self.k * dlambda) * self.I

    @verified_against_fortran(
        "yu_drb_dtheta",
        test="tests/benchmarks/yu_kinematic/test_numerical_vs_fortran.py::TestJacobianBlocks::test_dRbeta_dtheta",
    )
    def dRbeta_dtheta(self, dlambda):
        return self.k * self.b * dlambda / self.Y * self.I

    @verified_against_fortran(
        "yu_drb_dlambda",
        test="tests/benchmarks/yu_kinematic/test_numerical_vs_fortran.py::TestJacobianBlocks::test_dRbeta_dlambda",
    )
    def dRbeta_dlambda(self, xi, beta, dlambda):
        return -self.k * self.b / self.Y * xi + self.k * beta

    @verified_against_fortran(
        "yu_drt_dstress",
        test="tests/benchmarks/yu_kinematic/test_numerical_vs_fortran.py::TestJacobianBlocks::test_dRtheta_dstress",
    )
    def dRtheta_dstress(self, theta, theta_max, R, R_n, dlambda):
        _, _, C_k, _, a, _ = self._prepare_Rtheta(theta, theta_max, R, R_n, dlambda)
        return -a * C_k * dlambda / self.Y * self.I_dev()

    @verified_against_fortran(
        "yu_drt_dbeta",
        test="tests/benchmarks/yu_kinematic/test_numerical_vs_fortran.py::TestJacobianBlocks::test_dRtheta_dbeta",
    )
    def dRtheta_dbeta(self, theta, theta_max, R, R_n, dlambda):
        _, _, C_k, _, a, _ = self._prepare_Rtheta(theta, theta_max, R, R_n, dlambda)
        return a * C_k * dlambda / self.Y * self.I

    @verified_against_fortran(
        "yu_drt_dtheta",
        test="tests/benchmarks/yu_kinematic/test_numerical_vs_fortran.py::TestJacobianBlocks::test_dRtheta_dtheta",
    )
    def dRtheta_dtheta(self, theta, theta_max, R, R_n, dlambda):
        theta_bar, theta_flow, C_k, _, a, _ = self._prepare_Rtheta(theta, theta_max, R, R_n, dlambda)
        if theta_bar < 1e-14:
            return (1 + a * C_k * dlambda / self.Y) * self.I
        return (1 + a * C_k * dlambda / self.Y + C_k * dlambda * np.sqrt(a / theta_bar)) * self.I - (
            np.sqrt(1.5) * C_k * dlambda * np.sqrt(a / theta_bar) / (2 * theta_bar)
        ) * np.outer(theta_flow / np.sqrt(1.5), theta)
    
    @verified_against_fortran(
        "yu_drt_dlambda",
        test="tests/benchmarks/yu_kinematic/test_numerical_vs_fortran.py::TestJacobianBlocks::test_dRtheta_dlambda",
    )
    def dRtheta_dlambda(self, xi, theta, theta_max, R, R_n, dlambda):
        theta_bar, _, C_k, _, a, a_prime = self._prepare_Rtheta(theta, theta_max, R, R_n, dlambda)
        if theta_bar < 1e-14:
            return (-a * C_k / self.Y - C_k * dlambda / self.Y * a_prime) * xi
        fr = (
            - a * C_k / self.Y * xi
            - C_k * dlambda / self.Y * a_prime * xi
            + C_k * np.sqrt(a / theta_bar) * theta
            + C_k * dlambda * np.sqrt(1 / (theta_bar * a)) * a_prime / 2 * theta
        )
        return fr

    @verified_against_fortran(
        "yu_drl_dstress",
        test="tests/benchmarks/yu_kinematic/test_numerical_vs_fortran.py::TestJacobianBlocks::test_dRyield_dstress",
    )
    def dRyield_dstress(self, xi):
        _, flow = self.calc_norm_n_flow(xi)
        return flow

    @verified_against_fortran(
        "yu_drl_dbeta",
        test="tests/benchmarks/yu_kinematic/test_numerical_vs_fortran.py::TestJacobianBlocks::test_dRyield_dbeta",
    )
    def dRyield_dbeta(self, xi):
        _, flow = self.calc_norm_n_flow(xi)
        return -flow

    @verified_against_fortran(
        "yu_drl_dtheta",
        test="tests/benchmarks/yu_kinematic/test_numerical_vs_fortran.py::TestJacobianBlocks::test_dRyield_dtheta",
    )
    def dRyield_dtheta(self, xi):
        _, flow = self.calc_norm_n_flow(xi)
        return -flow

    @verified_against_fortran(
        "yu_drl_dlambda",
        test="tests/benchmarks/yu_kinematic/test_numerical_vs_fortran.py::TestJacobianBlocks::test_dRyield_dlambda",
    )
    def dRyield_dlambda(self):
        return np.array([0.0])

    @verified_against_fortran(
        "yu_calc_jacobian",
        test="tests/benchmarks/yu_kinematic/test_numerical_vs_fortran.py::TestResidualAndJacobian::test_residual_and_jacobian_match_fortran",
    )
    def calc_jacobian(self, state_new, state_n, stress_trial, dlambda):
        C = self.elastic_stiffness(state_new)
        xi = self.dev(state_new["stress"]) - state_new["theta"] - state_new["beta"]
        Rs_s = self.dRstress_dstress(C, xi, dlambda)
        Rs_b = self.dRstress_dbeta(C, xi, dlambda)
        Rs_t = self.dRstress_dtheta(C, xi, dlambda)
        Rs_l = self.dRstress_dlambda(C, xi, state_new["eps_eq"], dlambda)
        Rs = np.hstack((Rs_s, Rs_l[:, np.newaxis], Rs_t, Rs_b))
        Rb_s = self.dRbeta_dstress(dlambda)
        Rb_b = self.dRbeta_dbeta(dlambda)
        Rb_t = self.dRbeta_dtheta(dlambda)
        Rb_l = self.dRbeta_dlambda(xi, state_new["beta"], dlambda)
        Rb = np.hstack((Rb_s, Rb_l[:, np.newaxis], Rb_t, Rb_b))
        Rt_s = self.dRtheta_dstress(state_new["theta"], state_new["theta_max"], state_new["R"], state_n["R"], dlambda)
        Rt_b = self.dRtheta_dbeta(state_new["theta"], state_new["theta_max"], state_new["R"], state_n["R"], dlambda)
        Rt_t = self.dRtheta_dtheta(state_new["theta"], state_new["theta_max"], state_new["R"], state_n["R"], dlambda)
        Rt_l = self.dRtheta_dlambda(xi, state_new["theta"], state_new["theta_max"], state_new["R"], state_n["R"], dlambda)
        Rt = np.hstack((Rt_s, Rt_l[:, np.newaxis], Rt_t, Rt_b))
        Rl_s = self.dRyield_dstress(xi)
        Rl_b = self.dRyield_dbeta(xi)
        Rl_t = self.dRyield_dtheta(xi)
        Rl_l = self.dRyield_dlambda()
        Rl = np.hstack((Rl_s, Rl_l, Rl_t, Rl_b))
        return np.vstack((Rs, Rl.reshape(1, -1), Rt, Rb))

    @verified_against_fortran(
        "yu_calc_ddsdde",
        test="tests/benchmarks/yu_kinematic/test_numerical_vs_fortran.py::TestReturnMapping::test_single_plastic_step_ddsdde",
    )
    def calc_ddsdde(self, state_new, state_n, stress_trial, dlambda):
        C = self.elastic_stiffness(state_new)
        C_inv = anp.linalg.inv(C)
        xi = self.dev(state_new["stress"]) - state_new["theta"] - state_new["beta"]
        Rs_s = C_inv @ self.dRstress_dstress(C, xi, dlambda)
        Rs_b = C_inv @ self.dRstress_dbeta(C, xi, dlambda)
        Rs_t = C_inv @ self.dRstress_dtheta(C, xi, dlambda)
        Rs_l = C_inv @ self.dRstress_dlambda(C, xi, state_new["eps_eq"], dlambda)
        Rs = np.hstack((Rs_s, Rs_l[:, np.newaxis], Rs_t, Rs_b))
        Rb_s = self.dRbeta_dstress(dlambda)
        Rb_b = self.dRbeta_dbeta(dlambda)
        Rb_t = self.dRbeta_dtheta(dlambda)
        Rb_l = self.dRbeta_dlambda(xi, state_new["beta"], dlambda)
        Rb = np.hstack((Rb_s, Rb_l[:, np.newaxis], Rb_t, Rb_b))
        Rt_s = self.dRtheta_dstress(state_new["theta"], state_new["theta_max"], state_new["R"], state_n["R"], dlambda)
        Rt_b = self.dRtheta_dbeta(state_new["theta"], state_new["theta_max"], state_new["R"], state_n["R"], dlambda)
        Rt_t = self.dRtheta_dtheta(state_new["theta"], state_new["theta_max"], state_new["R"], state_n["R"], dlambda)
        Rt_l = self.dRtheta_dlambda(xi, state_new["theta"], state_new["theta_max"], state_new["R"], state_n["R"], dlambda)
        Rt = np.hstack((Rt_s, Rt_l[:, np.newaxis], Rt_t, Rt_b))
        Rl_s = self.dRyield_dstress(xi)
        Rl_b = self.dRyield_dbeta(xi)
        Rl_t = self.dRyield_dtheta(xi)
        Rl_l = self.dRyield_dlambda()
        Rl = np.hstack((Rl_s, Rl_l, Rl_t, Rl_b))
        jac = np.vstack((Rs, Rl.reshape(1, -1), Rt, Rb))
        jac_inv = anp.linalg.inv(jac)
        return np.array(jac_inv[:6, :6])


class YUKinematicPS(YUKinematic):
    def __init__(self, *, E: float, nu: float, Y: float, C_1: float, C_2: float,
                 B: float, Rsat: float, k: float, b: float,
                 h: float, Ea: float, xi: float):
        super().__init__(dimension=PLANE_STRESS, E=E, nu=nu, Y=Y, C_1=C_1, C_2=C_2,
                 B=B, Rsat=Rsat, k=k, b=b, h=h, Ea=Ea, xi=xi)


class YUKinematic1D(YUKinematic):
    def __init__(self, *, E: float, nu: float, Y: float, C_1: float, C_2: float,
                 B: float, Rsat: float, k: float, b: float,
                 h: float, Ea: float, xi: float):
        super().__init__(dimension=UNIAXIAL_1D, E=E, nu=nu, Y=Y, C_1=C_1, C_2=C_2,
                 B=B, Rsat=Rsat, k=k, b=b, h=h, Ea=Ea, xi=xi)
