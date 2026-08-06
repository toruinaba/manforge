from copy import deepcopy
import numpy as np
import autograd.numpy as anp
from manforge.utils.smooth import smooth_sqrt, smooth_max, smooth_heaviside
from manforge.core.material import MaterialModel, verified_against_fortran
from manforge.core.result import ReturnMappingResult
from manforge.core.state import Explicit, Implicit, NTENS, SCALAR
from manforge.core.dimension import (
    SOLID_3D, PLANE_STRESS_P, UNIAXIAL_1D, StressDimension,
)

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

    @property
    def k_eff(self):
        """Boundary-surface hardening rate as it enters the R evolution law.

        The norm-form and quadratic-form yield functions scale dlambda
        differently, so each dimension supplies its own effective rate; the
        evolution law itself is shared.
        """
        return self.k

    def _stagnation_update(self, r_n, R_n, g_xi, g_stag, d_beta, dlambda):
        """Return the gated ``(delta_q, delta_r, delta_R)``.

        The activity gate is applied HERE and nowhere else -- callers add the
        returned increments raw.  Applying it again outside would square the
        sigmoid, halving the increment on the transition band and changing its
        shape, which would show up as a spurious formulation difference when
        comparing against a subclass that gates differently.

        Subclasses override this to change how the stagnation surface evolves.
        mu is an implementation detail of this method, not part of the
        contract: an override may compute the increments without it.
        """
        # Dead band: shift by +1e-10 so boundary noise activates.
        g_flag = smooth_heaviside(g_stag + 1.0e-10)
        Gn = self.deviatoric_inner_product(g_xi, g_xi)
        Fn = self.deviatoric_inner_product(g_xi, d_beta)
        mu = 0.0
        if r_n >= 1e-14:
            for _ in range(10):
                H_mu = smooth_sqrt(r_n * r_n + 6 * self.h * Fn / (1 + mu))
                F_mu = 3 * Gn - r_n * (r_n + H_mu) * (1 + mu) * (1 + mu) - 3 * self.h * Fn * (1 + mu)
                # F_mu decreases in mu, so F_mu(0) < 0 puts the root at mu < 0:
                # beta is inside the surface and the stagnation state holds, so
                # mu = 0 is the answer.  Only the magnitude test may stop the
                # iteration otherwise -- a signed test would accept the first
                # step past the root, leaving beta off the surface by ~1e-1.
                if F_mu < 0.0 and mu <= 0.0:
                    mu = 0.0
                    break
                if abs(F_mu) < 1.0e-12 * max(abs(3 * Gn), 1.0):
                    break
                F_mu_prime = 3 * self.h * Fn / H_mu * (r_n - H_mu) - 2 * r_n * (1 + mu) * (r_n + H_mu)
                mu -= F_mu / F_mu_prime
            else:
                raise ValueError("Not converged mu")
        delta_q = mu * g_xi / (1 + mu)
        delta_r = 0.5 * (r_n + smooth_sqrt(r_n * r_n + 6 * self.h * Fn / (1 + mu))) - r_n
        k_eff = self.k_eff
        delta_R = (R_n + k_eff * self.Rsat * dlambda) / (1 + k_eff * dlambda) - R_n
        return g_flag * delta_q, g_flag * delta_r, g_flag * delta_R

    def update_state(self, dlambda, state_new, state_n, *, stress_trial=None, strain_inc=None):
        R_n = state_n["R"]
        q_n = state_n["q"]
        r_n = state_n["r"]
        beta_new = state_new["beta"]
        d_beta = beta_new - state_n["beta"]
        theta_new = state_new["theta"]
        theta_norm = self.vonmises_norm(theta_new)
        g_xi = beta_new - q_n
        g_stag = self.vonmises_norm(g_xi) - r_n
        delta_q, delta_r, delta_R = self._stagnation_update(
            r_n, R_n, g_xi, g_stag, d_beta, dlambda)
        return [
            self.R(R_n + delta_R),
            self.q(q_n + delta_q),
            self.r(r_n + delta_r),
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

    def flow(self, state):
        """Hand-derived flow direction, matching :meth:`calc_jacobian`.

        ``calc_norm_n_flow`` returns ``1.5·T@ξ/‖ξ‖`` — already engineering shear.

        This differs from the autograd default: ``grad(yield_function)(σ)``
        applies ``I_dev`` via the chain rule, so ``dRstress/dtheta`` would not
        match ``calc_jacobian`` when θ perturbs ξ off the deviatoric manifold.
        """
        xi = self.dev(state["stress"]) - state["theta"] - state["beta"]
        return self.strain_flow(self.calc_norm_n_flow(xi)[1])

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
            d_beta = state_new["beta"] - state_n["beta"]
            g_xi = state_new["beta"] - state_n["q"]
            g_stag = self.vonmises_norm(g_xi) - state_n["r"]
            # The gate inside _stagnation_update is re-evaluated every
            # iteration.  A hard branch here would be discontinuous across
            # iterations, and the one-way latch that used to guard it kept R
            # evolving on steps whose converged g_stag is negative — 0.30 MPa
            # of stress error via a = B + R − Y in R_theta, so the two routes
            # solved different systems.
            delta_q, delta_r, delta_R = self._stagnation_update(
                state_n["r"], state_n["R"], g_xi, g_stag, d_beta, dlambda)
            state_new["R"] = state_n["R"] + delta_R
            state_new["q"] = state_n["q"] + delta_q
            state_new["r"] = state_n["r"] + delta_r
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

    def _dflow_dxi(self, xi):
        """∂flow/∂ξ  (no I_dev projection; use for dRstress/dtheta and dRstress/dbeta).

        flow = 1.5·T@ξ / |ξ|_VM,  dflow/dξ = 1.5/|ξ| · (T − outer(n̂, n̂))
        where n̂ = flow/√1.5  (unit-norm direction).
        """
        xi_norm, flow = self.calc_norm_n_flow(xi)
        return 1.5 / xi_norm * (self.T - np.outer(flow / np.sqrt(1.5), flow / np.sqrt(1.5)))

    @verified_against_fortran(
        "yu_prepare_rstress",
        test="tests/benchmarks/yu_kinematic/test_numerical_vs_fortran.py::TestHelpers::test_prepare_rstress",
    )
    def _prepare_Rstress(self, xi):
        return self._dflow_dxi(xi) @ self.I_dev()

    @verified_against_fortran(
        "yu_prepare_rtheta",
        test="tests/benchmarks/yu_kinematic/test_numerical_vs_fortran.py::TestHelpers::test_prepare_rtheta",
    )
    def _prepare_Rtheta(self, theta, theta_max, R, R_n, dlambda, g_flag=None):
        theta_bar = self.vonmises_norm(theta)
        theta_flow = self.T @ theta / theta_bar * 1.5
        C_k = self.C_1 - (self.C_1 - self.C_2) * smooth_heaviside(theta_max - (self.B - self.Y))
        s = 1 / (1 + self.k * dlambda)
        a = self.B + R - self.Y
        active = (g_flag if g_flag is not None else
                  float(abs(R - R_n) > 1.0e-15 * max(abs(R_n), 1.0)))
        a_prime = (-self.k * s * s * (R_n + self.k * self.Rsat * dlambda) + s * self.k * self.Rsat) * active
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
        # ∂ξ/∂β = -I (no dev projection), so use _dflow_dxi without I_dev
        return -dlambda * C @ self._dflow_dxi(xi)

    @verified_against_fortran(
        "yu_drs_dtheta",
        test="tests/benchmarks/yu_kinematic/test_numerical_vs_fortran.py::TestJacobianBlocks::test_dRstress_dtheta",
    )
    def dRstress_dtheta(self, C, xi, dlambda):
        # ∂ξ/∂θ = -I (no dev projection), so use _dflow_dxi without I_dev
        return -dlambda * C @ self._dflow_dxi(xi)

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
        # θ ⊗ ∂θ̄/∂θ — the gradient goes in the second slot.  Transposing this
        # is invisible under uniaxial or pure-shear loading, because T's shear
        # weighting only makes outer(Tθ, θ) ≠ outer(θ, Tθ) when direct and
        # shear components are simultaneously nonzero.
        return (1 + a * C_k * dlambda / self.Y + C_k * dlambda * np.sqrt(a / theta_bar)) * self.I - (
            np.sqrt(1.5) * C_k * dlambda * np.sqrt(a / theta_bar) / (2 * theta_bar)
        ) * np.outer(theta, theta_flow / np.sqrt(1.5))
    
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
        Rt_s = self.dRtheta_dstress(state_new["theta"], state_n["theta_max"], state_new["R"], state_n["R"], dlambda)
        Rt_b = self.dRtheta_dbeta(state_new["theta"], state_n["theta_max"], state_new["R"], state_n["R"], dlambda)
        Rt_t = self.dRtheta_dtheta(state_new["theta"], state_n["theta_max"], state_new["R"], state_n["R"], dlambda)
        Rt_l = self.dRtheta_dlambda(xi, state_new["theta"], state_n["theta_max"], state_new["R"], state_n["R"], dlambda)
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
        # C_n (step-start stiffness) is used for the rhs scaling in the consistent
        # tangent: J·dx/dε = [C_n; 0; ...] → ddsdde = J^{-1}[0:6,0:6]·C_n.
        # Using C(state_new) here was a bug when E varies with eps_eq.
        C_n = self.elastic_stiffness(state_n)
        C_n_inv = anp.linalg.inv(C_n)
        xi = self.dev(state_new["stress"]) - state_new["theta"] - state_new["beta"]
        Rs_s = C_n_inv @ self.dRstress_dstress(C, xi, dlambda)
        Rs_b = C_n_inv @ self.dRstress_dbeta(C, xi, dlambda)
        Rs_t = C_n_inv @ self.dRstress_dtheta(C, xi, dlambda)
        Rs_l = C_n_inv @ self.dRstress_dlambda(C, xi, state_new["eps_eq"], dlambda)
        Rs = np.hstack((Rs_s, Rs_l[:, np.newaxis], Rs_t, Rs_b))
        Rb_s = self.dRbeta_dstress(dlambda)
        Rb_b = self.dRbeta_dbeta(dlambda)
        Rb_t = self.dRbeta_dtheta(dlambda)
        Rb_l = self.dRbeta_dlambda(xi, state_new["beta"], dlambda)
        Rb = np.hstack((Rb_s, Rb_l[:, np.newaxis], Rb_t, Rb_b))
        Rt_s = self.dRtheta_dstress(state_new["theta"], state_n["theta_max"], state_new["R"], state_n["R"], dlambda)
        Rt_b = self.dRtheta_dbeta(state_new["theta"], state_n["theta_max"], state_new["R"], state_n["R"], dlambda)
        Rt_t = self.dRtheta_dtheta(state_new["theta"], state_n["theta_max"], state_new["R"], state_n["R"], dlambda)
        Rt_l = self.dRtheta_dlambda(xi, state_new["theta"], state_n["theta_max"], state_new["R"], state_n["R"], dlambda)
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
    """YUKinematic specialised for plane stress under the P-metric convention.

    Stress-like quantities (σ, θ, β, q) store the raw in-plane tensor with the
    33 component identically zero, so deviatoric contractions go through
    ``PLANE_STRESS_P``'s P metric rather than an explicit projection.  This
    differs from ``PLANE_STRESS``, which treats the stored components as part
    of a 3D deviator and reconstructs θ33 = −(θ11 + θ22).

    The yield function is the quadratic form ``f = ½ ξᵀPξ − ⅓Y²``, so Δλ here
    is (2/3)Y times smaller than the norm-form Δλ used by ``YUKinematic``.
    Every 2/3 and Y factor below follows from that substitution.

    Only the autograd path is provided: the framework builds the NR Jacobian
    and consistent tangent by differentiating the methods below.
    """

    def __init__(self, *, E: float, nu: float, Y: float, C_1: float, C_2: float,
                 B: float, Rsat: float, k: float, b: float,
                 h: float, Ea: float, xi: float):
        super().__init__(dimension=PLANE_STRESS_P, E=E, nu=nu, Y=Y, C_1=C_1, C_2=C_2,
                 B=B, Rsat=Rsat, k=k, b=b, h=h, Ea=Ea, xi=xi)

    @property
    def P(self):
        """Plane-stress deviatoric metric; delegates to the dimension."""
        return self.dimension.P

    @property
    def k_eff(self):
        """(2/3)·Y·k: the quadratic-form dlambda is (2/3)Y times smaller."""
        return 2.0 / 3.0 * self.Y * self.k

    def yield_function(self, state):
        s_xi = self.dev(state["stress"]) - state["theta"] - state["beta"]
        return 0.5 * self.deviatoric_inner_product(s_xi, s_xi) - self.Y * self.Y / 3.0

    def update_state(self, dlambda, state_new, state_n, *, stress_trial=None, strain_inc=None):
        R_n = state_n["R"]
        q_n = state_n["q"]
        r_n = state_n["r"]
        s_xi = self.dev(state_new["stress"]) - state_new["theta"] - state_new["beta"]
        g = self.deviatoric_inner_product(s_xi, s_xi)
        # Rescales the quadratic-form Δλ to the norm-form increment: on the
        # yield surface √(2/3·g) = (2/3)Y.
        delta_eps_eq = dlambda * smooth_sqrt(2.0 / 3.0 * g)
        beta_new = state_new["beta"]
        d_beta = beta_new - state_n["beta"]
        theta_new = state_new["theta"]
        theta_norm = self.vonmises_norm(theta_new)
        g_xi = beta_new - q_n
        g_stag = self.vonmises_norm(g_xi) - r_n
        delta_q, delta_r, delta_R = self._stagnation_update(
            r_n, R_n, g_xi, g_stag, d_beta, dlambda)
        return [
            self.R(R_n + delta_R),
            self.q(q_n + delta_q),
            self.r(r_n + delta_r),
            self.eps_eq(state_n["eps_eq"] + delta_eps_eq),
            self.theta_max(smooth_max(state_n["theta_max"], theta_norm))
        ]

    def state_residual(self, state_new, dlambda, state_n, *, stress_trial, strain_inc=None):
        theta_new = state_new["theta"]
        beta_new = state_new["beta"]
        R_new = state_new["R"]
        theta_max = state_n["theta_max"]
        s_xi = self.dev(state_new["stress"]) - theta_new - beta_new
        a = self.B + R_new - self.Y
        theta_norm = self.vonmises_norm(theta_new)
        C_k = self.C_1 - (self.C_1 - self.C_2) * smooth_heaviside(theta_max - (self.B - self.Y))
        R_stress = self.default_stress_residual(state_new, dlambda, stress_trial)
        R_theta = theta_new - state_n["theta"] - 2.0 / 3.0 * (
            C_k * a * s_xi - C_k * self.Y * smooth_sqrt(a / theta_norm) * theta_new
        ) * dlambda
        R_beta = beta_new - state_n["beta"] - 2.0 / 3.0 * (
            self.k * self.b * s_xi - self.k * self.Y * beta_new
        ) * dlambda
        return [self.stress(R_stress), self.theta(R_theta), self.beta(R_beta)]

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
            state_new["stress"] -= dx[0:3]
            state_new["theta"] -= dx[4:7]
            state_new["beta"] -= dx[7:]
            dlambda -= dx[3]
            eta = state_new["stress"] - state_new["theta"] - state_new["beta"]
            theta_norm = self.vonmises_norm(state_new["theta"])
            g = self.deviatoric_inner_product(eta, eta)
            delta_eps_eq = dlambda * smooth_sqrt(2.0 / 3.0 * g)
            d_beta = state_new["beta"] - state_n["beta"]
            g_xi = state_new["beta"] - state_n["q"]
            g_stag = self.vonmises_norm(g_xi) - state_n["r"]
            # Same gate as update_state; see YUKinematic3D for why the
            # hard-branch latch was removed.
            delta_q, delta_r, delta_R = self._stagnation_update(
                state_n["r"], state_n["R"], g_xi, g_stag, d_beta, dlambda)
            state_new["R"] = state_n["R"] + delta_R
            state_new["q"] = state_n["q"] + delta_q
            state_new["r"] = state_n["r"] + delta_r
            state_new["eps_eq"] = state_n["eps_eq"] + delta_eps_eq
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
            converged=converged
        )

    def user_defined_tangent(self, stress, state, dlambda, C, state_n, stress_trial=None, strain_inc=None):
        ddsdde = self.calc_ddsdde(state, state_n, stress_trial, dlambda)
        return ddsdde
            
    def calc_residual(self, state_new, state_n, stress_trial, dlambda):
        C = self.elastic_stiffness(state_new)
        theta_new = state_new["theta"]
        beta_new = state_new["beta"]
        R_new = state_new["R"]
        theta_max = state_n["theta_max"]
        s_xi = self.dev(state_new["stress"]) - theta_new - beta_new
        a = self.B + R_new - self.Y
        theta_norm = self.vonmises_norm(theta_new)
        C_k = self.C_1 - (self.C_1 - self.C_2) * smooth_heaviside(theta_max - (self.B - self.Y))
        R_stress = self.default_stress_residual(state_new, dlambda, stress_trial)
        R_theta = theta_new - state_n["theta"] - 2.0 / 3.0 * (
            C_k * a * s_xi - C_k * self.Y * smooth_sqrt(a / theta_norm) * theta_new
        ) * dlambda
        R_beta = beta_new - state_n["beta"] - 2.0 / 3.0 * (
            self.k * self.b * s_xi - self.k * self.Y * beta_new
        ) * dlambda
        R_yield = self.yield_function(state_new)
        r_vector = anp.hstack((R_stress, R_yield, R_theta, R_beta))
        return r_vector

    def calc_fy_fs(self, state):
        stress = state["stress"]
        beta = state["beta"]
        theta = state["theta"]
        R = state["R"]
        eta = stress - theta - beta
        return self.P @ eta

    def calc_fy_ft(self, state):
        stress = state["stress"]
        beta = state["beta"]
        theta = state["theta"]
        R = state["R"]
        eta = stress - theta - beta
        return -self.P @ eta

    def calc_fy_fb(self, state):
        stress = state["stress"]
        beta = state["beta"]
        theta = state["theta"]
        R = state["R"]
        eta = stress - theta - beta
        return -self.P @ eta

    def calc_fy_fl(self, state):
        return 0.0

    def calc_fe_fs(self, state, dlambda, state_n):
        stress = state["stress"]
        beta = state["beta"]
        theta = state["theta"]
        eps_eq = state["eps_eq"]
        eps_eq_n = state_n["eps_eq"]
        eta = stress - beta - theta
        d_eps_eq = dlambda * np.sqrt(eta @ (self.P @ eta))
        C = self.elastic_stiffness(state)
        f = self._calc_E_factor(eps_eq)
        fb = -self.xi * (1 - self.Ea / self.E) * np.exp(-self.xi * (eps_eq))
        deq_ds = 2 / 3 * dlambda * self.P @ eta / smooth_sqrt(2 / 3 * eta @ self.P @ eta) 
        dC_deq = fb / f * C @ deq_ds
        return np.eye(3) + dlambda * C @ self.P + dlambda * np.outer(dC_deq, self.P @ eta)

    def calc_fe_ft(self, state, dlambda, state_n):
        stress = state["stress"]
        beta = state["beta"]
        theta = state["theta"]
        eps_eq = state["eps_eq"]
        eps_eq_n = state_n["eps_eq"]
        eta = stress - beta - theta
        d_eps_eq = dlambda * np.sqrt(eta @ (self.P @ eta))
        C = self.elastic_stiffness(state)
        f = self._calc_E_factor(eps_eq)
        fb = -self.xi * (1 - self.Ea / self.E) * np.exp(-self.xi * (eps_eq))
        deq_ds = 2 / 3 * dlambda * self.P @ eta / smooth_sqrt(2 / 3 * eta @ self.P @ eta) 
        dC_deq = fb / f * C @ deq_ds
        return - dlambda * C @ self.P - dlambda * np.outer(dC_deq, self.P @ eta)

    def calc_fe_fb(self, state, dlambda, state_n):
        stress = state["stress"]
        beta = state["beta"]
        theta = state["theta"]
        eps_eq = state["eps_eq"]
        eps_eq_n = state_n["eps_eq"]
        eta = stress - beta - theta
        d_eps_eq = dlambda * np.sqrt(eta @ (self.P @ eta))
        C = self.elastic_stiffness(state)
        f = self._calc_E_factor(eps_eq)
        fb = -self.xi * (1 - self.Ea / self.E) * np.exp(-self.xi * (eps_eq))
        deq_ds = 2 / 3 * dlambda * self.P @ eta / smooth_sqrt(2 / 3 * eta @ self.P @ eta) 
        dC_deq = fb / f * C @ deq_ds
        return - dlambda * C @ self.P - dlambda * np.outer(dC_deq, self.P @ eta)

    def calc_fe_fl(self, state, dlambda, state_n):
        stress = state["stress"]
        beta = state["beta"]
        theta = state["theta"]
        eps_eq = state["eps_eq"]
        eps_eq_n = state_n["eps_eq"]
        eta = stress - beta - theta
        d_eps_eq = dlambda * np.sqrt(eta @ (self.P @ eta))
        C = self.elastic_stiffness(state)
        f = self._calc_E_factor(eps_eq)
        fb = -self.xi * (1 - self.Ea / self.E) * np.exp(-self.xi * (eps_eq))
        deq_dl =smooth_sqrt(2 / 3 * eta @ self.P @ eta) 
        dC_dl = fb / f * deq_dl * C
        return C @ self.P @ eta + dlambda * dC_dl @ self.P @ eta

    def _C_k(self, state_n):
        """Kinematic hardening rate.

        ``theta_max`` comes from ``state_n``: the residual holds C_k fixed at
        its step-start value, so its Jacobian must too.  Reading the updated
        ``theta_max`` here makes the tangent jump by C_1/C_2 (×10) on steps
        whose theta_max crosses B − Y.
        """
        return self.C_1 - (self.C_1 - self.C_2) * smooth_heaviside(
            state_n["theta_max"] - (self.B - self.Y)
        )

    def calc_ft_fs(self, state, dlambda, state_n):
        a = self.B + state["R"] - self.Y
        C_k = self._C_k(state_n)
        return - 2 / 3 * C_k * a * dlambda * np.eye(3)

    def calc_ft_ft(self, state, dlambda, state_n):
        theta = state["theta"]
        theta_bar = self.vonmises_norm(theta)
        a = self.B + state["R"] - self.Y
        C_k = self._C_k(state_n)
        f1 = (
            1 + 
            2 / 3 * C_k * a * dlambda +
            2 / 3 * C_k * self.Y * dlambda * smooth_sqrt(a / theta_bar)
        )
        f2 = - C_k * self.Y * dlambda / 3 / theta_bar * smooth_sqrt(a / theta_bar)
        dthb_dth = np.sqrt(1.5) * self.P @ theta / smooth_sqrt(self.deviatoric_inner_product(theta, theta))
        return f1 * np.eye(3) + f2 * np.outer(theta, dthb_dth)

    def calc_ft_fb(self, state, dlambda, state_n):
        a = self.B + state["R"] - self.Y
        C_k = self._C_k(state_n)
        return 2 / 3 * C_k * a * dlambda * np.eye(3)

    def calc_ft_fl(self, state, dlambda, state_n):
        stress = state["stress"]
        beta = state["beta"]
        theta = state["theta"]
        eta = stress - beta - theta
        theta_bar = self.vonmises_norm(theta)
        R = state["R"]
        R_n = state_n["R"]
        a = self.B + R - self.Y
        C_k = self._C_k(state_n)
        s =  1 / (1 + 2 / 3 * self.k * self.Y * dlambda)
        ds_dl = - 2 / 3 * self.k * self.Y * s * s
        # R only evolves while the stagnation surface is active (the g_flag gate in
        # update_state), so da/dΔλ vanishes when it is not.  Same test as
        # YUKinematic3D._prepare_Rtheta.
        active = float(abs(R - R_n) > 1.0e-15 * max(abs(R_n), 1.0))
        da_dl = active * (ds_dl * (
            R_n + 2 / 3 * self.k * self.Y * self.Rsat * dlambda
        ) + 2 / 3 * s * self.k * self.Y * self.Rsat)
        f1 = - 2 / 3 * C_k * dlambda * da_dl
        f2 = - 2 / 3 * C_k * a
        f3 = 2 / 3 * C_k * self.Y * (
            dlambda / 2 * smooth_sqrt(1 / a / theta_bar) * da_dl +
            smooth_sqrt(a / theta_bar)
        )
        return f1 * eta + f2 * eta + f3 * theta

    def calc_fb_fs(self, state, dlambda):
        return - 2 / 3 * self.k * self.b * dlambda * np.eye(3)

    def calc_fb_ft(self, state, dlambda):
        return 2 / 3 * self.k * self.b * dlambda * np.eye(3)

    def calc_fb_fb(self, state, dlambda):
        return (1 + 2 / 3 * self.k * self.b * dlambda + 2 / 3 * self.k * self.Y * dlambda) * np.eye(3)

    def calc_fb_fl(self, state, dlambda):
        stress = state["stress"]
        beta = state["beta"]
        theta = state["theta"]
        eta = stress - beta - theta
        return 2 / 3 * (self.k * self.Y * beta - self.k * self.b * eta)

    def calc_jacobian(self, state_new, state_n, stress_trial, dlambda):
        Rs_s = self.calc_fe_fs(state_new, dlambda, state_n)
        Rs_b = self.calc_fe_fb(state_new, dlambda, state_n)
        Rs_t = self.calc_fe_ft(state_new, dlambda, state_n)
        Rs_l = self.calc_fe_fl(state_new, dlambda, state_n)
        Rs = np.hstack((Rs_s, Rs_l[:, np.newaxis], Rs_t, Rs_b))
        Rb_s = self.calc_fb_fs(state_new, dlambda)
        Rb_b = self.calc_fb_fb(state_new, dlambda)
        Rb_t = self.calc_fb_ft(state_new, dlambda)
        Rb_l = self.calc_fb_fl(state_new, dlambda)
        Rb = np.hstack((Rb_s, Rb_l[:, np.newaxis], Rb_t, Rb_b))
        Rt_s = self.calc_ft_fs(state_new, dlambda, state_n)
        Rt_b = self.calc_ft_fb(state_new, dlambda, state_n)
        Rt_t = self.calc_ft_ft(state_new, dlambda, state_n)
        Rt_l = self.calc_ft_fl(state_new, dlambda, state_n)
        Rt = np.hstack((Rt_s, Rt_l[:, np.newaxis], Rt_t, Rt_b))
        Ry_s = self.calc_fy_fs(state_new)
        Ry_b = self.calc_fy_fb(state_new)
        Ry_t = self.calc_fy_ft(state_new)
        Ry_l = self.calc_fy_fl(state_new)
        Ry = np.hstack((Ry_s, Ry_l, Ry_t, Ry_b))
        return np.vstack((Rs, Ry.reshape(1, -1), Rt, Rb))

    def calc_ddsdde(self, state_new, state_n, stress_trial, dlambda):
        # C_n (step-start stiffness) is used for the rhs scaling in the consistent
        # tangent: J·dx/dε = [C_n; 0; ...] → ddsdde = J^{-1}[0:3,0:3]·C_n.
        # Using C(state_new) here was a bug when E varies with eps_eq.
        Cinv = np.linalg.inv(self.elastic_stiffness(state_n))
        Rs_s = Cinv @ self.calc_fe_fs(state_new, dlambda, state_n)
        Rs_b = Cinv @ self.calc_fe_fb(state_new, dlambda, state_n)
        Rs_t = Cinv @ self.calc_fe_ft(state_new, dlambda, state_n)
        Rs_l = Cinv @ self.calc_fe_fl(state_new, dlambda, state_n)
        Rs = np.hstack((Rs_s, Rs_l[:, np.newaxis], Rs_t, Rs_b))
        Rb_s = self.calc_fb_fs(state_new, dlambda)
        Rb_b = self.calc_fb_fb(state_new, dlambda)
        Rb_t = self.calc_fb_ft(state_new, dlambda)
        Rb_l = self.calc_fb_fl(state_new, dlambda)
        Rb = np.hstack((Rb_s, Rb_l[:, np.newaxis], Rb_t, Rb_b))
        Rt_s = self.calc_ft_fs(state_new, dlambda, state_n)
        Rt_b = self.calc_ft_fb(state_new, dlambda, state_n)
        Rt_t = self.calc_ft_ft(state_new, dlambda, state_n)
        Rt_l = self.calc_ft_fl(state_new, dlambda, state_n)
        Rt = np.hstack((Rt_s, Rt_l[:, np.newaxis], Rt_t, Rt_b))
        Ry_s = self.calc_fy_fs(state_new)
        Ry_b = self.calc_fy_fb(state_new)
        Ry_t = self.calc_fy_ft(state_new)
        Ry_l = self.calc_fy_fl(state_new)
        Ry = np.hstack((Ry_s, Ry_l, Ry_t, Ry_b))
        jac = np.vstack((Rs, Ry.reshape(1, -1), Rt, Rb))
        jac_inv = np.linalg.inv(jac)
        return jac_inv[:3, :3]




class YUKinematic1D(YUKinematic):
    def __init__(self, *, E: float, nu: float, Y: float, C_1: float, C_2: float,
                 B: float, Rsat: float, k: float, b: float,
                 h: float, Ea: float, xi: float):
        super().__init__(dimension=UNIAXIAL_1D, E=E, nu=nu, Y=Y, C_1=C_1, C_2=C_2,
                 B=B, Rsat=Rsat, k=k, b=b, h=h, Ea=Ea, xi=xi)
