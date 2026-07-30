import numpy as np
import numpy.testing as npt

from manforge.models.yu_kinematic import YUKinematicPS
from manforge.simulation.integrator import PythonNumericalIntegrator
from manforge.simulation.driver import MixedDriver, StrainDriver
from manforge.simulation.types import FieldHistory, FieldType
from manforge.verification import JacobianChecker



class YUChecker:

    def __init__(self, model, result):
        self.model = model
        self.state = result.state
        self.state_n = result._state_n
        self.dlambda = result.dlambda
        self.stress_trial = result.stress_trial
        self.C = model.elastic_stiffness(result.state)
        self.C_inv = np.linalg.inv(self.C)
        self.I_dev = model.I_dev()
        self.stress = self.state["stress"]
        self.theta = self.state["theta"]
        self.beta = self.state["beta"]
        self.eps_eq = self.state["eps_eq"]
        self.R = self.state["R"]
        self.R_n = self.state_n["R"]
        self.dev_stress = model.dev(self.stress)
        self.xi = self.stress - self.theta - self.beta
        self.theta_max = self.state_n["theta_max"]
    
    def assertion_array(self, actual, expected, atol=1e-10):
        assertion = False
        try:
            npt.assert_allclose(actual, expected, atol=atol)
            assertion = True
        except Exception as e:
            print(e)
        return assertion

    def check_all(self, jac):
        # Rstress
        assert_Rs_s = self.assertion_array(self.model.dRstress_dstress(self.C, self.xi, self.dlambda), jac.part["stress"]["stress"])
        assert_Rs_b = self.assertion_array(self.model.dRstress_dbeta(self.C, self.xi, self.dlambda), jac.part["stress"]["beta"])
        assert_Rs_t = self.assertion_array(self.model.dRstress_dtheta(self.C, self.xi, self.dlambda), jac.part["stress"]["theta"])
        assert_Rs_l = self.assertion_array(self.model.dRstress_dlambda(self.C, self.xi, self.eps_eq, self.dlambda), jac.part["stress"]["dlambda"])
        # Rbeta
        assert_Rb_s = self.assertion_array(self.model.dRbeta_dstress(self.dlambda), jac.part["beta"]["stress"])
        assert_Rb_b = self.assertion_array(self.model.dRbeta_dbeta(self.dlambda), jac.part["beta"]["beta"])
        assert_Rb_t = self.assertion_array(self.model.dRbeta_dtheta(self.dlambda), jac.part["beta"]["theta"])
        assert_Rb_l = self.assertion_array(self.model.dRbeta_dlambda(self.xi, self.beta, self.dlambda), jac.part["beta"]["dlambda"])
        # Rtheta
        assert_Rt_s = self.assertion_array(self.model.dRtheta_dstress(self.theta, self.theta_max, self.R, self.R_n, self.dlambda), jac.part["theta"]["stress"])
        assert_Rt_b = self.assertion_array(self.model.dRtheta_dbeta(self.theta, self.theta_max, self.R, self.R_n, self.dlambda), jac.part["theta"]["beta"])
        assert_Rt_t = self.assertion_array(self.model.dRtheta_dtheta(self.theta, self.theta_max, self.R, self.R_n, self.dlambda), jac.part["theta"]["theta"])
        assert_Rt_l = self.assertion_array(self.model.dRtheta_dlambda(self.xi, self.theta, self.theta_max, self.R, self.R_n, self.dlambda), jac.part["theta"]["dlambda"])
        # stress
        assert_Rl_s = self.assertion_array(self.model.dRyield_dstress(self.xi), jac.part["dlambda"]["stress"])
        assert_Rl_b = self.assertion_array(self.model.dRyield_dbeta(self.xi), jac.part["dlambda"]["beta"])
        assert_Rl_t = self.assertion_array(self.model.dRyield_dtheta(self.xi), jac.part["dlambda"]["theta"])
        assert_Rl_l = self.assertion_array(self.model.dRyield_dlambda(), jac.part["dlambda"]["dlambda"])
        return np.array([
            [assert_Rs_s, assert_Rs_b, assert_Rs_t, assert_Rs_l],
            [assert_Rb_s, assert_Rb_b, assert_Rb_t, assert_Rb_l],
            [assert_Rt_s, assert_Rt_b, assert_Rt_t, assert_Rt_l],
            [assert_Rl_s, assert_Rl_b, assert_Rl_t, assert_Rl_l],
        ])


modelps = YUKinematicPS(
    E=206_000, nu=0.3, Y=360.0, C_1=2000.0, C_2=200.0,
    B=435.0, Rsat=255.0, k=26.0, b=66.0,
    h=0.4, Ea=159_000, xi=61.0)

idx = 0
intps = PythonNumericalIntegrator(modelps)
driverps = MixedDriver(intps, prescribed_strain_idx=[idx])

hist = FieldHistory.cyclic_strain([0.05, -0.05, 0.05, -0.05], n_per_segment=50)
run_g = driverps.iter_run(hist)

strains = []
stress = []
i = 0
for g in run_g:
    if g.result.is_plastic:
        print(f"check {i}")
        yu_checker = YUChecker(modelps, g.result)
        jac = JacobianChecker(modelps).compute(g.result, g.result._state_n)
        #assert_res = yu_checker.check_all(jac)
        i += 1
        stress = yu_checker.stress
        beta = yu_checker.beta 
        theta = yu_checker.theta
        print(stress)
        xi1 = modelps.dev(stress) - beta - theta
        xi2 = stress - beta - theta
        mises1 = modelps.vonmises_norm(xi1)
        mises2 = np.sqrt(modelps.calc_g(xi2) * 3 / 2)
        print(f"mises1: {mises1}\nmises2: {mises2}")
        quit()
        # ddsdde = g.result.ddsdde
        # calced_ddsdde = yu_checker.model.calc_ddsdde(g.result.state, g.result._state_n, g.result.stress_trial, g.result.dlambda)
        # try:
        #     npt.assert_allclose(ddsdde, calced_ddsdde, rtol=1e-06)
        # except Exception as e:
        #     print(e)
        if not assert_res.all():
            print(f"Asertion result: ")
            print(assert_res)
            print(f"dlambda: {g.result.dlambda}")
            print(f"n_iteration: {g.result.n_iterations}")
            print(f"r_hist: {g.result.residual_history}")
        if i == 1:
            quit()

    strains.append(g.strain[idx])
    stress.append(g.result.stress[idx])

from matplotlib import pyplot as plt

fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(strains, stress)
plt.show()

