from datetime import datetime
import numpy as np
import numpy.testing as npt

from manforge.models.yu_kinematic import YUKinematicPS
from manforge.simulation.integrator import PythonNumericalIntegrator, PythonAnalyticalIntegrator
from manforge.simulation.driver import MixedDriver, StrainDriver
from manforge.simulation.types import FieldHistory, FieldType
from manforge.verification import JacobianChecker


modelps = YUKinematicPS(
    E=206_000, nu=0.3, Y=360.0, C_1=2000.0, C_2=200.0,
    B=435.0, Rsat=255.0, k=26.0, b=66.0,
    h=0.4, Ea=159_000, xi=61.0)

d_id = 2
intps_a = PythonAnalyticalIntegrator(modelps)
driverps_a = MixedDriver(intps_a, prescribed_strain_idx=[d_id])

intps_n = PythonNumericalIntegrator(modelps)
driverps_n = MixedDriver(intps_n, prescribed_strain_idx=[d_id])
hist = FieldHistory.cyclic_strain([0.05, -0.05, 0.05, -0.05], n_per_segment=50)

start_n = datetime.now()
print("Numerical calculation start")
res_n = driverps_n.run(hist)
end_n = datetime.now()
t_n = end_n - start_n
print(f"Numerical calculation end. total time: {t_n}")

start_a = datetime.now()
print("Analytical calculation start")
res_a = driverps_a.run(hist)
end_a = datetime.now()
t_a = end_a - start_a
print(f"Analytical calculation end. total time: {t_a}")

for idx in range(1, len(res_n.step_results)):
    step = res_n.step_results[idx]
    state = step.state
    state_n = res_n.step_results[idx - 1].state
    dlambda = step.dlambda
    if step.is_plastic:
        jac = JacobianChecker(modelps).compute(step, state_n)
        fy_fsn = jac.part["dlambda"]["stress"]
        fy_fsa = modelps.calc_fy_fs(state)
        npt.assert_allclose(fy_fsn, fy_fsa)
        fy_ftn = jac.part["dlambda"]["theta"]
        fy_fta = modelps.calc_fy_ft(state)
        npt.assert_allclose(fy_ftn, fy_fta)
        fy_fbn = jac.part["dlambda"]["beta"]
        fy_fba = modelps.calc_fy_fb(state)
        npt.assert_allclose(fy_fbn, fy_fba)
        fy_fln = jac.part["dlambda"]["dlambda"]
        fy_fla = modelps.calc_fy_fl(state)
        npt.assert_allclose(fy_fln, fy_fla)
        fe_fsn = jac.part["stress"]["stress"]
        fe_fsa = modelps.calc_fe_fs(state, dlambda, state_n)
        npt.assert_allclose(fe_fsn, fe_fsa)
        fe_ftn = jac.part["stress"]["theta"]
        fe_fta = modelps.calc_fe_ft(state, dlambda, state_n)
        npt.assert_allclose(fe_ftn, fe_fta)
        fe_fbn = jac.part["stress"]["beta"]
        fe_fba = modelps.calc_fe_fb(state, dlambda, state_n)
        npt.assert_allclose(fe_fbn, fe_fba)
        fe_fln = jac.part["stress"]["dlambda"]
        fe_fla = modelps.calc_fe_fl(state, dlambda, state_n)
        npt.assert_allclose(fe_fln, fe_fla)
        ft_fsn = jac.part["theta"]["stress"]
        ft_fsa = modelps.calc_ft_fs(state, dlambda)
        npt.assert_allclose(ft_fsn, ft_fsa)
        ft_ftn = jac.part["theta"]["theta"]
        ft_fta = modelps.calc_ft_ft(state, dlambda)
        npt.assert_allclose(ft_ftn, ft_fta)
        ft_fbn = jac.part["theta"]["beta"]
        ft_fba = modelps.calc_ft_fb(state, dlambda)
        npt.assert_allclose(ft_fbn, ft_fba)
        ft_fln = jac.part["theta"]["dlambda"]
        ft_fla = modelps.calc_ft_fl(state, dlambda, state_n)
        npt.assert_allclose(ft_fln, ft_fla)
        fb_fsn = jac.part["beta"]["stress"]
        fb_fsa = modelps.calc_fb_fs(state, dlambda)
        npt.assert_allclose(fb_fsn, fb_fsa)
        fb_ftn = jac.part["beta"]["theta"]
        fb_fta = modelps.calc_fb_ft(state, dlambda)
        npt.assert_allclose(fb_ftn, fb_fta)
        fb_fbn = jac.part["beta"]["beta"]
        fb_fba = modelps.calc_fb_fb(state, dlambda)
        npt.assert_allclose(fb_fbn, fb_fba)
        fb_fln = jac.part["beta"]["dlambda"]
        fb_fla = modelps.calc_fb_fl(state, dlambda)
        npt.assert_allclose(fb_fbn, fb_fba)
        break

from matplotlib import pyplot as plt

fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(res_n.strain[:, d_id], res_n.stress[:, d_id], label="Numerical")
ax.plot(res_a.strain[:, d_id], res_a.stress[:, d_id], label="Analytical")
ax.legend()
plt.show()
#
