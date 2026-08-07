"""Shared setup for the YU UMAT timing harnesses.

PARAMS mirrors tests/benchmarks/yu_kinematic/conftest.py, duplicated on purpose:
importing a pytest conftest from a standalone script is fragile, and these
numbers only need to be a representative parameter set, not the same one the
correctness tests use.
"""

import numpy as np

from manforge.simulation.driver import StrainDriver
from manforge.simulation.integrator import PythonAnalyticalIntegrator

PARAMS = dict(
    E=206_000, nu=0.3, Y=360.0, C_1=2000.0, C_2=200.0,
    B=435.0, Rsat=255.0, k=26.0, b=66.0, h=0.4, Ea=159_000, xi=61.0,
)

N_STEPS = 40
STEP = 30  # well inside the plastic range of the pull below


def props(model):
    """PROPS in the order the Fortran subroutines expect."""
    return (model.E, model.nu, model.Y, model.B, model.C_1, model.C_2,
            model.Rsat, model.k, model.b, model.h, model.Ea, model.xi)


def plastic_point(model_cls, ntens):
    """Step-start state and increment for a plastic step of a uniaxial pull.

    Timing has to happen at a plastic point: the elastic branch returns before
    the NR loop, so an elastic state would report a misleadingly small number
    that says nothing about the cost being measured.
    """
    model = model_cls(**PARAMS)
    driver = StrainDriver(PythonAnalyticalIntegrator(model))
    data = np.zeros((N_STEPS, ntens))
    data[:, 0] = np.linspace(0.0, 8e-3, N_STEPS)

    state_n = None
    for step in driver.iter_run(data):
        if step.i == STEP:
            assert step.result.is_plastic, "bench point fell on the elastic branch"
            return model, state_n, data[STEP] - data[STEP - 1], step.result
        state_n = step.result.state
    raise RuntimeError(f"step {STEP} not reached")
