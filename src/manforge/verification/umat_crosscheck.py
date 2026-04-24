"""Multi-step crosscheck: Python Driver vs Fortran UMAT.

Provides :func:`crosscheck_umat`, a harness that drives both a Python
constitutive model (via any :class:`~manforge.simulation.driver.DriverBase`
subclass) and a compiled Fortran UMAT through the same loading history and
compares the resulting stress trajectories element-wise.

Intended audience
-----------------
Both library-internal validation (J2 reference implementation) and
external users who build their own model + UMAT:

::

    # External user workflow
    # 1. Compile your UMAT core with the manforge CLI
    #    $ uv run manforge build my_model.f90 --name my_model

    from manforge.verification import crosscheck_umat, FortranUMAT
    from manforge.simulation import StrainDriver
    from manforge.simulation.types import FieldHistory, FieldType
    import numpy as np

    model   = MyModel(E=210_000.0, nu=0.3, sigma_y0=250.0, H=1_000.0)
    fortran = FortranUMAT("my_model")
    load    = FieldHistory(FieldType.STRAIN, "eps",
                           np.linspace([0]*6, [1e-3,0,0,0,0,0], 20))

    result = crosscheck_umat(
        StrainDriver(), model, fortran,
        umat_subroutine="my_model_core",
        load=load,
        param_fn=lambda m: (m.E, m.nu, m.sigma_y0, m.H),
    )
    assert result.passed
    print(f"max stress rel error = {result.max_stress_rel_err:.2e}")

Fortran subroutine requirements
--------------------------------
``umat_subroutine`` must be a **f2py-callable core-logic subroutine**, NOT
the full ABAQUS ``umat`` entry point (which has 20+ ``inout`` arguments).
The expected interface is::

    subroutine my_model_core(<material params>, stress_in, <state vars in>, dstran, &
                             stress_out, <state vars out> [, ddsdde])
        real(8), intent(in)  :: <material params>       ! scalars
        real(8), intent(in)  :: stress_in(ntens)
        real(8), intent(in)  :: <state vars in>         ! scalar or array each
        real(8), intent(in)  :: dstran(ntens)
        real(8), intent(out) :: stress_out(ntens)
        real(8), intent(out) :: <state vars out>        ! same shapes as inputs
        real(8), intent(out) :: ddsdde(ntens, ntens)    ! optional, may be absent

The recommended pattern (used by the J2 reference) is to write the algorithm
in a standalone ``my_model_core`` subroutine and have the full ``umat``
delegate to it — see ``fortran/j2_isotropic_3d.f90`` for an example.

Default state hook convention
------------------------------
When ``state_to_args`` / ``parse_umat_return`` are omitted the defaults assume:

* **Pack**: each entry in ``model.state_names`` is passed as one argument —
  scalar → ``float``, ndarray → ``np.float64`` array of the same shape.
* **Unpack**: Fortran returns ``(stress, state[0], state[1], ..., <trailing>)``,
  i.e. state variables follow ``stress`` in ``state_names`` order.  Trailing
  outputs (e.g. ``ddsdde``) are discarded.

If your UMAT uses a different argument order, supply explicit hooks::

    result = crosscheck_umat(
        ...,
        state_to_args=lambda s: (s["ep"], s["alpha"]),           # custom order
        parse_umat_return=lambda ret: (ret[0], {"ep": float(ret[2]),
                                                "alpha": np.asarray(ret[1])}),
    )
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np


@dataclass
class UMATCrosscheckResult:
    """Result of a multi-step Python vs Fortran UMAT crosscheck.

    Attributes
    ----------
    passed : bool
        True when ``max_stress_rel_err < rtol``.
    max_stress_rel_err : float
        Maximum element-wise relative stress error across all steps,
        computed as ``max |s_f - s_py| / (|s_py| + 1)``.
    stress_py : np.ndarray, shape (N, ntens)
        Stress history from the Python driver.
    stress_f : np.ndarray, shape (N, ntens)
        Stress history from the Fortran UMAT.
    strain : np.ndarray, shape (N, ntens)
        Cumulative total strain applied at each step (identical for both
        sides; shown for convenience).
    """

    passed: bool
    max_stress_rel_err: float
    stress_py: np.ndarray
    stress_f: np.ndarray
    strain: np.ndarray


def _default_state_to_args(
    state: dict[str, Any], state_names: list[str]
) -> tuple:
    out = []
    for name in state_names:
        v = state[name]
        if np.ndim(v) == 0:
            out.append(float(v))
        else:
            out.append(np.asarray(v, dtype=np.float64))
    return tuple(out)


def _default_parse_umat_return(
    ret: tuple,
    state_names: list[str],
    initial_state: dict[str, Any],
) -> tuple[np.ndarray, dict[str, Any]]:
    stress = np.asarray(ret[0], dtype=np.float64)
    state_out: dict[str, Any] = {}
    for i, name in enumerate(state_names, start=1):
        ref = initial_state[name]
        v = ret[i]
        if np.ndim(ref) == 0:
            state_out[name] = float(v)
        else:
            state_out[name] = np.asarray(v, dtype=np.float64).reshape(
                np.asarray(ref).shape
            )
    return stress, state_out


def crosscheck_umat(
    driver,
    model,
    fortran,
    *,
    umat_subroutine: str,
    load,
    param_fn: Callable[[Any], tuple],
    state_to_args: Callable[[dict], tuple] | None = None,
    parse_umat_return: Callable[[tuple], tuple[np.ndarray, dict]] | None = None,
    rtol: float = 1e-6,
) -> UMATCrosscheckResult:
    """Compare Python Driver and Fortran UMAT over a multi-step load history.

    Both sides receive the same strain increment at every step.  The Python
    driver runs first (via ``driver.iter_run``); at each converged step its
    cumulative strain is differenced to obtain ``dstran``, which is then
    passed to the Fortran UMAT.  This approach works for both
    :class:`~manforge.simulation.driver.StrainDriver` (direct strain control)
    and :class:`~manforge.simulation.driver.StressDriver` (the driver's inner
    NR loop solves for the converged ``dstran``; the UMAT is called once per
    step with that converged increment).

    Parameters
    ----------
    driver
        An instantiated driver, e.g. ``StrainDriver()`` or ``StressDriver()``.
    model
        Python constitutive model (``MaterialModel`` subclass).
    fortran
        :class:`~manforge.verification.FortranUMAT` wrapping the compiled
        f2py module.
    umat_subroutine : str
        Name of the f2py-callable subroutine inside *fortran*'s module.
    load
        :class:`~manforge.simulation.types.FieldHistory` with the loading
        history (``FieldType.STRAIN`` or ``FieldType.STRESS``).
    param_fn : callable
        ``param_fn(model) -> tuple`` — material parameters in the order
        expected by *umat_subroutine*.  Required because parameter ordering
        is UMAT-specific.
    state_to_args : callable, optional
        ``state_to_args(state_dict) -> tuple`` — packs the Python state dict
        into positional arguments for the Fortran call.  Defaults to
        ``state_names``-order, one argument per variable.
    parse_umat_return : callable, optional
        ``parse_umat_return(ret) -> (stress_ndarray, state_dict)`` — unpacks
        the raw f2py return tuple.  Defaults to ``(stress, *state in
        state_names order, <trailing discarded>)``.
    rtol : float, optional
        Relative tolerance for the ``passed`` flag.  Default ``1e-6``.

    Returns
    -------
    UMATCrosscheckResult
    """
    state_names: list[str] = list(model.state_names)
    initial_state: dict = model.initial_state()

    _state_to_args = (
        state_to_args
        if state_to_args is not None
        else lambda s: _default_state_to_args(s, state_names)
    )
    _parse_umat_return = (
        parse_umat_return
        if parse_umat_return is not None
        else lambda ret: _default_parse_umat_return(ret, state_names, initial_state)
    )

    ntens: int = model.stress_state.ntens
    stress_f = np.zeros(ntens, dtype=np.float64)
    state_f: dict = model.initial_state()
    eps_prev = np.zeros(ntens, dtype=np.float64)

    stress_py_hist: list[np.ndarray] = []
    stress_f_hist: list[np.ndarray] = []
    strain_hist: list[np.ndarray] = []

    for step in driver.iter_run(model, load):
        dstran = np.asarray(step.strain, dtype=np.float64) - eps_prev
        eps_prev = np.asarray(step.strain, dtype=np.float64).copy()

        state_tup = _state_to_args(state_f)
        returns = fortran.call(
            umat_subroutine,
            *param_fn(model),
            stress_f,
            *state_tup,
            dstran,
        )
        stress_f, state_f = _parse_umat_return(returns)

        stress_py_hist.append(np.asarray(step.result.stress, dtype=np.float64))
        stress_f_hist.append(stress_f.copy())
        strain_hist.append(np.asarray(step.strain, dtype=np.float64).copy())

    stress_py = np.vstack(stress_py_hist)
    stress_f_arr = np.vstack(stress_f_hist)
    strain = np.vstack(strain_hist)

    max_err = float(
        np.max(np.abs(stress_f_arr - stress_py) / (np.abs(stress_py) + 1.0))
    )

    return UMATCrosscheckResult(
        passed=max_err < rtol,
        max_stress_rel_err=max_err,
        stress_py=stress_py,
        stress_f=stress_f_arr,
        strain=strain,
    )
