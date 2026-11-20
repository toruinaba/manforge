"""Tune a model against a test record by hand — a marimo notebook.

Move a slider, the model is run again and drawn over the record.  When the
curve is close, fit from where the sliders stand.

Usage
-----
    uv sync --extra notebook
    uv run marimo edit examples/notebook_tuning.py     # or: marimo run ...
"""

import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import warnings
    from pathlib import Path

    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np

    import returnmap.models as models
    from returnmap.fitting import FitTarget, ParameterSpace, SlowFitWarning, fit_params
    from returnmap.simulation import FieldHistory, FieldType, PythonIntegrator, StrainDriver

    return (
        FieldHistory,
        FieldType,
        FitTarget,
        ParameterSpace,
        Path,
        PythonIntegrator,
        SlowFitWarning,
        StrainDriver,
        fit_params,
        mo,
        models,
        np,
        plt,
        warnings,
    )


@app.cell
def _(mo):
    mo.md("""
    # Tuning a model against a test record

    The record is a CSV with a header line and two columns: axial strain and
    axial stress.  The models are the uniaxial (`*1D`) ones of
    `returnmap.models`, so the stress is uniaxial stress — a
    tension-compression test.
    """)
    return


@app.cell
def _(Path, mo, models):
    record_path = mo.ui.text(
        value=str(Path(mo.notebook_dir() or ".") / "data" / "cyclic_tension_compression.csv"),
        label="Test record (CSV)", full_width=True,
    )
    model_name = mo.ui.dropdown(
        options=[name for name in models.__all__ if name.endswith("1D")],
        value="AFKinematic1D", label="Model",
    )
    every = mo.ui.slider(1, 10, value=3, label="Use every n-th sample", show_value=True)
    mo.vstack([record_path, mo.hstack([model_name, every], justify="start", gap=2)])
    return every, model_name, record_path


@app.cell
def _(every, np, record_path):
    record = np.loadtxt(record_path.value, delimiter=",", skiprows=1)
    strain, stress = record[:: every.value, 0], record[:: every.value, 1]
    return record, strain, stress


@app.cell
def _(mo, model_name, models):
    model_cls = getattr(models, model_name.value)

    def slider_for(name):
        meta = model_cls.param_meta[name]
        lower, upper = meta.bounds
        step = (upper - lower) / 400.0
        if meta.default > 0.0:
            # A quarter to three times the typical value, inside the plausible
            # range, in steps of 1 % of it — so the typical value is on a step.
            step = meta.default / 100.0
            lower, upper = max(lower, 25 * step), min(upper, 300 * step)
        unit = f" [{meta.unit}]" if meta.unit else ""
        return mo.ui.slider(
            start=lower, stop=upper, step=step, value=meta.default,
            label=f"`{name}`{unit} — {meta.doc}",
            debounce=True, include_input=True, full_width=True,
        )

    sliders = mo.ui.dictionary({name: slider_for(name) for name in model_cls.param_names})
    mo.vstack([sliders[name] for name in model_cls.param_names])
    return model_cls, sliders


@app.cell
def _(
    FieldHistory,
    FieldType,
    PythonIntegrator,
    StrainDriver,
    model_cls,
    np,
    sliders,
    strain,
    stress,
):
    model = model_cls(**sliders.value)
    # Non-convergence is a result here, not an error: a slider can easily be
    # dragged to values the model cannot be integrated at.
    integrator = PythonIntegrator(model, raise_on_nonconverged=False)
    result = StrainDriver(integrator).run(FieldHistory(FieldType.STRAIN, strain))
    n_done = len(result.stress)
    rms = (
        float(np.sqrt(np.mean((result.stress[:, 0] - stress[:n_done]) ** 2)))
        if n_done else float("nan")
    )
    return model, n_done, result, rms


@app.cell
def _(mo, model, n_done, plt, record, result, rms, strain):
    INK, SECONDARY, MUTED, GRID, AXIS, SURFACE = (
        "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb")

    fig, ax = plt.subplots(figsize=(7.5, 4.6), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    ax.scatter(record[:, 0] * 100, record[:, 1], s=8, color=MUTED, alpha=0.6,
               linewidths=0, label="test record", zorder=2)
    ax.plot(result.strain[:, 0] * 100, result.stress[:, 0], color="#2a78d6",
            linewidth=2, label=type(model).__name__, zorder=3)
    ax.axhline(0.0, color=AXIS, linewidth=0.8, zorder=1)
    ax.axvline(0.0, color=AXIS, linewidth=0.8, zorder=1)
    ax.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
    ax.tick_params(colors=MUTED, labelcolor=SECONDARY)
    ax.set_xlabel("Axial strain [%]", color=SECONDARY)
    ax.set_ylabel("Axial stress [MPa]", color=SECONDARY)
    ax.legend(frameon=False, labelcolor=INK, loc="lower right")
    fig.tight_layout()

    status = (
        mo.md(f"**RMS stress error: {rms:.1f} MPa** over {n_done} steps")
        if n_done == len(strain) else
        mo.callout(
            mo.md(f"The model stopped converging at step {n_done} of {len(strain)}. "
                  f"RMS over the completed steps: {rms:.1f} MPa."),
            kind="warn",
        )
    )
    mo.vstack([status, fig])
    return


@app.cell
def _(mo, model):
    arguments = ", ".join(f"{name}={value:g}" for name, value in model.params.items())
    mo.md(f"""
    To continue in a script:

    ```python
    from returnmap.models import {type(model).__name__}
    model = {type(model).__name__}({arguments})
    ```
    """)
    return


@app.cell
def _(mo, model_cls):
    free = mo.ui.multiselect(
        options=list(model_cls.param_names),
        value=[name for name in model_cls.param_names if name not in ("E", "nu")],
        label="Parameters to fit (the others stay where their sliders are)",
    )
    fit_button = mo.ui.run_button(label="Fit from here")
    mo.vstack([
        mo.md("## Fit\nStarts from the slider values and searches inside the slider ranges."),
        free, fit_button,
    ])
    return fit_button, free


@app.cell
def _(
    FitTarget,
    ParameterSpace,
    SlowFitWarning,
    fit_button,
    fit_params,
    free,
    mo,
    model,
    np,
    sliders,
    strain,
    stress,
    warnings,
):
    mo.stop(not fit_button.value, mo.md("*Press the button to fit.*"))
    mo.stop(not free.value, mo.md("*Choose at least one parameter to fit.*"))

    space = ParameterSpace(
        free={
            name: (model.params[name], (sliders[name].start, sliders[name].stop))
            for name in free.value
        },
        fixed={name: value for name, value in model.params.items() if name not in free.value},
    )
    with warnings.catch_warnings():
        # The time one evaluation takes is visible on the progress bar instead.
        warnings.simplefilter("ignore", SlowFitWarning)
        with mo.status.spinner(title="Fitting…") as spinner:
            fit = fit_params(
                model, [FitTarget.stress_strain(strain, stress)], space,
                nr_jacobian="broyden", tol=1e-6,
                progress=lambda n, residual, best: spinner.update(
                    subtitle=f"evaluation {n}: RMS {np.sqrt(best):.2f} MPa"),
            )

    fitted_arguments = ", ".join(f"{k}={v:g}" for k, v in fit.model_params.items())
    mo.vstack([
        mo.md(f"**RMS stress error after the fit: {np.sqrt(fit.residual):.2f} MPa** "
              f"({fit.n_evaluations} evaluations)"),
        mo.ui.table(
            [{"parameter": name,
              "sliders": float(f"{model.params[name]:.5g}"),
              "fitted": float(f"{fit.model_params[name]:.5g}")}
             for name in free.value],
            selection=None, show_download=False, show_search=False,
        ),
        mo.md(f"```python\nmodel = {type(model).__name__}({fitted_arguments})\n```"),
    ])
    return


if __name__ == "__main__":
    app.run()
