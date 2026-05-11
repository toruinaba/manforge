"""Mixin classes for Armstrong-Frederick and Ohno-Wang kinematic hardening.

Both mixins implement the physics in terms of three abstract operators that
stress-state base classes provide:

* ``vonmises_relative(stress, alpha)`` — σ_vm(σ − α) with α deviatoric
* ``flow_direction(stress, alpha)``    — n̂ = ∂f/∂σ
* ``alpha_norm(alpha)``               — ‖α‖_vm with α deviatoric

The AF evolution equation in original form:

    dα = (2/3) C_k dp n̂ − γ dp α

gives, after backward-Euler discretisation:

    α_{n+1} = (α_n + (2/3) C_k Δλ n̂) / (1 + γ Δλ)

The OW variant weights the dynamic recovery by the current backstress norm:

    R_α = α_{n+1} − α_n − (2/3) C_k Δλ n̂ + γ Δλ ‖α_{n+1}‖ α_{n+1} = 0

The (2/3) factor is explicit here; it combines with (3/2) inside ``n̂``
to recover the classic kinematic-hardening slope C_k under uniaxial loading.

Note on 1D α storage
---------------------
In 1D the stored quantity is the *deviatoric component* α11_dev, not the
effective backstress α_eff = (3/2)·α11_dev used in some formulations.
The operators in ``MaterialModel1D`` account for this convention so that the
unified update equations above produce physically correct results in all
dimensions.
"""

from __future__ import annotations


class KinematicAFMixin:
    """Armstrong-Frederick kinematic hardening — dimension-agnostic physics.

    Requires the concrete class to also inherit from a stress-state base
    (``MaterialModel3D``, ``MaterialModelPS``, or ``MaterialModel1D``) which
    provides ``vonmises_relative``, ``flow_direction``, and ``alpha_norm``.

    The concrete subclass must declare ``alpha`` and ``ep`` as state fields
    and provide ``C_k``, ``gamma``, and ``sigma_y0`` as instance attributes.
    """

    def yield_function(self, state) -> object:
        """J2 yield function: f = σ_vm(σ − α) − σ_y0."""
        return self.vonmises_relative(state["stress"], state["alpha"]) - self.sigma_y0  # type: ignore[attr-defined]

    def update_state(self, dlambda, state_n, state_trial) -> list:
        """Armstrong-Frederick backstress update (backward Euler, original form).

        α_{n+1} = (α_n + (2/3) C_k Δλ n̂) / (1 + γ Δλ)

        where n̂ = flow_direction(σ_trial, α_n) is the unit normal evaluated at
        the trial relative stress.
        """
        alpha_n = state_n["alpha"]
        n_hat = self.flow_direction(state_trial["stress"], alpha_n)  # type: ignore[attr-defined]
        alpha_new = (
            alpha_n + (2.0 / 3.0) * self.C_k * dlambda * n_hat  # type: ignore[attr-defined]
        ) / (1.0 + self.gamma * dlambda)  # type: ignore[attr-defined]
        return [self.alpha(alpha_new), self.ep(state_n["ep"] + dlambda)]  # type: ignore[attr-defined]


class KinematicOWMixin:
    """Ohno-Wang kinematic hardening — dimension-agnostic physics.

    Requires the concrete class to also inherit from a stress-state base
    (``MaterialModel3D``, ``MaterialModelPS``, or ``MaterialModel1D``) which
    provides ``vonmises_relative``, ``flow_direction``, and ``alpha_norm``.

    The concrete subclass must declare ``stress``, ``alpha``, and ``ep`` as
    ``Implicit`` state fields and provide ``C_k``, ``gamma``, and ``sigma_y0``
    as instance attributes.
    """

    def yield_function(self, state) -> object:
        """J2 yield function: f = σ_vm(σ − α) − σ_y0."""
        return self.vonmises_relative(state["stress"], state["alpha"]) - self.sigma_y0  # type: ignore[attr-defined]

    def state_residual(self, state_new, dlambda, state_n, state_trial, *, stress_trial) -> list:
        """Ohno-Wang implicit backstress residual (backward Euler).

        R_α = α_{n+1} − α_n − (2/3) C_k Δλ n̂ + γ Δλ ‖α_{n+1}‖ α_{n+1} = 0

        The flow direction n̂ is evaluated at the *new* relative stress for
        full backward-Euler consistency.
        """
        alpha_new = state_new["alpha"]
        n_hat = self.flow_direction(state_new["stress"], alpha_new)  # type: ignore[attr-defined]
        a_norm = self.alpha_norm(alpha_new)  # type: ignore[attr-defined]
        R_stress = self.default_stress_residual(state_new, dlambda, stress_trial)  # type: ignore[attr-defined]
        R_alpha = (
            alpha_new
            - state_n["alpha"]
            - (2.0 / 3.0) * self.C_k * dlambda * n_hat  # type: ignore[attr-defined]
            + self.gamma * dlambda * a_norm * alpha_new  # type: ignore[attr-defined]
        )
        R_ep = state_new["ep"] - state_n["ep"] - dlambda
        return [self.stress(R_stress), self.alpha(R_alpha), self.ep(R_ep)]  # type: ignore[attr-defined]
