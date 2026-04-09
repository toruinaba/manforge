"""Abstract base class for constitutive material models."""

from abc import ABC, abstractmethod

import jax.numpy as jnp

from manforge.autodiff.operators import identity_voigt


class MaterialModel(ABC):
    """Abstract base class for constitutive material models.

    Subclasses must define :attr:`param_names` and :attr:`state_names`,
    and implement the three abstract methods.  The framework then provides
    return mapping, consistent tangent computation, and parameter fitting.

    Attributes
    ----------
    param_names : list[str]
        Names of material parameters (keys expected in ``params`` dicts).
    state_names : list[str]
        Names of internal state variables (keys in ``state`` dicts).
    ntens : int
        Number of stress/strain components. Default 6 (3-D full).
    """

    param_names: list[str]
    state_names: list[str]
    ntens: int = 6

    # ------------------------------------------------------------------
    # Abstract interface — must be implemented by subclasses
    # ------------------------------------------------------------------

    @abstractmethod
    def elastic_stiffness(self, params: dict) -> jnp.ndarray:
        """Return the elastic stiffness tensor in Voigt notation.

        Parameters
        ----------
        params : dict
            Material parameters keyed by :attr:`param_names`.

        Returns
        -------
        jnp.ndarray, shape (ntens, ntens)
            Elastic stiffness C (σ = C : ε, engineering shear convention).
        """

    @abstractmethod
    def yield_function(
        self,
        stress: jnp.ndarray,
        state: dict,
        params: dict,
    ) -> jnp.ndarray:
        """Evaluate the yield function f(σ, q, params).

        The material is in the elastic domain when f ≤ 0.

        Parameters
        ----------
        stress : jnp.ndarray, shape (ntens,)
            Stress in Voigt notation.
        state : dict
            Internal state variables.
        params : dict
            Material parameters.

        Returns
        -------
        jnp.ndarray, scalar
            Yield function value.
        """

    @abstractmethod
    def hardening_increment(
        self,
        dlambda: jnp.ndarray,
        state: dict,
        params: dict,
    ) -> dict:
        """Return updated state variables after a plastic increment.

        Parameters
        ----------
        dlambda : jnp.ndarray, scalar
            Plastic multiplier increment Δλ ≥ 0.
        state : dict
            State at the beginning of the increment.
        params : dict
            Material parameters.

        Returns
        -------
        dict
            Updated state ``state_{n+1}``.
        """

    # ------------------------------------------------------------------
    # Default helpers provided by the framework
    # ------------------------------------------------------------------

    def isotropic_C(self, lam: float, mu: float) -> jnp.ndarray:
        """Build the isotropic elastic stiffness tensor.

        Uses the engineering-shear Voigt convention (σ = C ε, where ε uses
        engineering shear strains γ = 2ε_ij):

            C = λ δ⊗δ + μ diag([2, 2, 2, 1, 1, 1])

        Normal–normal block: C[i,i] = λ + 2μ, C[i,j] = λ (i≠j, i,j<3)
        Shear block        : C[k,k] = μ (k = 3,4,5)

        Parameters
        ----------
        lam : float
            First Lamé constant λ.
        mu : float
            Shear modulus μ.

        Returns
        -------
        jnp.ndarray, shape (6, 6)
        """
        delta = identity_voigt()
        scale = jnp.array([2.0, 2.0, 2.0, 1.0, 1.0, 1.0])
        return lam * jnp.outer(delta, delta) + mu * jnp.diag(scale)

    def initial_state(self) -> dict:
        """Return zero-initialised state dict.

        Returns
        -------
        dict
            ``{name: jnp.array(0.0) for name in self.state_names}``
        """
        return {name: jnp.array(0.0) for name in self.state_names}
