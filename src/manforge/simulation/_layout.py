"""ResidualLayout — central descriptor for the NR unknown/residual vector layout.

Unknown vector:   x = [σ (ntens) | Δλ (1) | q_implicit_non_stress (n_imp, declaration order)]
Residual vector:  R = [R_σ (ntens) | R_Δλ (1) | R_q_non_stress (n_imp, declaration order)]

All slice/pack/unpack logic lives here so that _residual.py, integrator/base.py,
and verification/jacobian.py share a single source of truth.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import autograd.numpy as anp
import numpy as np

if TYPE_CHECKING:
    pass


@dataclass(frozen=True)
class ResidualLayout:
    """Descriptor for the augmented NR residual system.

    Parameters
    ----------
    ntens : int
        Number of stress/strain components.
    stress_kind : str
        ``"implicit"`` when stress is declared ``Implicit``; ``"explicit"`` otherwise.
    implicit_keys : tuple[str, ...]
        Names of non-stress implicit state variables, **in declaration order**.
    explicit_keys : tuple[str, ...]
        Names of non-stress explicit state variables, in declaration order.
    _shapes : tuple[tuple[str, tuple[int, ...]], ...]
        ``(name, shape)`` pairs for each key in ``implicit_keys``, in the same order.
        Shapes are concrete Python tuples (no NTENS sentinel).
    _residual_names : tuple[tuple[str, str], ...]
        ``(state_name, residual_name)`` pairs for each slot in canonical order:
        ``[stress, dlambda, *implicit_keys]``.  Residual names are used as row
        labels in :class:`~manforge.verification.jacobian.JacobianBlocks`.part.
    dlambda_residual_name : str
        Residual-row label for the Δλ slot (default ``"dlambda"``).
    """

    ntens: int
    stress_kind: str                                        # "implicit" | "explicit"
    implicit_keys: tuple                                    # declaration order, stress excluded
    explicit_keys: tuple                                    # declaration order, stress excluded
    _shapes: tuple                                          # (name, shape) for implicit_keys
    _residual_names: tuple = ()                             # (state_name, residual_name) pairs
    dlambda_residual_name: str = "dlambda"

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def from_model(cls, model) -> "ResidualLayout":
        """Build a ResidualLayout from a MaterialModel instance."""
        ntens = model.ntens
        stress_kind = model.state_fields["stress"].kind
        implicit_keys = tuple(
            k for k in model.state_names
            if k != "stress" and model.state_fields[k].kind == "implicit"
        )
        explicit_keys = tuple(
            k for k in model.state_names
            if k != "stress" and model.state_fields[k].kind == "explicit"
        )
        shapes = tuple(
            (k, model.state_fields[k].resolve_shape(ntens))
            for k in implicit_keys
        )
        dl_rname = getattr(model, "dlambda_residual_name", "dlambda")
        residual_names = (
            ("stress", model.state_fields["stress"].effective_residual_name),
            ("dlambda", dl_rname),
            *((k, model.state_fields[k].effective_residual_name) for k in implicit_keys),
        )
        return cls(
            ntens=ntens,
            stress_kind=stress_kind,
            implicit_keys=implicit_keys,
            explicit_keys=explicit_keys,
            _shapes=shapes,
            _residual_names=residual_names,
            dlambda_residual_name=dl_rname,
        )

    # ------------------------------------------------------------------
    # Derived properties
    # ------------------------------------------------------------------

    @property
    def is_stress_implicit(self) -> bool:
        return self.stress_kind == "implicit"

    @property
    def n_implicit(self) -> int:
        return sum(int(np.prod(s)) for _, s in self._shapes)

    @property
    def n_unknown(self) -> int:
        return self.ntens + 1 + self.n_implicit

    # ------------------------------------------------------------------
    # Slice helpers
    # ------------------------------------------------------------------

    def stress_slice(self) -> slice:
        return slice(0, self.ntens)

    def dlambda_index(self) -> int:
        return self.ntens

    def state_slice(self, name: str) -> slice:
        """Return the slice of the implicit-state block for *name*.

        The slice is relative to the start of the state block (i.e. offset ntens+1
        must be added by the caller when indexing into the full x or R vector).
        """
        offset = 0
        for k, shp in self._shapes:
            size = int(np.prod(shp))
            if k == name:
                return slice(offset, offset + size)
            offset += size
        raise KeyError(f"ResidualLayout: {name!r} is not an implicit non-stress key")

    def slot_slice(self, name: str) -> slice:
        """Return the absolute slice in the x / R vector for the named slot.

        Accepts ``"stress"``, ``"dlambda"``, or any key in :attr:`implicit_keys`.
        Unlike :meth:`state_slice`, the returned slice is relative to the start
        of the full vector (no caller-side offset needed).
        """
        if name == "stress":
            return slice(0, self.ntens)
        if name == "dlambda":
            return slice(self.ntens, self.ntens + 1)
        sl = self.state_slice(name)
        q0 = self.ntens + 1
        return slice(q0 + sl.start, q0 + sl.stop)

    def slot_shape(self, name: str) -> tuple:
        """Return the concrete shape for the named slot.

        ``"stress"`` → ``(ntens,)``; ``"dlambda"`` → ``()``; implicit keys
        → the resolved shape from ``_shapes``.
        """
        if name == "stress":
            return (self.ntens,)
        if name == "dlambda":
            return ()
        for k, shp in self._shapes:
            if k == name:
                return shp
        raise KeyError(f"ResidualLayout: {name!r} is not a known slot name")

    def residual_name_for(self, state_name: str) -> str:
        """Return the residual-row label for the given state name."""
        for s, r in self._residual_names:
            if s == state_name:
                return r
        raise KeyError(f"ResidualLayout: {state_name!r} not found in _residual_names")

    def residual_names(self) -> tuple:
        """Residual-row labels in canonical order: (stress, dlambda, *implicit_keys)."""
        return tuple(r for _, r in self._residual_names)

    # ------------------------------------------------------------------
    # Pack / unpack
    # ------------------------------------------------------------------

    def pack(self, sigma, dlambda, q_imp: dict):
        """Assemble the unknown vector x from its components.

        Parameters
        ----------
        sigma : array-like, shape (ntens,)
        dlambda : scalar
        q_imp : dict mapping implicit_keys → array
            Only non-stress implicit states.  Missing keys default to zero.
        """
        parts = [anp.array(sigma), anp.atleast_1d(anp.array(dlambda))]
        for k, shp in self._shapes:
            v = q_imp.get(k, np.zeros(shp))
            parts.append(anp.reshape(v, (-1,)) if shp else anp.atleast_1d(v))
        return anp.concatenate(parts) if len(parts) > 1 else parts[0]

    def unpack(self, x) -> tuple:
        """Decompose x into (sigma, dlambda, q_imp_dict).

        Returns
        -------
        sigma : array, shape (ntens,)
        dlambda : scalar
        q_imp : dict[str, array]
        """
        ntens = self.ntens
        sigma = x[:ntens]
        dlambda = x[ntens]
        q_imp: dict = {}
        idx = ntens + 1
        for k, shp in self._shapes:
            size = int(np.prod(shp))
            chunk = x[idx: idx + size]
            q_imp[k] = chunk.reshape(shp) if shp else chunk[0]
            idx += size
        return sigma, dlambda, q_imp

    def pack_residual(self, R_stress, R_dlambda, R_state: dict):
        """Assemble the residual vector R from its components.

        Parameters
        ----------
        R_stress : array-like, shape (ntens,)
        R_dlambda : scalar
        R_state : dict mapping implicit_keys → array (may be empty)
        """
        parts = [anp.array(R_stress), anp.atleast_1d(R_dlambda)]
        for k, shp in self._shapes:
            v = R_state[k]
            parts.append(anp.reshape(v, (-1,)) if shp else anp.atleast_1d(v))
        return anp.concatenate(parts)
