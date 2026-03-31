"""How to extend a pseudoatomic basis set with additional orbitals."""

import enum
from abc import ABC, abstractmethod

from pydantic import Field

from pao_plusplus.basis import (
    AngularMomentum,
    AtomicBasis,
    PseudoatomicBasis,
    ordered_subshells,
)
from pao_plusplus.pydantic import BaseModel


class BasisExtensionType(enum.Enum):
    """Type of basis extension."""

    SUBSHELL = "subshell"
    POLARIZATION = "polarization"
    S = "s"
    P = "p"
    D = "d"
    F = "f"
    G = "g"

    @property
    def angular_momentum(self) -> AngularMomentum | None:
        """Return the angular momentum if this is a channel-specific extension."""
        _map = {"s": AngularMomentum.S, "p": AngularMomentum.P, "d": AngularMomentum.D,
                "f": AngularMomentum.F, "g": AngularMomentum.G}
        return _map.get(self.value)


class BasisExtension(BaseModel, ABC):
    """An extension to a basis set."""

    @abstractmethod
    def extend(self, basis: AtomicBasis | PseudoatomicBasis) -> PseudoatomicBasis:
        """Extend the given basis set and return the new basis set."""


class BasisExtensionViaAddition(BasisExtension):
    """Add the next subshell to an atomic basis set."""

    increment: int = Field(default=1, description="number of subshells to add")

    def extend_atomic(self, basis: AtomicBasis) -> AtomicBasis:
        """Extend the provided basis by adding the next subshell(s), returning an AtomicBasis.

        First checks for gaps in Madelung order between basis subshells
        (e.g. 5s missing between 4p and 4d for Pd), filtering out core
        subshells (n < min n of basis). If no valid gaps, adds the next
        subshell after the outermost.
        """
        indices = sorted(ordered_subshells.index(s) for s in basis.subshells)
        i_innermost = indices[0]
        i_outermost = indices[-1]
        min_n = min(s.n for s in basis.subshells)

        # For each l channel, record the max n present in the basis
        max_n_per_l: dict[AngularMomentum, int] = {}
        for s in basis.subshells:
            if s.l not in max_n_per_l or s.n > max_n_per_l[s.l]:
                max_n_per_l[s.l] = s.n

        # Look for valid gaps between innermost and outermost basis entries
        # Skip core subshells: n < min_n, or same l channel already has higher n
        gaps = []
        for subshell in ordered_subshells[i_innermost:i_outermost]:
            if subshell in basis:
                continue
            if subshell.n < min_n:
                continue
            if subshell.l in max_n_per_l and subshell.n < max_n_per_l[subshell.l]:
                continue
            gaps.append(subshell)

        # Use gaps first, then continue past outermost
        candidates = gaps + ordered_subshells[i_outermost + 1 :]
        to_add = candidates[: self.increment]
        if len(to_add) < self.increment:
            raise ValueError(f"Cannot add {self.increment} subshell(s) beyond the current basis.")
        return basis.extend(to_add)

    def extend(self, basis: AtomicBasis | PseudoatomicBasis) -> PseudoatomicBasis:
        """Extend the provided basis by adding the next subshell(s)."""
        if isinstance(basis, PseudoatomicBasis):
            raise TypeError(
                "Cannot extend pseudoatomic bases by addition because we can't know"
                " what l channel to add to"
            )
        return self.extend_atomic(basis).to_pseudoatomic_basis()


class BasisExtensionViaChannel(BasisExtension):
    """Add orbitals in a specific angular momentum channel."""

    channel: AngularMomentum = Field(description="angular momentum channel to extend")
    increment: int = Field(default=1, description="number of radial functions to add")

    def extend(self, basis: AtomicBasis | PseudoatomicBasis) -> PseudoatomicBasis:
        """Extend the provided basis by adding orbitals in the specified channel."""
        if isinstance(basis, AtomicBasis):
            basis = basis.to_pseudoatomic_basis()
        return basis.extend(**{self.channel.name.lower(): self.increment})


class BasisExtensionViaPolarization(BasisExtension):
    """Add polarization orbitals to a pseudoatomic basis set."""

    increment: int = Field(
        default=1, description="number of polarization orbitals to add per angular momentum channel"
    )

    def extend(self, basis: AtomicBasis | PseudoatomicBasis) -> PseudoatomicBasis:
        """Extend the provided basis by adding polarization orbitals."""
        if isinstance(basis, AtomicBasis):
            basis = basis.to_pseudoatomic_basis()

        new_basis = basis

        increment = self.increment
        while increment > 0:
            channels_to_increment = [l for l in AngularMomentum if l <= new_basis.l_max + 1]
            for l in channels_to_increment:
                new_basis = new_basis.extend(**{l.name.lower(): 1})
            increment -= 1

        return new_basis


