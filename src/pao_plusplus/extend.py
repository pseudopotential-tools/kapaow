"""How to extend a pseudoatomic basis set with additional orbitals."""

from abc import ABC, abstractmethod

from pydantic import Field

from pao_plusplus.basis import (
    AngularMomentum,
    AtomicBasis,
    PseudoatomicBasis,
    Subshell,
    ordered_subshells,
)
from pao_plusplus.pydantic import BaseModel


class BasisExtension(BaseModel, ABC):
    """An extension to a basis set."""

    @abstractmethod
    def extend(self, basis: AtomicBasis | PseudoatomicBasis) -> PseudoatomicBasis:
        """Extend the given basis set and return the new basis set."""


class BasisExtensionViaAddition(BasisExtension):
    """Add the next subshell to an atomic basis set."""

    increment: int = Field(default=1, description="number of subshells to add")

    def extend(self, basis: AtomicBasis | PseudoatomicBasis) -> PseudoatomicBasis:
        """Extend the provided basis by adding the next subshell(s)."""
        if isinstance(basis, PseudoatomicBasis):
            raise TypeError(
                "Cannot extend pseudoatomic bases by addition because we can't know"
                " what l channel to add to"
            )
        # Go through all possible subshells in reverse order
        outermost_subshell: Subshell | None = None
        for subshell in ordered_subshells[::-1]:
            if subshell in basis:
                outermost_subshell = subshell
                break

        if outermost_subshell is None:
            raise ValueError("Basis set is empty.")

        i_outermost = ordered_subshells.index(outermost_subshell)
        to_add = ordered_subshells[i_outermost + 1 : i_outermost + 1 + self.increment]

        new_basis = basis.extend(to_add)

        return new_basis.to_pseudoatomic_basis()


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
