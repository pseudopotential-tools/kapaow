"""How to extend a pseudoatomic basis set with additional orbitals."""

from __future__ import annotations
from pao_plusplus.pydantic import BaseModel
from pydantic import field_validator, model_validator
from enum import IntEnum
from abc import ABC, abstractmethod
from typing import Any, Self

class AngularMomentum(IntEnum):
    """Angular momentum quantum numbers."""
    S = 0
    P = 1
    D = 2
    F = 3
    G = 4

    def __add__(self, other: int | AngularMomentum) -> AngularMomentum:
        if isinstance(other, AngularMomentum):
            other = other.value
        return AngularMomentum(self.value + other)

class Basis(BaseModel, ABC):
    """A basis set."""

    @property
    @abstractmethod
    def l_max(self) -> AngularMomentum:
        """The maximum angular momentum quantum number in the basis set."""
        ...
    
    @property
    @abstractmethod
    def n_max(self) -> int:
        """The maximum principal quantum number in the basis set."""
        ...

class PseudoatomicBasis(Basis):
    """A pseudoatomic basis set.
    
    Only need to keep track of the number of orbitals per angular momentum channel."""
    number_of_orbitals: dict[AngularMomentum, int]
    
    @field_validator("number_of_orbitals", mode="before")
    @classmethod
    def ensure_all_channels_present(cls, v: dict[AngularMomentum, int]) -> dict[AngularMomentum, int]:
        """Ensure all angular momentum channels are present in the dictionary."""
        for ang_mtm in AngularMomentum:
            if ang_mtm not in v:
                v[ang_mtm] = 0
        return v
    
    @field_validator("number_of_orbitals", mode="before")
    @classmethod
    def coerce_keys(cls, v: Any) -> Any:
        """Coerce integer keys to AngMtm."""
        if isinstance(v, dict):
            coerced_v: dict[AngularMomentum, int] = {AngularMomentum(k) if isinstance(k, int) else k: val for k, val in v.items()}
            return coerced_v
        return v

    @property
    def l_max(self) -> AngularMomentum:
        """The maximum angular momentum quantum number in the basis set
        
        i.e. return the largest l for which number_of_orbitals[l] > 0
        """
        return max([l for l, count in self.number_of_orbitals.items() if count > 0], key=lambda l: l.value)
    
    @property
    def n_max(self) -> int:
        """The maximum principal quantum number in the basis set.
        
        Note that this is for the pseudo-wavefunction, so n_max is 1 if the basis has only 2s and 2p orbitals."""
        return max(self.number_of_orbitals.values()) if self.number_of_orbitals else 0

    def __len__(self) -> int:
        return sum(self.number_of_orbitals.values())

    def extend(self, s: int = 0, p: int = 0, d: int = 0, f: int = 0, g: int = 0) -> "PseudoatomicBasis":
        """Return a new PseudoatomicBasis with added orbitals."""
        new_number_of_orbitals = self.number_of_orbitals.copy()
        new_number_of_orbitals[AngularMomentum.S] += s
        new_number_of_orbitals[AngularMomentum.P] += p
        new_number_of_orbitals[AngularMomentum.D] += d
        new_number_of_orbitals[AngularMomentum.F] += f
        new_number_of_orbitals[AngularMomentum.G] += g
        return PseudoatomicBasis(number_of_orbitals=new_number_of_orbitals)

    @property
    def l_values(self) -> list[int]:
        """List of l values in the basis set, repeated according to the number of orbitals."""
        l_vals: list[int] = []
        for l in AngularMomentum:
            l_vals += [l.value] * self.number_of_orbitals[l]
        return l_vals

class Subshell(BaseModel):
    """A subshell in a pseudoatomic basis set."""
    n: int
    l: AngularMomentum

    @field_validator("l", mode="before")
    @classmethod
    def coerce_l(cls, v: Any) -> Any:
        """Coerce integer l to AngMtm."""
        if isinstance(v, int):
            return AngularMomentum(v)
        return v
    
    @model_validator(mode="after")
    def validate_n_l(self) -> Self:
        """Validate that n and l are consistent."""
        if self.n <= self.l.value:
            raise ValueError(f"Invalid subshell with n={self.n} and l={self.l}: n must be greater than l")
        return self

# Order of subshells to add following the Madelung rule
ordered_subshells = [Subshell(n=1, l=AngularMomentum.S),  # 1s
                     Subshell(n=2, l=AngularMomentum.S),  # 2s
                     Subshell(n=2, l=AngularMomentum.P),  # 2p
                     Subshell(n=3, l=AngularMomentum.S),  # 3s
                     Subshell(n=3, l=AngularMomentum.P),  # 3p
                     Subshell(n=4, l=AngularMomentum.S),  # 4s
                     Subshell(n=3, l=AngularMomentum.D),  # 3d
                     Subshell(n=4, l=AngularMomentum.P),  # 4p
                     Subshell(n=5, l=AngularMomentum.S),  # 5s
                     Subshell(n=4, l=AngularMomentum.D),  # 4d
                     Subshell(n=5, l=AngularMomentum.P),  # 5p
                     Subshell(n=6, l=AngularMomentum.S),  # 6s
                     Subshell(n=4, l=AngularMomentum.F),  # 4f
                     Subshell(n=5, l=AngularMomentum.D),  # 5d
                     Subshell(n=6, l=AngularMomentum.P),  # 6p
                     Subshell(n=7, l=AngularMomentum.S)]  # 7s


class AtomicBasis(Basis):
    """An atomic basis set.
    
    Need to keep track of the (n, l) values of each subshell."""

    subshells: list[Subshell]

    @property
    def l_max(self) -> AngularMomentum:
        """The maximum angular momentum quantum number in the basis set."""
        ang_mtms = [s.l for s in self.subshells]
        return max(ang_mtms, key=lambda l: l.value)
    
    @property
    def n_max(self) -> int:
        """The maximum principal quantum number in the basis set."""
        n_values = [s.n for s in self.subshells]
        return max(n_values) if n_values else 0
    
    def to_pseudoatomic_basis(self) -> PseudoatomicBasis:
        """Convert to a PseudoatomicBasis."""
        number_of_orbitals: dict[AngularMomentum, int] = {}
        for subshell in self.subshells:
            if subshell.l not in number_of_orbitals:
                number_of_orbitals[subshell.l] = 0
            number_of_orbitals[subshell.l] += 1
        return PseudoatomicBasis(number_of_orbitals=number_of_orbitals)

    def extend(self, subshells: list[Subshell]) -> "AtomicBasis":
        """Return a new AtomicBasis with an added subshell."""
        return AtomicBasis(subshells=self.subshells + subshells)
    
    def __contains__(self, subshell: Subshell) -> bool:
        return subshell in self.subshells
    
    def __len__(self) -> int:
        return len(self.subshells)


