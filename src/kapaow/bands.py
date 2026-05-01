"""Utilities for computing the number of bands needed for QE calculations."""

from __future__ import annotations

from pathlib import Path

from kapaow.basis import AtomicBasis
from kapaow.extend import BasisExtension


def orbitals_per_atom(upf_path: Path, extension: BasisExtension | None = None) -> int:
    """Compute the total number of orbitals per atom for a given UPF and optional extension."""
    atomic_basis = AtomicBasis.from_upf(upf_path)
    if extension is not None:
        pseudo_basis = extension.extend(atomic_basis)
    else:
        pseudo_basis = atomic_basis.to_pseudoatomic_basis()
    return pseudo_basis.total_number_of_orbitals


def compute_num_target_bands(
    structure_file: Path,
    orbitals_per_element: dict[str, int],
) -> int:
    """Compute the number of target bands for a given material.

    This is the total number of PAO orbitals across all atoms in the unit cell.

    Parameters
    ----------
    structure_file
        Path to a structure file readable by ASE.
    orbitals_per_element
        ``{element_symbol: orbitals_per_atom}`` mapping for every species
        present in the structure.
    """
    import ase.io

    atoms = ase.io.read(str(structure_file))
    return sum(orbitals_per_element[sym] for sym in atoms.get_chemical_symbols())


def compute_min_nbnd(num_target_bands: int) -> int:
    """Compute the minimum number of bands for a QE calculation.

    Ensures at least 50% more bands than the target, or at minimum 4 extra.
    """
    return max(int(1.5 * num_target_bands), num_target_bands + 4)
