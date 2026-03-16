"""Projectability module for pao_plusplus."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
from qe_wavefunctions.atomic_wfcs import AtomicWFC
from qe_wavefunctions.qe_input_wfcs import QEInputWFC
from qe_wavefunctions.qe_projections import compute_atomic_projections

from pao_plusplus.fat_bands import build_atoms_dict


@dataclass
class CachedMaterial:
    """Pre-loaded QE wavefunctions and structural data for a material.

    Stores everything that stays constant across optimizer steps so that
    expensive disk I/O is done only once.
    """

    atoms_dict: dict[str, Any]
    lattice_vectors: npt.NDArray[np.float64]
    kpoint_data: list[tuple[Any, Any, Any]]  # (kpt, miller, wfcs) per k-point


def preload_material(
    pwi_file: Path,
    outdir: Path,
    prefix: str,
) -> CachedMaterial:
    """Pre-load QE wavefunctions for all k-points.

    Call once after the QE calculation completes; the returned object
    can be reused across many projectability evaluations.
    """
    atoms_dict, lattice_vectors = build_atoms_dict(pwi_file)

    qe_wfc = QEInputWFC(
        outdir=str(outdir), prefix=prefix, lattice_vectors=lattice_vectors
    )

    kpoint_data = []
    ik = 1
    while True:
        try:
            kpt, _, miller, wfcs = qe_wfc.get_wfc(ik)
        except FileNotFoundError:
            break
        kpoint_data.append((kpt, miller, wfcs))
        ik += 1

    return CachedMaterial(atoms_dict, lattice_vectors, kpoint_data)


def compute_projectability_cached(
    cached: CachedMaterial,
    bessel_files: dict[str, Path],
    num_target_bands: int,
) -> float:
    """Compute projectability using pre-loaded wavefunctions.

    This avoids re-reading QE wavefunction files from disk on every call.
    Only the atomic Bessel files (which change when the confining potential
    changes) are reloaded.
    """
    atomic_wfc = AtomicWFC(
        atoms_dict=cached.atoms_dict, lattice_vectors=cached.lattice_vectors
    )
    species_list = list(bessel_files.keys())
    file_list = [str(bessel_files[s]) for s in species_list]
    atomic_wfc.load_atomic_wfcs(file_list)

    proj_matrices = []
    for kpt, miller, wfcs in cached.kpoint_data:
        _, a_mn, c_mn = compute_atomic_projections(atomic_wfc, kpt, miller, wfcs)
        proj_matrices.append((np.conj(c_mn).T @ a_mn).real)

    return projectability_score(proj_matrices, num_target_bands)


def projectability_score(
    proj_matrices: list[npt.NDArray[np.float64]], num_target_bands: int
) -> float:
    """Calculate the projectability score from the n smallest eigenvalues of Re(C†A).

    Parameters
    ----------
    proj_matrices
        List of Re(C†A) matrices, one per k-point, each of shape
        (num_bands, num_bands).
    num_target_bands
        Number of bands expected to be well-described by the PAO basis.

    Returns
    -------
    float
        Average of the ``num_target_bands`` largest eigenvalues of Re(C†A)
        across all k-points.
    """
    num_kpoints = len(proj_matrices)
    total = 0.0
    for P in proj_matrices:
        eigvals = np.sort(np.linalg.eigvalsh(P))
        total += np.sum(eigvals[-num_target_bands:])
    return float(total / num_target_bands / num_kpoints)


def compute_projectability(
    pwi_file: Path,
    outdir: Path,
    prefix: str,
    bessel_files: dict[str, Path],
    num_target_bands: int,
) -> float:
    """Compute the projectability of PAOs against QE nscf wavefunctions.

    Assumes the wavefunctions come from an nscf calculation on the full
    (unsymmetrised) k-grid, so all k-points carry equal weight.

    Parameters
    ----------
    pwi_file
        QE input file (used for atom positions and lattice vectors).
    outdir
        Directory containing ``prefix.save/`` with nscf wfc files.
    prefix
        QE calculation prefix.
    bessel_files
        ``{species: Path}`` mapping to Bessel HDF5 files.
    num_target_bands
        Number of bands expected to be well-described by the PAO basis.

    Returns
    -------
    float
        Average projectability score.
    """
    atoms_dict, lattice_vectors = build_atoms_dict(pwi_file)

    qe_wfc = QEInputWFC(
        outdir=str(outdir), prefix=prefix, lattice_vectors=lattice_vectors
    )

    atomic_wfc = AtomicWFC(atoms_dict=atoms_dict, lattice_vectors=lattice_vectors)
    species_list = list(bessel_files.keys())
    file_list = [str(bessel_files[s]) for s in species_list]
    atomic_wfc.load_atomic_wfcs(file_list)

    # Iterate over all available k-points
    proj_matrices = []
    ik = 1
    while True:
        try:
            kpt, _, miller, wfcs = qe_wfc.get_wfc(ik)
        except FileNotFoundError:
            break
        _, a_mn, c_mn = compute_atomic_projections(atomic_wfc, kpt, miller, wfcs)
        proj_matrices.append((np.conj(c_mn).T @ a_mn).real)
        ik += 1

    return projectability_score(proj_matrices, num_target_bands)


def check_onsite_overlap(
    pwi_file: Path,
    outdir: Path,
    prefix: str,
    bessel_files: dict[str, Path],
) -> None:
    """Check that (1/Nk) sum_k S(k) ≈ I on each atomic site's orbital block.

    For properly normalized atomic orbitals, the on-site block of the
    k-averaged overlap matrix should be the identity.
    """
    atoms_dict, lattice_vectors = build_atoms_dict(pwi_file)

    qe_wfc = QEInputWFC(
        outdir=str(outdir), prefix=prefix, lattice_vectors=lattice_vectors
    )

    atomic_wfc = AtomicWFC(atoms_dict=atoms_dict, lattice_vectors=lattice_vectors)
    species_list = list(bessel_files.keys())
    file_list = [str(bessel_files[s]) for s in species_list]
    atomic_wfc.load_atomic_wfcs(file_list)

    s_sum: npt.NDArray[np.complex128] | None = None
    num_kpoints = 0

    ik = 1
    while True:
        try:
            kpt, _, miller, wfcs = qe_wfc.get_wfc(ik)
        except FileNotFoundError:
            break
        s_mn, _, _ = compute_atomic_projections(atomic_wfc, kpt, miller, wfcs)
        if s_sum is None:
            s_sum = np.zeros_like(s_mn)
        s_sum += s_mn
        num_kpoints += 1
        ik += 1

    assert s_sum is not None
    s_avg = s_sum / num_kpoints

    # Extract on-site blocks using start_indices, skipping empty orbitals
    for ispec, species in enumerate(species_list):
        lmax = atomic_wfc.lmax_species[ispec]
        nmax = atomic_wfc.nmax_species[ispec]
        norb_per_atom = (lmax + 1) ** 2 * (nmax + 1)
        for iat in range(atomic_wfc.num_atoms[ispec]):
            atom_idx = sum(atomic_wfc.num_atoms[:ispec]) + iat
            base = atomic_wfc.start_indices[atom_idx]
            block = s_avg[base:base + norb_per_atom, base:base + norb_per_atom]
            diag = np.diag(block).real
            # Mask out empty orbital slots (nan from zero-norm orbitals)
            active = ~np.isnan(diag)
            active_block = block[np.ix_(active, active)]
            identity = np.eye(active_block.shape[0])
            err = np.max(np.abs(active_block - identity))
            print(f"{species} atom {iat}: max|S_onsite - I| = {err:.6e} ({active.sum()}/{norb_per_atom} active orbitals)")
            print(f"  diagonal: {np.diag(active_block).real}")
