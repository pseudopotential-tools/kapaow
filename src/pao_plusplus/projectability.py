"""Projectability module for pao_plusplus."""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
from qe_wavefunctions.atomic_wfcs import AtomicWFC
from qe_wavefunctions.qe_input_wfcs import QEInputWFC
from qe_wavefunctions.qe_projections import compute_atomic_projections

from pao_plusplus.fat_bands import build_atoms_dict

logger = logging.getLogger(__name__)


def active_orbital_mask(
    s_diag: npt.NDArray[np.float64],
) -> npt.NDArray[np.bool_]:
    """Identify populated (non-empty) orbital slots from the diagonal of S.

    The ``qe_wavefunctions`` library allocates ``(lmax+1)^2 * (nmax+1)``
    orbital slots per atom, but unpopulated slots appear as either ``NaN``
    or ``0`` on the overlap diagonal.  This function returns a boolean mask
    that is ``True`` for genuinely populated orbitals.

    Parameters
    ----------
    s_diag
        Real part of the diagonal of the overlap matrix S, shape
        ``(num_orbitals,)``.

    Returns
    -------
    np.ndarray
        Boolean mask of shape ``(num_orbitals,)``.
    """
    return ~np.isnan(s_diag) & (s_diag > 1e-10)


def _make_qe_input_wfc(
    wfc_dir: Path,
    lattice_vectors: npt.NDArray[np.float64],
) -> QEInputWFC:
    """Create a QEInputWFC that reads wfcN.hdf5 directly from wfc_dir.

    QEInputWFC normally looks in ``outdir/prefix.save/``, but AiiDA
    dumps put wfc files flat in the outputs directory.  We override
    ``outdir`` after construction to point directly at wfc_dir.
    """
    qe_wfc = QEInputWFC(
        outdir=str(wfc_dir),
        prefix="dummy",
        lattice_vectors=lattice_vectors,
    )
    qe_wfc.outdir = str(wfc_dir)
    return qe_wfc


@dataclass
class CachedMaterial:
    """Pre-loaded QE wavefunctions and structural data for a material.

    Stores everything that stays constant across optimizer steps so that
    expensive disk I/O is done only once.
    """

    atoms_dict: dict[str, Any]
    lattice_vectors: npt.NDArray[np.float64]
    kpoint_data: list[tuple[Any, Any, Any]]  # (kpt, miller, wfcs) per k-point
    kpoint_weights: npt.NDArray[np.float64]


def preload_material(
    pwi_file: Path,
    wfc_dir: Path,
    kpoint_weights: npt.NDArray[np.float64],
) -> CachedMaterial:
    """Pre-load QE wavefunctions for all k-points.

    Call once after the QE calculation completes; the returned object
    can be reused across many projectability evaluations.

    Args:
        pwi_file: QE input file (for atom positions and lattice vectors).
        wfc_dir: Directory containing wfcN.hdf5 files directly.
        kpoint_weights: Array of k-point weights (from AiiDA output_kpoints).
    """
    atoms_dict, lattice_vectors = build_atoms_dict(pwi_file)

    qe_wfc = _make_qe_input_wfc(wfc_dir, lattice_vectors)

    kpoint_data = []
    ik = 1
    while True:
        try:
            kpt, _, miller, wfcs = qe_wfc.get_wfc(ik)
        except FileNotFoundError:
            break
        kpoint_data.append((kpt, miller, wfcs))
        ik += 1

    if not kpoint_data:
        raise FileNotFoundError(
            f"No wavefunction files found in {wfc_dir}. "
            f"Directory exists: {wfc_dir.exists()}. "
            f"Contents: {list(wfc_dir.iterdir()) if wfc_dir.exists() else 'N/A'}"
        )

    if len(kpoint_data) != len(kpoint_weights):
        raise ValueError(
            f"Number of wfc files ({len(kpoint_data)}) does not match "
            f"number of k-point weights ({len(kpoint_weights)})"
        )

    return CachedMaterial(atoms_dict, lattice_vectors, kpoint_data, kpoint_weights)


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
        atoms_dict=cached.atoms_dict,
        lattice_vectors=cached.lattice_vectors,
    )
    species_list = list(bessel_files.keys())
    file_list = [str(bessel_files[s]) for s in species_list]
    atomic_wfc.load_atomic_wfcs(file_list)

    proj_matrices = []
    for kpt, miller, wfcs in cached.kpoint_data:
        _, a_mn, c_mn = compute_atomic_projections(
            atomic_wfc,
            kpt,
            miller,
            wfcs,
        )
        proj_matrices.append((np.conj(c_mn).T @ a_mn).real)

    return projectability_score(proj_matrices, num_target_bands, cached.kpoint_weights)


def proj_matrices_from_amn(
    amn: npt.NDArray[np.complex128],
    cmn: npt.NDArray[np.complex128],
) -> list[npt.NDArray[np.float64]]:
    r"""Compute Re(C^dag A) matrices at each k-point from stacked Amn/Cmn arrays.

    Parameters
    ----------
    amn
        Array of shape (num_kpoints, num_orbitals, num_bands).
    cmn
        Array of shape (num_kpoints, num_orbitals, num_bands).

    Returns
    -------
    list
        List of Re(C^dag A) matrices, one per k-point.
    """
    return [(np.conj(cmn[ik]).T @ amn[ik]).real for ik in range(amn.shape[0])]


def projectability_eigenvalues(
    proj_matrices: list[npt.NDArray[np.float64]],
) -> npt.NDArray[np.float64]:
    """Compute gauge-invariant projectability eigenvalues at each k-point.

    Diagonalises ``Re(C^dag A)`` at each k-point and returns eigenvalues
    sorted in descending order.

    Parameters
    ----------
    proj_matrices
        List of Re(C^dag A) matrices, one per k-point, each of shape
        (num_bands, num_bands).

    Returns
    -------
    np.ndarray
        Array of shape (num_kpoints, num_bands) with eigenvalues sorted
        descending at each k-point.
    """
    result = []
    for p_mat in proj_matrices:
        eigvals = np.sort(np.linalg.eigvalsh(p_mat))[::-1]
        result.append(eigvals)
    return np.array(result)


def projectability_score(
    proj_matrices: list[npt.NDArray[np.float64]],
    num_target_bands: int,
    kpoint_weights: npt.NDArray[np.float64],
) -> float:
    """Calculate the weighted projectability score from eigenvalues of Re(C^dag A).

    Parameters
    ----------
    proj_matrices
        List of Re(C^dag A) matrices, one per k-point, each of shape
        (num_bands, num_bands).
    num_target_bands
        Number of bands expected to be well-described by the PAO basis.
    kpoint_weights
        Array of k-point weights (should sum to ~1 when normalized).

    Returns
    -------
    float
        Weighted average of the ``num_target_bands`` largest eigenvalues
        of Re(C^dag A) across all k-points.
    """
    eigvals = projectability_eigenvalues(proj_matrices)
    weights = kpoint_weights / np.sum(kpoint_weights)
    total = 0.0
    for ev, w in zip(eigvals, weights, strict=True):
        total += w * np.sum(ev[:num_target_bands])
    return float(total / num_target_bands)


def suggest_disentanglement_thresholds(
    amn: npt.NDArray[np.complex128],
    cmn: npt.NDArray[np.complex128],
    otsu_bins: int = 5,
) -> tuple[float, float]:
    """Suggest dis_proj_min and dis_proj_max using multi-Otsu thresholding.

    Computes the projectability eigenvalues from the A and C matrices,
    pools them across the BZ, and applies multi-class Otsu thresholding.
    The first and last thresholds are returned as dis_proj_min and
    dis_proj_max, so that increasing ``otsu_bins`` widens the
    disentanglement window.

    Parameters
    ----------
    amn
        Projection matrices, shape ``(num_kpoints, num_orbitals, num_bands)``.
    cmn
        ``S^{-1} A`` matrices, shape ``(num_kpoints, num_orbitals, num_bands)``.
    otsu_bins
        Number of Otsu classes (must be >= 3).  More classes resolve finer
        structure, widening the disentanglement window.

    Returns
    -------
    dis_proj_min, dis_proj_max
        Suggested thresholds (both in [0, 1]).
    """
    from skimage.filters import threshold_multiotsu

    proj_mats = proj_matrices_from_amn(amn, cmn)
    eigvals = projectability_eigenvalues(proj_mats)
    pooled = np.clip(eigvals.ravel(), 0.0, 1.0)

    thresholds = threshold_multiotsu(pooled, classes=otsu_bins, nbins=64)
    # Always use first and last thresholds to maximise the middle region
    dis_proj_min = float(thresholds[0])
    dis_proj_max = float(thresholds[-1])

    return dis_proj_min, dis_proj_max


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
        outdir=str(outdir),
        prefix=prefix,
        lattice_vectors=lattice_vectors,
    )

    atomic_wfc = AtomicWFC(
        atoms_dict=atoms_dict,
        lattice_vectors=lattice_vectors,
    )
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
        _, a_mn, c_mn = compute_atomic_projections(
            atomic_wfc,
            kpt,
            miller,
            wfcs,
        )
        proj_matrices.append((np.conj(c_mn).T @ a_mn).real)
        ik += 1

    equal_weights = np.ones(len(proj_matrices))
    return projectability_score(proj_matrices, num_target_bands, equal_weights)


def check_onsite_overlap(
    pwi_file: Path,
    outdir: Path,
    prefix: str,
    bessel_files: dict[str, Path],
) -> None:
    r"""Check that (1/Nk) sum_k S(k) ~ I on each atomic site's orbital block.

    For properly normalized atomic orbitals, the on-site block of the
    k-averaged overlap matrix should be the identity.
    """
    atoms_dict, lattice_vectors = build_atoms_dict(pwi_file)

    qe_wfc = QEInputWFC(
        outdir=str(outdir),
        prefix=prefix,
        lattice_vectors=lattice_vectors,
    )

    atomic_wfc = AtomicWFC(
        atoms_dict=atoms_dict,
        lattice_vectors=lattice_vectors,
    )
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
        s_mn, _, _ = compute_atomic_projections(
            atomic_wfc,
            kpt,
            miller,
            wfcs,
        )
        if s_sum is None:
            s_sum = np.zeros_like(s_mn)
        s_sum += s_mn
        num_kpoints += 1
        ik += 1

    if s_sum is None:
        raise RuntimeError("No k-points were found; s_sum is None.")
    s_avg = s_sum / num_kpoints

    # Extract on-site blocks using start_indices, skipping empty orbitals
    for ispec, species in enumerate(species_list):
        lmax = atomic_wfc.lmax_species[ispec]
        nmax = atomic_wfc.nmax_species[ispec]
        norb_per_atom = (lmax + 1) ** 2 * (nmax + 1)
        for iat in range(atomic_wfc.num_atoms[ispec]):
            atom_idx = sum(atomic_wfc.num_atoms[:ispec]) + iat
            base = atomic_wfc.start_indices[atom_idx]
            end = base + norb_per_atom
            block = s_avg[base:end, base:end]
            diag = np.diag(block).real
            active = active_orbital_mask(diag)
            active_block = block[np.ix_(active, active)]
            identity = np.eye(active_block.shape[0])
            err = np.max(np.abs(active_block - identity))
            logger.info(
                "%s atom %d: max|S_onsite - I| = %.6e (%d/%d active orbitals)",
                species,
                iat,
                err,
                active.sum(),
                norb_per_atom,
            )
            logger.info(
                "  diagonal: %s",
                np.diag(active_block).real,
            )
