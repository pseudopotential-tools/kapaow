"""Compare A_mn^k and U_mn^k matrices across different PAO choices.

Uses the regular NSCF k-point mesh (not the bands k-path) so that the
Frobenius distance is a proper Brillouin-zone average.  The projection
matrix is orthogonalized via S^{-1/2} before the SVD so that the singular
values correspond to the square roots of the projectabilities.
"""

from __future__ import annotations

import itertools
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import numpy.typing as npt

logger = logging.getLogger(__name__)


def orthogonalize_amn(
    smn: npt.NDArray[np.complex128],
    amn: npt.NDArray[np.complex128],
) -> npt.NDArray[np.complex128]:
    r"""Orthogonalize A_mn^k using the overlap matrix S.

    Computes S^{-1/2} A at each k-point so that the result lives in an
    orthonormal orbital basis.  Empty orbital slots (NaN on the diagonal
    of S) are dropped, so the returned array may have fewer orbital rows
    than the input.  The singular values of S^{-1/2} A are the square
    roots of the projectability eigenvalues.

    Parameters
    ----------
    smn
        Overlap matrices, shape ``(num_kpoints, num_orbitals, num_orbitals)``.
    amn
        Projection matrices, shape ``(num_kpoints, num_orbitals, num_bands)``.

    Returns
    -------
    np.ndarray
        Orthogonalized projection matrices, shape
        ``(num_kpoints, num_active_orbitals, num_bands)``.
    """
    num_kpoints = smn.shape[0]

    from kapaow._experimental.projectability import active_orbital_mask

    # The set of active orbitals is the same at every k-point because it
    # depends only on which radial functions have nonzero norm.
    active = active_orbital_mask(np.diag(smn[0]).real)
    n_active = int(active.sum())
    num_bands = amn.shape[2]

    result = np.empty((num_kpoints, n_active, num_bands), dtype=amn.dtype)
    for ik in range(num_kpoints):
        s_active = smn[ik][np.ix_(active, active)]
        a_active = amn[ik][active, :]

        eigvals, eigvecs = np.linalg.eigh(s_active)
        # S^{-1/2} = Q diag(1/sqrt(d)) Q†
        s_inv_sqrt = eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.conj().T
        result[ik] = s_inv_sqrt @ a_active
    return result


def compute_u_from_svd(
    amn_orth: npt.NDArray[np.complex128],
) -> npt.NDArray[np.complex128]:
    r"""Compute the unitary polar factor from the SVD of an orthogonalized A.

    For each k-point, decomposes \tilde{A}^k = U \Sigma V^\dagger and
    returns U V^\dagger, which has the same shape as A and is the closest
    unitary (in Frobenius norm) to \tilde{A}.

    Parameters
    ----------
    amn_orth
        Orthogonalized projection matrices, shape
        ``(num_kpoints, num_orbitals, num_bands)``.

    Returns
    -------
    np.ndarray
        Polar factor, same shape as *amn_orth*.
    """
    num_kpoints = amn_orth.shape[0]
    u_mn = np.empty_like(amn_orth)
    for ik in range(num_kpoints):
        u, _s, vh = np.linalg.svd(amn_orth[ik], full_matrices=False)
        u_mn[ik] = u @ vh
    return u_mn


def _bz_averaged_frobenius_norm(
    m: npt.NDArray[np.complex128],
    weights: npt.NDArray[np.float64],
) -> float:
    r"""BZ-averaged Frobenius norm: ``\sqrt{ \sum_k w_k \| M^k \|_F^2 }``."""
    frob_sq = np.sum(np.abs(m) ** 2, axis=(1, 2))
    return float(np.sqrt(np.dot(weights, frob_sq)))


def bz_averaged_frobenius_distance(
    m1: npt.NDArray[np.complex128],
    m2: npt.NDArray[np.complex128],
    kpoint_weights: npt.NDArray[np.float64],
    normalized: bool = False,
) -> float:
    r"""Compute the BZ-averaged Frobenius distance between two matrix stacks.

    Returns ``\sqrt{ \sum_k w_k \| M_1^k - M_2^k \|_F^2 }`` where the
    weights are normalised to sum to one.

    When *normalized* is ``True``, divides by the average of the two
    matrices' BZ-averaged Frobenius norms, giving a scale-free relative
    distance.

    Parameters
    ----------
    m1, m2
        Arrays of shape ``(num_kpoints, rows, cols)``.
    kpoint_weights
        Array of k-point weights (will be normalised internally).
    normalized
        If ``True``, return the relative distance.

    Returns
    -------
    float
        (Optionally normalised) weighted RMS Frobenius distance.
    """
    weights = kpoint_weights / np.sum(kpoint_weights)
    diff = m1 - m2
    frob_sq = np.sum(np.abs(diff) ** 2, axis=(1, 2))
    d = float(np.sqrt(np.dot(weights, frob_sq)))
    if normalized:
        norm1 = _bz_averaged_frobenius_norm(m1, weights)
        norm2 = _bz_averaged_frobenius_norm(m2, weights)
        d /= 0.5 * (norm1 + norm2)
    return d


def bz_averaged_principal_angles(
    m1: npt.NDArray[np.complex128],
    m2: npt.NDArray[np.complex128],
    kpoint_weights: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    r"""Compute BZ-averaged principal angles between two matrix stacks.

    At each k-point, computes the singular values of M_1^\dagger M_2 to
    obtain cos(θ_i), then returns the weighted average of the principal
    angles across the BZ.

    Parameters
    ----------
    m1, m2
        Arrays of shape ``(num_kpoints, num_orbitals, num_bands)``.
        Must have the same shape.
    kpoint_weights
        Array of k-point weights (will be normalised internally).

    Returns
    -------
    np.ndarray
        Array of shape ``(num_orbitals,)`` with the BZ-averaged
        principal angles in radians, sorted ascending.
    """
    weights = kpoint_weights / np.sum(kpoint_weights)
    num_kpoints = m1.shape[0]
    n_orb = m1.shape[1]

    angles_sum = np.zeros(n_orb)
    for ik in range(num_kpoints):
        svs = np.linalg.svd(m1[ik] @ m2[ik].conj().T, compute_uv=False)
        # Clamp to [0, 1] for numerical safety
        svs = np.clip(svs, 0.0, 1.0)
        angles_sum += weights[ik] * np.sort(np.arccos(svs))
    return angles_sum


@dataclass
class GaugeComparisonResult:
    """Results from a gauge matrix comparison."""

    matrix_labels: list[str]
    """Labels for each matrix in the Frobenius distance table."""
    distance_table: npt.NDArray[np.float64]
    """Symmetric pairwise Frobenius distance table."""
    principal_angle_comparisons: list[
        tuple[str, str, npt.NDArray[np.float64], npt.NDArray[np.float64]]
    ]
    """For each cross-set pair: (label_i, label_j, angles_from_A, angles_from_U)."""


def compare_matrices(
    config_path: Path,
    working_dir: Path | None = None,
    num_bands: int | None = None,
) -> GaugeComparisonResult:
    r"""Compare A and U matrices across PAO sets on the NSCF k-mesh.

    For each PAO set, computes:

    - :math:`\tilde{A}^k = S^{-1/2} A^k` (orthogonalized projection)
    - :math:`U^k = (UV^\dagger)` from the SVD of :math:`\tilde{A}^k`

    Returns a Frobenius distance table and, for each cross-set pair,
    the BZ-averaged principal angles from both :math:`\tilde{A}^\dagger`
    and :math:`U^\dagger` overlaps.

    Parameters
    ----------
    config_path
        Path to the TOML configuration file.
    working_dir
        Working directory for intermediate files.
    num_bands
        Manual override for the number of bands.
    """
    from kapaow._experimental.projectability import _make_qe_input_wfc
    from kapaow._experimental.workflows import run_qe_workflow
    from kapaow.fat_bands import (
        build_atoms_dict,
        compute_amn_from_wfc,
        prepare_comparison_sets,
    )

    if working_dir is None:
        working_dir = Path("tmp") / "gauge_comparison" / config_path.stem

    prep = prepare_comparison_sets(config_path, working_dir, num_bands=num_bands)

    # Run SCF + NSCF on the regular k-mesh
    qe_result = run_qe_workflow(
        prep.config.structure,
        working_dir,
        min_nbnd=prep.min_nbnd,
        periodic=prep.config.periodic,
    )
    kpoint_weights = qe_result.kpoint_weights

    atoms_dict, lattice_vectors = build_atoms_dict(qe_result.nscf_input_file)
    qe_wfc = _make_qe_input_wfc(qe_result.nscf_wfc_dir, lattice_vectors)
    num_kpoints = len(kpoint_weights)

    # Compute S^{-1/2} A and U for each set
    all_matrices: list[npt.NDArray[np.complex128]] = []
    matrix_labels: list[str] = []
    # Keep per-set A and U separately for principal angle comparison
    set_amn: list[npt.NDArray[np.complex128]] = []
    set_umn: list[npt.NDArray[np.complex128]] = []

    for i, bessel_files in enumerate(prep.all_bessel):
        smn, amn, _cmn, _channel_indices = compute_amn_from_wfc(
            qe_wfc=qe_wfc,
            bessel_files=bessel_files,
            atoms_dict=atoms_dict,
            lattice_vectors=lattice_vectors,
            num_kpoints=num_kpoints,
        )
        amn_orth = orthogonalize_amn(smn, amn)
        u_mn = compute_u_from_svd(amn_orth)

        set_amn.append(amn_orth)
        set_umn.append(u_mn)

        all_matrices.append(amn_orth)
        matrix_labels.append(f"A: {prep.labels[i]}")
        all_matrices.append(u_mn)
        matrix_labels.append(f"U: {prep.labels[i]}")

    # Build pairwise Frobenius distance table
    n = len(all_matrices)
    distance_table = np.zeros((n, n))
    for i, j in itertools.combinations(range(n), 2):
        if all_matrices[i].shape != all_matrices[j].shape:
            distance_table[i, j] = np.nan
            distance_table[j, i] = np.nan
        else:
            d = bz_averaged_frobenius_distance(all_matrices[i], all_matrices[j], kpoint_weights)
            distance_table[i, j] = d
            distance_table[j, i] = d

    # Principal angles for cross-set pairs
    pa_comparisons = []
    for i, j in itertools.combinations(range(len(set_amn)), 2):
        if set_amn[i].shape != set_amn[j].shape:
            continue
        angles_a = bz_averaged_principal_angles(set_amn[i], set_amn[j], kpoint_weights)
        angles_u = bz_averaged_principal_angles(set_umn[i], set_umn[j], kpoint_weights)
        pa_comparisons.append((prep.labels[i], prep.labels[j], angles_a, angles_u))

    return GaugeComparisonResult(
        matrix_labels=matrix_labels,
        distance_table=distance_table,
        principal_angle_comparisons=pa_comparisons,
    )


def format_distance_table(
    labels: list[str],
    table: npt.NDArray[np.float64],
) -> str:
    """Format a pairwise distance table as a human-readable string.

    Parameters
    ----------
    labels
        Row/column labels.
    table
        Symmetric distance matrix.

    Returns
    -------
    str
        Formatted table string.
    """
    n = len(labels)
    col_width = max(len(lbl) for lbl in labels)
    val_width = max(col_width, 10)

    header = " " * (col_width + 2)
    header += "  ".join(f"{lbl:>{val_width}}" for lbl in labels)
    lines = [header]
    lines.append("-" * len(header))

    for i in range(n):
        row = f"{labels[i]:<{col_width}}  "
        cells = []
        for j in range(n):
            if i == j:
                cells.append(f"{'---':>{val_width}}")
            elif np.isnan(table[i, j]):
                cells.append(f"{'n/a':>{val_width}}")
            else:
                cells.append(f"{table[i, j]:>{val_width}.6f}")
        row += "  ".join(cells)
        lines.append(row)

    return "\n".join(lines)
