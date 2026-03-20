"""Unweighted band distance metric for comparing DFT and Wannier-interpolated bands.

Unlike the Fermi-Dirac-weighted eta of Qiao et al. (2023), this metric treats
all bands equally -- occupied and empty states contribute the same weight.
This is appropriate when the quality of empty-state interpolation matters
(e.g. for Koopmans spectral functionals).

Both band structures must be evaluated on the same k-point grid (same number
of k-points in the same order). This is ensured by passing the DFT bands
k-points as explicit ``bands_kpoints`` to Wannier90.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt


@dataclass
class BandDistanceResult:
    """Result of an unweighted band distance calculation."""

    eta: float
    """RMS band distance (eV), averaged over k-points and bands."""
    max_dist: float
    """Maximum absolute band difference (eV) across all k-points and bands."""
    max_dist_k: int
    """K-point index at which the maximum distance occurs."""
    max_dist_band: int
    """Band index at which the maximum distance occurs."""
    per_band_eta: npt.NDArray[np.float64]
    """RMS distance per band (eV), shape (num_bands,)."""


def bands_distance(
    dft_energies: npt.NDArray[np.float64],
    wannier_energies: npt.NDArray[np.float64],
) -> BandDistanceResult:
    """Compute unweighted RMS band distance between DFT and Wannier bands.

    Both band structures must be on the same k-point grid (same number of
    k-points). Only the lowest ``min(num_dft_bands, num_wannier_bands)``
    bands are compared.

    Args:
        dft_energies: DFT band energies relative to Fermi level,
            shape (num_kpoints, num_dft_bands).
        wannier_energies: Wannier band energies relative to Fermi level,
            shape (num_kpoints, num_wannier_bands).

    Returns:
        BandDistanceResult with the computed metrics.

    Raises:
        ValueError: If the number of k-points differs.
    """
    if dft_energies.shape[0] != wannier_energies.shape[0]:
        raise ValueError(
            f"K-point count mismatch: DFT has {dft_energies.shape[0]}, "
            f"Wannier has {wannier_energies.shape[0]}. "
            "Both must use the same k-grid."
        )

    num_bands = min(dft_energies.shape[1], wannier_energies.shape[1])
    dft_trimmed = dft_energies[:, :num_bands]
    wan_trimmed = wannier_energies[:, :num_bands]

    diff = dft_trimmed - wan_trimmed  # (num_kpoints, num_bands)

    # Per-band RMS
    per_band_eta = np.sqrt(np.mean(diff**2, axis=0))

    # Global RMS
    eta = float(np.sqrt(np.mean(diff**2)))

    # Maximum absolute difference
    abs_diff = np.abs(diff)
    max_idx = np.unravel_index(np.argmax(abs_diff), abs_diff.shape)
    max_dist = float(abs_diff[max_idx])

    return BandDistanceResult(
        eta=eta,
        max_dist=max_dist,
        max_dist_k=int(max_idx[0]),
        max_dist_band=int(max_idx[1]),
        per_band_eta=per_band_eta,
    )
