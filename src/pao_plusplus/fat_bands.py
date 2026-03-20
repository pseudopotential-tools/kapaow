"""Fat band computation and plotting using qe_wavefunctions for Amn."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.colors as mcolors
import numpy as np
import numpy.typing as npt
from ase.io.espresso import read_espresso_in
from ase.units import Bohr
from matplotlib import pyplot as plt
from matplotlib.collections import PolyCollection
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedFormatter, FixedLocator
from qe_wavefunctions.atomic_wfcs import AtomicWFC
from qe_wavefunctions.qe_input_wfcs import QEInputWFC
from qe_wavefunctions.qe_projections import compute_atomic_projections
from scipy.interpolate import make_interp_spline

from pao_plusplus.workflows import BandPlotData

L_LABELS = {0: "s", 1: "p", 2: "d", 3: "f", 4: "g"}

# Small offset so that log10(0 + _EPS) and log10(1 - 0 + _EPS) are finite.
_EPS = 1e-3


def _proj_transform(x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Nonlinear transform that stretches projectability near 0 and 1.

    Uses a shifted logit: ``log10(x + eps) - log10(1 - x + eps)``.
    This is finite at x = 0 and x = 1.
    """
    x = np.asarray(x, dtype=np.float64)
    return np.log10(x + _EPS) - np.log10(1 - x + _EPS)


def get_channel_indices(
    atomic_wfc: AtomicWFC,
    species_names: list[str],
) -> dict[tuple[str, int], list[int]]:
    """Map each (species, l) pair to its global orbital indices in the Amn matrix.

    Sums over all atoms, m values, and radial indices n for each (species, l).

    Parameters
    ----------
    atomic_wfc
        AtomicWFC object with loaded wavefunctions.
    species_names
        List of species names in the same order as in the atoms_dict.

    Returns
    -------
    dict
        ``{(species, l): [orbital_indices]}`` mapping.
    """
    channels: dict[tuple[str, int], list[int]] = {}
    for ispec in range(atomic_wfc.num_species):
        species = species_names[ispec]
        lmax = atomic_wfc.lmax_species[ispec]
        nmax = atomic_wfc.nmax_species[ispec]
        for iat in range(atomic_wfc.num_atoms[ispec]):
            atom_idx = sum(atomic_wfc.num_atoms[:ispec]) + iat
            base = atomic_wfc.start_indices[atom_idx]
            for l in range(lmax + 1):
                key = (species, l)
                if key not in channels:
                    channels[key] = []
                for n in range(nmax + 1):
                    for m in range(-l, l + 1):
                        flat = (l**2 + l + m) * (nmax + 1) + n
                        channels[key].append(base + flat)
    return channels


def compute_amn(
    qe_outdir: Path,
    prefix: str,
    bessel_files: dict[str, Path],
    atoms_dict: dict,
    lattice_vectors: npt.NDArray[np.float64],
    num_kpoints: int,
) -> tuple[
    npt.NDArray[np.complex128],
    npt.NDArray[np.complex128],
    dict[tuple[str, int], list[int]],
]:
    """Compute the Amn projection matrix at each k-point.

    Parameters
    ----------
    qe_outdir
        Path to the QE output directory (containing prefix.save/).
    prefix
        QE calculation prefix.
    bessel_files
        Mapping of species name to Bessel HDF5 file path.
    atoms_dict
        Atomic structure info dict keyed by species.
    lattice_vectors
        Real-space lattice vectors as a 3x3 array.
    num_kpoints
        Number of k-points in the bands calculation.

    Returns
    -------
    amn
        Array of shape (num_kpoints, num_orbitals, num_bands) with complex Amn values.
    cmn
        Array of shape (num_kpoints, num_orbitals, num_bands) with ``S^{-1} A`` values.
    channel_indices
        ``{(species, l): [orbital_indices]}`` mapping for per-channel decomposition.
    """
    qe_wfc = QEInputWFC(
        outdir=str(qe_outdir),
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

    channel_indices = get_channel_indices(atomic_wfc, species_list)

    amn_list = []
    cmn_list = []
    for ik in range(1, num_kpoints + 1):
        kpt, _kvec, miller, wfcs = qe_wfc.get_wfc(ik)
        _, a_mn, c_mn = compute_atomic_projections(atomic_wfc, kpt, miller, wfcs)
        amn_list.append(a_mn)
        cmn_list.append(c_mn)

    return np.array(amn_list), np.array(cmn_list), channel_indices


def compute_amn_from_wfc(
    qe_wfc: QEInputWFC,
    bessel_files: dict[str, Path],
    atoms_dict: dict,
    lattice_vectors: npt.NDArray[np.float64],
    num_kpoints: int,
) -> tuple[
    npt.NDArray[np.complex128],
    npt.NDArray[np.complex128],
    dict[tuple[str, int], list[int]],
]:
    """Compute the Amn projection matrix using a pre-configured QEInputWFC.

    Like :func:`compute_amn` but accepts a QEInputWFC directly, which allows
    working with flat wfc directories from AiiDA dumps.
    """
    atomic_wfc = AtomicWFC(
        atoms_dict=atoms_dict,
        lattice_vectors=lattice_vectors,
    )
    species_list = list(bessel_files.keys())
    file_list = [str(bessel_files[s]) for s in species_list]
    atomic_wfc.load_atomic_wfcs(file_list)

    channel_indices = get_channel_indices(atomic_wfc, species_list)

    amn_list = []
    cmn_list = []
    for ik in range(1, num_kpoints + 1):
        kpt, _kvec, miller, wfcs = qe_wfc.get_wfc(ik)
        _, a_mn, c_mn = compute_atomic_projections(atomic_wfc, kpt, miller, wfcs)
        amn_list.append(a_mn)
        cmn_list.append(c_mn)

    return np.array(amn_list), np.array(cmn_list), channel_indices


def compute_projectability_per_channel(
    amn: npt.NDArray[np.complex128],
    cmn: npt.NDArray[np.complex128],
    channel_indices: dict[tuple[str, int], list[int]],
) -> dict[tuple[str, int], npt.NDArray[np.float64]]:
    """Compute projectability decomposed by (species, l) channel.

    Uses ``Re(C_mn* A_mn)`` to correctly account for non-orthogonality.

    Parameters
    ----------
    amn
        Array of shape (num_kpoints, num_orbitals, num_bands).
    cmn
        Array of shape (num_kpoints, num_orbitals, num_bands).
    channel_indices
        ``{(species, l): [orbital_indices]}`` mapping.

    Returns
    -------
    dict
        ``{(species, l): array of shape (num_kpoints, num_bands)}`` with per-channel
        projectability values.
    """
    result = {}
    for key, indices in channel_indices.items():
        result[key] = np.sum(np.conj(cmn[:, indices, :]) * amn[:, indices, :], axis=1).real
    return result


def build_atoms_dict(
    pwi_file: Path,
) -> tuple[dict, npt.NDArray[np.float64]]:
    """Build the atoms_dict and lattice_vectors from a QE input file.

    Parameters
    ----------
    pwi_file
        Path to the QE .pwi input file.

    Returns
    -------
    atoms_dict
        ``{species: {'num_atoms': int, 'positions': [[x,y,z], ...]}}``.
    lattice_vectors
        3x3 array of lattice vectors in Angstrom.
    """
    atoms = read_espresso_in(str(pwi_file))
    # Convert to Bohr: QE stores xk in Bohr^-1 and the Bessel qgrid is in Bohr^-1,
    # so the reciprocal lattice (and hence q_magnitudes) must also be in Bohr^-1.
    lattice_vectors = np.array(atoms.cell) / Bohr

    # Build atoms_dict grouped by species
    atoms_dict: dict[str, dict] = {}
    scaled_positions = atoms.get_scaled_positions()
    for symbol, pos in zip(atoms.get_chemical_symbols(), scaled_positions, strict=True):
        if symbol not in atoms_dict:
            atoms_dict[symbol] = {"num_atoms": 0, "positions": []}
        atoms_dict[symbol]["num_atoms"] += 1
        atoms_dict[symbol]["positions"].append(pos.tolist())

    return atoms_dict, lattice_vectors


def generate_fat_bands_plot(
    bands_calc_dir: Path,
    band_plot_data: BandPlotData,
    bessel_files: dict[str, Path],
    emin: float | None = None,
    emax: float | None = None,
    filename: Path | None = None,
) -> None:
    """Generate a fat bands plot from an AiiDA-dumped bands calculation.

    Parameters
    ----------
    bands_calc_dir
        Path to the dumped PwCalculation directory (containing inputs/ and
        outputs/ subdirectories). The wfc HDF5 files should be flat in
        outputs/.
    band_plot_data
        Pre-computed band structure data (x-distances, energies, labels)
        from the AiiDA BandsData node.
    bessel_files
        ``{species: Path}`` mapping to Bessel HDF5 files.
    emin, emax
        Energy range relative to the Fermi level.
    filename
        If provided, save the figure to this path.
    """
    from pao_plusplus.projectability import _make_qe_input_wfc

    pwi_file = bands_calc_dir / "inputs" / "aiida.in"
    wfc_dir = bands_calc_dir / "outputs"

    atoms_dict, lattice_vectors = build_atoms_dict(pwi_file)

    # Use the flat wfc directory (AiiDA dump puts wfc files directly in outputs/)
    qe_wfc = _make_qe_input_wfc(wfc_dir, lattice_vectors)

    num_kpoints = band_plot_data.energies.shape[0]
    amn, cmn, channel_indices = compute_amn_from_wfc(
        qe_wfc=qe_wfc,
        bessel_files=bessel_files,
        atoms_dict=atoms_dict,
        lattice_vectors=lattice_vectors,
        num_kpoints=num_kpoints,
    )
    channel_proj = compute_projectability_per_channel(amn, cmn, channel_indices)

    plot_fat_bands(
        band_plot_data,
        channel_proj,
        emin=emin,
        emax=emax,
        filename=filename,
    )


def _split_kpath_segments(xcoords: npt.NDArray[np.float64]) -> list[slice]:
    """Split k-path into continuous segments at duplicate x-coordinates."""
    break_indices = list(np.where(np.diff(xcoords) == 0)[0] + 1)
    seg_slices = []
    start = 0
    for brk in break_indices:
        seg_slices.append(slice(start, brk))
        start = brk
    seg_slices.append(slice(start, len(xcoords)))
    return seg_slices


def _get_channel_color(species: str, l: int) -> tuple[float, float, float]:
    """Return RGB color for a (species, l) channel."""
    _tab10 = plt.cm.tab10.colors
    default_l_colors = {
        0: _tab10[2],  # green (s)
        1: _tab10[0],  # blue (p)
        2: _tab10[4],  # purple (d)
        3: _tab10[5],  # brown (f)
    }
    oxygen_l_colors = {
        0: _tab10[1],  # orange (s)
        1: _tab10[3],  # red (p)
    }
    palette = oxygen_l_colors if species == "O" else default_l_colors
    return mcolors.to_rgb(palette.get(l, "#7f7f7f"))


def _compute_vertex_offsets(
    pts_disp: npt.NDArray[np.float64],
    hw_disp: float,
) -> npt.NDArray[np.float64]:
    """Compute per-vertex perpendicular offset vectors in display space."""
    dx_disp = np.diff(pts_disp[:, 0])
    dy_disp = np.diff(pts_disp[:, 1])
    seg_len = np.hypot(dx_disp, dy_disp)
    seg_len = np.where(seg_len == 0, 1, seg_len)
    seg_nx = -dy_disp / seg_len
    seg_ny = dx_disp / seg_len

    n_pts = len(pts_disp)
    offsets = np.empty((n_pts, 2))

    cos_first = max(abs(seg_ny[0]), 0.3)
    cos_last = max(abs(seg_ny[-1]), 0.3)
    offsets[0] = np.array([0, hw_disp / cos_first])
    offsets[-1] = np.array([0, hw_disp / cos_last])

    for j in range(1, n_pts - 1):
        n_prev = np.array([seg_nx[j - 1], seg_ny[j - 1]])
        n_next = np.array([seg_nx[j], seg_ny[j]])
        bisector = n_prev + n_next
        bisector_len = np.linalg.norm(bisector)
        if bisector_len < 1e-12:
            offsets[j] = hw_disp * n_prev
        else:
            bisector /= bisector_len
            cos_half = max(np.dot(bisector, n_prev), 0.3)
            offsets[j] = (hw_disp / cos_half) * bisector

    return offsets


def _build_fat_band_quads(
    ax: Any,
    x_fine: npt.NDArray[np.float64],
    e_fine: npt.NDArray[np.float64],
    p_fine: npt.NDArray[np.float64],
    rgb: tuple[float, float, float],
    half_w: float,
) -> None:
    """Build and add trapezoid quad PolyCollection for one channel segment."""
    disp_transform = ax.transData
    inv_transform = disp_transform.inverted()
    pts_data = np.column_stack([x_fine, e_fine])
    pts_disp = disp_transform.transform(pts_data)

    origin_disp = disp_transform.transform([[0, 0]])[0]
    hw_point = disp_transform.transform([[0, half_w]])[0]
    hw_disp = abs(hw_point[1] - origin_disp[1])

    offsets = _compute_vertex_offsets(pts_disp, hw_disp)

    verts = []
    face_colors = []
    for i in range(len(x_fine) - 1):
        corners_disp = np.array(
            [
                pts_disp[i] - offsets[i],
                pts_disp[i] + offsets[i],
                pts_disp[i + 1] + offsets[i + 1],
                pts_disp[i + 1] - offsets[i + 1],
            ]
        )
        corners_data = inv_transform.transform(corners_disp)
        verts.append(corners_data.tolist())
        alpha = (p_fine[i] + p_fine[i + 1]) / 2
        face_colors.append((*rgb, float(alpha)))

    pc = PolyCollection(
        verts,
        facecolors=face_colors,
        edgecolors="none",
        zorder=2,
    )
    ax.add_collection(pc)


def _configure_proj_panel(ax_proj: Any) -> None:
    """Configure the projectability side panel with nonlinear x-axis."""
    tick_values = [0, 0.01, 0.1, 0.5, 0.9, 0.99, 1]
    tick_labels = ["0", "0.01", "0.1", "0.5", "0.9", "0.99", "1"]
    transformed_ticks = _proj_transform(np.array(tick_values))
    ax_proj.set_xlim(transformed_ticks[0], transformed_ticks[-1])
    ax_proj.set_xlabel("Projectability")
    ax_proj.axvline(
        _proj_transform(np.array([1.0]))[0],
        color="k",
        ls=":",
        lw=0.5,
    )
    ax_proj.tick_params(
        labelleft=False,
        labelsize="x-small",
        labelrotation=90,
    )
    ax_proj.xaxis.set_major_locator(FixedLocator(transformed_ticks))
    ax_proj.xaxis.set_major_formatter(FixedFormatter(tick_labels))
    ax_proj.xaxis.set_minor_locator(FixedLocator([]))


def plot_fat_bands(
    band_plot_data: BandPlotData,
    channel_projectabilities: dict[tuple[str, int], npt.NDArray[np.float64]],
    emin: float | None = None,
    emax: float | None = None,
    filename: Path | None = None,
) -> None:
    """Plot fat bands with per-(species, l) channel colors.

    Alpha encodes projectability. Non-oxygen species use
    red/green/blue for s/p/d; oxygen uses cyan/magenta/yellow.

    Parameters
    ----------
    band_plot_data
        Pre-computed band structure data from AiiDA BandsData node.
    channel_projectabilities
        ``{(species, l): array of shape (num_kpoints, num_bands)}`` from
        :func:`compute_projectability_per_channel`.
    emin, emax
        Energy range relative to the reference. If None, determined from the
        data with padding.
    filename
        If provided, save the figure to this path.
    """
    xcoords = band_plot_data.x
    # (num_kpoints, num_bands), already Fermi-shifted
    energies = band_plot_data.energies
    labels = band_plot_data.labels

    padding = 0.025 * (energies.max() - energies.min())
    if emin is None:
        emin = float(energies.min()) - padding
    if emax is None:
        emax = float(energies.max()) + padding

    # Create figure with two subplots sharing y-axis
    from pao_plusplus.plotting import REVTEX_COLUMN_WIDTH

    fig, (ax, ax_proj) = plt.subplots(
        1,
        2,
        sharey=True,
        width_ratios=[4, 1],
        figsize=(REVTEX_COLUMN_WIDTH, REVTEX_COLUMN_WIDTH * 0.75),
        gridspec_kw={"wspace": 0.1},
    )

    # Set up band structure axes (replaces BandStructurePlot.prepare_plot)
    ax.set_xlim(xcoords[0], xcoords[-1])
    ax.set_ylim(emin, emax)
    ax.set_ylabel("Energy (eV)")
    ax.axhline(0, color="k", ls="--", lw=0.5)

    # Draw vertical lines and labels at high-symmetry points
    label_positions = [pos for pos, _ in labels]
    label_strings = [lbl for _, lbl in labels]
    for pos in label_positions:
        ax.axvline(pos, color="k", ls="-", lw=0.5, alpha=0.5)
    ax.set_xticks(label_positions)
    ax.set_xticklabels(label_strings)

    seg_slices = _split_kpath_segments(xcoords)
    channel_keys = sorted(channel_projectabilities.keys())
    total_proj = sum(channel_projectabilities.values())
    half_w = (emax - emin) * 0.004

    num_bands = energies.shape[1]
    for band_idx in range(num_bands):
        band_energies = energies[:, band_idx]

        for seg_sl in seg_slices:
            x_seg = xcoords[seg_sl]
            e_seg = band_energies[seg_sl]
            if len(x_seg) < 2:
                continue

            k = min(3, len(x_seg) - 1)
            x_fine = np.linspace(x_seg[0], x_seg[-1], len(x_seg) * 3)
            e_fine = make_interp_spline(x_seg, e_seg, k=k)(x_fine)
            ax.plot(x_fine, e_fine, color=(0.8, 0.8, 0.8), linewidth=0.5, zorder=1)

            for species, l in channel_keys:
                proj = channel_projectabilities[(species, l)][seg_sl, band_idx]
                p_fine = np.clip(make_interp_spline(x_seg, proj, k=k)(x_fine), 0, 1)
                rgb = _get_channel_color(species, l)
                _build_fat_band_quads(ax, x_fine, e_fine, p_fine, rgb, half_w)

        band_proj = total_proj[:, band_idx]
        ax_proj.scatter(
            _proj_transform(band_proj),
            band_energies,
            s=0.5,
            color="k",
            alpha=1,
        )

    # Legend
    legend_handles = []
    for species, l in channel_keys:
        rgb = _get_channel_color(species, l)
        label = f"{species} {L_LABELS.get(l, '?')}"
        legend_handles.append(Line2D([0], [0], color=rgb, linewidth=2, label=label))
    ax.legend(
        handles=legend_handles,
        loc="lower right",
        bbox_to_anchor=(1, 1),
        fontsize="small",
        frameon=False,
        ncol=len(legend_handles),
    )

    _configure_proj_panel(ax_proj)

    fig.subplots_adjust(left=0.15, bottom=0.15, right=0.95)
    if filename is not None:
        fig.savefig(filename, dpi=300)
    plt.close(fig)
