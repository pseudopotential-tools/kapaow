"""Fat band computation and plotting using qe_wavefunctions for Amn."""

from pathlib import Path

import numpy as np
import numpy.typing as npt
from ase_koopmans.io.espresso import read_espresso_in, read_espresso_out
from ase_koopmans.units import Bohr
from scipy.interpolate import make_interp_spline
from ase_koopmans.spectrum.band_structure import BandStructure, BandStructurePlot
from matplotlib import pyplot as plt
from matplotlib.collections import PolyCollection
import matplotlib.colors as mcolors
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator, FixedFormatter
from qe_wavefunctions.atomic_wfcs import AtomicWFC
from qe_wavefunctions.qe_input_wfcs import QEInputWFC
from qe_wavefunctions.qe_projections import compute_atomic_projections


def find_qe_files(
    qe_dir: Path, prefix: str | None = None
) -> tuple[Path, Path, Path, str]:
    """Locate QE input, output, and save directory within a calculation directory.

    Parameters
    ----------
    qe_dir
        Directory containing the QE bands calculation.
    prefix
        QE prefix. If None, auto-detected from the .save directory name.

    Returns
    -------
    pwi_file, pwo_file, outdir, prefix
    """
    pwi_files = list(qe_dir.glob("*.pwi"))
    if len(pwi_files) != 1:
        raise FileNotFoundError(
            f"Expected exactly one .pwi file in {qe_dir}, found {len(pwi_files)}: {pwi_files}"
        )
    pwi_file = pwi_files[0]

    pwo_files = list(qe_dir.glob("*.pwo"))
    if len(pwo_files) != 1:
        raise FileNotFoundError(
            f"Expected exactly one .pwo file in {qe_dir}, found {len(pwo_files)}: {pwo_files}"
        )
    pwo_file = pwo_files[0]

    # Find the .save directory and extract prefix
    save_dirs = list(qe_dir.rglob("*.save"))
    if len(save_dirs) == 0:
        raise FileNotFoundError(f"No .save directory found under {qe_dir}")
    save_dir = save_dirs[0]
    detected_prefix = save_dir.name.removesuffix(".save")
    outdir = save_dir.parent

    return pwi_file, pwo_file, outdir, prefix or detected_prefix


def read_fermi_energy(pwo_file: Path) -> float:
    """Read the Fermi energy from a QE output file.

    Parameters
    ----------
    pwo_file
        Path to the QE .pwo output file (typically from an nscf calculation).

    Returns
    -------
    float
        Fermi energy in eV.
    """
    atoms = list(read_espresso_out(str(pwo_file)))[-1]
    return float(atoms.calc.eFermi)


def read_qe_bands(
    pwo_file: Path, pwi_file: Path, reference: float | None = None
) -> BandStructure:
    """Read a QE bands calculation and return a BandStructure object.

    Parameters
    ----------
    pwo_file
        Path to the QE .pwo output file from a bands calculation.
    pwi_file
        Path to the QE .pwi input file (used for cell and k-path info).
    reference
        Reference energy (e.g. Fermi level) in eV. If None, attempts to read
        from the bands output file, falling back to 0.0.

    Returns
    -------
    BandStructure
        Band structure with eigenvalues and k-path.
    """
    atoms_in = read_espresso_in(str(pwi_file))
    atoms_out = list(read_espresso_out(str(pwo_file)))[-1]

    calc = atoms_out.calc

    # Determine number of spins
    spins = sorted({kpt.s for kpt in calc.kpts})

    # Group eigenvalues by spin
    eigenvalues_by_spin = []
    for s in spins:
        eigs = np.array([kpt.eps_n for kpt in calc.kpts if kpt.s == s])
        eigenvalues_by_spin.append(eigs)
    energies = np.array(eigenvalues_by_spin)

    # Extract k-points (fractional coordinates)
    kpts_frac = np.array([kpt.k for kpt in calc.kpts if kpt.s == spins[0]])

    # Build a BandPath with properly detected kinks and path segments
    from ase_koopmans.dft.kpoints import find_bandpath_kinks, resolve_custom_points

    path = atoms_in.cell.bandpath(npoints=0)
    kinks = find_bandpath_kinks(atoms_in.cell, kpts_frac, eps=1e-5)
    pathspec = resolve_custom_points(kpts_frac[kinks], path.special_points, eps=1e-5)
    path._kpts = kpts_frac
    path._path = pathspec

    if reference is None:
        efermi = getattr(calc, 'eFermi', None)
        if efermi is None:
            raise ValueError(
                f"No Fermi energy found in {pwo_file}. "
                "Provide a reference energy explicitly (e.g. from an nscf calculation)."
            )
        reference = float(efermi)

    return BandStructure(path=path, energies=energies, reference=reference)



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
) -> tuple[npt.NDArray[np.complex128], npt.NDArray[np.complex128], dict[tuple[str, int], list[int]]]:
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
        Atomic structure info: ``{species: {'num_atoms': int, 'positions': [[x,y,z], ...]}}``.
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

    atomic_wfc = AtomicWFC(atoms_dict=atoms_dict, lattice_vectors=lattice_vectors)
    species_list = list(bessel_files.keys())
    file_list = [str(bessel_files[s]) for s in species_list]
    atomic_wfc.load_atomic_wfcs(file_list)

    channel_indices = get_channel_indices(atomic_wfc, species_list)

    amn_list = []
    cmn_list = []
    for ik in range(1, num_kpoints + 1):
        kpt, kvec, miller, wfcs = qe_wfc.get_wfc(ik)
        _, a_mn, c_mn = compute_atomic_projections(atomic_wfc, kpt, miller, wfcs)
        amn_list.append(a_mn)
        cmn_list.append(c_mn)

    return np.array(amn_list), np.array(cmn_list), channel_indices


def compute_projectability_per_band(
    amn: npt.NDArray[np.complex128],
    cmn: npt.NDArray[np.complex128],
) -> npt.NDArray[np.float64]:
    """Compute total projectability for each band at each k-point.

    Uses ``Re(C_mn* A_mn)`` to correctly account for the non-orthogonality
    of the atomic projectors, where ``C_mn = S^{-1} A_mn``.

    Parameters
    ----------
    amn
        Array of shape (num_kpoints, num_orbitals, num_bands).
    cmn
        Array of shape (num_kpoints, num_orbitals, num_bands).

    Returns
    -------
    np.ndarray
        Array of shape (num_kpoints, num_bands) with projectability values in [0, 1].
    """
    return np.sum(np.conj(cmn) * amn, axis=1).real


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
        result[key] = np.sum(
            np.conj(cmn[:, indices, :]) * amn[:, indices, :], axis=1
        ).real
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
    koopmans_dir: Path,
    bessel_files: dict[str, Path],
    prefix: str | None = None,
    emin: float | None = None,
    emax: float | None = None,
    filename: Path | None = None,
    title: str | None = None,
) -> None:
    """Generate a fat bands plot from a koopmans workflow directory.

    Parameters
    ----------
    koopmans_dir
        Directory containing 01-scf/, 02-nscf/, and 03-bands/ subdirectories.
    bessel_files
        ``{species: Path}`` mapping to Bessel HDF5 files.
    prefix
        QE calculation prefix. If None, auto-detected.
    emin, emax
        Energy range relative to the Fermi level.
    filename
        If provided, save the figure to this path.
    title
        If provided, set as the figure title.
    """
    bands_dir = koopmans_dir / "03-bands"
    nscf_dir = koopmans_dir / "02-nscf"

    pwi_file, pwo_file, outdir, detected_prefix = find_qe_files(bands_dir, prefix)
    prefix = prefix or detected_prefix

    nscf_pwo = list(nscf_dir.glob("*.pwo"))
    if len(nscf_pwo) != 1:
        raise FileNotFoundError(
            f"Expected exactly one .pwo file in {nscf_dir}, found {len(nscf_pwo)}"
        )
    fermi_energy = read_fermi_energy(nscf_pwo[0])

    bs = read_qe_bands(pwo_file, pwi_file, reference=fermi_energy)
    atoms_dict, lattice_vectors = build_atoms_dict(pwi_file)

    num_kpoints = bs.energies.shape[1]
    amn, cmn, channel_indices = compute_amn(
        qe_outdir=outdir,
        prefix=prefix,
        bessel_files=bessel_files,
        atoms_dict=atoms_dict,
        lattice_vectors=lattice_vectors,
        num_kpoints=num_kpoints,
    )
    channel_proj = compute_projectability_per_channel(amn, cmn, channel_indices)

    plot_fat_bands(bs, channel_proj, emin=emin, emax=emax, filename=filename, title=title)


def plot_fat_bands(
    bs: BandStructure,
    channel_projectabilities: dict[tuple[str, int], npt.NDArray[np.float64]],
    emin: float | None = None,
    emax: float | None = None,
    filename: Path | None = None,
    title: str | None = None,
) -> None:
    """Plot fat bands with per-(species, l) channel colors and alpha encoding projectability.

    Non-oxygen species use red/green/blue for s/p/d; oxygen uses cyan/magenta/yellow.

    Parameters
    ----------
    bs
        Band structure with eigenvalues and k-path.
    channel_projectabilities
        ``{(species, l): array of shape (num_kpoints, num_bands)}`` from
        :func:`compute_projectability_per_channel`.
    emin, emax
        Energy range relative to the reference. If None, determined from the
        data with 0.5 eV padding.
    filename
        If provided, save the figure to this path.
    title
        If provided, set as the figure title.
    """
    bs = bs.subtract_reference()
    padding = 0.025 * (bs.energies.max() - bs.energies.min())
    if emin is None:
        emin = float(bs.energies.min()) - padding
    if emax is None:
        emax = float(bs.energies.max()) + padding

    # Create figure with two subplots sharing y-axis
    from pao_plusplus.plotting import REVTEX_COLUMN_WIDTH
    fig, (ax, ax_proj) = plt.subplots(
        1, 2, sharey=True, width_ratios=[4, 1],
        figsize=(REVTEX_COLUMN_WIDTH, REVTEX_COLUMN_WIDTH * 0.75),
        gridspec_kw={"wspace": 0.05},
    )

    bsp = BandStructurePlot(bs)
    ax = bsp.prepare_plot(ax=ax, emin=emin, emax=emax)

    xcoords = np.asarray(bsp.xcoords)
    e_skn = bs.energies

    # Split k-path into continuous segments at discontinuities (duplicate x-coords)
    break_indices = list(np.where(np.diff(xcoords) == 0)[0] + 1)
    seg_slices = []
    start = 0
    for brk in break_indices:
        seg_slices.append(slice(start, brk))
        start = brk
    seg_slices.append(slice(start, len(xcoords)))

    # Color palettes per species group
    _tab10 = plt.cm.tab10.colors
    DEFAULT_L_COLORS = {
        0: _tab10[2],  # green (s)
        1: _tab10[0],  # blue (p)
        2: _tab10[4],  # purple (d)
        3: _tab10[5],  # brown (f)
        4: _tab10[6],  # pink (g)
    }
    OXYGEN_L_COLORS = {
        0: _tab10[1],  # orange (s)
        1: _tab10[3],  # red (p)
    }

    def _get_color(species: str, l: int) -> tuple[float, float, float]:
        palette = OXYGEN_L_COLORS if species == "O" else DEFAULT_L_COLORS
        return mcolors.to_rgb(palette.get(l, "#7f7f7f"))

    channel_keys = sorted(channel_projectabilities.keys())

    # Compute total projectability (sum over channels)
    total_proj = sum(channel_projectabilities.values())

    # Half-width for fat-band quads in energy units
    half_w = (emax - emin) * 0.004

    for spin_idx, e_kn in enumerate(e_skn):
        for band_idx in range(e_kn.shape[1]):
            energies = e_kn[:, band_idx]

            for seg_sl in seg_slices:
                x_seg = xcoords[seg_sl]
                e_seg = energies[seg_sl]
                if len(x_seg) < 2:
                    continue

                k = min(3, len(x_seg) - 1)
                x_fine = np.linspace(x_seg[0], x_seg[-1], len(x_seg) * 3)
                e_fine = make_interp_spline(x_seg, e_seg, k=k)(x_fine)

                # Thin background line
                ax.plot(x_fine, e_fine, color=(0.8, 0.8, 0.8), linewidth=0.5, zorder=1)

                for species, l in channel_keys:
                    proj = channel_projectabilities[(species, l)][seg_sl, band_idx]
                    p_fine = make_interp_spline(x_seg, proj, k=k)(x_fine)
                    p_fine = np.clip(p_fine, 0, 1)

                    # Build trapezoid quads with uniform perpendicular thickness.
                    # At each interior vertex, offset along the bisector of the
                    # adjacent segment normals so that adjacent trapezoids share
                    # edges exactly (no gaps or overlaps).
                    verts = []
                    face_colors = []
                    rgb = _get_color(species, l)

                    # Work in display coordinates for correct aspect ratio
                    disp_transform = ax.transData
                    inv_transform = disp_transform.inverted()
                    pts_data = np.column_stack([x_fine, e_fine])
                    pts_disp = disp_transform.transform(pts_data)

                    # Per-segment unit normals in display space
                    dx_disp = np.diff(pts_disp[:, 0])
                    dy_disp = np.diff(pts_disp[:, 1])
                    seg_len = np.hypot(dx_disp, dy_disp)
                    seg_len = np.where(seg_len == 0, 1, seg_len)
                    # Normal perpendicular to segment (rotated 90° CCW)
                    seg_nx = -dy_disp / seg_len
                    seg_ny = dx_disp / seg_len

                    # half_w in data coords -> display coords
                    origin_disp = disp_transform.transform([[0, 0]])[0]
                    hw_point = disp_transform.transform([[0, half_w]])[0]
                    hw_disp = abs(hw_point[1] - origin_disp[1])

                    # Compute offset vectors at each vertex (bisector-based)
                    n_pts = len(x_fine)
                    offsets_disp = np.empty((n_pts, 2))

                    # Endpoints: vertical edges, scaled so perpendicular width is preserved.
                    # For a segment at angle θ, a vertical cut has height hw_disp / cos(θ)
                    # where cos(θ) = |ny| (the y-component of the unit normal).
                    cos_first = abs(seg_ny[0]) if abs(seg_ny[0]) > 0.3 else 0.3
                    cos_last = abs(seg_ny[-1]) if abs(seg_ny[-1]) > 0.3 else 0.3
                    offsets_disp[0] = np.array([0, hw_disp / cos_first])
                    offsets_disp[-1] = np.array([0, hw_disp / cos_last])

                    # Interior points: bisector of adjacent normals, scaled so
                    # that the perpendicular distance to each segment is hw_disp
                    for j in range(1, n_pts - 1):
                        n_prev = np.array([seg_nx[j - 1], seg_ny[j - 1]])
                        n_next = np.array([seg_nx[j], seg_ny[j]])
                        bisector = n_prev + n_next
                        bisector_len = np.linalg.norm(bisector)
                        if bisector_len < 1e-12:
                            # Segments are collinear; use either normal
                            offsets_disp[j] = hw_disp * n_prev
                        else:
                            bisector /= bisector_len
                            # Scale so perpendicular distance to segment = hw_disp
                            # dot(bisector, n_prev) = cos(half-angle)
                            cos_half = np.dot(bisector, n_prev)
                            cos_half = max(cos_half, 0.3)  # clamp to avoid extreme miter spikes
                            offsets_disp[j] = (hw_disp / cos_half) * bisector

                    # Build quads using precomputed per-vertex offsets
                    for i in range(n_pts - 1):
                        corners_disp = np.array([
                            pts_disp[i] - offsets_disp[i],
                            pts_disp[i] + offsets_disp[i],
                            pts_disp[i + 1] + offsets_disp[i + 1],
                            pts_disp[i + 1] - offsets_disp[i + 1],
                        ])
                        corners_data = inv_transform.transform(corners_disp)
                        verts.append(corners_data.tolist())

                        alpha = (p_fine[i] + p_fine[i + 1]) / 2
                        face_colors.append((*rgb, float(alpha)))

                    pc = PolyCollection(verts, facecolors=face_colors, edgecolors="none", zorder=2)
                    ax.add_collection(pc)

            # Side panel: scatter total projectability vs energy
            band_proj = total_proj[:, band_idx]
            ax_proj.scatter(_proj_transform(band_proj), energies, s=0.5, color="k", alpha=1)

    # Legend: group by species, show species name + l label
    legend_handles = []
    for species, l in channel_keys:
        rgb = _get_color(species, l)
        label = f"{species} {L_LABELS.get(l, '?')}"
        legend_handles.append(
            Line2D([0], [0], color=rgb, linewidth=2, label=label)
        )
    ax.legend(handles=legend_handles, loc="upper right", fontsize="small")

    # Configure side panel with nonlinear (shifted-logit) x-axis
    tick_values = [0, 0.01, 0.1, 0.5, 0.9, 0.99, 1]
    tick_labels = ["0", "0.01", "0.1", "0.5", "0.9", "0.99", "1"]
    transformed_ticks = _proj_transform(np.array(tick_values))
    ax_proj.set_xlim(transformed_ticks[0], transformed_ticks[-1])
    ax_proj.set_xlabel("Projectability")
    ax_proj.axvline(_proj_transform(np.array([1.0]))[0], color="k", ls=":", lw=0.5)
    ax_proj.tick_params(labelleft=False, labelsize="x-small", labelrotation=90)
    ax_proj.xaxis.set_major_locator(FixedLocator(transformed_ticks))
    ax_proj.xaxis.set_major_formatter(FixedFormatter(tick_labels))
    ax_proj.xaxis.set_minor_locator(FixedLocator([]))  # disable minor ticks

    if title is not None:
        fig.suptitle(title)

    bsp.finish_plot(filename=str(filename) if filename else None, show=False, loc=None)

    plt.close(fig)
