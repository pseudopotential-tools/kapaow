"""Fat band computation and plotting using qe_wavefunctions for Amn."""

from pathlib import Path

import numpy as np
import numpy.typing as npt
from ase_koopmans.io.espresso import read_espresso_in, read_espresso_out
from ase_koopmans.units import Bohr
from scipy.interpolate import make_interp_spline
from ase_koopmans.spectrum.band_structure import BandStructure, BandStructurePlot
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.collections import LineCollection
from matplotlib.colorbar import Colorbar
from matplotlib.colors import Normalize
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


def get_channel_indices(
    atomic_wfc: AtomicWFC,
) -> dict[int, list[int]]:
    """Map each angular momentum l to its global orbital indices in the Amn matrix.

    Sums over all atoms, m values, and radial indices n for each l.

    Returns
    -------
    dict
        ``{l: [orbital_indices]}`` mapping.
    """
    channels: dict[int, list[int]] = {}
    for ispec in range(atomic_wfc.num_species):
        lmax = atomic_wfc.lmax_species[ispec]
        nmax = atomic_wfc.nmax_species[ispec]
        for iat in range(atomic_wfc.num_atoms[ispec]):
            atom_idx = sum(atomic_wfc.num_atoms[:ispec]) + iat
            base = atomic_wfc.start_indices[atom_idx]
            for l in range(lmax + 1):
                if l not in channels:
                    channels[l] = []
                for n in range(nmax + 1):
                    for m in range(-l, l + 1):
                        flat = (l**2 + l + m) * (nmax + 1) + n
                        channels[l].append(base + flat)
    return channels


def compute_amn(
    qe_outdir: Path,
    prefix: str,
    bessel_files: dict[str, Path],
    atoms_dict: dict,
    lattice_vectors: npt.NDArray[np.float64],
    num_kpoints: int,
) -> tuple[npt.NDArray[np.complex128], npt.NDArray[np.complex128], dict[int, list[int]]]:
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
        ``{l: [orbital_indices]}`` mapping for per-channel decomposition.
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

    channel_indices = get_channel_indices(atomic_wfc)

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
    channel_indices: dict[int, list[int]],
) -> dict[int, npt.NDArray[np.float64]]:
    """Compute projectability decomposed by l channel.

    Uses ``Re(C_mn* A_mn)`` to correctly account for non-orthogonality.

    Parameters
    ----------
    amn
        Array of shape (num_kpoints, num_orbitals, num_bands).
    cmn
        Array of shape (num_kpoints, num_orbitals, num_bands).
    channel_indices
        ``{l: [orbital_indices]}`` mapping.

    Returns
    -------
    dict
        ``{l: array of shape (num_kpoints, num_bands)}`` with per-channel
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


def plot_fat_bands(
    bs: BandStructure,
    channel_projectabilities: dict[int, npt.NDArray[np.float64]],
    emin: float | None = None,
    emax: float | None = None,
    filename: Path | None = None,
) -> Axes:
    """Plot fat bands with per-l channel colors and alpha encoding projectability.

    Includes a side panel showing total projectability vs energy.

    Parameters
    ----------
    bs
        Band structure with eigenvalues and k-path.
    channel_projectabilities
        ``{l: array of shape (num_kpoints, num_bands)}`` from
        :func:`compute_projectability_per_channel`.
    emin, emax
        Energy range relative to the reference. If None, determined from the
        data with 0.5 eV padding.
    filename
        If provided, save the figure to this path.

    Returns
    -------
    Axes
        The matplotlib axes with the fat bands plot.
    """
    import matplotlib.colors as mcolors

    bs = bs.subtract_reference()
    padding = 0.5
    if emin is None:
        emin = float(bs.energies.min()) - padding
    if emax is None:
        emax = float(bs.energies.max()) + padding

    # Create figure with two subplots sharing y-axis
    fig, (ax, ax_proj) = plt.subplots(
        1, 2, sharey=True, width_ratios=[4, 1],
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

    # Use matplotlib's tab10 palette: red, green, blue for s, p, d
    _tab10 = plt.cm.tab10.colors
    L_COLORS = {
        0: _tab10[3],  # red (s)
        1: _tab10[2],  # green (p)
        2: _tab10[0],  # blue (d)
        3: _tab10[4],  # purple (f)
        4: _tab10[5],  # brown (g)
    }
    channel_keys = sorted(channel_projectabilities.keys())

    # Compute total projectability (sum over channels)
    total_proj = sum(channel_projectabilities.values())

    for spin_idx, e_kn in enumerate(e_skn):
        for band_idx in range(e_kn.shape[1]):
            energies = e_kn[:, band_idx]

            for seg_sl in seg_slices:
                x_seg = xcoords[seg_sl]
                e_seg = energies[seg_sl]
                if len(x_seg) < 2:
                    continue

                k = min(3, len(x_seg) - 1)
                x_fine = np.linspace(x_seg[0], x_seg[-1], len(x_seg) * 10)
                e_fine = make_interp_spline(x_seg, e_seg, k=k)(x_fine)
                points = np.column_stack([x_fine, e_fine]).reshape(-1, 1, 2)
                segments = np.concatenate([points[:-1], points[1:]], axis=1)

                for key in channel_keys:
                    proj = channel_projectabilities[key][seg_sl, band_idx]
                    p_fine = make_interp_spline(x_seg, proj, k=k)(x_fine)
                    seg_alpha = np.clip((p_fine[:-1] + p_fine[1:]) / 2, 0, 1)

                    rgb = mcolors.to_rgb(L_COLORS.get(key, "#7f7f7f"))
                    colors = np.zeros((len(segments), 4))
                    colors[:, :3] = rgb
                    colors[:, 3] = seg_alpha

                    lc = LineCollection(segments, colors=colors)
                    lc.set_linewidth(3)
                    ax.add_collection(lc)

            # Side panel: scatter total projectability vs energy
            band_proj = total_proj[:, band_idx]
            ax_proj.scatter(band_proj, energies, s=0.5, color="k", alpha=1)

    # Legend for channels
    from matplotlib.lines import Line2D
    legend_handles = [
        Line2D([0], [0], color=L_COLORS.get(key, "#7f7f7f"), linewidth=2,
               label=f"l={key} ({L_LABELS.get(key, '?')})")
        for key in channel_keys
    ]
    ax.legend(handles=legend_handles, loc="upper right", fontsize="small")

    # Configure side panel
    ax_proj.set_xlim(0, 1)
    ax_proj.set_xlabel("Projectability")
    ax_proj.axvline(1, color="k", ls=":", lw=0.5)
    ax_proj.tick_params(labelleft=False)

    bsp.finish_plot(filename=str(filename) if filename else None, show=False, loc=None)

    return ax
