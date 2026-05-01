"""Fat band computation and plotting using qe_wavefunctions for Amn."""

from __future__ import annotations

import logging
from dataclasses import dataclass
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

logger = logging.getLogger(__name__)

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
    smn
        Array of shape (num_kpoints, num_orbitals, num_orbitals) with the
        overlap matrix at each k-point.
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

    smn_list = []
    amn_list = []
    cmn_list = []
    for ik in range(1, num_kpoints + 1):
        kpt, _kvec, miller, wfcs = qe_wfc.get_wfc(ik)
        s_mn, a_mn, c_mn = compute_atomic_projections(atomic_wfc, kpt, miller, wfcs)
        smn_list.append(s_mn)
        amn_list.append(a_mn)
        cmn_list.append(c_mn)

    return np.array(smn_list), np.array(amn_list), np.array(cmn_list), channel_indices


def compute_amn_from_wfc(
    qe_wfc: QEInputWFC,
    bessel_files: dict[str, Path],
    atoms_dict: dict,
    lattice_vectors: npt.NDArray[np.float64],
    num_kpoints: int,
) -> tuple[
    npt.NDArray[np.complex128],
    npt.NDArray[np.complex128],
    npt.NDArray[np.complex128],
    dict[tuple[str, int], list[int]],
]:
    """Compute the Amn projection matrix using a pre-configured QEInputWFC.

    Like :func:`compute_amn` but accepts a QEInputWFC directly, which allows
    working with flat wfc directories from AiiDA dumps.

    Returns
    -------
    smn
        Overlap matrices, shape ``(num_kpoints, num_orbitals, num_orbitals)``.
    amn
        Projection matrices, shape ``(num_kpoints, num_orbitals, num_bands)``.
    cmn
        ``S^{-1} A`` matrices, shape ``(num_kpoints, num_orbitals, num_bands)``.
    channel_indices
        ``{(species, l): [orbital_indices]}`` mapping.
    """
    atomic_wfc = AtomicWFC(
        atoms_dict=atoms_dict,
        lattice_vectors=lattice_vectors,
    )
    species_list = list(bessel_files.keys())
    file_list = [str(bessel_files[s]) for s in species_list]
    atomic_wfc.load_atomic_wfcs(file_list)

    channel_indices = get_channel_indices(atomic_wfc, species_list)

    smn_list = []
    smn_list = []
    amn_list = []
    cmn_list = []
    for ik in range(1, num_kpoints + 1):
        kpt, _kvec, miller, wfcs = qe_wfc.get_wfc(ik)
        s_mn, a_mn, c_mn = compute_atomic_projections(atomic_wfc, kpt, miller, wfcs)
        smn_list.append(s_mn)
        amn_list.append(a_mn)
        cmn_list.append(c_mn)

    return np.array(smn_list), np.array(amn_list), np.array(cmn_list), channel_indices


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


def build_atoms_dict_from_structure(
    structure_file: Path,
) -> tuple[dict, npt.NDArray[np.float64]]:
    """Build atoms_dict and lattice_vectors from any ASE-readable structure file.

    Uses AiiDA StructureData as an intermediary to avoid direct ASE IO imports
    for file reading.

    Parameters
    ----------
    structure_file
        Path to a CIF, XSF, or other structure file.

    Returns
    -------
    atoms_dict
        ``{species: {'num_atoms': int, 'positions': [[x,y,z], ...]}}``.
    lattice_vectors
        3x3 array of lattice vectors in Bohr.
    """
    from pao_plusplus.workflows import structure_to_aiida

    _, structure = structure_to_aiida(structure_file)
    atoms = structure.get_ase()
    lattice_vectors = np.array(atoms.cell) / Bohr

    atoms_dict: dict[str, dict] = {}
    scaled_positions = atoms.get_scaled_positions()
    for symbol, pos in zip(atoms.get_chemical_symbols(), scaled_positions, strict=True):
        if symbol not in atoms_dict:
            atoms_dict[symbol] = {"num_atoms": 0, "positions": []}
        atoms_dict[symbol]["num_atoms"] += 1
        atoms_dict[symbol]["positions"].append(pos.tolist())

    return atoms_dict, lattice_vectors


def _count_pao_orbitals(element: str, pao_config: Any) -> int:
    """Count the total number of orbitals from a PaoConfig.

    Each selected orbital contributes (2l+1) orbitals (the m quantum numbers).
    """
    from pao_plusplus.data.openmx import fetch_pao
    from pao_plusplus.openmx import parse_select, read_openmx_pao

    pao = read_openmx_pao(fetch_pao(element, pao_config.rc))
    if pao_config.select:
        selected = parse_select(pao_config.select)
    else:
        selected = [1] * pao.lmax

    total = 0
    for l_val, count in enumerate(selected):
        total += count * (2 * l_val + 1)
    return total


def generate_fat_bands_from_config(
    config_path: Path,
    working_dir: Path | None = None,
    emin: float | None = None,
    emax: float | None = None,
    filename: Path | None = None,
    num_bands: int | None = None,
) -> list[Path]:
    """Generate fat bands plot(s) from a TOML configuration file.

    Accepts both single-entry (``[Element]``) and multi-entry
    (``[[Element]]``) TOML configs.  Runs a single DFT bands calculation
    and produces one fat bands plot per comparison set.

    Parameters
    ----------
    config_path
        Path to the TOML configuration file.
    working_dir
        Working directory for intermediate files. Defaults to a directory
        alongside the TOML file.
    emin, emax
        Energy range relative to the Fermi level.
    filename
        Output file path.  When there are multiple sets, a ``_set0``,
        ``_set1``, ... suffix is inserted before the extension.
    num_bands
        Manual override for the number of bands. Takes precedence over both
        the TOML ``num_bands`` field and the automatic calculation.

    Returns
    -------
    list[Path]
        Paths to the generated plot files.
    """
    if working_dir is None:
        working_dir = Path("tmp") / "fat_bands" / config_path.stem

    set_results, band_plot_data = compute_amn_for_comparison_sets(
        config_path, working_dir, num_bands=num_bands
    )

    output_files: list[Path] = []
    for i, r in enumerate(set_results):
        channel_proj = compute_projectability_per_channel(r.amn, r.cmn, r.channel_indices)

        if len(set_results) == 1:
            out = filename
        elif filename is not None:
            stem = filename.stem
            out = filename.with_stem(f"{stem}_{i}_{r.label}")
        else:
            out = None

        plot_fat_bands(
            band_plot_data,
            channel_proj,
            emin=emin,
            emax=emax,
            filename=out,
        )
        if out is not None:
            output_files.append(out)

    return output_files


def _generate_bessel_files(
    elements: dict[str, Any],
    solver_dir: Path,
) -> dict[str, Path]:
    """Generate Bessel .h5 files for a set of element configs.

    Shared helper for fat-bands and projectability comparison.
    """
    from pao_plusplus.config import PaoConfig, UpfConfig
    from pao_plusplus.solve import solve_and_export

    bessel_files: dict[str, Path] = {}
    for element, elem_config in elements.items():
        if isinstance(elem_config, UpfConfig):
            logger.info(
                "Solving pseudoatomic problem for %s (rc=%.2f, ri_factor=%.4f)",
                element, elem_config.rc, elem_config.ri_factor,
            )
            _, bessel = solve_and_export(
                upf_path=elem_config.upf,
                rc=elem_config.rc,
                ri_factor=elem_config.ri_factor,
                extension=elem_config.get_extension(),
                working_dir=solver_dir,
                dat_filename=f"{element}.dat",
            )
            if bessel is None:
                raise RuntimeError(
                    f"solve_and_export for {element} did not produce a Bessel file. "
                    f"Check the UPF file: {elem_config.upf}"
                )
            bessel_files[element] = bessel
        elif isinstance(elem_config, PaoConfig):
            from pao_plusplus.data.openmx import fetch_pao
            from pao_plusplus.openmx import pao_to_bessel, parse_select, read_openmx_pao

            pao_path = fetch_pao(element, elem_config.rc)
            logger.info("Converting %s to Bessel for %s", pao_path.name, element)
            pao = read_openmx_pao(pao_path)
            selected = parse_select(elem_config.select) if elem_config.select else None
            bessel_path = solver_dir / f"{element}.h5"
            pao_to_bessel(pao, bessel_path, selected=selected)
            bessel_files[element] = bessel_path
    return bessel_files


@dataclass
class ComparisonSetResult:
    """Amn/Cmn matrices and metadata for one comparison set."""

    amn: npt.NDArray[np.complex128]
    """Shape ``(num_kpoints, num_orbitals, num_bands)``."""
    cmn: npt.NDArray[np.complex128]
    """Shape ``(num_kpoints, num_orbitals, num_bands)``."""
    channel_indices: dict[tuple[str, int], list[int]]
    label: str


@dataclass
class PreparedComparisonSets:
    """Parsed config, bessel files, and labels for comparison sets."""

    config: Any
    """The :class:`ProjectabilityComparisonConfig`."""
    all_bessel: list[dict[str, Path]]
    """Bessel files for each comparison set."""
    labels: list[str]
    """Label for each comparison set."""
    min_nbnd: int
    """Minimum number of bands to request from the DFT calculation."""


def prepare_comparison_sets(
    config_path: Path,
    working_dir: Path,
    num_bands: int | None = None,
) -> PreparedComparisonSets:
    """Parse a comparison TOML config and generate bessel files for each set.

    This handles config parsing, building comparison sets by broadcasting
    single-entry elements, generating bessel/dat files, computing the
    required number of bands, and assembling labels.  It does **not** run
    any DFT calculation or compute Amn matrices.

    Parameters
    ----------
    config_path
        Path to the TOML configuration file.
    working_dir
        Working directory for intermediate files.
    num_bands
        Manual override for the number of bands.
    """
    from pao_plusplus.bands import compute_min_nbnd, compute_num_target_bands, orbitals_per_atom
    from pao_plusplus.config import (
        PaoConfig,
        ProjectabilityComparisonConfig,
        UpfConfig,
    )

    config = ProjectabilityComparisonConfig.from_toml(config_path)
    num_sets = config.num_sets

    working_dir.mkdir(parents=True, exist_ok=True)

    # Build the comparison sets by broadcasting single entries
    sets: list[dict[str, UpfConfig | PaoConfig]] = []
    for i in range(num_sets):
        s: dict[str, UpfConfig | PaoConfig] = {}
        for element, configs in config.elements.items():
            s[element] = configs[i] if i < len(configs) else configs[0]
        sets.append(s)

    # Generate bessel files for each set
    all_bessel: list[dict[str, Path]] = []
    for i, elem_set in enumerate(sets):
        solver_dir = working_dir / "projectors" / f"set_{i}"
        solver_dir.mkdir(parents=True, exist_ok=True)
        all_bessel.append(_generate_bessel_files(elem_set, solver_dir))

    # Build labels
    labels: list[str] = []
    for i in range(num_sets):
        set_labels = []
        for element, configs in config.elements.items():
            if len(configs) > 1:
                cfg = configs[i] if i < len(configs) else configs[0]
                set_labels.append(cfg.label or f"{element} set {i}")
        labels.append(", ".join(set_labels) if set_labels else f"Set {i}")

    # Compute nbnd from the largest set (most orbitals), or use manual override
    effective_num_bands = num_bands or config.num_bands
    if effective_num_bands is not None:
        min_nbnd = effective_num_bands
        logger.info("Comparison: using manual num_bands override = %d", min_nbnd)
    else:
        max_orbitals_per_el: dict[str, int] = {}
        for elem_set in sets:
            for element, elem_config in elem_set.items():
                if isinstance(elem_config, UpfConfig):
                    n = orbitals_per_atom(elem_config.upf, elem_config.get_extension())
                else:
                    n = _count_pao_orbitals(element, elem_config)
                max_orbitals_per_el[element] = max(max_orbitals_per_el.get(element, 0), n)

        ntb = compute_num_target_bands(config.structure, max_orbitals_per_el)
        min_nbnd = compute_min_nbnd(ntb)
        logger.info("Comparison: max_orbitals_per_el=%s, ntb=%d, min_nbnd=%d", max_orbitals_per_el, ntb, min_nbnd)

    return PreparedComparisonSets(config, all_bessel, labels, min_nbnd)


def compute_amn_for_comparison_sets(
    config_path: Path,
    working_dir: Path,
    num_bands: int | None = None,
) -> tuple[list[ComparisonSetResult], BandPlotData]:
    """Shared pipeline: parse config, run DFT bands, compute Amn for each PAO set.

    Reads a TOML config with ``[[Element]]`` syntax, runs a single DFT
    bands calculation, and computes A_mn^k / C_mn^k for each Cartesian-product
    combination of PAO choices.

    Parameters
    ----------
    config_path
        Path to the TOML configuration file.
    working_dir
        Working directory for intermediate files.
    num_bands
        Manual override for the number of bands.

    Returns
    -------
    results
        One :class:`ComparisonSetResult` per comparison set.
    band_plot_data
        Band structure data shared across all sets.
    """
    from pao_plusplus.projectability import _make_qe_input_wfc
    from pao_plusplus.workflows import run_bands_workflow

    prep = prepare_comparison_sets(config_path, working_dir, num_bands=num_bands)

    bands_result = run_bands_workflow(
        prep.config.structure,
        working_dir,
        min_nbnd=prep.min_nbnd,
        kpath=prep.config.kpath,
        periodic=prep.config.periodic,
    )
    logger.info(
        "Comparison: energies shape=%s, num_kpoints=%d, num_bands=%d",
        bands_result.band_plot_data.energies.shape,
        bands_result.band_plot_data.energies.shape[0],
        bands_result.band_plot_data.energies.shape[1],
    )

    # Compute Amn for each set
    bands_calc_dir = bands_result.bands_calc_dir
    pwi_file = bands_calc_dir / "inputs" / "aiida.in"
    wfc_dir = bands_calc_dir / "outputs"
    atoms_dict, lattice_vectors = build_atoms_dict(pwi_file)
    qe_wfc = _make_qe_input_wfc(wfc_dir, lattice_vectors)
    num_kpoints = bands_result.band_plot_data.energies.shape[0]

    results: list[ComparisonSetResult] = []
    for i, bessel_files in enumerate(prep.all_bessel):
        _smn, amn, cmn, channel_indices = compute_amn_from_wfc(
            qe_wfc=qe_wfc,
            bessel_files=bessel_files,
            atoms_dict=atoms_dict,
            lattice_vectors=lattice_vectors,
            num_kpoints=num_kpoints,
        )
        results.append(ComparisonSetResult(amn, cmn, channel_indices, prep.labels[i]))

    return results, bands_result.band_plot_data


def generate_projectability_comparison(
    config_path: Path,
    working_dir: Path | None = None,
    emin: float | None = None,
    emax: float | None = None,
    filename: Path | None = None,
    num_bands: int | None = None,
) -> None:
    """Compare total projectability across different basis sets.

    Reads a TOML config where elements can have multiple entries (using
    ``[[Element]]`` syntax).  Runs a single DFT bands calculation, then
    computes and overlays total projectability for each comparison set.

    The ``otsu_bins`` field from the TOML config controls the number of
    Otsu classes used for threshold detection on the plot.

    Parameters
    ----------
    config_path
        Path to the TOML configuration file.
    working_dir
        Working directory for intermediate files.
    emin, emax
        Energy range relative to the Fermi level.
    filename
        If provided, save the figure to this path.
    num_bands
        Manual override for the number of bands. Takes precedence over both
        the TOML ``num_bands`` field and the automatic calculation.
    """
    from pao_plusplus.config import ProjectabilityComparisonConfig
    from pao_plusplus.projectability import suggest_disentanglement_thresholds

    config = ProjectabilityComparisonConfig.from_toml(config_path)

    if working_dir is None:
        working_dir = Path("tmp") / "proj_comparison" / config_path.stem

    set_results, band_plot_data = compute_amn_for_comparison_sets(
        config_path, working_dir, num_bands=num_bands
    )

    total_projs: list[npt.NDArray[np.float64]] = []
    labels: list[str] = []
    thresholds: list[tuple[float, float]] = []
    for r in set_results:
        channel_proj = compute_projectability_per_channel(r.amn, r.cmn, r.channel_indices)
        total_projs.append(sum(channel_proj.values()))
        labels.append(r.label)
        otsu_min, otsu_max = suggest_disentanglement_thresholds(r.amn, r.cmn, otsu_bins=config.otsu_bins)
        explicit_min = config.wannier90.get("dis_proj_min")
        effective_min = explicit_min if explicit_min is not None else otsu_min
        thresholds.append((effective_min, otsu_max))

    plot_projectability_comparison(
        band_plot_data,
        total_projs,
        labels=labels,
        thresholds=thresholds,
        emin=emin,
        emax=emax,
        filename=filename,
    )


def plot_projectability_comparison(
    band_plot_data: BandPlotData,
    total_projectabilities: list[npt.NDArray[np.float64]],
    labels: list[str],
    thresholds: list[tuple[float, float]] | None = None,
    emin: float | None = None,
    emax: float | None = None,
    filename: Path | None = None,
) -> None:
    """Plot total projectability from multiple basis sets.

    Energy along the x-axis, linear projectability on the y-axis.
    A marginal histogram on the right shows the distribution of
    projectability values for each set.

    Parameters
    ----------
    band_plot_data
        Band structure data (shared across all sets).
    total_projectabilities
        List of total projectability arrays, each shape ``(num_kpoints, num_bands)``.
    labels
        Legend label for each set.
    thresholds
        If provided, one ``(dis_proj_min, dis_proj_max)`` pair per set.
        Points are plotted with different markers for the three Otsu
        regions: excluded (below min), disentangled (between), frozen
        (above max).
    emin, emax
        Energy range relative to the Fermi level.
    filename
        If provided, save the figure to this path.
    """
    from pao_plusplus.plotting import REVTEX_COLUMN_WIDTH

    energies = band_plot_data.energies
    padding = 0.025 * (energies.max() - energies.min())
    if emin is None:
        emin = float(energies.min()) - padding
    if emax is None:
        emax = float(energies.max())

    from matplotlib import cm

    from pao_plusplus.plotting import COLORMAP

    fig, (ax, ax_hist) = plt.subplots(
        1,
        2,
        sharey=True,
        width_ratios=[3, 1],
        figsize=(REVTEX_COLUMN_WIDTH, REVTEX_COLUMN_WIDTH * 0.6),
        gridspec_kw={"wspace": 0.1},
    )

    # Filter to energy window
    mask = (energies >= emin) & (energies <= emax)

    n_sets = len(total_projectabilities)
    cmap = cm.get_cmap(COLORMAP)
    colors = [cmap(x) for x in np.linspace(0.25, 0.75, max(n_sets, 2))]
    from scipy.stats import gaussian_kde

    kde_y = np.linspace(0, 1, 200)
    y_max = 0.0
    kde_max = 0.0
    # Markers for the three Otsu regions: excluded, disentangled, frozen
    _REGION_MARKERS = ["v", "o", "^"]

    for i, (total_proj, label) in enumerate(zip(total_projectabilities, labels, strict=True)):
        color = colors[i]

        # Scatter: energy on x, projectability on y
        e_flat = energies[mask].ravel()
        p_flat = total_proj[mask].ravel()
        y_max = max(y_max, float(p_flat.max()))

        if thresholds is not None:
            t_min, t_max = thresholds[i]
            regions = [
                (p_flat < t_min, _REGION_MARKERS[0]),
                ((p_flat >= t_min) & (p_flat <= t_max), _REGION_MARKERS[1]),
                (p_flat > t_max, _REGION_MARKERS[2]),
            ]
            for region_mask, marker in regions:
                if region_mask.any():
                    ax.scatter(
                        e_flat[region_mask],
                        p_flat[region_mask],
                        s=2,
                        color=color,
                        alpha=0.5,
                        edgecolors="none",
                        marker=marker,
                        rasterized=False,
                    )
            # Draw threshold lines on both axes
            for a in (ax, ax_hist):
                a.axhline(t_min, color=color, ls="--", lw=0.5, alpha=0.7)
                a.axhline(t_max, color=color, ls="--", lw=0.5, alpha=0.7)
        else:
            ax.scatter(
                e_flat,
                p_flat,
                s=2,
                color=color,
                alpha=0.5,
                edgecolors="none",
                rasterized=False,
            )

        # Marginal KDE
        kde = gaussian_kde(p_flat, bw_method=0.04)
        kde_vals = kde(kde_y)
        kde_max = max(kde_max, float(kde_vals.max()))
        ax_hist.fill_betweenx(kde_y, kde_vals, color=color, alpha=0.3)
        ax_hist.plot(kde_vals, kde_y, color=color, linewidth=0.8, label=label)

    ax.set_xlim(emin, emax)
    ax.set_ylim(0, 1.0)
    ax.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_xlabel("energy (eV)")
    ax.set_ylabel(r"$(A^\dagger S^{-1} A)_{mm}(\mathbf{k})$")

    # Custom legend with combined scatter circle + histogram rectangle
    from matplotlib.legend_handler import HandlerBase
    from matplotlib.patches import Circle, FancyBboxPatch

    class _ScatterHistHandler(HandlerBase):
        """Draw a filled circle and a filled rectangle side-by-side."""

        def __init__(self, color, alpha=0.5):
            self._color = color
            self._alpha = alpha
            super().__init__()

        def create_artists(
            self, legend, orig_handle, xdescent, ydescent, width, height, fontsize, trans
        ):
            cx = width * 0.25
            cy = height / 2
            circle = Circle(
                (cx, cy),
                radius=height * 0.35,
                facecolor=self._color,
                edgecolor="none",
                alpha=self._alpha,
                transform=trans,
            )
            rect = FancyBboxPatch(
                (width * 0.45, 0),
                width * 0.5,
                height,
                boxstyle="square,pad=0",
                facecolor=self._color,
                edgecolor="none",
                alpha=self._alpha,
                transform=trans,
            )
            return [circle, rect]

    handles = [plt.Line2D([], [], color="none", label=lbl) for lbl in labels]
    handler_map = {
        h: _ScatterHistHandler(colors[i]) for i, h in enumerate(handles)
    }
    ax.legend(
        handles=handles,
        handler_map=handler_map,
        fontsize="small",
        frameon=False,
        loc="lower left",
        bbox_to_anchor=(0, 1.05),
        ncol=2,
        borderpad=0,
        borderaxespad=0,
        handletextpad=0.4,
        columnspacing=1.0,
    )

    ax_hist.set_xscale("log")
    ax_hist.set_xlim(left=kde_max * 1e-3, right=kde_max)
    ax_hist.tick_params(labelleft=False, labelbottom=False)
    from matplotlib.ticker import LogLocator
    ax_hist.xaxis.set_minor_locator(LogLocator(subs=[2, 4, 6, 8], numticks=12))

    fig.subplots_adjust(left=0.15, bottom=0.18, right=0.99, top=0.9)
    if filename is not None:
        from pao_plusplus.plotting import savefig
        savefig(fig, filename)
    plt.close(fig)


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
    _smn, amn, cmn, channel_indices = compute_amn_from_wfc(
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


_ELEMENT_PALETTES = [
    {0: "tab:green", 1: "tab:blue", 2: "tab:purple", 3: "tab:pink"},
    {0: "tab:orange", 1: "tab:red", 2: "tab:brown", 3: "tab:olive"},
]


def _get_channel_color(species_index: int, l: int) -> tuple[float, float, float]:
    """Return RGB color for a channel given the element's index and l."""
    palette = _ELEMENT_PALETTES[species_index % len(_ELEMENT_PALETTES)]
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
    """Configure the projectability side panel with linear x-axis."""
    ax_proj.set_xlim(0, 1)
    ax_proj.set_xlabel("projectability")
    ax_proj.tick_params(
        labelleft=False,
        labelsize="x-small",
        labelrotation=90,
    )
    ax_proj.set_xticks([0, 0.5, 1.0])


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

    # Discrete colormap for channels
    from matplotlib import cm

    from pao_plusplus.plotting import COLORMAP

    cmap = cm.get_cmap(COLORMAP)
    n_channels = len(channel_keys)
    channel_colors = {
        key: mcolors.to_rgb(cmap(x))
        for key, x in zip(channel_keys, np.linspace(0.0, 1.0, max(n_channels, 2)))
    }
    proj_color = cmap(0.5)

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
                rgb = channel_colors[(species, l)]
                _build_fat_band_quads(ax, x_fine, e_fine, p_fine, rgb, half_w)

        band_proj = total_proj[:, band_idx]
        ax_proj.scatter(
            band_proj,
            band_energies,
            s=2,
            color=proj_color,
            alpha=0.5,
            edgecolors="none",
        )

    # Legend
    legend_handles = []
    single_species = len({s for s, _ in channel_keys}) == 1
    for species, l in channel_keys:
        rgb = channel_colors[(species, l)]
        if single_species:
            label = f"${L_LABELS.get(l, '?')}$"
        else:
            label = f"{species} ${L_LABELS.get(l, '?')}$"
        legend_handles.append(Line2D([0], [0], color=rgb, linewidth=2, label=label))
    ax.legend(
        handles=legend_handles,
        loc="lower left",
        bbox_to_anchor=(0, 1.025),
        fontsize="small",
        frameon=False,
        ncol=len(legend_handles),
        borderpad=0,
        borderaxespad=0,
        handletextpad=0.4,
        columnspacing=1.0,
    )

    _configure_proj_panel(ax_proj)

    fig.subplots_adjust(left=0.15, bottom=0.15, right=0.99, top=0.925)
    if filename is not None:
        from pao_plusplus.plotting import savefig
        savefig(fig, filename)
    plt.close(fig)
