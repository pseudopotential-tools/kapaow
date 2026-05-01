"""Workflow module for kapaow using AiiDA workgraphs."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import ase
import ase.io
import numpy as np
import numpy.typing as npt

logger = logging.getLogger(__name__)

PSEUDO_LIBRARY = "PseudoDojo/0.4/PBE/SR/standard/upf"


@dataclass
class QEWorkflowResult:
    """Results from a QE workflow run."""

    atoms: ase.Atoms
    nscf_input_file: Path
    nscf_wfc_dir: Path
    output_dir: Path
    kpoint_weights: npt.NDArray[np.float64]
    fermi_energy: float


def structure_to_aiida(
    structure_file: Path,
    periodic: tuple[bool, bool, bool] = (True, True, True),
) -> tuple[ase.Atoms, Any]:
    """Read a structure file and convert to AiiDA StructureData.

    Requires that the AiiDA profile has already been loaded.

    Args:
        structure_file: Path to CIF, XSF, or any ASE-readable structure file.
        periodic: Periodic boundary conditions along (a, b, c). The ASE Atoms'
            ``pbc`` is set accordingly before building the StructureData so
            that aiida-quantumespresso's PwBaseWorkChain automatically injects
            the matching ``SYSTEM.assume_isolated`` (e.g. ``'2D'`` for
            ``(True, True, False)``).

    Returns:
        Tuple of (ASE Atoms, AiiDA StructureData).
    """
    from aiida import orm

    atoms = ase.io.read(str(structure_file))
    atoms.pbc = periodic
    structure = orm.StructureData(ase=atoms)
    return atoms, structure


def _find_nscf_calc_dir(dump_dir: Path) -> Path:
    """Find the NSCF PwCalculation directory within an AiiDA dump.

    Searches for a directory matching the pattern ``*nscf*PwCalculation``
    in the dump tree.

    Args:
        dump_dir: Root of the dumped workgraph output.

    Returns:
        Path to the NSCF calculation directory (containing inputs/ and outputs/).
    """
    matches = [
        d for d in dump_dir.rglob("*PwCalculation") if any("nscf" in p.name for p in d.parents)
    ]
    if not matches:
        raise FileNotFoundError(
            f"No NSCF PwCalculation directory found in {dump_dir}. "
            f"Contents: {list(dump_dir.rglob('*'))}"
        )
    return matches[0]



def run_qe_workflow(
    structure_file: Path,
    working_dir: Path,
    min_nbnd: int | None = None,
    periodic: tuple[bool, bool, bool] = (True, True, True),
) -> QEWorkflowResult:
    """Run SCF + NSCF via AiiDA PwScfNscfTask.

    AiiDA handles caching of identical calculations automatically.

    Args:
        structure_file: Path to a structure file (CIF, XSF, etc.).
        working_dir: Working directory for this material.
        min_nbnd: Minimum number of bands.

    Returns:
        QEWorkflowResult with paths to NSCF outputs.
    """

    from aiida import orm
    from aiida_koopmans.workgraphs.pw import PwScfNscfTask
    from koopmans.aiida.dumping import dump_workgraph
    from koopmans.aiida.progress import run_with_progress
    from koopmans.aiida.setup import load_koopmans_profile

    load_koopmans_profile()

    atoms, structure = structure_to_aiida(structure_file, periodic=periodic)
    pw_code = orm.load_code("pw@localhost")

    # Retrieve NSCF wavefunctions (needed for projectability computation)
    nscf_overrides: dict[str, Any] = {
        "pw": {
            "metadata": {
                "options": {
                    "additional_retrieve_list": ["out/aiida.save/wfc*.hdf5"],
                }
            }
        }
    }
    if min_nbnd is not None:
        nscf_overrides["pw"]["parameters"] = {"SYSTEM": {"nbnd": min_nbnd}}

    overrides: dict[str, Any] = {"nscf": nscf_overrides}

    wg = PwScfNscfTask.build(
        code=pw_code,
        structure=structure,
        pseudo_family=PSEUDO_LIBRARY,
        overrides=overrides,
    )
    run_with_progress(wg)

    process_node = wg.process
    if not process_node.is_finished_ok:
        raise RuntimeError(
            f"Workflow failed with exit status {process_node.exit_status}: "
            f"{process_node.exit_message}"
        )

    output_dir = working_dir / "scf_and_nscf"
    dump_workgraph(process_node, output_dir)

    nscf_calc_dir = _find_nscf_calc_dir(output_dir)

    # Extract k-point weights from AiiDA output (stored in BandsData)
    nscf_bands = process_node.outputs.nscf_output_band
    _, weights = nscf_bands.get_kpoints(also_weights=True)

    # Extract Fermi energy from NSCF output parameters
    nscf_params = process_node.outputs.nscf_output_parameters.get_dict()
    fermi_energy = nscf_params["fermi_energy"]

    result = QEWorkflowResult(
        atoms=atoms,
        nscf_input_file=nscf_calc_dir / "inputs" / "aiida.in",
        nscf_wfc_dir=nscf_calc_dir / "outputs",
        output_dir=output_dir,
        kpoint_weights=np.array(weights),
        fermi_energy=fermi_energy,
    )

    return result


@dataclass
class BandPlotData:
    """Pre-computed band structure data for plotting."""

    x: npt.NDArray[np.float64]
    """K-path distances (1D, length num_kpoints)."""
    energies: npt.NDArray[np.float64]
    """Band energies relative to Fermi level (num_kpoints x num_bands)."""
    labels: list[tuple[float, str]]
    """High-symmetry point labels as (x_position, label_string)."""


@dataclass
class BandsWorkflowResult:
    """Results from a PwBandsWorkChain run."""

    bands_calc_dir: Path
    output_dir: Path
    fermi_energy: float
    band_plot_data: BandPlotData
    bands_kpoints_pk: int | None = None
    """AiiDA PK of a KpointsData node with the explicit k-path used for bands."""
    reference_bands_pk: int | None = None
    """AiiDA PK of the BandsData node with DFT reference bands."""


def _find_bands_calc_dir(dump_dir: Path) -> Path:
    """Find the bands PwCalculation directory within a PwBandsWorkChain dump.

    Looks for a PwCalculation whose immediate PwBaseWorkChain parent
    contains "bands" (e.g. ``03-bands-PwBaseWorkChain``), excluding the
    SCF sub-step which also lives under the top-level PwBandsWorkChain.
    """
    matches = [
        d
        for d in dump_dir.rglob("*PwCalculation")
        if any("bands" in p.name and "PwBaseWorkChain" in p.name for p in d.parents)
    ]
    if not matches:
        raise FileNotFoundError(
            f"No bands PwCalculation directory found in {dump_dir}. "
            f"Contents: {list(dump_dir.rglob('*'))}"
        )
    return matches[0]


def _build_kpoints_from_path(
    structure: Any,
    kpath: list[list[str]],
    reference_distance: float = 0.025,
) -> Any:
    """Build an explicit KpointsData from a user-specified k-path.

    Uses seekpath to look up the fractional coordinates of named high-symmetry
    points, then generates an explicit list of k-points along the requested
    path segments only (bypassing seekpath's automatic path selection).

    Each inner list is a continuous path through the listed points.  Multiple
    inner lists create discontinuities in the band plot.

    Args:
        structure: AiiDA StructureData node.
        kpath: List of continuous segments, e.g.
            ``[["GAMMA", "M", "K", "GAMMA"]]`` for a single continuous path,
            or ``[["GAMMA", "M"], ["X", "GAMMA"]]`` for a path with a
            discontinuity between M and X.
        reference_distance: Target spacing between k-points in reciprocal
            Angstrom (default matches seekpath's default of 0.025).

    Returns:
        A KpointsData node with explicit k-points and labels.
    """
    from aiida import orm

    # Get the point coordinates from seekpath (but ignore its path)
    cell = np.array(structure.cell)
    reciprocal = 2 * np.pi * np.linalg.inv(cell).T

    result = _seekpath_get_point_coords(structure)
    point_coords = result["point_coords"]

    # Validate all labels exist
    all_labels = set()
    for segment in kpath:
        all_labels.update(segment)
    missing = all_labels - set(point_coords.keys())
    if missing:
        raise ValueError(
            f"Unknown k-point labels: {missing}. "
            f"Available labels: {sorted(point_coords.keys())}"
        )

    # For lower-dimensional structures, check that every *requested* label
    # has zero fractional coordinate along each aperiodic axis. Seekpath's
    # full catalog will contain points outside the periodic subspace; we
    # only care about the ones the user actually picked.
    pbc = tuple(structure.pbc)
    if pbc != (True, True, True):
        aperiodic_axes = [i for i, p in enumerate(pbc) if not p]
        bad: list[tuple[str, list[float]]] = []
        for label in sorted(all_labels):
            coord = point_coords[label]
            if any(coord[i] != 0 for i in aperiodic_axes):
                bad.append((label, list(coord)))
        if bad:
            raise ValueError(
                f"Requested k-points have non-zero components along "
                f"aperiodic axes {aperiodic_axes} for a structure with "
                f"pbc={pbc}: {bad}."
            )

    # Build explicit k-points along each continuous segment
    all_kpoints: list[list[float]] = []
    labels: list[tuple[int, str]] = []

    for segment in kpath:
        if len(segment) < 2:
            raise ValueError(f"Each path segment needs at least 2 labels, got {segment}")

        for j in range(len(segment) - 1):
            start_label = segment[j]
            end_label = segment[j + 1]
            start_frac = np.array(point_coords[start_label])
            end_frac = np.array(point_coords[end_label])

            # Compute distance in cartesian reciprocal space
            delta_frac = end_frac - start_frac
            delta_cart = delta_frac @ reciprocal
            distance = float(np.linalg.norm(delta_cart))
            npoints = max(2, int(np.ceil(distance / reference_distance)))

            # Skip the first point for non-first sub-segments within a
            # continuous segment (it's already the last point of the previous)
            i_start = 1 if j > 0 else 0

            if i_start == 0:
                labels.append((len(all_kpoints), start_label))

            for i in range(i_start, npoints):
                t = i / (npoints - 1)
                all_kpoints.append((start_frac + t * delta_frac).tolist())

            labels.append((len(all_kpoints) - 1, end_label))

    kpoints_data = orm.KpointsData()
    kpoints_data.set_cell_from_structure(structure)
    kpoints_data.set_kpoints(all_kpoints)
    kpoints_data.labels = labels
    return kpoints_data


def _seekpath_get_point_coords(structure: Any) -> dict[str, Any]:
    """Get high-symmetry point coordinates from seekpath without generating a path.

    Seekpath is only implemented for fully periodic structures, so for
    lower-dimensional systems the structure is temporarily promoted to
    ``(True, True, True)`` for the lookup. Validation that the *requested*
    high-symmetry points are compatible with the original periodicity is
    performed by the caller.

    Args:
        structure: AiiDA StructureData node.

    Returns:
        Dict with 'point_coords' mapping label -> [kx, ky, kz] in fractional coords.
    """
    from aiida import orm
    from aiida.tools import get_explicit_kpoints_path

    if tuple(structure.pbc) != (True, True, True):
        atoms = structure.get_ase()
        atoms.pbc = (True, True, True)
        structure = orm.StructureData(ase=atoms)

    result = get_explicit_kpoints_path(structure)
    params = result["parameters"].get_dict()
    return {"point_coords": params["point_coords"]}


def run_bands_workflow(
    structure_file: Path,
    working_dir: Path,
    min_nbnd: int | None = None,
    kpath: list[list[str]] | None = None,
    periodic: tuple[bool, bool, bool] = (True, True, True),
) -> BandsWorkflowResult:
    """Run SCF + bands along high-symmetry k-path via PwBandsWorkChain.

    AiiDA handles caching of identical calculations automatically.

    Args:
        structure_file: Path to a structure file (CIF, XSF, etc.).
        working_dir: Working directory for this material.
        min_nbnd: Minimum number of bands for the bands calculation.
        kpath: Explicit k-path as a list of segment pairs, e.g.
            ``[["GAMMA", "M"], ["M", "K"], ["K", "GAMMA"]]``.
            If provided, seekpath is bypassed entirely.

    Returns:
        BandsWorkflowResult with path to the bands calculation dump.
    """

    from aiida import orm
    from aiida_koopmans.workgraphs.pw import PwBandsTaskViaBuilder
    from koopmans.aiida.dumping import dump_workgraph
    from koopmans.aiida.progress import run_with_progress
    from koopmans.aiida.setup import load_koopmans_profile

    load_koopmans_profile()

    _, structure = structure_to_aiida(structure_file, periodic=periodic)
    pw_code = orm.load_code("pw@localhost")

    # Build explicit KpointsData if a manual k-path was provided
    bands_kpoints = None
    if kpath is not None:
        bands_kpoints = _build_kpoints_from_path(structure, kpath)

    # Retrieve bands wavefunctions (needed for fat bands Amn computation)
    bands_pw_overrides: dict[str, Any] = {
        "metadata": {
            "options": {
                "additional_retrieve_list": ["out/aiida.save/wfc*.hdf5"],
            }
        }
    }
    overrides: dict[str, Any] = {
        "bands": {"pw": bands_pw_overrides},
        # Use the same kpoints_distance as aiida-wannier90-workflows (0.2)
        # so the SCF Fermi energies are consistent across workflows
        "scf": {"kpoints_distance": 0.2},
    }
    if min_nbnd is not None:
        bands_pw_overrides["parameters"] = {"SYSTEM": {"nbnd": min_nbnd}}

    wg = PwBandsTaskViaBuilder.build(
        code=pw_code,
        structure=structure,
        pseudo_family=PSEUDO_LIBRARY,
        overrides=overrides,
        bands_kpoints=bands_kpoints,
    )
    run_with_progress(wg)

    process_node = wg.process
    if not process_node.is_finished_ok:
        raise RuntimeError(
            f"Bands workflow failed with exit status {process_node.exit_status}: "
            f"{process_node.exit_message}"
        )

    output_dir = working_dir / "bands"
    dump_workgraph(process_node, output_dir)

    bands_calc_dir = _find_bands_calc_dir(output_dir)

    # Extract Fermi energy from SCF output parameters
    scf_params = process_node.outputs.scf_parameters.get_dict()
    fermi_energy = scf_params["fermi_energy"]

    # Extract band plot data from BandsData node
    bands_node = process_node.outputs.band_structure
    plot_info = bands_node._get_bandplot_data(
        cartesian=True,
        prettify_format="latex_seekpath",
        join_symbol="|",
        y_origin=fermi_energy,
    )
    band_x = np.array(plot_info["x"])
    band_energies = np.array(plot_info["y"])
    band_labels = [(float(pos), str(lbl)) for pos, lbl in plot_info["labels"]]

    # Build a KpointsData with the exact k-points used by the bands calculation.
    # This can be passed to Wannier90 as bands_kpoints so both use the same grid.
    kpoints_array = bands_node.get_kpoints()
    kpoint_labels = bands_node.labels
    bands_kpoints = orm.KpointsData()
    bands_kpoints.set_kpoints(kpoints_array)
    bands_kpoints.labels = kpoint_labels
    bands_kpoints.store()
    bands_kpoints_pk = bands_kpoints.pk
    reference_bands_pk = bands_node.pk

    band_plot_data = BandPlotData(
        x=band_x,
        energies=band_energies,
        labels=band_labels,
    )

    return BandsWorkflowResult(
        bands_calc_dir=bands_calc_dir,
        output_dir=output_dir,
        fermi_energy=fermi_energy,
        band_plot_data=band_plot_data,
        bands_kpoints_pk=bands_kpoints_pk,
        reference_bands_pk=reference_bands_pk,
    )


def _build_external_projectors(proj_dir: Path) -> dict[str, list[dict[str, Any]]]:
    """Build the external_projectors metadata dict from .dat files in a directory.

    The Wannier90WorkChain needs this to count projections for determining
    num_bands. Each orbital gets ``l`` (for counting) and a dummy ``label``.

    Args:
        proj_dir: Directory containing {Element}.dat files.

    Returns:
        Dict mapping element symbols to lists of orbital descriptors.
    """
    from kapaow.io import read_wannier90_dat_file

    l_to_letter = {0: "s", 1: "p", 2: "d", 3: "f", 4: "g"}
    projectors: dict[str, list[dict[str, Any]]] = {}

    for dat_file in sorted(proj_dir.glob("*.dat")):
        element = dat_file.stem
        _, _, l_values, _ = read_wannier90_dat_file(dat_file)

        l_count: dict[int, int] = {}
        orbitals: list[dict[str, Any]] = []
        for l in l_values:
            l_count[l] = l_count.get(l, 0) + 1
            n = l_count[l]
            orbitals.append({"label": f"{n}{l_to_letter[l]}", "l": l, "alpha": "external"})

        projectors[element] = orbitals

    return projectors


def get_kpoint_path(structure_file: Path) -> dict[str, Any]:
    """Get a high-symmetry k-point path for a structure using seekpath.

    Args:
        structure_file: Path to a structure file (CIF, XSF, etc.).

    Returns:
        Dict with 'path' and 'point_coords' keys for Wannier90.
    """
    from aiida_quantumespresso.calculations.functions.seekpath_structure_analysis import (
        seekpath_structure_analysis,
    )
    from koopmans.aiida.setup import load_koopmans_profile

    load_koopmans_profile()

    _, structure = structure_to_aiida(structure_file)
    result = seekpath_structure_analysis(structure)
    params = result["parameters"].get_dict()
    return {
        "path": params["path"],
        "point_coords": params["point_coords"],
    }


def _build_projector_rotation(
    structure_file: Path,
    proj_dir: Path,
    bond_cutoff: float | None,
) -> Any:
    """Build the symmetry-adapted rotation matrix as an ArrayData node.

    Returns an ``orm.ArrayData`` with the complex unitary ``B`` stored
    under key ``"B"``, ready to be passed as ``projector_rotation``.
    """
    from aiida import orm as _orm

    from kapaow.fat_bands import build_atoms_dict_from_structure
    from kapaow.symmetrize import symmetry_adapted_rotation

    atoms_dict, lattice_vectors = build_atoms_dict_from_structure(structure_file)
    B, labels = symmetry_adapted_rotation(
        structure_file=structure_file,
        proj_dir=proj_dir,
        atoms_dict=atoms_dict,
        lattice_vectors=lattice_vectors,
        hybridize=True,
        bond_cutoff=bond_cutoff,
    )
    logger.info(
        "Symmetrized projector basis: %d orbitals (%d hybrid, %d irrep)",
        len(labels),
        sum(1 for lab in labels if lab.kind == "hybrid"),
        sum(1 for lab in labels if lab.kind == "irrep"),
    )
    rotation = _orm.ArrayData()
    rotation.set_array("B", B)
    return rotation


def run_wannierize_workflow(
    structure_file: Path,
    proj_dir: Path,
    working_dir: Path,
    kpoint_path: dict[str, Any] | None = None,
    bands_kpoints_pk: int | None = None,
    dis_proj_max: float = 0.8,
    dis_proj_min: float | None = None,
    dis_froz_max: float | None = None,
    extra_w90_params: dict[str, Any] | None = None,
    min_nbnd: int | None = None,
    periodic: tuple[bool, bool, bool] = (True, True, True),
    symmetrize: bool = False,
    bond_cutoff: float | None = None,
) -> Any:
    """Run the full Wannierize workflow via AiiDA.

    AiiDA caching automatically reuses QE results (SCF/NSCF) from previous
    runs. Always runs as a metal to enable disentanglement.

    Args:
        structure_file: Path to a structure file (CIF, XSF, etc.).
        proj_dir: Path to directory containing external projector .dat files.
        working_dir: Working directory for wannierization outputs.
        kpoint_path: If provided, enable Wannier band interpolation along this
            k-path. Dict with 'path' and 'point_coords' keys (from seekpath).
            Mutually exclusive with bands_kpoints_pk.
        bands_kpoints_pk: If provided, PK of a KpointsData node with explicit
            k-points for Wannier band interpolation. This ensures the Wannier
            bands use the exact same k-grid as a prior DFT bands calculation.
            Mutually exclusive with kpoint_path.
        dis_proj_max: Disentanglement projection maximum.
        dis_proj_min: Disentanglement projection minimum. If None, the
            Wannier90 default is used.
        dis_froz_max: If provided, upper bound of the frozen energy window
            (eV, absolute).  Switches from projectability-only to
            projectability + energy frozen window disentanglement.
        min_nbnd: If provided, minimum number of bands for the NSCF step.

    Returns:
        The AiiDA process node for the completed workflow.
    """
    from aiida_koopmans.workgraphs.wannier90 import Wannier90TaskViaBuilder
    from aiida_quantumespresso.common.types import ElectronicType
    from aiida_wannier90_workflows.common.types import (
        WannierFrozenType,
        WannierProjectionType,
    )
    from koopmans.aiida.dumping import dump_workgraph
    from koopmans.aiida.progress import run_with_progress
    from koopmans.aiida.setup import load_koopmans_profile

    load_koopmans_profile()

    _, structure = structure_to_aiida(structure_file, periodic=periodic)
    codes = _load_codes(
        required=("pw", "pw2wannier90", "wannier90"),
        optional=("projwfc",),
    )

    external_projectors = _build_external_projectors(proj_dir)

    if kpoint_path is not None and bands_kpoints_pk is not None:
        raise ValueError("Cannot specify both kpoint_path and bands_kpoints_pk.")

    w90_params: dict[str, Any] = {"dis_proj_max": dis_proj_max}
    if dis_proj_min is not None:
        w90_params["dis_proj_min"] = dis_proj_min
    if dis_froz_max is not None:
        w90_params["dis_froz_max"] = dis_froz_max
    if kpoint_path is not None or bands_kpoints_pk is not None:
        w90_params["bands_plot"] = True
    if min_nbnd is not None:
        w90_params["num_bands"] = min_nbnd
    if extra_w90_params:
        w90_params.update(extra_w90_params)

    # Load the KpointsData node if a PK was provided
    bands_kpoints_node = None
    if bands_kpoints_pk is not None:
        from aiida import orm as _orm

        bands_kpoints_node = _orm.load_node(bands_kpoints_pk)

    projector_rotation = _build_projector_rotation(
        structure_file=structure_file,
        proj_dir=proj_dir,
        bond_cutoff=bond_cutoff,
    ) if symmetrize else None

    overrides: dict[str, Any] = {
        "wannier90": {
            "wannier90": {
                "parameters": w90_params,
                "settings": {"parse_iteration_data": True},
            },
        },
    }

    wg = Wannier90TaskViaBuilder.build(
        codes=codes,
        structure=structure,
        pseudo_family=PSEUDO_LIBRARY,
        overrides=overrides,
        projection_type=WannierProjectionType.ATOMIC_PROJECTORS_EXTERNAL,
        external_projectors_path=str(proj_dir.resolve()),
        external_projectors=external_projectors,
        electronic_type=ElectronicType.METAL,
        frozen_type=(
            WannierFrozenType.FIXED_PLUS_PROJECTABILITY
            if dis_froz_max is not None
            else WannierFrozenType.PROJECTABILITY
        ),
        kpoint_path=kpoint_path,
        bands_kpoints=bands_kpoints_node,
        projector_rotation=projector_rotation,
    )
    run_with_progress(wg)

    process_node = wg.process
    if not process_node.is_finished_ok:
        raise RuntimeError(
            f"Wannierize workflow failed with exit status {process_node.exit_status}: "
            f"{process_node.exit_message}"
        )

    output_dir = working_dir / "wannierize"
    dump_workgraph(process_node, output_dir)

    return process_node


def run_wannierize_optimize_workflow(
    structure_file: Path,
    proj_dir: Path,
    working_dir: Path,
    bands_kpoints_pk: int,
    reference_bands_pk: int,
    dis_proj_max_range: list[float] | None = None,
    dis_proj_min_range: list[float] | None = None,
    dis_froz_max: float | None = None,
    extra_w90_params: dict[str, Any] | None = None,
    strategy: str = "bayesian",
    max_iterations: int = 5,
    mu_shift: float = 2.0,
    sigma: float = 10.0,
    mu_reference: str = "cbm",
    min_nbnd: int | None = None,
    periodic: tuple[bool, bool, bool] = (True, True, True),
    symmetrize: bool = False,
    bond_cutoff: float | None = None,
) -> Any:
    """Run Wannier90 optimization workflow via AiiDA.

    Uses the specified strategy (Bayesian or grid search) to find the
    optimal dis_proj_max (and optionally dis_proj_min) that minimizes
    the bands distance.

    Args:
        structure_file: Path to a structure file (CIF, XSF, etc.).
        proj_dir: Path to directory containing external projector .dat files.
        working_dir: Working directory for outputs.
        bands_kpoints_pk: PK of KpointsData with explicit k-points from DFT bands.
        reference_bands_pk: PK of BandsData with DFT reference bands.
        dis_proj_max_range: [min, max] bounds for dis_proj_max optimization.
            Defaults to [0.6, 0.95].
        dis_proj_min_range: [min, max] bounds for dis_proj_min. If None or a
            single-element list, dis_proj_min is held fixed at the default.
        max_iterations: Maximum Bayesian optimization iterations.

    Returns:
        The AiiDA process node for the completed optimization workflow.
    """
    from aiida import orm
    from aiida_koopmans.workgraphs.wannier90 import Wannier90OptimizeTaskViaBuilder
    from aiida_quantumespresso.common.types import ElectronicType
    from aiida_wannier90_workflows.common.types import (
        OptimizeMetric,
        OptimizeMuReference,
        OptimizeStrategy,
        WannierFrozenType,
        WannierProjectionType,
    )
    from koopmans.aiida.dumping import dump_workgraph
    from koopmans.aiida.progress import run_with_progress
    from koopmans.aiida.setup import load_koopmans_profile

    load_koopmans_profile()

    if dis_proj_max_range is None:
        dis_proj_max_range = [0.6, 0.95]
    if dis_proj_min_range is None:
        dis_proj_min_range = [0.01]  # single value = held fixed

    _, structure = structure_to_aiida(structure_file, periodic=periodic)
    codes = _load_codes(
        required=("pw", "pw2wannier90", "wannier90"),
        optional=("projwfc",),
    )

    external_projectors = _build_external_projectors(proj_dir)
    bands_kpoints_node = orm.load_node(bands_kpoints_pk)
    reference_bands_node = orm.load_node(reference_bands_pk)

    w90_params: dict[str, Any] = {}
    if min_nbnd is not None:
        w90_params["num_bands"] = min_nbnd
    if dis_froz_max is not None:
        w90_params["dis_froz_max"] = dis_froz_max
    if extra_w90_params:
        w90_params.update(extra_w90_params)

    overrides: dict[str, Any] = {
        "wannier90": {
            "wannier90": {
                "parameters": w90_params,
                "settings": {"parse_iteration_data": True},
            },
        },
    }

    projector_rotation = _build_projector_rotation(
        structure_file=structure_file,
        proj_dir=proj_dir,
        bond_cutoff=bond_cutoff,
    ) if symmetrize else None

    wg = Wannier90OptimizeTaskViaBuilder.build(
        codes=codes,
        structure=structure,
        pseudo_family=PSEUDO_LIBRARY,
        overrides=overrides,
        reference_bands=reference_bands_node,
        optimize_strategy=OptimizeStrategy(strategy),
        optimize_metric=OptimizeMetric.UNWEIGHTED_RMS,
        optimize_mu_shift=mu_shift,
        optimize_sigma=sigma,
        optimize_mu_reference=OptimizeMuReference(mu_reference),
        optimize_max_iterations=max_iterations,
        optimize_disprojmax_range=dis_proj_max_range,
        optimize_disprojmin_range=dis_proj_min_range,
        projection_type=WannierProjectionType.ATOMIC_PROJECTORS_EXTERNAL,
        external_projectors_path=str(proj_dir.resolve()),
        external_projectors=external_projectors,
        electronic_type=ElectronicType.METAL,
        frozen_type=(
            WannierFrozenType.FIXED_PLUS_PROJECTABILITY
            if dis_froz_max is not None
            else WannierFrozenType.PROJECTABILITY
        ),
        bands_kpoints=bands_kpoints_node,
        projector_rotation=projector_rotation,
    )
    run_with_progress(wg)

    process_node = wg.process
    if not process_node.is_finished_ok:
        raise RuntimeError(
            f"Optimize workflow failed with exit status {process_node.exit_status}: "
            f"{process_node.exit_message}"
        )

    output_dir = working_dir / "wannierize_optimize"
    dump_workgraph(process_node, output_dir)

    return process_node


def _load_codes(
    required: tuple[str, ...] = ("pw",),
    optional: tuple[str, ...] = ("dos", "projwfc", "pw2wannier90", "wannier90"),
) -> dict[str, Any]:
    """Load AiiDA codes by name.

    Args:
        required: Code names that must be available.
        optional: Code names that are loaded if available, silently skipped otherwise.
    """
    from aiida import orm

    codes: dict[str, orm.AbstractCode] = {}
    for name in required:
        try:
            codes[name] = orm.load_code(f"{name}@localhost")
        except Exception as exc:
            raise ValueError(
                f"Could not load {name} code: {exc}\n"
                "Please run 'koopmans install' first to set up the AiiDA backend."
            ) from exc

    for name in optional:
        try:
            codes[name] = orm.load_code(f"{name}@localhost")
        except Exception:
            logging.debug("Optional code %s not available", name)

    return codes
