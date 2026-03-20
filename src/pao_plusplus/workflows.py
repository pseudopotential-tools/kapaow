"""Workflow module for pao_plusplus using AiiDA workgraphs."""

from __future__ import annotations

import json
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
_QE_CACHE_FILE = "workflow_cache.json"
_BANDS_CACHE_FILE = "bands_cache.json"


@dataclass
class QEWorkflowResult:
    """Results from a QE workflow run."""

    atoms: ase.Atoms
    nscf_input_file: Path
    nscf_wfc_dir: Path
    output_dir: Path
    kpoint_weights: npt.NDArray[np.float64]
    fermi_energy: float


def structure_to_aiida(structure_file: Path) -> tuple[ase.Atoms, Any]:
    """Read a structure file and convert to AiiDA StructureData.

    Requires that the AiiDA profile has already been loaded.

    Args:
        structure_file: Path to CIF, XSF, or any ASE-readable structure file.

    Returns:
        Tuple of (ASE Atoms, AiiDA StructureData).
    """
    from aiida import orm

    atoms = ase.io.read(str(structure_file))
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
        d for d in dump_dir.rglob("*PwCalculation")
        if any("nscf" in p.name for p in d.parents)
    ]
    if not matches:
        raise FileNotFoundError(
            f"No NSCF PwCalculation directory found in {dump_dir}. "
            f"Contents: {list(dump_dir.rglob('*'))}"
        )
    return matches[0]


def _try_load_qe_cache(working_dir: Path, structure_file: Path) -> QEWorkflowResult | None:
    """Try to load a cached QEWorkflowResult from disk."""
    cache_file = working_dir / _QE_CACHE_FILE
    if not cache_file.exists():
        return None
    try:
        meta = json.loads(cache_file.read_text())
        nscf_input_file = working_dir / meta["nscf_input_file"]
        nscf_wfc_dir = working_dir / meta["nscf_wfc_dir"]
        output_dir = working_dir / meta["output_dir"]
        if not nscf_input_file.exists() or not nscf_wfc_dir.exists():
            return None
        atoms = ase.io.read(str(structure_file))
        result = QEWorkflowResult(
            atoms=atoms,
            nscf_input_file=nscf_input_file,
            nscf_wfc_dir=nscf_wfc_dir,
            output_dir=output_dir,
            kpoint_weights=np.array(meta["kpoint_weights"]),
            fermi_energy=meta["fermi_energy"],
        )
        logger.info("Loaded cached QE result from %s", cache_file)
        return result
    except (json.JSONDecodeError, KeyError, FileNotFoundError):
        return None


def _save_qe_cache(working_dir: Path, result: QEWorkflowResult) -> None:
    """Save QE workflow metadata to disk for fast re-loading."""
    cache_file = working_dir / _QE_CACHE_FILE
    meta = {
        "nscf_input_file": str(result.nscf_input_file.relative_to(working_dir)),
        "nscf_wfc_dir": str(result.nscf_wfc_dir.relative_to(working_dir)),
        "output_dir": str(result.output_dir.relative_to(working_dir)),
        "kpoint_weights": result.kpoint_weights.tolist(),
        "fermi_energy": result.fermi_energy,
    }
    cache_file.write_text(json.dumps(meta))


def run_qe_workflow(
    structure_file: Path,
    working_dir: Path,
    min_nbnd: int | None = None,
) -> QEWorkflowResult:
    """Run SCF + NSCF via AiiDA PwScfNscfTask.

    On subsequent calls with the same working_dir, returns cached results
    from disk without touching AiiDA.

    Args:
        structure_file: Path to a structure file (CIF, XSF, etc.).
        working_dir: Working directory for this material.
        min_nbnd: Minimum number of bands.

    Returns:
        QEWorkflowResult with paths to NSCF outputs.
    """
    cached = _try_load_qe_cache(working_dir, structure_file)
    if cached is not None:
        return cached

    from aiida import orm

    from aiida_koopmans.workgraphs.pw import PwScfNscfTask
    from koopmans.aiida.dumping import dump_workgraph
    from koopmans.aiida.progress import run_with_progress
    from koopmans.aiida.setup import load_koopmans_profile

    load_koopmans_profile()

    atoms, structure = structure_to_aiida(structure_file)
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

    _save_qe_cache(working_dir, result)
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


def _find_bands_calc_dir(dump_dir: Path) -> Path:
    """Find the bands PwCalculation directory within a PwBandsWorkChain dump.

    Looks for a PwCalculation whose immediate PwBaseWorkChain parent
    contains "bands" (e.g. ``03-bands-PwBaseWorkChain``), excluding the
    SCF sub-step which also lives under the top-level PwBandsWorkChain.
    """
    matches = [
        d for d in dump_dir.rglob("*PwCalculation")
        if any("bands" in p.name and "PwBaseWorkChain" in p.name for p in d.parents)
    ]
    if not matches:
        raise FileNotFoundError(
            f"No bands PwCalculation directory found in {dump_dir}. "
            f"Contents: {list(dump_dir.rglob('*'))}"
        )
    return matches[0]


def _try_load_bands_cache(working_dir: Path) -> BandsWorkflowResult | None:
    """Try to load a cached BandsWorkflowResult from disk."""
    cache_file = working_dir / _BANDS_CACHE_FILE
    if not cache_file.exists():
        return None
    try:
        meta = json.loads(cache_file.read_text())
        bands_calc_dir = working_dir / meta["bands_calc_dir"]
        output_dir = working_dir / meta["output_dir"]
        if not (bands_calc_dir / "inputs" / "aiida.in").exists():
            return None

        band_x = np.load(working_dir / "band_x.npy")
        band_energies = np.load(working_dir / "band_energies.npy")
        band_labels = [(float(pos), str(lbl)) for pos, lbl in meta["band_labels"]]

        logger.info("Loaded cached bands result from %s", cache_file)
        return BandsWorkflowResult(
            bands_calc_dir=bands_calc_dir,
            output_dir=output_dir,
            fermi_energy=meta["fermi_energy"],
            band_plot_data=BandPlotData(
                x=band_x,
                energies=band_energies,
                labels=band_labels,
            ),
            bands_kpoints_pk=meta.get("bands_kpoints_pk"),
        )
    except (json.JSONDecodeError, KeyError, FileNotFoundError):
        return None


def run_bands_workflow(
    structure_file: Path,
    working_dir: Path,
    min_nbnd: int | None = None,
) -> BandsWorkflowResult:
    """Run SCF + bands along high-symmetry k-path via PwBandsWorkChain.

    On subsequent calls with the same working_dir, returns cached results
    from disk without touching AiiDA.

    Args:
        structure_file: Path to a structure file (CIF, XSF, etc.).
        working_dir: Working directory for this material.
        min_nbnd: Minimum number of bands for the bands calculation.

    Returns:
        BandsWorkflowResult with path to the bands calculation dump.
    """
    cached = _try_load_bands_cache(working_dir)
    if cached is not None:
        return cached

    from aiida import orm

    from aiida_koopmans.workgraphs.pw import PwBandsTaskViaBuilder
    from koopmans.aiida.dumping import dump_workgraph
    from koopmans.aiida.progress import run_with_progress
    from koopmans.aiida.setup import load_koopmans_profile

    load_koopmans_profile()

    _, structure = structure_to_aiida(structure_file)
    pw_code = orm.load_code("pw@localhost")

    # Retrieve bands wavefunctions (needed for fat bands Amn computation)
    bands_pw_overrides: dict[str, Any] = {
        "metadata": {
            "options": {
                "additional_retrieve_list": ["out/aiida.save/wfc*.hdf5"],
            }
        }
    }
    overrides: dict[str, Any] = {"bands": {"pw": bands_pw_overrides}}
    if min_nbnd is not None:
        bands_pw_overrides["parameters"] = {"SYSTEM": {"nbnd": min_nbnd}}

    wg = PwBandsTaskViaBuilder.build(
        code=pw_code,
        structure=structure,
        pseudo_family=PSEUDO_LIBRARY,
        overrides=overrides,
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
        cartesian=True, prettify_format="latex_seekpath", join_symbol="|",
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

    # Save band arrays to disk
    np.save(working_dir / "band_x.npy", band_x)
    np.save(working_dir / "band_energies.npy", band_energies)

    # Write cache with paths and labels for fast reload
    cache_file = working_dir / _BANDS_CACHE_FILE
    cache_file.write_text(json.dumps({
        "bands_calc_dir": str(bands_calc_dir.relative_to(working_dir)),
        "output_dir": str(output_dir.relative_to(working_dir)),
        "fermi_energy": fermi_energy,
        "band_labels": band_labels,
        "bands_kpoints_pk": bands_kpoints_pk,
    }))

    band_plot_data = BandPlotData(
        x=band_x, energies=band_energies, labels=band_labels,
    )

    return BandsWorkflowResult(
        bands_calc_dir=bands_calc_dir,
        output_dir=output_dir,
        fermi_energy=fermi_energy,
        band_plot_data=band_plot_data,
        bands_kpoints_pk=bands_kpoints_pk,
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
    from pao_plusplus.io import read_wannier90_dat_file

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


def run_wannierize_workflow(
    structure_file: Path,
    proj_dir: Path,
    working_dir: Path,
    kpoint_path: dict[str, Any] | None = None,
    bands_kpoints_pk: int | None = None,
    dis_proj_max: float = 0.8,
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

    Returns:
        The AiiDA process node for the completed workflow.
    """
    from aiida_quantumespresso.common.types import ElectronicType
    from aiida_wannier90_workflows.common.types import (
        WannierFrozenType,
        WannierProjectionType,
    )

    from aiida_koopmans.workgraphs.wannier90 import Wannier90TaskViaBuilder
    from koopmans.aiida.dumping import dump_workgraph
    from koopmans.aiida.progress import run_with_progress
    from koopmans.aiida.setup import load_koopmans_profile

    load_koopmans_profile()

    _, structure = structure_to_aiida(structure_file)
    codes = _load_codes(
        required=("pw", "pw2wannier90", "wannier90"),
        optional=("projwfc",),
    )

    external_projectors = _build_external_projectors(proj_dir)

    if kpoint_path is not None and bands_kpoints_pk is not None:
        raise ValueError("Cannot specify both kpoint_path and bands_kpoints_pk.")

    w90_params: dict[str, Any] = {"dis_proj_max": dis_proj_max}
    if kpoint_path is not None or bands_kpoints_pk is not None:
        w90_params["bands_plot"] = True

    # Load the KpointsData node if a PK was provided
    bands_kpoints_node = None
    if bands_kpoints_pk is not None:
        from aiida import orm as _orm
        bands_kpoints_node = _orm.load_node(bands_kpoints_pk)

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
        frozen_type=WannierFrozenType.PROJECTABILITY,
        kpoint_path=kpoint_path,
        bands_kpoints=bands_kpoints_node,
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
            pass

    return codes
