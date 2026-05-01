"""Benchmark rival projectors by running wannierization and comparing metrics."""

from __future__ import annotations

import logging
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

logger = logging.getLogger(__name__)


@dataclass
class WannierBenchmarkResult:
    """Metrics extracted from a single wannierization run."""

    dat_file: Path
    """Path to the .dat file used for the benchmarked species."""

    omega_i: float
    """Gauge-invariant part of the spread (Ang^2)."""
    omega_d: float
    """Diagonal (gauge-dependent) part of the spread (Ang^2)."""
    omega_od: float
    """Off-diagonal (gauge-dependent) part of the spread (Ang^2)."""
    omega_total: float
    """Total spread (Ang^2)."""

    wf_spreads: list[float]
    """Per-Wannier-function spreads (Ang^2)."""
    wf_centres: list[tuple[float, float, float]]
    """Per-Wannier-function centres (Ang)."""

    converged: bool
    """Whether wannierisation converged within the iteration limit."""

    warnings: list[str]
    """Warnings from the wannier90 output."""

    label: str
    """Pretty display name for this projector (from config file)."""

    raw_parameters: dict[str, Any] = field(repr=False)
    """Full output_parameters dict from AiiDA for further inspection."""

    dis_iterations: npt.NDArray[np.float64] | None = field(default=None, repr=False)
    """Disentanglement iteration numbers."""
    dis_omega_i: npt.NDArray[np.float64] | None = field(default=None, repr=False)
    """Omega_I per disentanglement iteration."""

    spread_cycles: npt.NDArray[np.float64] | None = field(default=None, repr=False)
    """Spread minimisation cycle numbers."""
    spread_omega_total: npt.NDArray[np.float64] | None = field(default=None, repr=False)
    """Omega_Total per spread minimisation cycle."""

    band_x: npt.NDArray[np.float64] | None = field(default=None, repr=False)
    """K-path distances for Wannier-interpolated bands."""
    band_energies: npt.NDArray[np.float64] | None = field(default=None, repr=False)
    """Wannier-interpolated band energies relative to Fermi level, in eV."""
    band_labels: list[tuple[float, str]] | None = field(default=None, repr=False)
    """High-symmetry point labels as (x_position, label_string) with unicode."""

    @property
    def display_name(self) -> str:
        """Return the label."""
        return self.label


from pao_plusplus.config import BenchmarkConfig, PaoConfig, UpfConfig  # noqa: F401


def generate_dat_files(
    config: BenchmarkConfig,
    working_dir: Path,
) -> tuple[list[tuple[str, dict[str, Path]]], list[dict[str, Path]]]:
    """Generate .dat and Bessel files for each comparison set.

    Builds the Cartesian product of per-element configs (elements with
    multiple entries vary; single-entry elements are fixed).

    Returns
    -------
    combinations
        List of ``(label, {element: dat_path})`` tuples suitable for
        :func:`run_benchmark`.
    bessel_combinations
        List of ``{element: bessel_h5_path}`` dicts, one per combination,
        in the same order as *combinations*.  Used to compute fat bands.
    """
    import itertools

    from pao_plusplus.io import format_wannier90_dat
    from pao_plusplus.openmx import convert_to_wannier90, pao_to_bessel, parse_select, read_openmx_pao
    from pao_plusplus.solve import solve_and_export

    solver_dir = working_dir / "projectors"
    solver_dir.mkdir(parents=True, exist_ok=True)

    # Generate .dat and bessel .h5 for each (element, config_index) pair
    dat_paths: dict[str, list[tuple[str, Path]]] = {}
    bessel_paths: dict[str, list[Path]] = {}
    for element, configs in config.elements.items():
        dat_paths[element] = []
        bessel_paths[element] = []
        for i, elem_config in enumerate(configs):
            label = elem_config.label or f"{element} set {i}"
            if isinstance(elem_config, UpfConfig):
                result, bessel_path = solve_and_export(
                    upf_path=elem_config.upf,
                    rc=elem_config.rc,
                    ri_factor=elem_config.ri_factor,
                    extension=elem_config.get_extension(),
                    working_dir=solver_dir,
                    dat_filename=f"{element}_set{i}.dat",
                )
                dat_path = solver_dir / f"{element}_set{i}.dat"
                if bessel_path is None:
                    raise RuntimeError(
                        f"solve_and_export for {element} did not produce a Bessel file. "
                        f"Check the UPF file: {elem_config.upf}"
                    )
            elif isinstance(elem_config, PaoConfig):
                from pao_plusplus.data.openmx import fetch_pao

                pao_path = fetch_pao(element, elem_config.rc)
                pao = read_openmx_pao(pao_path)
                selected = parse_select(elem_config.select) if elem_config.select else None
                x, r, l_values, orbitals = convert_to_wannier90(pao, selected)
                dat_path = solver_dir / f"{element}_set{i}.dat"
                dat_path.write_text(format_wannier90_dat(x, r, l_values, orbitals))
                bessel_path = solver_dir / f"{element}_set{i}.h5"
                pao_to_bessel(pao, bessel_path, selected=selected)
            else:
                raise ValueError(f"Unknown config type for {element}: {type(elem_config)}")
            dat_paths[element].append((label, dat_path))
            bessel_paths[element].append(bessel_path)

    # Build Cartesian product of combinations
    elements = list(dat_paths.keys())
    per_element_choices = [dat_paths[el] for el in elements]
    per_element_bessel = [bessel_paths[el] for el in elements]
    varying = [el for el in elements if len(dat_paths[el]) > 1]

    combinations: list[tuple[str, dict[str, Path]]] = []
    bessel_combinations: list[dict[str, Path]] = []
    for combo, bessel_combo in zip(
        itertools.product(*per_element_choices),
        itertools.product(*per_element_bessel),
        strict=True,
    ):
        path_map = {el: path for el, (_, path) in zip(elements, combo, strict=True)}
        bessel_map = {el: bp for el, bp in zip(elements, bessel_combo, strict=True)}
        if len(varying) <= 1:
            if varying:
                idx = elements.index(varying[0])
                combo_label = combo[idx][0]
            else:
                combo_label = combo[0][0]
        else:
            parts = [f"{el}: {combo[elements.index(el)][0]}" for el in varying]
            combo_label = ", ".join(parts)
        combinations.append((combo_label, path_map))
        bessel_combinations.append(bessel_map)
    return combinations, bessel_combinations


def extract_benchmark_result(
    process_node: Any,
    dat_file: Path,
    label: str,
    use_optimal: bool = False,
    dft_fermi_energy: float | None = None,
) -> WannierBenchmarkResult:
    """Extract benchmark metrics from a completed Wannier90 process node.

    Args:
        process_node: AiiDA ProcessNode from a completed Wannier90WorkChain
            or Wannier90OptimizeWorkChain.
        dat_file: Path to the rival .dat file used in this run.
        label: Pretty display name for this projector.
        use_optimal: If True, extract from ``wannier90_optimal`` outputs
            (used with Wannier90OptimizeWorkChain).
        dft_fermi_energy: Fermi energy from the DFT bands calculation. If
            provided, the Wannier bands are additionally shifted to align
            with the DFT Fermi level reference.

    Returns:
        WannierBenchmarkResult with all extracted metrics.
    """
    w90_key = "wannier90_optimal" if use_optimal else "wannier90"
    params = getattr(process_node.outputs, w90_key).output_parameters.get_dict()

    wf_output = params.get("wannier_functions_output", [])
    wf_spreads = [wf["wf_spreads"] for wf in wf_output]
    wf_centres = [tuple(wf["wf_centres"]) for wf in wf_output]

    warnings = params.get("warnings", [])
    converged = not any("num_iter was reached" in w for w in warnings)

    # Extract iteration data from ArrayData outputs (if available)
    dis_iterations = None
    dis_omega_i = None
    spread_cycles = None
    spread_omega_total = None

    w90_outputs = getattr(process_node.outputs, w90_key)
    if hasattr(w90_outputs, "disentanglement_data"):
        dis_node = w90_outputs.disentanglement_data
        dis_iterations = dis_node.get_array("iteration")
        dis_omega_i = dis_node.get_array("omega_i")

    if hasattr(w90_outputs, "spread_data"):
        spread_node = w90_outputs.spread_data
        spread_cycles = spread_node.get_array("cycle")
        spread_omega_total = spread_node.get_array("omega_total")

    # Extract Wannier-interpolated bands (if band interpolation was enabled)
    band_x = None
    band_energies = None
    band_labels = None
    if hasattr(w90_outputs, "interpolated_bands"):
        bands_node = w90_outputs.interpolated_bands
        # Shift Wannier bands by their own SCF Fermi energy
        scf_params = process_node.outputs.scf.output_parameters.get_dict()
        w90_fermi_energy = scf_params["fermi_energy"]

        plot_info = bands_node._get_bandplot_data(
            cartesian=True,
            prettify_format="latex_seekpath",
            join_symbol="|",
            y_origin=w90_fermi_energy,
        )
        band_x = np.array(plot_info["x"])
        band_energies = np.array(plot_info["y"])
        band_labels = [(float(pos), str(lbl)) for pos, lbl in plot_info["labels"]]
        # Correct for the difference between the Wannier and DFT Fermi energies
        # so that both band sets are on the same zero reference
        if dft_fermi_energy is not None:
            band_energies += dft_fermi_energy - w90_fermi_energy

    return WannierBenchmarkResult(
        dat_file=dat_file,
        label=label,
        omega_i=params.get("Omega_I", float("nan")),
        omega_d=params.get("Omega_D", float("nan")),
        omega_od=params.get("Omega_OD", float("nan")),
        omega_total=params.get("Omega_total", float("nan")),
        wf_spreads=wf_spreads,
        wf_centres=wf_centres,
        converged=converged,
        warnings=warnings,
        dis_iterations=dis_iterations,
        dis_omega_i=dis_omega_i,
        spread_cycles=spread_cycles,
        spread_omega_total=spread_omega_total,
        band_x=band_x,
        band_energies=band_energies,
        band_labels=band_labels,
        raw_parameters=params,
    )


@dataclass
class OptimizeTrialResult:
    """One trial from the Bayesian optimization of dis_proj_max."""

    dis_proj_min: float
    dis_proj_max: float
    bands_distance: float | None
    """Bands distance metric (eV), or None if the trial failed."""


def extract_optimize_trajectory(
    optimize_process_node: Any,
) -> list[OptimizeTrialResult]:
    """Extract the optimization trajectory from a Wannier90OptimizeWorkChain.

    Iterates over the called Wannier90BaseWorkChain children (excluding the
    initial wannierization) and extracts dis_proj_min, dis_proj_max, and bands
    distance for each trial.

    Args:
        optimize_process_node: AiiDA ProcessNode from a completed
            Wannier90OptimizeWorkChain.

    Returns:
        List of OptimizeTrialResult, one per optimization iteration.
    """
    from aiida_wannier90_workflows.utils.bands.distance import bands_distance_fermi_dirac
    from aiida_wannier90_workflows.workflows.optimize import _resolve_mu
    from aiida_wannier90_workflows.common.types import OptimizeMuReference

    # The optimize workchain sits inside a workgraph task. We need to find
    # the actual Wannier90OptimizeWorkChain node.
    optimize_wc = _find_optimize_workchain(optimize_process_node)

    # Get reference bands from the workchain inputs
    ref_bands = optimize_wc.inputs.optimize_reference_bands

    # Get mu/sigma/mu_reference from the workchain inputs
    mu_shift = optimize_wc.inputs.optimize_mu_shift.value
    sigma = optimize_wc.inputs.optimize_sigma.value
    mu_ref = OptimizeMuReference(optimize_wc.inputs.optimize_mu_reference.value)

    # Find all Wannier90BaseWorkChain children that are optimization trials.
    # Use the called property directly instead of QueryBuilder to avoid tag issues.
    all_called = sorted(optimize_wc.called, key=lambda n: n.ctime)
    trial_workchains = [
        n for n in all_called
        if getattr(n, "process_label", "") == "Wannier90BaseWorkChain"
    ]

    trials = []
    for wc in trial_workchains:
        # Extract dis_proj_max/min from the W90 calculation parameters
        # The BaseWorkChain wraps a Wannier90Calculation
        try:
            w90_calc = [
                c for c in wc.called
                if c.process_label == "Wannier90Calculation"
            ][-1]
            w90_params = w90_calc.inputs.parameters.get_dict()
            dis_proj_max = w90_params.get("dis_proj_max", float("nan"))
            dis_proj_min = w90_params.get("dis_proj_min", float("nan"))
        except (IndexError, AttributeError):
            logger.warning("Could not extract W90 parameters from %s", wc.pk)
            continue

        # Compute bands distance if interpolated bands are available
        bands_dist = None
        if wc.is_finished_ok and hasattr(wc.outputs, "interpolated_bands"):
            try:
                mu = _resolve_mu(mu_ref, mu_shift, w90_params, ref_bands)
                bands_dist = bands_distance_fermi_dirac(
                    ref_bands, wc.outputs.interpolated_bands,
                    mu=mu, sigma=sigma,
                )
            except Exception as exc:
                logger.warning("Failed to compute bands distance for %s: %s", wc.pk, exc)
        elif not wc.is_finished_ok:
            logger.info(
                "Trial %s not finished_ok (exit_status=%s): %s",
                wc.pk, wc.exit_status, wc.exit_message,
            )

        trials.append(OptimizeTrialResult(
            dis_proj_min=dis_proj_min,
            dis_proj_max=dis_proj_max,
            bands_distance=bands_dist,
        ))

    return trials


def _find_optimize_workchain(process_node: Any) -> Any:
    """Find the Wannier90OptimizeWorkChain within a workgraph process node.

    The optimize workflow is typically wrapped in a workgraph task, so the
    top-level process_node is a WorkGraph. This function traverses to find
    the actual Wannier90OptimizeWorkChain.
    """
    from aiida.orm import QueryBuilder, WorkChainNode

    # First check if the process_node itself is the optimize workchain
    if getattr(process_node, "process_label", "") == "Wannier90OptimizeWorkChain":
        return process_node

    # Search called descendants
    qb = QueryBuilder()
    qb.append(WorkChainNode, filters={"id": process_node.pk}, tag="parent")
    qb.append(
        WorkChainNode,
        with_incoming="parent",
        filters={"attributes.process_label": "Wannier90OptimizeWorkChain"},
        project=["*"],
    )
    results = qb.all()
    if not results:
        # Try one more level deep (workgraph -> task -> optimize workchain)
        qb2 = QueryBuilder()
        qb2.append(WorkChainNode, filters={"id": process_node.pk}, tag="root")
        qb2.append(WorkChainNode, with_incoming="root", tag="mid")
        qb2.append(
            WorkChainNode,
            with_incoming="mid",
            filters={"attributes.process_label": "Wannier90OptimizeWorkChain"},
            project=["*"],
        )
        results = qb2.all()

    if not results:
        raise ValueError(
            f"Could not find Wannier90OptimizeWorkChain in descendants of <{process_node.pk}>"
        )
    return results[0][0]


def _prepare_proj_dir(
    dat_map: dict[str, Path],
    dest_dir: Path,
) -> Path:
    """Create a projector directory for one wannierization run.

    Copies each .dat file as ``{element}.dat`` into a ``projectors``
    subdirectory of *dest_dir*.  Using a deterministic path (rather than a
    random tmpdir) ensures that AiiDA can cache calculations when the
    projector files are unchanged.

    Args:
        dat_map: Mapping of element symbol to .dat file path.
        dest_dir: Parent directory; projectors are placed in ``dest_dir/projectors``.

    Returns:
        Path to the projectors directory.
    """
    proj_dir = dest_dir / "projectors"
    proj_dir.mkdir(parents=True, exist_ok=True)
    for element, dat_path in dat_map.items():
        shutil.copy2(dat_path, proj_dir / f"{element}.dat")
    return proj_dir


def run_benchmark(
    structure_file: Path,
    combinations: list[tuple[str, dict[str, Path]]],
    working_dir: Path,
    bessel_combinations: list[dict[str, Path]] | None = None,
    kpoint_path: dict[str, Any] | None = None,
    bands_kpoints_pk: int | None = None,
    dis_proj_max: float | None = None,
    dis_proj_min: float | None = None,
    dis_froz_max: float | None = None,
    extra_w90_params: dict[str, Any] | None = None,
    optimize_strategy: str | None = None,
    otsu_bins: int = 5,
    reference_bands_pk: int | None = None,
    min_nbnd: int | None = None,
    fermi_energy: float | None = None,
    periodic: tuple[bool, bool, bool] = (True, True, True),
    symmetrize: bool = False,
    bond_cutoff: float | None = None,
) -> list[WannierBenchmarkResult]:
    """Run wannierization for each projector combination and collect metrics.

    Args:
        structure_file: Path to structure file (CIF, XSF, etc.).
        combinations: List of ``(label, {element: dat_path})`` tuples, as
            produced by :attr:`BenchmarkConfig.combinations`.
        working_dir: Base working directory for outputs.
        bessel_combinations: List of ``{element: bessel_h5_path}`` dicts,
            one per combination. Required when ``optimize_strategy="otsu"``.
        kpoint_path: If provided, enable Wannier band interpolation along this
            k-path (from seekpath). Dict with 'path' and 'point_coords' keys.
            Mutually exclusive with bands_kpoints_pk.
        bands_kpoints_pk: If provided, PK of a KpointsData node with explicit
            k-points from a prior DFT bands calculation. Ensures Wannier and
            DFT bands use the exact same k-grid.
        dis_proj_max: Disentanglement projection maximum.
        dis_proj_min: Disentanglement projection minimum.
        dis_froz_max: Frozen window upper bound (eV, absolute).
        optimize_strategy: If set (``"bayesian"``, ``"grid"``, or
            ``"otsu"``), determine thresholds automatically.  For
            ``"otsu"``, computes per-combination Otsu thresholds from the
            Amn matrices. For ``"bayesian"``/``"grid"``, runs the AiiDA
            optimize workflow.
        reference_bands_pk: PK of the DFT reference BandsData node. Required
            when *optimize_strategy* is ``"bayesian"`` or ``"grid"``.
        min_nbnd: If provided, minimum number of bands for the NSCF step.
        fermi_energy: DFT Fermi energy for shifting Wannier bands.

    Returns:
        List of WannierBenchmarkResult, one per combination.
    """
    from pao_plusplus.workflows import run_wannierize_optimize_workflow, run_wannierize_workflow

    optimize = optimize_strategy in ("bayesian", "grid")
    otsu = optimize_strategy == "otsu"

    if optimize and reference_bands_pk is None:
        raise ValueError(
            "reference_bands_pk is required when optimize_strategy is 'bayesian' or 'grid'."
        )
    if optimize and bands_kpoints_pk is None:
        raise ValueError(
            "bands_kpoints_pk is required when optimize_strategy is 'bayesian' or 'grid'."
        )
    if otsu and bessel_combinations is None:
        raise ValueError(
            "bessel_combinations is required when optimize_strategy='otsu'."
        )

    # Pre-compute NSCF wavefunctions for Otsu (shared across combinations)
    _otsu_qe_result = None
    if otsu:
        from pao_plusplus.workflows import run_qe_workflow

        _otsu_qe_result = run_qe_workflow(
            structure_file, working_dir, min_nbnd=min_nbnd, periodic=periodic
        )

    working_dir.mkdir(parents=True, exist_ok=True)
    results: list[WannierBenchmarkResult] = []

    for i, (label, dat_map) in enumerate(combinations):
        logger.info("Running wannierization %d/%d: %s", i + 1, len(combinations), label)

        run_dir = working_dir / f"run_{i:03d}"
        proj_dir = _prepare_proj_dir(dat_map, dest_dir=run_dir)

        # Use the first dat file as the representative for WannierBenchmarkResult
        first_dat = next(iter(dat_map.values()))

        # Resolve per-combination thresholds
        combo_dis_proj_max = dis_proj_max
        combo_dis_proj_min = dis_proj_min
        if otsu:
            from pao_plusplus.fat_bands import build_atoms_dict, compute_amn_from_wfc
            from pao_plusplus.projectability import (
                _make_qe_input_wfc,
                suggest_disentanglement_thresholds,
            )

            atoms_dict, lattice_vectors = build_atoms_dict(
                _otsu_qe_result.nscf_input_file
            )
            qe_wfc = _make_qe_input_wfc(
                _otsu_qe_result.nscf_wfc_dir, lattice_vectors
            )
            num_kpoints = len(_otsu_qe_result.kpoint_weights)
            _smn, amn, cmn, _ch = compute_amn_from_wfc(
                qe_wfc=qe_wfc,
                bessel_files=bessel_combinations[i],
                atoms_dict=atoms_dict,
                lattice_vectors=lattice_vectors,
                num_kpoints=num_kpoints,
            )
            otsu_min, otsu_max = (
                suggest_disentanglement_thresholds(amn, cmn, otsu_bins=otsu_bins)
            )
            combo_dis_proj_max = otsu_max
            # An explicit dis_proj_min overrides the Otsu value
            if dis_proj_min is not None:
                combo_dis_proj_min = dis_proj_min
            else:
                combo_dis_proj_min = otsu_min
            logger.info(
                "Otsu thresholds for %s: dis_proj_min=%.4f%s, dis_proj_max=%.4f",
                label, combo_dis_proj_min,
                " (explicit override)" if dis_proj_min is not None else " (otsu)",
                combo_dis_proj_max,
            )

        if optimize:
            # Explicit values become single-element ranges (held fixed);
            # None values become None ranges (optimized).
            dis_proj_max_range = [combo_dis_proj_max] if combo_dis_proj_max is not None else [0.6, 0.95]
            dis_proj_min_range = [combo_dis_proj_min] if combo_dis_proj_min is not None else None
            process_node = run_wannierize_optimize_workflow(
                structure_file=structure_file,
                proj_dir=proj_dir,
                working_dir=run_dir,
                bands_kpoints_pk=bands_kpoints_pk,
                reference_bands_pk=reference_bands_pk,
                dis_proj_max_range=dis_proj_max_range,
                dis_proj_min_range=dis_proj_min_range,
                dis_froz_max=dis_froz_max,

                extra_w90_params=extra_w90_params,
                strategy=optimize_strategy,
                min_nbnd=min_nbnd,
                periodic=periodic,
                symmetrize=symmetrize,
                bond_cutoff=bond_cutoff,
            )
            result = extract_benchmark_result(
                process_node, first_dat, label=label, use_optimal=True,
                dft_fermi_energy=fermi_energy,
            )
        else:
            process_node = run_wannierize_workflow(
                structure_file=structure_file,
                proj_dir=proj_dir,
                working_dir=run_dir,
                kpoint_path=kpoint_path,
                bands_kpoints_pk=bands_kpoints_pk,
                dis_proj_max=combo_dis_proj_max if combo_dis_proj_max is not None else 0.8,
                dis_proj_min=combo_dis_proj_min,
                dis_froz_max=dis_froz_max,

                extra_w90_params=extra_w90_params,
                min_nbnd=min_nbnd,
                periodic=periodic,
                symmetrize=symmetrize,
                bond_cutoff=bond_cutoff,
            )
            result = extract_benchmark_result(
                process_node, first_dat, label=label,
                dft_fermi_energy=fermi_energy,
            )

        results.append(result)

    return results


def format_benchmark_table(results: list[WannierBenchmarkResult]) -> str:
    """Format benchmark results as a human-readable comparison table.

    Args:
        results: List of WannierBenchmarkResult from run_benchmark.

    Returns:
        Formatted table string.
    """
    if not results:
        return "No results to display."

    lines: list[str] = []

    # Header
    lines.append(
        f"{'Projector':<40s} {'Conv':>4s} {'Omega_I':>8s} "
        f"{'Omega_D':>8s} {'Omega_OD':>8s} {'Omega_tot':>8s} "
        f"{'max(s)':>8s} {'mean(s)':>8s}"
    )
    lines.append("-" * len(lines[0]))

    for r in results:
        name = r.display_name
        if len(name) > 39:
            name = "..." + name[-36:]
        conv = "yes" if r.converged else "NO"
        max_spread = max(r.wf_spreads) if r.wf_spreads else float("nan")
        mean_spread = (sum(r.wf_spreads) / len(r.wf_spreads)) if r.wf_spreads else float("nan")
        lines.append(
            f"{name:<40s} {conv:>4s} {r.omega_i:8.4f} "
            f"{r.omega_d:8.4f} {r.omega_od:8.4f} "
            f"{r.omega_total:8.4f} {max_spread:8.4f} "
            f"{mean_spread:8.4f}"
        )

    # Per-WF detail
    lines.append("")
    lines.append("Per-Wannier-function spreads:")
    for r in results:
        name = r.display_name
        if len(name) > 39:
            name = "..." + name[-36:]
        spreads_str = ", ".join(f"{s:.4f}" for s in r.wf_spreads)
        lines.append(f"  {name}: [{spreads_str}]")

    return "\n".join(lines)


def _annotate_convergence(
    ax: Any,
    n_iter: int,
    y_final: float,
    color: str,
) -> None:
    """Add a vertical arrow annotation at x=n_iter on the given axis."""
    ax.annotate(
        f"{n_iter} iterations",
        xy=(n_iter, y_final),
        xycoords="data",
        xytext=(0, 10),
        textcoords="offset points",
        fontsize=5,
        color=color,
        ha="center",
        va="bottom",
        rotation=90,
        bbox={
            "boxstyle": "round,pad=0.15",
            "fc": "white",
            "ec": "none",
            "alpha": 0.8,
        },
        arrowprops={
            "arrowstyle": "->",
            "color": color,
            "lw": 0.8,
        },
    )


def _sqrt_shifted_scale(offset: float) -> tuple:
    """Return (forward, inverse) functions for a sqrt(y - offset) scale."""

    def forward(y: npt.ArrayLike) -> npt.ArrayLike:
        with np.errstate(invalid="ignore"):
            return np.sqrt(y - offset)

    def inverse(y_t: npt.ArrayLike) -> npt.ArrayLike:
        return y_t**2 + offset

    return forward, inverse


def _plot_disentanglement_panel(
    ax: Any,
    results: list[WannierBenchmarkResult],
) -> None:
    """Plot the disentanglement convergence panel (Omega_I vs iteration)."""
    best_dis = min(r.dis_omega_i[-1] for r in results if r.dis_iterations is not None)
    margin_dis = 0.02
    data_max = max(
        float(np.max(r.dis_omega_i)) for r in results if r.dis_iterations is not None
    )
    for r in results:
        if r.dis_iterations is not None:
            (line,) = ax.plot(
                r.dis_iterations,
                r.dis_omega_i,
                label=r.display_name,
            )
            n_iter = int(r.dis_iterations[-1])
            _annotate_convergence(ax, n_iter, r.dis_omega_i[-1], line.get_color())
    ax.set_ylim(best_dis - margin_dis, data_max)
    ax.set_yscale("function", functions=_sqrt_shifted_scale(best_dis - margin_dis))
    ax.set_xscale("log")
    ax.set_xlim(left=1)
    ax.set_xlabel("Disentanglement iteration")
    ax.set_ylabel(r"$\Omega_\mathrm{I}$ ($\AA^2$)")
    ax.legend(fontsize=6, loc="lower right", bbox_to_anchor=(1, 1))


def _plot_spread_panel(
    ax: Any,
    results: list[WannierBenchmarkResult],
) -> None:
    """Plot the spread minimisation convergence panel (Omega_Total vs cycle)."""
    best_spread = min(r.spread_omega_total[-1] for r in results if r.spread_cycles is not None)
    margin_spread = 0.02
    data_max = max(
        float(np.max(r.spread_omega_total)) for r in results if r.spread_cycles is not None
    )
    for r in results:
        if r.spread_cycles is not None:
            (line,) = ax.plot(
                r.spread_cycles,
                r.spread_omega_total,
            )
            n_iter = int(r.spread_cycles[-1])
            _annotate_convergence(ax, n_iter, r.spread_omega_total[-1], line.get_color())
    ax.set_ylim(best_spread - margin_spread, data_max)
    ax.set_yscale("function", functions=_sqrt_shifted_scale(best_spread - margin_spread))
    ax.set_xscale("log")
    ax.set_xlim(left=1)
    ax.set_xlabel("Spread minimisation iteration")
    ax.set_ylabel(r"$\Omega_\mathrm{total}$ ($\AA^2$)")


def plot_convergence(
    results: list[WannierBenchmarkResult],
    filename: Path | str = "convergence.svg",
) -> None:
    """Plot disentanglement and spread minimisation convergence for each rival.

    Produces a two-panel figure:
      - Top: Omega_I vs disentanglement iteration
      - Bottom: Omega_Total vs spread minimisation cycle

    Args:
        results: List of WannierBenchmarkResult (must contain iteration data).
        filename: Path to save the figure.
    """
    import matplotlib.pyplot as plt

    from pao_plusplus.plotting import REVTEX_COLUMN_WIDTH

    has_dis = any(r.dis_iterations is not None for r in results)
    has_spread = any(r.spread_cycles is not None for r in results)
    n_panels = has_dis + has_spread
    if n_panels == 0:
        logger.warning("No iteration data found in any result; nothing to plot.")
        return

    fig, axes = plt.subplots(
        n_panels,
        1,
        figsize=(REVTEX_COLUMN_WIDTH, 2.5 * n_panels),
        squeeze=False,
    )
    axes = axes[:, 0]
    ax_idx = 0

    if has_dis:
        _plot_disentanglement_panel(axes[ax_idx], results)
        ax_idx += 1

    if has_spread:
        _plot_spread_panel(axes[ax_idx], results)

    from pao_plusplus.plotting import savefig as _savefig

    fig.tight_layout()
    _savefig(fig, filename)
    plt.close(fig)
    logger.info("Convergence plot saved to %s", filename)


_BAND_LINESTYLES = ["--", "-.", ":", (0, (3, 1, 1, 1, 1, 1))]


def _discontinuity_positions(
    labels: list[tuple[float, str]] | None,
) -> list[float]:
    """Return x-positions where the k-path is discontinuous.

    Discontinuities are identified by labels containing ``"|"``
    (e.g. ``"X|Y"``), which seekpath uses to mark path breaks.
    """
    if labels is None:
        return []
    return [pos for pos, lbl in labels if "|" in lbl]


def _split_segments(
    x: npt.NDArray[np.float64],
    energies: npt.NDArray[np.float64],
    disc_positions: list[float],
) -> list[tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]]:
    """Split band arrays at discontinuity positions.

    Returns a list of (x_segment, energies_segment) tuples.  Each segment
    contains the k-points strictly between consecutive discontinuities.
    """
    if not disc_positions:
        return [(x, energies)]

    segments: list[tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]] = []
    prev = 0
    for disc_x in disc_positions:
        # Find the first index at the discontinuity x-value and include it
        # in the current segment (side="left" + 1), so the duplicate point
        # at the same x starts the *next* segment.
        idx = int(np.searchsorted(x, disc_x, side="left")) + 1
        if idx > prev:
            segments.append((x[prev:idx], energies[prev:idx]))
        prev = idx
    if prev < len(x):
        segments.append((x[prev:], energies[prev:]))
    return segments


def _plot_dft_bands(ax: Any, dft_band_plot_data: Any) -> None:
    """Plot DFT bands as grey background lines on *ax*."""
    dft_x = dft_band_plot_data.x
    dft_energies = dft_band_plot_data.energies  # already Fermi-shifted
    disc = _discontinuity_positions(dft_band_plot_data.labels)
    for seg_x, seg_e in _split_segments(dft_x, dft_energies, disc):
        for band_idx in range(seg_e.shape[1]):
            ax.plot(seg_x, seg_e[:, band_idx], color="0.75", lw=0.5, zorder=1)


def _plot_wannier_bands(
    ax: Any,
    band_results: list[WannierBenchmarkResult],
    colors: list[str],
) -> None:
    """Plot each rival's Wannier-interpolated bands on *ax*."""
    for i, (r, color) in enumerate(zip(band_results, colors, strict=False)):
        ls = _BAND_LINESTYLES[i % len(_BAND_LINESTYLES)]
        disc = _discontinuity_positions(r.band_labels)
        for seg_x, seg_e in _split_segments(r.band_x, r.band_energies, disc):
            for band_idx in range(seg_e.shape[1]):
                ax.plot(
                    seg_x,
                    seg_e[:, band_idx],
                    color=color,
                    ls=ls,
                    lw=0.5,
                    zorder=2,
                )


def _configure_band_axis(
    ax: Any,
    ref: WannierBenchmarkResult,
    emin: float,
    emax: float,
) -> None:
    """Set axis limits, Fermi line, and high-symmetry labels on *ax*."""
    ax.set_xlim(ref.band_x[0], ref.band_x[-1])
    ax.set_ylim(emin, emax)
    ax.axhline(0, color="k", ls="--", lw=0.3)

    if ref.band_labels is not None:
        label_positions = [pos for pos, _ in ref.band_labels]
        label_strings = [lbl for _, lbl in ref.band_labels]
        for pos in label_positions:
            ax.axvline(pos, color="k", ls="-", lw=0.3, alpha=0.5)
        ax.set_xticks(label_positions)
        ax.set_xticklabels(label_strings)

    ax.set_ylabel("Energy (eV)")


def _plot_fat_bands_on_axis(
    ax: Any,
    dft_band_plot_data: Any,
    channel_projectabilities: dict[tuple[str, int], npt.NDArray[np.float64]],
    emin: float,
    emax: float,
) -> dict[tuple[str, Any], tuple[float, float, float]]:
    """Draw DFT bands with per-channel fat band overlays on *ax*.

    Returns the ``channel_colors`` mapping so callers can build a
    combined legend.
    """
    from pao_plusplus.fat_bands import draw_fat_bands_on_axis

    return draw_fat_bands_on_axis(
        ax,
        dft_band_plot_data.x,
        dft_band_plot_data.energies,
        channel_projectabilities,
        emin,
        emax,
    )


def _plot_projectability_panel(
    ax_proj: Any,
    dft_band_plot_data: Any,
    channel_projectabilities: dict[tuple[str, int], npt.NDArray[np.float64]],
) -> None:
    """Plot total projectability scatter on the side panel."""
    from matplotlib import cm

    from pao_plusplus.plotting import COLORMAP

    energies = dft_band_plot_data.energies
    total_proj = sum(channel_projectabilities.values())
    cmap = cm.get_cmap(COLORMAP)
    proj_color = cmap(0.5)

    for band_idx in range(energies.shape[1]):
        ax_proj.scatter(
            total_proj[:, band_idx],
            energies[:, band_idx],
            s=2,
            color=proj_color,
            alpha=0.5,
            edgecolors="none",
        )


def _build_band_legend(
    ax: Any,
    band_results: list[WannierBenchmarkResult],
    colors: list[str],
    dft_band_plot_data: Any | None,
    extra_handles: list[Any] | None = None,
) -> None:
    """Build and attach a legend to *ax* for band comparison plots.

    If *extra_handles* is provided (e.g. fat-band channel entries), they
    are appended so that a single combined legend is drawn.
    """
    from matplotlib.lines import Line2D

    handles: list[Any] = []
    if dft_band_plot_data is not None:
        handles.append(Line2D([], [], color="0.75", lw=1, label="DFT"))
    for i, (r, color) in enumerate(zip(band_results, colors, strict=False)):
        ls = _BAND_LINESTYLES[i % len(_BAND_LINESTYLES)]
        handles.append(
            Line2D([], [], color=color, ls=ls, lw=1, label=r.display_name),
        )
    if extra_handles:
        handles.extend(extra_handles)
    ax.legend(
        handles=handles, fontsize=6, loc="lower right", bbox_to_anchor=(1, 1),
        ncol=len(handles),
    )


def plot_bands_comparison(
    results: list[WannierBenchmarkResult],
    dft_band_plot_data: Any | None = None,
    channel_projectabilities: dict[tuple[str, int], npt.NDArray[np.float64]] | None = None,
    emin: float | None = None,
    emax: float | None = None,
    filename: Path | str = "bands_comparison.svg",
) -> None:
    """Plot Wannier-interpolated bands overlaid on DFT bands.

    All rivals are plotted on a single axis: DFT bands as grey lines,
    each rival's Wannier-interpolated bands as a different colour.
    Both DFT and Wannier bands follow the same seekpath k-path
    (potentially at different k-point densities).

    When *channel_projectabilities* is provided (single projector set),
    fat bands are drawn on the DFT bands and a projectability side
    panel is added.

    Args:
        results: List of WannierBenchmarkResult (must contain band_energies).
        dft_band_plot_data: BandPlotData from run_bands_workflow. If provided,
            DFT bands are plotted as grey background lines.
        channel_projectabilities: Per-(species, l) projectability arrays of
            shape ``(num_kpoints, num_bands)``.  If provided, fat bands are
            rendered on the DFT bands.
        emin: Lower energy limit relative to Fermi level (eV). If None,
            determined from the data with padding.
        emax: Upper energy limit relative to Fermi level (eV). If None,
            determined from the data with padding.
        filename: Path to save the figure.
    """
    import matplotlib.pyplot as plt

    from pao_plusplus.plotting import REVTEX_COLUMN_WIDTH

    band_results = [r for r in results if r.band_energies is not None]
    if not band_results:
        logger.warning("No interpolated bands found in any result; nothing to plot.")
        return

    # Determine energy range from all Wannier bands with padding
    all_energies = np.concatenate(
        [r.band_energies.ravel() for r in band_results],
    )
    padding = 0.025 * (all_energies.max() - all_energies.min())
    if emin is None:
        emin = float(all_energies.min()) - padding
    if emax is None:
        emax = float(all_energies.max()) + padding

    has_fat_bands = channel_projectabilities is not None and dft_band_plot_data is not None

    if has_fat_bands:
        fig, (ax, ax_proj) = plt.subplots(
            1,
            2,
            sharey=True,
            width_ratios=[4, 1],
            figsize=(REVTEX_COLUMN_WIDTH, REVTEX_COLUMN_WIDTH * 0.75),
            gridspec_kw={"wspace": 0.1},
        )
        channel_colors = _plot_fat_bands_on_axis(
            ax, dft_band_plot_data, channel_projectabilities, emin, emax
        )
        _plot_projectability_panel(ax_proj, dft_band_plot_data, channel_projectabilities)
    else:
        channel_colors = None
        fig, ax = plt.subplots(
            figsize=(REVTEX_COLUMN_WIDTH, REVTEX_COLUMN_WIDTH * 0.75),
        )
        if dft_band_plot_data is not None:
            _plot_dft_bands(ax, dft_band_plot_data)

    colors = [f"C{i}" for i in range(len(band_results))]
    _plot_wannier_bands(ax, band_results, colors)
    _configure_band_axis(ax, band_results[0], emin, emax)
    extra_handles = None
    if channel_colors is not None:
        from pao_plusplus.fat_bands import build_fat_bands_legend_handles

        extra_handles = build_fat_bands_legend_handles(channel_colors)
    _build_band_legend(
        ax, band_results, colors, dft_band_plot_data, extra_handles=extra_handles
    )

    if has_fat_bands:
        from pao_plusplus.fat_bands import _configure_proj_panel

        _configure_proj_panel(ax_proj)
        fig.subplots_adjust(left=0.15, bottom=0.15, right=0.99, top=0.925)
    else:
        fig.tight_layout()
    from pao_plusplus.plotting import savefig as _savefig

    _savefig(fig, filename)
    plt.close(fig)
    logger.info("Bands comparison plot saved to %s", filename)


def plot_optimize_trajectory(
    trials: list[OptimizeTrialResult],
    filename: Path | str = "optimize_trajectory.svg",
) -> None:
    """Plot bands distance vs dis_proj_max for each Bayesian optimization trial.

    Args:
        trials: List of OptimizeTrialResult from extract_optimize_trajectory.
        filename: Path to save the figure.
    """
    import matplotlib.pyplot as plt

    from pao_plusplus.plotting import REVTEX_COLUMN_WIDTH

    valid = [t for t in trials if t.bands_distance is not None]
    if not valid:
        logger.warning("No valid optimization trials to plot.")
        return

    dis_proj_max = [t.dis_proj_max for t in valid]
    bands_dist = [t.bands_distance for t in valid]

    fig, ax = plt.subplots(figsize=(REVTEX_COLUMN_WIDTH, REVTEX_COLUMN_WIDTH * 0.6))

    # Color points by iteration order
    for i, (x, y) in enumerate(zip(dis_proj_max, bands_dist)):
        ax.plot(x, y, "o", color=f"C{i}", markersize=5, zorder=3)
        ax.annotate(str(i + 1), (x, y), fontsize=6, textcoords="offset points",
                    xytext=(4, 4))

    # Highlight the best trial
    best_idx = int(np.argmin(bands_dist))
    ax.plot(
        dis_proj_max[best_idx], bands_dist[best_idx],
        "*", color="C3", markersize=12, zorder=4, label=f"best (iter {best_idx + 1})",
    )

    ax.set_xlabel(r"$\mathrm{dis\_proj\_max}$")
    ax.set_ylabel("Bands distance (eV)")
    from pao_plusplus.plotting import savefig as _savefig

    ax.legend(fontsize=7)
    fig.tight_layout()
    _savefig(fig, filename)
    plt.close(fig)
    logger.info("Optimize trajectory plot saved to %s", filename)
