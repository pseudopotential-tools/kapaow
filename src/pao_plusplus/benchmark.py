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
    """Wannier-interpolated band energies relative to Fermi level (num_kpoints x num_bands), in eV."""
    band_labels: list[tuple[float, str]] | None = field(default=None, repr=False)
    """High-symmetry point labels as (x_position, label_string) with unicode."""


def extract_benchmark_result(
    process_node: Any,
    dat_file: Path,
) -> WannierBenchmarkResult:
    """Extract benchmark metrics from a completed Wannier90WorkChain process node.

    Args:
        process_node: AiiDA ProcessNode from a completed Wannier90WorkChain.
        dat_file: Path to the rival .dat file used in this run.

    Returns:
        WannierBenchmarkResult with all extracted metrics.
    """
    params = process_node.outputs.wannier90.output_parameters.get_dict()

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

    w90_outputs = process_node.outputs.wannier90
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
        # Get Fermi energy from the SCF step for shifting
        scf_params = process_node.outputs.scf.output_parameters.get_dict()
        fermi_energy = scf_params["fermi_energy"]
        # Use AiiDA's bandplot helper for proper x-distances and labels
        plot_info = bands_node._get_bandplot_data(
            cartesian=True, prettify_format="latex_seekpath", join_symbol="|",
            y_origin=fermi_energy,
        )
        band_x = np.array(plot_info["x"])
        band_energies = np.array(plot_info["y"])
        band_labels = [(float(pos), str(lbl)) for pos, lbl in plot_info["labels"]]

    return WannierBenchmarkResult(
        dat_file=dat_file,
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


def _prepare_proj_dir(
    rival_dat: Path,
    element: str,
    fixed_dats: dict[str, Path],
    dest_dir: Path,
) -> Path:
    """Create a projector directory for one wannierization run.

    Copies the rival .dat as ``{element}.dat`` and each fixed .dat as
    ``{species}.dat`` into a ``projectors`` subdirectory of *dest_dir*.
    Using a deterministic path (rather than a random tmpdir) ensures that
    AiiDA can cache calculations when the projector files are unchanged.

    Args:
        rival_dat: Path to the rival projector .dat file.
        element: Element symbol for the species being benchmarked.
        fixed_dats: Mapping of species symbol to .dat file for non-rival species.
        dest_dir: Parent directory; projectors are placed in ``dest_dir/projectors``.

    Returns:
        Path to the projectors directory.
    """
    proj_dir = dest_dir / "projectors"
    proj_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(rival_dat, proj_dir / f"{element}.dat")
    for species, dat_path in fixed_dats.items():
        shutil.copy2(dat_path, proj_dir / f"{species}.dat")
    return proj_dir


def run_benchmark(
    structure_file: Path,
    element: str,
    rival_dats: list[Path],
    fixed_dats: dict[str, Path],
    working_dir: Path,
    kpoint_path: dict[str, Any] | None = None,
    bands_kpoints_pk: int | None = None,
    dis_proj_max: float = 0.8,
) -> list[WannierBenchmarkResult]:
    """Run wannierization for each rival projector and collect metrics.

    Args:
        structure_file: Path to structure file (CIF, XSF, etc.).
        element: Element symbol for the species being benchmarked.
        rival_dats: List of rival .dat files (all for the same element).
        fixed_dats: Mapping of species symbol to .dat file for other species.
        working_dir: Base working directory for outputs.
        kpoint_path: If provided, enable Wannier band interpolation along this
            k-path (from seekpath). Dict with 'path' and 'point_coords' keys.
            Mutually exclusive with bands_kpoints_pk.
        bands_kpoints_pk: If provided, PK of a KpointsData node with explicit
            k-points from a prior DFT bands calculation. Ensures Wannier and
            DFT bands use the exact same k-grid.
        dis_proj_max: Disentanglement projection maximum.

    Returns:
        List of WannierBenchmarkResult, one per rival.
    """
    from pao_plusplus.workflows import run_wannierize_workflow

    working_dir.mkdir(parents=True, exist_ok=True)
    results: list[WannierBenchmarkResult] = []

    for i, rival_dat in enumerate(rival_dats):
        logger.info("Running wannierization %d/%d: %s", i + 1, len(rival_dats), rival_dat.name)

        run_dir = working_dir / f"rival_{i:03d}_{rival_dat.stem}"
        proj_dir = _prepare_proj_dir(rival_dat, element, fixed_dats, dest_dir=run_dir)

        process_node = run_wannierize_workflow(
            structure_file=structure_file,
            proj_dir=proj_dir,
            working_dir=run_dir,
            kpoint_path=kpoint_path,
            bands_kpoints_pk=bands_kpoints_pk,
            dis_proj_max=dis_proj_max,
        )
        result = extract_benchmark_result(process_node, rival_dat)
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
    lines.append(f"{'Projector':<40s} {'Conv':>4s} {'Ω_I':>8s} {'Ω_D':>8s} "
                 f"{'Ω_OD':>8s} {'Ω_tot':>8s} {'max(σ)':>8s} {'mean(σ)':>8s}")
    lines.append("-" * len(lines[0]))

    for r in results:
        name = r.dat_file.name
        if len(name) > 39:
            name = "…" + name[-38:]
        conv = "yes" if r.converged else "NO"
        max_spread = max(r.wf_spreads) if r.wf_spreads else float("nan")
        mean_spread = (sum(r.wf_spreads) / len(r.wf_spreads)) if r.wf_spreads else float("nan")
        lines.append(
            f"{name:<40s} {conv:>4s} {r.omega_i:8.4f} {r.omega_d:8.4f} "
            f"{r.omega_od:8.4f} {r.omega_total:8.4f} {max_spread:8.4f} {mean_spread:8.4f}"
        )

    # Per-WF detail
    lines.append("")
    lines.append("Per-Wannier-function spreads:")
    for r in results:
        name = r.dat_file.name
        if len(name) > 39:
            name = "…" + name[-38:]
        spreads_str = ", ".join(f"{s:.4f}" for s in r.wf_spreads)
        lines.append(f"  {name}: [{spreads_str}]")

    return "\n".join(lines)


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
        n_panels, 1,
        figsize=(REVTEX_COLUMN_WIDTH, 2.5 * n_panels),
        squeeze=False,
    )
    axes = axes[:, 0]
    ax_idx = 0

    def _annotate_convergence(ax: Any, n_iter: int, y_final: float, color: str) -> None:
        """Add a vertical arrow at x=n_iter pointing to y_final with iteration count above."""
        ax.annotate(
            f"{n_iter} iterations",
            xy=(n_iter, y_final), xycoords="data",
            xytext=(0, 10), textcoords="offset points",
            fontsize=5, color=color, ha="center", va="bottom",
            rotation=90,
            bbox={"boxstyle": "round,pad=0.15", "fc": "white", "ec": "none", "alpha": 0.8},
            arrowprops={"arrowstyle": "->", "color": color, "lw": 0.8},
        )

    def _sqrt_shifted_scale(offset: float) -> tuple:
        """Return (forward, inverse) functions for a sqrt(y - offset) scale."""
        def forward(y):
            with np.errstate(invalid="ignore"):
                return np.sqrt(y - offset)
        def inverse(y_t):
            return y_t ** 2 + offset
        return forward, inverse

    if has_dis:
        ax = axes[ax_idx]
        best_dis = min(r.dis_omega_i[-1] for r in results if r.dis_iterations is not None)
        worst_dis = max(r.dis_omega_i[0] for r in results if r.dis_iterations is not None)
        margin_dis = 0.02
        for r in results:
            if r.dis_iterations is not None:
                line, = ax.plot(
                    r.dis_iterations, r.dis_omega_i,
                    label=r.dat_file.stem,
                )
                n_iter = int(r.dis_iterations[-1])
                _annotate_convergence(ax, n_iter, r.dis_omega_i[-1], line.get_color())
        ax.set_yscale("function", functions=_sqrt_shifted_scale(best_dis - margin_dis))
        ax.set_ylim(bottom=best_dis - margin_dis)
        ax.set_xscale("log")
        ax.set_xlim(left=1)
        ax.set_xlabel("Disentanglement iteration")
        ax.set_ylabel(r"$\Omega_\mathrm{I}$ ($\AA^2$)")
        ax.legend(fontsize=6, loc="lower right", bbox_to_anchor=(1, 1))
        ax_idx += 1

    if has_spread:
        ax = axes[ax_idx]
        best_spread = min(r.spread_omega_total[-1] for r in results if r.spread_cycles is not None)
        worst_spread = max(r.spread_omega_total[0] for r in results if r.spread_cycles is not None)
        margin_spread = 0.02
        for r in results:
            if r.spread_cycles is not None:
                line, = ax.plot(
                    r.spread_cycles, r.spread_omega_total,
                )
                n_iter = int(r.spread_cycles[-1])
                _annotate_convergence(ax, n_iter, r.spread_omega_total[-1], line.get_color())
        ax.set_yscale("function", functions=_sqrt_shifted_scale(best_spread - margin_spread))
        ax.set_ylim(bottom=best_spread - margin_spread)
        ax.set_xscale("log")
        ax.set_xlim(left=1)
        ax.set_xlabel("Spread minimisation iteration")
        ax.set_ylabel(r"$\Omega_\mathrm{total}$ ($\AA^2$)")

    fig.tight_layout()
    fig.savefig(filename, dpi=300)
    plt.close(fig)
    logger.info("Convergence plot saved to %s", filename)


def plot_bands_comparison(
    results: list[WannierBenchmarkResult],
    dft_band_plot_data: Any | None = None,
    emin: float | None = None,
    emax: float | None = None,
    filename: Path | str = "bands_comparison.svg",
) -> None:
    """Plot Wannier-interpolated bands overlaid on DFT bands.

    All rivals are plotted on a single axis: DFT bands as grey lines,
    each rival's Wannier-interpolated bands as a different colour.
    Both DFT and Wannier bands follow the same seekpath k-path
    (potentially at different k-point densities).

    Args:
        results: List of WannierBenchmarkResult (must contain band_energies).
        dft_band_plot_data: BandPlotData from run_bands_workflow. If provided,
            DFT bands are plotted as grey background lines.
        emin: Lower energy limit relative to Fermi level (eV). If None,
            determined from the data with padding.
        emax: Upper energy limit relative to Fermi level (eV). If None,
            determined from the data with padding.
        filename: Path to save the figure.
    """
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    from pao_plusplus.plotting import REVTEX_COLUMN_WIDTH

    band_results = [r for r in results if r.band_energies is not None]
    if not band_results:
        logger.warning("No interpolated bands found in any result; nothing to plot.")
        return

    # Determine energy range from all Wannier bands with padding
    all_energies = np.concatenate([r.band_energies.ravel() for r in band_results])
    padding = 0.025 * (all_energies.max() - all_energies.min())
    if emin is None:
        emin = float(all_energies.min()) - padding
    if emax is None:
        emax = float(all_energies.max()) + padding

    fig, ax = plt.subplots(
        figsize=(REVTEX_COLUMN_WIDTH, REVTEX_COLUMN_WIDTH * 0.75),
    )

    # Plot DFT bands as grey background
    if dft_band_plot_data is not None:
        dft_x = dft_band_plot_data.x
        dft_energies = dft_band_plot_data.energies  # already Fermi-shifted
        for band_idx in range(dft_energies.shape[1]):
            ax.plot(dft_x, dft_energies[:, band_idx], color="0.75", lw=0.5, zorder=1)

    # Plot each rival's Wannier bands in a different colour and linestyle
    colors = [f"C{i}" for i in range(len(band_results))]
    linestyles = ["--", "-.", ":", (0, (3, 1, 1, 1, 1, 1))]
    for i, (r, color) in enumerate(zip(band_results, colors)):
        ls = linestyles[i % len(linestyles)]
        for band_idx in range(r.band_energies.shape[1]):
            ax.plot(r.band_x, r.band_energies[:, band_idx], color=color, ls=ls, lw=0.5, zorder=2)

    # Use labels from the first rival (all share the same seekpath)
    ref = band_results[0]
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

    # Legend
    handles = []
    if dft_band_plot_data is not None:
        handles.append(Line2D([], [], color="0.75", lw=1, label="DFT"))
    for i, (r, color) in enumerate(zip(band_results, colors)):
        ls = linestyles[i % len(linestyles)]
        handles.append(Line2D([], [], color=color, ls=ls, lw=1, label=r.dat_file.stem))
    ax.legend(handles=handles, fontsize=6, loc="lower right", bbox_to_anchor=(1, 1))

    fig.tight_layout()
    fig.savefig(filename, dpi=300)
    plt.close(fig)
    logger.info("Bands comparison plot saved to %s", filename)
