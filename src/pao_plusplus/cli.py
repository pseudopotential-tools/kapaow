"""Command line interface for :mod:`pao_plusplus`.

Why does this file exist, and why not put this in ``__main__``? You might be tempted to
import things from ``__main__`` later, but that will cause problems--the code will get
executed twice:

- When you run ``python3 -m pao_plusplus`` python will execute``__main__.py`` as a
  script. That means there won't be any ``pao_plusplus.__main__`` in ``sys.modules``.
- When you import __main__ it will get executed again (as a module) because there's no
  ``pao_plusplus.__main__`` in ``sys.modules``.

.. seealso::

    https://click.palletsprojects.com/en/8.1.x/setuptools/#setuptools-integration
"""

import sys
import traceback
from collections.abc import Callable
from pathlib import Path
from types import TracebackType

from bayes_opt import BayesianOptimization
import click
import ipdb
from upf_tools import UPFDict

from pao_plusplus.extend import (
    BasisExtension,
    BasisExtensionViaAddition,
    BasisExtensionViaPolarization,
)
from pao_plusplus.optimize import optimize as internal_optimize
from pao_plusplus.optimize import plot_optimizer
from pao_plusplus.plotting import plot_wannier90_dat_files
from pao_plusplus.periodic_table import plot_periodic_table
from pao_plusplus.solve import DEFAULT_RC_MAX, DEFAULT_RI_FACTOR_MAX, solve_and_export

__all__ = [
    "main",
]


def add_option(func: Callable) -> click.Command:
    """Reusable Click option for adding basis functions."""
    return click.option(
        "--add",
        type=click.Choice(["subshell", "polarization"]),
        multiple=True,
        help="Add basis functions to the PAO basis (repeatable, e.g. --add subshell --add subshell).",
    )(func)


def get_extension(add: tuple[str, ...]) -> BasisExtension | None:
    """Convert the add flags into the corresponding extension object(s).

    Counts occurrences of each type and creates extensions with the
    appropriate increment.  Only one type may be used at a time.
    """
    if not add:
        return None
    kinds = set(add)
    if len(kinds) > 1:
        raise click.UsageError("Cannot mix --add subshell and --add polarization in the same call.")
    kind = kinds.pop()
    count = len(add)
    if kind == "subshell":
        return BasisExtensionViaAddition(increment=count)
    elif kind == "polarization":
        return BasisExtensionViaPolarization(increment=count)
    return None


def _describe_extension(upf: Path, extension: BasisExtension) -> str:
    """Return a filename-friendly description of what an extension adds.

    For subshell additions, compares the original and extended bases to
    produce e.g. ``"_with_2p_and_3s"``.  For polarization, returns
    ``"_polarized"`` (or ``"_polarized_x2"`` etc.).
    """
    from pao_plusplus.basis import AtomicBasis, Subshell

    l_letters = {0: "s", 1: "p", 2: "d", 3: "f", 4: "g"}

    if isinstance(extension, BasisExtensionViaAddition):
        upf_dict = UPFDict.from_upf(upf)
        original = AtomicBasis(
            subshells=[Subshell(n=chi["n"], l=chi["l"]) for chi in upf_dict["pswfc"]["chi"]]
        )
        extended = extension.extend_atomic(original)
        added = [s for s in extended.subshells if s not in original.subshells]
        names = [f"{s.n}{l_letters[s.l]}" for s in added]
        return "_with_" + "_and_".join(names)
    elif isinstance(extension, BasisExtensionViaPolarization):
        if extension.increment == 1:
            return "_polarized"
        return f"_polarized_x{extension.increment}"
    return ""


def enable_postmortem_debugger() -> None:
    """Replace sys.excepthook to start pdb on uncaught exceptions."""

    def _excepthook(
        exc_type: type[BaseException], exc_value: BaseException, exc_tb: TracebackType | None
    ) -> None:
        traceback.print_exception(exc_type, exc_value, exc_tb)
        click.echo(click.style("\nEntering post-mortem debugging...", fg="yellow"))
        ipdb.post_mortem(exc_tb)

    sys.excepthook = _excepthook


@click.group()
@click.version_option()
@click.option("--debug", is_flag=True, help="Enable debug mode.")
@click.pass_context
def main(ctx: click.Context, debug: bool) -> None:
    """CLI for pao_plusplus."""
    if debug:
        enable_postmortem_debugger()
    ctx.ensure_object(dict)
    ctx.obj["debug"] = debug


# ---------------------------------------------------------------------------
# solve
# ---------------------------------------------------------------------------


@main.command()
@click.argument("upf", type=click.Path(exists=True, path_type=Path))
@click.option("--rc", type=float, default=None, help="Confinement radius.")
@click.option(
    "--ri-factor", type=float, default=None, help="Inner radius factor for the confining potential."
)
@add_option
def solve(upf: Path, rc: float | None, ri_factor: float | None, add: tuple[str, ...]) -> None:
    """Solve for the pseudoatomic orbitals of a UPF file.

    Without --add, simply extracts the PAOs already present in the UPF.
    With --add, solves for additional orbitals under confinement (--rc and
    --ri-factor control the confining potential).
    """
    extension = get_extension(add)

    if extension is None:
        # No extension: just extract the PAOs from the UPF
        if rc is not None or ri_factor is not None:
            raise click.UsageError("--rc and --ri-factor require --add.")
        upf_dict = UPFDict.from_upf(upf)
        click.echo(upf_dict.to_dat())
        return

    if rc is None:
        rc = DEFAULT_RC_MAX
    if ri_factor is None:
        ri_factor = DEFAULT_RI_FACTOR_MAX

    ext_tag = _describe_extension(upf, extension)
    suffix = f"{ext_tag}_rc_{rc}_ri-factor_{ri_factor}".replace('.', '_')
    dat_file = Path(upf.stem + suffix).with_suffix(".dat")

    solve_and_export(
        upf, rc=rc, ri_factor=ri_factor, extension=extension, dat_filename=dat_file
    )


# ---------------------------------------------------------------------------
# optimize (group with projectability and spread subcommands)
# ---------------------------------------------------------------------------


@main.group()
def optimize() -> None:
    """Optimize PAO parameters."""
    pass


@optimize.command()
@click.argument("upf", type=click.Path(exists=True, path_type=Path))
@add_option
def projectability(upf: Path, add: tuple[str, ...]) -> None:
    """Optimize PAOs to maximise their projectability via Bayesian optimization."""
    extension = get_extension(add)
    internal_optimize(upf, extension)


@optimize.command()
@click.argument("upf", type=click.Path(exists=True, path_type=Path))
@click.option("--rc", type=float, multiple=True, default=None,
              help="Confinement radii to scan (can be specified multiple times).")
@click.option("--ri-factor", type=float, multiple=True, default=None,
              help="Inner radius factors to scan (can be specified multiple times).")
@click.option("-o", "--output", type=click.Path(path_type=Path), default="spread.svg",
              help="Save the Pareto plot to this file.")
@click.option("--loglog", is_flag=True, default=False, help="Use log-log axes.")
@click.option("--logy", is_flag=True, default=False, help="Use log scale for the y-axis only.")
@click.option("--json", "json_path", type=click.Path(path_type=Path), default=None,
              help="Save the grid data to a JSON file.")
@add_option
def spread(upf: Path, rc: tuple[float, ...], ri_factor: tuple[float, ...],
           output: Path, loglog: bool, logy: bool, json_path: Path | None, add: tuple[str, ...]) -> None:
    """Optimize PAO spread by scanning rc and ri_factor, producing a Pareto front."""
    from pao_plusplus.pareto import compute_pareto_front, dump_pareto_json, plot_pareto

    extension = get_extension(add)
    rc_values = list(rc) if rc else None
    ri_factor_values = list(ri_factor) if ri_factor else None

    spreads, max_energy_shifts, metadata = compute_pareto_front(
        upf, extension=extension,
        rc_values=rc_values, ri_factor_values=ri_factor_values,
        loglog=loglog,
    )

    # Always dump JSON (to a default path if not specified), then plot from it
    effective_json = json_path if json_path is not None else output.with_suffix(".json")
    dump_pareto_json(spreads, max_energy_shifts, metadata, effective_json, upf_path=upf)
    plot_pareto(effective_json, filename=output, loglog=loglog, logy=logy)
    click.echo(f"Pareto plot saved to {output}")


# ---------------------------------------------------------------------------
# benchmark
# ---------------------------------------------------------------------------


@main.command()
@click.argument("structure", type=click.Path(exists=True, path_type=Path))
@click.argument("element", type=str)
@click.argument("rival_dats", nargs=-1, type=click.Path(exists=True, path_type=Path))
@click.option(
    "--fixed", multiple=True, type=click.Path(exists=True, path_type=Path),
    help="Fixed projector .dat file for another species (repeatable). "
    "Element inferred from filename prefix before '_rc_'.",
)
@click.option(
    "-o", "--output", type=click.Path(path_type=Path), default=None,
    help="Working directory for benchmark outputs (default: tmp/benchmark/<structure_stem>).",
)
@click.option(
    "--dis-proj-max", type=float, default=0.8,
    help="Disentanglement projection maximum (default: 0.8).",
)
def benchmark(
    structure: Path, element: str, rival_dats: tuple[Path, ...],
    fixed: tuple[Path, ...], output: Path | None, dis_proj_max: float,
) -> None:
    """Benchmark rival projectors by wannierizing a structure.

    Runs wannierization (as a metal) for each rival projector .dat file and
    compares disentanglement difficulty, convergence, and Wannier function
    spreads.

    STRUCTURE is a CIF/XSF file. ELEMENT is the species being benchmarked
    (e.g. Li). RIVAL_DATS are .dat projector files for that element.

    \b
    Example:
        pao_plusplus benchmark LiF.cif Li Li_rc_8.dat Li_rc_10.dat --fixed F_rc_5.dat
    """
    from pao_plusplus.benchmark import (
        format_benchmark_table,
        plot_bands_comparison,
        plot_convergence,
        run_benchmark,
    )
    from pao_plusplus.workflows import run_bands_workflow

    if not rival_dats:
        raise click.UsageError("At least one rival .dat file is required.")

    # Parse fixed projectors: infer element from filename prefix before '_rc_'
    fixed_dats: dict[str, Path] = {}
    for fpath in fixed:
        fp = Path(fpath)
        stem = fp.stem
        if "_rc_" in stem:
            species = stem.split("_rc_")[0]
        else:
            species = stem.split("_")[0]
        if species in fixed_dats:
            raise click.UsageError(
                f"Multiple fixed projectors for {species}: {fixed_dats[species]} and {fp}"
            )
        fixed_dats[species] = fp

    working_dir = output or Path("tmp") / "benchmark" / structure.stem
    rival_paths = [Path(d) for d in rival_dats]

    # Run DFT bands first (cached if already done). This gives us the exact
    # k-points used by QE so Wannier90 can interpolate on the same grid.
    bands_result = run_bands_workflow(structure, working_dir)
    click.echo("DFT bands computed.")

    results = run_benchmark(
        structure_file=structure,
        element=element,
        rival_dats=rival_paths,
        fixed_dats=fixed_dats,
        working_dir=working_dir,
        bands_kpoints_pk=bands_result.bands_kpoints_pk,
        dis_proj_max=dis_proj_max,
    )

    click.echo(format_benchmark_table(results))

    # Include dis_proj_max in figure names to avoid overwriting
    dpm_tag = f"_dpm{dis_proj_max:.2f}".replace(".", "_")

    convergence_plot = working_dir / f"convergence{dpm_tag}.svg"
    plot_convergence(results, filename=convergence_plot)
    click.echo(f"Convergence plot saved to {convergence_plot}")

    # Plot Wannier-interpolated bands comparison against DFT bands
    bands_plot = working_dir / f"bands_comparison{dpm_tag}.svg"
    plot_bands_comparison(results, dft_band_plot_data=bands_result.band_plot_data, filename=bands_plot)
    click.echo(f"Bands comparison plot saved to {bands_plot}")


# ---------------------------------------------------------------------------
# animate
# ---------------------------------------------------------------------------


@main.command()
@click.argument("element_or_upf", type=str)
@click.option("--frames-per-segment", type=int, default=10, help="Number of frames per path segment.")
@click.option("--max-r-mid", type=float, default=15.0, help="Maximum midpoint of confining potential.")
@click.option("--min-r-mid", type=float, default=5.0, help="Minimum midpoint of confining potential.")
@click.option("--min-width", type=float, default=0.1, help="Minimum half-width of confining potential.")
@click.option("--max-width", type=float, default=5.0, help="Maximum half-width of confining potential.")
@click.option("--energy-shift-threshold", type=float, default=0.02,
              help="Energy shift threshold in Ry; orbitals above this are plotted in red.")
@add_option
@click.option("--output", "-o", type=click.Path(path_type=Path), default="output.gif",
              help="Save the output to this file.")
def animate(
    element_or_upf: str, frames_per_segment: int,
    max_r_mid: float, min_r_mid: float,
    min_width: float, max_width: float,
    energy_shift_threshold: float,
    output: Path, add: tuple[str, ...],
) -> None:
    """Generate a GIF showing PAOs under varying confinement.

    ELEMENT_OR_UPF is either an element symbol (e.g. Li) to use the bundled
    PseudoDojo pseudopotential, or a path to a UPF file.
    """
    from pao_plusplus.animate import generate_animation

    upf_candidate = Path(element_or_upf)
    if upf_candidate.is_file():
        upf_path = upf_candidate
    else:
        from pao_plusplus.data.pseudodojo import fetch_pseudopotential
        upf_path = fetch_pseudopotential(element_or_upf)

    extension = get_extension(add)

    def _on_frame(i: int, total: int, rc: float, ri_factor: float) -> None:
        click.echo(f"  Frame {i+1}/{total}: rc={rc:.2f}, ri_factor={ri_factor:.2f}")

    generate_animation(
        upf_path=upf_path,
        extension=extension,
        frames_per_segment=frames_per_segment,
        max_r_mid=max_r_mid,
        min_r_mid=min_r_mid,
        min_width=min_width,
        max_width=max_width,
        energy_shift_threshold=energy_shift_threshold,
        output=output,
        on_frame=_on_frame,
    )
    click.echo(f"GIF saved to {output}")


# ---------------------------------------------------------------------------
# plot (group)
# ---------------------------------------------------------------------------


@main.group()
def plot() -> None:
    """Plotting commands."""
    pass


def with_output_option(default_format: str) -> Callable:
    """Reusable Click option for output file paths."""
    def decorator(func: Callable) -> click.Command:
        return click.option(
            "--output",
            "-o",
            type=click.Path(path_type=Path),
            default=f"output{default_format}",
            help="Save the output to this file.",
        )(func)
    return decorator


@plot.command()
@click.argument("dat", nargs=-1, type=click.Path(exists=True, path_type=Path))
@with_output_option(default_format=".png")
def paos(dat: list[Path], output: Path) -> None:
    """Plot the pseudoatomic orbitals stored in a Wannier90 .dat file."""
    plot_wannier90_dat_files(dat, filename=output)


@plot.command(name="projectability-optimization")
@click.argument("log_file", type=click.Path(exists=True, path_type=Path))
@with_output_option(default_format=".png")
def projectability_optimization(log_file: Path, output: Path | None) -> None:
    """Plot the results of a projectability optimization."""
    plot_optimizer(log_file, filename=output)


@plot.command(name="spread-optimization")
@click.argument("json_file", type=click.Path(exists=True, path_type=Path))
@with_output_option(default_format=".svg")
@click.option("--loglog", is_flag=True, default=False, help="Use log-log axes.")
@click.option("--logy", is_flag=True, default=False, help="Use log scale for the y-axis only.")
def spread_optimization(json_file: Path, output: Path, loglog: bool, logy: bool) -> None:
    """Plot a spread optimization Pareto front from an existing JSON file."""
    from pao_plusplus.pareto import plot_pareto
    plot_pareto(json_file, filename=output, loglog=loglog, logy=logy)
    click.echo(f"Pareto plot saved to {output}")


@plot.command(name="periodic-table")
@click.argument("directory", type=click.Path(exists=True, path_type=Path))
@click.option("--color-by", type=click.Choice(["projectability", "spread"]), required=True,
              help="Metric to color the periodic table by.")
@click.option("--threshold", type=float, default=0.02,
              help="Energy shift threshold in Ry (only used with --color-by spread, default: 0.02).")
@with_output_option(default_format=".html")
def periodic_table_cmd(directory: Path, color_by: str, threshold: float, output: Path) -> None:
    """Plot a periodic table colored by a PAO quality metric.

    DIRECTORY contains per-element data files: optimizer log JSONs for
    --color-by projectability, or Pareto front JSONs for --color-by spread.
    """
    if color_by == "projectability":
        def extract_data(optimizer: BayesianOptimization) -> float:
            return optimizer.max["target"]

        plot_periodic_table(extract_data, directory, output=output)
    elif color_by == "spread":
        from pao_plusplus.periodic_table import plot_pareto_periodic_table
        plot_pareto_periodic_table(directory, output=output, threshold_ry=threshold)


@plot.command(name="fat-bands")
@click.argument("working_dir", type=click.Path(exists=True, path_type=Path))
@click.argument("bessel_files", nargs=-1, type=click.Path(exists=True, path_type=Path))
@click.option("--emin", type=float, default=None, help="Minimum energy relative to Fermi level (eV).")
@click.option("--emax", type=float, default=None, help="Maximum energy relative to Fermi level (eV).")
@with_output_option(default_format=".svg")
def fat_bands(
    working_dir: Path,
    bessel_files: tuple[Path, ...],
    emin: float | None,
    emax: float | None,
    output: Path,
) -> None:
    """Plot fat bands colored by projectability.

    WORKING_DIR is a material working directory from a previous optimize run
    (e.g. tmp/optimize/projectability/calculations/H) containing a
    bands_cache.json file.
    BESSEL_FILES are Bessel-format HDF5 atomic wavefunction files (one per species,
    in the same order as the species appear in the QE input).
    """
    from pao_plusplus.fat_bands import build_atoms_dict, generate_fat_bands_plot
    from pao_plusplus.workflows import _try_load_bands_cache

    bands_result = _try_load_bands_cache(working_dir)
    if bands_result is None:
        raise click.UsageError(
            f"No bands_cache.json found in {working_dir}. "
            "Run 'pao_plusplus optimize projectability' first to generate band structure data."
        )

    pwi_file = bands_result.bands_calc_dir / "inputs" / "aiida.in"
    atoms_dict, _ = build_atoms_dict(pwi_file)
    species_list = list(atoms_dict.keys())
    if len(bessel_files) != len(species_list):
        raise click.BadParameter(
            f"Expected {len(species_list)} Bessel files (one per species: {species_list}), "
            f"got {len(bessel_files)}."
        )
    bessel_map = dict(zip(species_list, bessel_files, strict=True))

    generate_fat_bands_plot(
        bands_calc_dir=bands_result.bands_calc_dir,
        band_plot_data=bands_result.band_plot_data,
        bessel_files=bessel_map,
        emin=emin, emax=emax, filename=output,
    )
    click.echo(f"Fat bands plot saved to {output}")


@plot.command(name="unshifted-vs-rc")
@click.argument("grid_directory", type=click.Path(exists=True, path_type=Path))
@click.option("--threshold", type=float, default=0.02,
              help="Energy shift threshold in Ry (default: 0.02).")
@with_output_option(default_format=".svg")
def unshifted_vs_rc(grid_directory: Path, threshold: float, output: Path) -> None:
    """Plot cumulative fraction of elements below energy shift threshold vs rc.

    GRID_DIRECTORY is a directory containing per-element grid JSON files.
    """
    from pao_plusplus.analysis import plot_cumulative_below_threshold
    plot_cumulative_below_threshold(
        grid_directory, threshold_ry=threshold, filename=output,
    )
    click.echo(f"Unshifted-vs-rc plot saved to {output}")


if __name__ == "__main__":
    main()
