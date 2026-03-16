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
from pao_plusplus.plotting import COLOR_ALERT, COLORMAP, plot_wannier90_dat_files
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
        default=None,
        help="Add basis functions to the PAO basis.",
    )(func)


def get_extension(add: str | None) -> BasisExtension | None:
    """Convert the add flag string into the corresponding extension object."""
    if add == "subshell":
        return BasisExtensionViaAddition(increment=1)
    elif add == "polarization":
        return BasisExtensionViaPolarization(increment=1)
    return None


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
@click.option("--qe-bin", type=click.Path(exists=True, path_type=Path), default=None,
              help="Path to directory containing QE executables (pw.x, pw2wannier90.x, etc.).")
@click.pass_context
def main(ctx: click.Context, debug: bool, qe_bin: Path | None) -> None:
    """CLI for pao_plusplus."""
    if debug:
        enable_postmortem_debugger()
    ctx.ensure_object(dict)
    ctx.obj["debug"] = debug
    ctx.obj["qe_bin"] = qe_bin


@main.command()
@click.argument("upf", type=click.Path(exists=True, path_type=Path))
def extract_paos(upf: Path) -> None:
    """Extract the pseudoatomic orbitals from a UPF file and return them in Wannier90 format."""
    upf_dict = UPFDict.from_upf(upf)
    click.echo(upf_dict.to_dat())


@main.command()
@click.argument("upf", type=click.Path(exists=True, path_type=Path))
@click.option("--rc", type=float, default=None, help="Confinement radius.")
@click.option(
    "--ri-factor", type=float, default=None, help="Inner radius factor for the confining potential."
)
@add_option
def solve(upf: Path, rc: float | None, ri_factor: float | None, add: str | None) -> None:
    """Solve for the pseudoatomic orbitals for a UPF."""
    extension = get_extension(add)

    suffix = f"_rc_{rc}_ri-factor_{ri_factor}".replace('.', '_')
    if extension is not None:
        suffix += f"_{add}"
    suffix = suffix.replace("None", "default")
    dat_file = Path(upf.stem + suffix).with_suffix(".dat")

    solve_and_export(
        upf, rc=rc, ri_factor=ri_factor, extension=extension, dat_filename=dat_file
    )


@main.command()
@click.argument("upf", type=click.Path(exists=True, path_type=Path))
@add_option
@click.pass_context
def optimize(ctx: click.Context, upf: Path, add: str | None) -> None:
    """Optimize a set of PAOs to maximise their projectability."""
    extension = get_extension(add)
    internal_optimize(upf, extension, qe_bin=ctx.obj["qe_bin"])


@main.command()
@click.argument("upf", type=click.Path(exists=True, path_type=Path))
@click.option("--rc", type=float, multiple=True, default=None,
              help="Confinement radii to scan (can be specified multiple times).")
@click.option("--ri-factor", type=float, multiple=True, default=None,
              help="Inner radius factors to scan (can be specified multiple times).")
@click.option("-o", "--output", type=click.Path(path_type=Path), default="pareto.svg",
              help="Save the Pareto plot to this file.")
@click.option("--loglog", is_flag=True, default=False, help="Use log-log axes.")
@click.option("--logy", is_flag=True, default=False, help="Use log scale for the y-axis only.")
@click.option("--json", "json_path", type=click.Path(path_type=Path), default=None,
              help="Save the grid data to a JSON file.")
@add_option
def pareto(upf: Path, rc: tuple[float, ...], ri_factor: tuple[float, ...],
           output: Path, loglog: bool, logy: bool, json_path: Path | None, add: str | None) -> None:
    """Generate a Pareto front of spread vs energy shift by scanning rc and ri_factor."""
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
def wannier90_dat(dat: list[Path], output: Path) -> None:
    """Plot the pseudoatomic orbitals stored in a Wannier90 .dat file."""
    plot_wannier90_dat_files(dat, filename=output)


@plot.command()
@click.argument("log_file", type=click.Path(exists=True, path_type=Path))
@with_output_option(default_format=".png")
def optimizer(log_file: Path, output: Path | None) -> None:
    """Plot the results of a PAO optimization."""
    plot_optimizer(log_file, filename=output)

@plot.command()
@click.argument("log_directory", type=click.Path(exists=True, path_type=Path))
@with_output_option(default_format=".html")
def periodic_table(log_directory: Path, output: Path) -> None:
    """Plot the periodic table of optimized PAO scores."""

    def extract_data(optimizer: BayesianOptimization) -> float:
        return optimizer.max["target"]

    plot_periodic_table(extract_data, log_directory, output=output)


@plot.command(name="pareto")
@click.argument("json_file", type=click.Path(exists=True, path_type=Path))
@with_output_option(default_format=".svg")
@click.option("--loglog", is_flag=True, default=False, help="Use log-log axes.")
@click.option("--logy", is_flag=True, default=False, help="Use log scale for the y-axis only.")
def plot_pareto_cmd(json_file: Path, output: Path, loglog: bool, logy: bool) -> None:
    """Plot a Pareto front from an existing JSON file."""
    from pao_plusplus.pareto import plot_pareto
    plot_pareto(json_file, filename=output, loglog=loglog, logy=logy)
    click.echo(f"Pareto plot saved to {output}")


@plot.command(name="pareto-periodic-table")
@click.argument("pareto_directory", type=click.Path(exists=True, path_type=Path))
@click.option("--threshold", type=float, default=0.02,
              help="Energy shift threshold in Ry (default: 0.02).")
@with_output_option(default_format=".svg")
def pareto_periodic_table(pareto_directory: Path, threshold: float, output: Path) -> None:
    """Plot a periodic table colored by smallest spread below an energy shift threshold.

    PARETO_DIRECTORY is a directory containing per-element Pareto front JSON files.
    """
    from pao_plusplus.periodic_table import plot_pareto_periodic_table
    plot_pareto_periodic_table(pareto_directory, output=output, threshold_ry=threshold)


@plot.command()
@click.argument("koopmans_dir", type=click.Path(exists=True, path_type=Path))
@click.argument("bessel_files", nargs=-1, type=click.Path(exists=True, path_type=Path))
@click.option("--prefix", type=str, default=None, help="QE calculation prefix (auto-detected if not given).")
@click.option("--emin", type=float, default=None, help="Minimum energy relative to reference (default: all bands - 0.5 eV).")
@click.option("--emax", type=float, default=None, help="Maximum energy relative to reference (default: all bands + 0.5 eV).")
@with_output_option(default_format=".svg")
def fat_bands(
    koopmans_dir: Path,
    bessel_files: tuple[Path, ...],
    prefix: str | None,
    emin: float | None,
    emax: float | None,
    output: Path,
) -> None:
    """Plot fat bands colored by projectability.

    KOOPMANS_DIR is a directory containing 01-scf/, 02-nscf/, and 03-bands/
    subdirectories from a koopmans workflow.
    BESSEL_FILES are Bessel-format HDF5 atomic wavefunction files (one per species,
    in the same order as the species appear in the QE input).
    """
    from pao_plusplus.fat_bands import build_atoms_dict, generate_fat_bands_plot

    bands_dir = koopmans_dir / "03-bands"
    nscf_dir = koopmans_dir / "02-nscf"

    if not bands_dir.is_dir():
        raise click.BadParameter(f"Expected 03-bands/ subdirectory in {koopmans_dir}")
    if not nscf_dir.is_dir():
        raise click.BadParameter(f"Expected 02-nscf/ subdirectory in {koopmans_dir}")

    # Map bessel files to species using the QE input file
    from pao_plusplus.fat_bands import find_qe_files

    pwi_file, _, _, _ = find_qe_files(bands_dir, prefix)
    atoms_dict, _ = build_atoms_dict(pwi_file)
    species_list = list(atoms_dict.keys())
    if len(bessel_files) != len(species_list):
        raise click.BadParameter(
            f"Expected {len(species_list)} Bessel files (one per species: {species_list}), "
            f"got {len(bessel_files)}."
        )
    bessel_map = dict(zip(species_list, bessel_files, strict=True))

    generate_fat_bands_plot(
        koopmans_dir, bessel_files=bessel_map, prefix=prefix,
        emin=emin, emax=emax, filename=output,
    )
    click.echo(f"Fat bands plot saved to {output}")


@plot.command(name="cumulative-unshifted")
@click.argument("grid_directory", type=click.Path(exists=True, path_type=Path))
@click.option("--threshold", type=float, default=0.02,
              help="Energy shift threshold in Ry (default: 0.02).")
@with_output_option(default_format=".svg")
def cumulative_unshifted(grid_directory: Path, threshold: float, output: Path) -> None:
    """Plot cumulative fraction of elements below energy shift threshold vs rc.

    GRID_DIRECTORY is a directory containing per-element grid JSON files.
    """
    from pao_plusplus.analysis import plot_cumulative_below_threshold
    plot_cumulative_below_threshold(
        grid_directory, threshold_ry=threshold, filename=output,
    )
    click.echo(f"Cumulative unshifted plot saved to {output}")


@plot.command(name="gif")
@click.argument("element_or_upf", type=str)
@click.option("--frames-per-segment", type=int, default=10, help="Number of frames per path segment.")
@click.option("--max-r-mid", type=float, default=15.0, help="Maximum midpoint of confining potential.")
@click.option("--min-r-mid", type=float, default=5.0, help="Minimum midpoint of confining potential.")
@click.option("--min-width", type=float, default=0.1, help="Minimum half-width of confining potential.")
@click.option("--max-width", type=float, default=5.0, help="Maximum half-width of confining potential.")
@click.option("--energy-shift-threshold", type=float, default=0.02,
              help="Energy shift threshold in Ry; orbitals above this are plotted in red.")
@add_option
@with_output_option(default_format=".gif")
def gif(
    element_or_upf: str, frames_per_segment: int,
    max_r_mid: float, min_r_mid: float,
    min_width: float, max_width: float,
    energy_shift_threshold: float,
    output: Path, add: str | None,
) -> None:
    """Generate a GIF showing PAOs under varying confinement.

    ELEMENT_OR_UPF is either an element symbol (e.g. Li) to use the bundled
    PseudoDojo pseudopotential, or a path to a UPF file.
    """
    import matplotlib.pyplot as plt
    import numpy as np
    from PIL import Image

    import matplotlib.cm as cm

    from pao_plusplus.io import read_wannier90_dat_file
    from pao_plusplus.optimize import ATOMIC_FEMDVR_PATCHES
    from pao_plusplus.plotting import plot_wannier90_dat_file

    cmap = cm.get_cmap(COLORMAP)
    color_original = cmap(0.2)
    color_added = cmap(0.6)
    color_potential = cmap(0.8)

    upf_candidate = Path(element_or_upf)
    if upf_candidate.is_file():
        upf_path = upf_candidate
        element = UPFDict.from_upf(upf_path)["header"]["element"].strip()
    else:
        from pao_plusplus.data.pseudodojo import fetch_pseudopotential
        element = element_or_upf
        upf_path = fetch_pseudopotential(element)

    from pao_plusplus.basis import AngularMomentum, AtomicBasis, Subshell

    extension = get_extension(add)
    upf_dict = UPFDict.from_upf(upf_path)
    original_basis = AtomicBasis(
        subshells=[Subshell(n=chi["n"], l=chi["l"]) for chi in upf_dict["pswfc"]["chi"]]
    )
    original_n_per_l = original_basis.to_pseudoatomic_basis().number_of_orbitals
    barrier_height = 10.0
    threshold_ha = energy_shift_threshold / 2  # Convert Ry to Hartree
    atomic_femdvr_config = ATOMIC_FEMDVR_PATCHES.get(element)

    # Build a closed loop in (mid, width) space
    n = frames_per_segment
    mid = ([max_r_mid] * n
           + np.linspace(max_r_mid, min_r_mid, n).tolist()
           + [min_r_mid] * n
           + np.linspace(min_r_mid, max_r_mid, n).tolist())
    width = (np.linspace(min_width, max_width, n).tolist()
             + [max_width] * n
             + np.linspace(max_width, min_width, n).tolist()
             + [min_width] * n)

    working_dir = Path("tmp") / "gif"
    working_dir.mkdir(parents=True, exist_ok=True)
    frames = []

    # Generate a reference (unconfined) dat file
    solve_and_export(
        upf_path, rc=DEFAULT_RC_MAX, ri_factor=DEFAULT_RI_FACTOR_MAX, extension=extension,
        working_dir=working_dir, dat_filename=f"{element}_reference.dat",
        atomic_femdvr_config=atomic_femdvr_config,
    )
    ref_dat = working_dir / f"{element}_reference.dat"
    _, _, _, ref_orbitals = read_wannier90_dat_file(ref_dat)

    for i, (m, w) in enumerate(zip(mid, width, strict=True)):
        rc = m + w
        ri_factor = (m - w) / rc
        dat_filename = f"{element}_frame_{i:03d}.dat"

        result, _ = solve_and_export(
            upf_path, rc=rc, ri_factor=ri_factor, extension=extension,
            working_dir=working_dir, dat_filename=dat_filename,
            atomic_femdvr_config=atomic_femdvr_config,
        )

        # Determine per-orbital colors: blue for original, tab:orange for added, red if shift too large
        _, _, l_values, _ = read_wannier90_dat_file(working_dir / dat_filename)
        energy_shifts = result.energy_shifts
        confined_colors = []
        ref_colors = []
        l_counter: dict[int, int] = {}
        for l in l_values:
            n = l_counter.get(l, 0)
            l_shifts = energy_shifts.get(str(l)) if energy_shifts else None
            n_orig = original_n_per_l.get(AngularMomentum(l), 0)
            is_added = n >= n_orig
            if is_added:
                confined_colors.append(color_added)
                ref_colors.append(color_added)
            elif abs(l_shifts[n]) > threshold_ha:
                confined_colors.append(COLOR_ALERT)
                ref_colors.append(color_original)
            else:
                confined_colors.append(color_original)
                ref_colors.append(color_original)
            l_counter[l] = n + 1

        axes = plot_wannier90_dat_file(working_dir / dat_filename, fix_sign=True, colors=confined_colors,
                                       reference_orbitals=ref_orbitals)
        plot_wannier90_dat_file(ref_dat, axes=axes, linestyle="--", colors=ref_colors,
                                reference_orbitals=ref_orbitals)

        for j, ax in enumerate(axes):
            ax.set_xlim([0, 20])
            ax.set_ylim([-2, 2])
            ax.set_ylabel("PAOs")
            ax2 = ax.twinx()
            r_start = rc * ri_factor
            r = np.linspace(r_start, rc, 100)
            V = barrier_height * np.sin((r - r_start) / (rc - r_start) * (np.pi / 2)) ** 2
            rmax = ax.get_xlim()[1]
            ax2.fill_between(
                r.tolist() + [rmax], 0, V.tolist() + [barrier_height],
                color=color_potential, alpha=0.3,
            )
            ax2.set_ylim([0, barrier_height])
            ax2.set_ylabel('confining potential (Ha)')
            # Legend on top subplot only
            if j == 0:
                from matplotlib.legend_handler import HandlerBase
                from matplotlib.lines import Line2D

                class _SolidDashedHandler(HandlerBase):
                    """Draw a solid line above a dashed line, like an equals sign."""
                    def __init__(self, color: str):
                        super().__init__()
                        self._color = color

                    def create_artists(self, legend, orig_handle, xdescent, ydescent,
                                       width, height, fontsize, trans):
                        spacing = height * 0.35
                        solid = Line2D([xdescent, xdescent + width],
                                       [height / 2 + spacing, height / 2 + spacing],
                                       color=self._color, linestyle="-",
                                       lw=1.5, transform=trans)
                        dashed = Line2D([xdescent, xdescent + width],
                                        [height / 2 - spacing, height / 2 - spacing],
                                        color=self._color, linestyle="--",
                                        lw=1.5, transform=trans)
                        return [solid, dashed]

                dummy_original = Line2D([], [], label="preexisting PAOs (confined / unconfined)")
                dummy_added = Line2D([], [], label="added PAO (confined / unconfined)")
                ax.legend(
                    handles=[dummy_original, dummy_added],
                    handler_map={
                        dummy_original: _SolidDashedHandler(color_original),
                        dummy_added: _SolidDashedHandler(color_added),
                    },
                    loc="lower right", bbox_to_anchor=(1.0, 1.0), ncol=1, fontsize=6,
                    handlelength=4, frameon=False,
                )

        axes[-1].set_xlabel("$r$ (Bohr)")
        frame_path = working_dir / f"frame_{i:03d}.png"
        plt.subplots_adjust(top=0.93, left=0.15, right=0.875, bottom=0.075)
        plt.savefig(frame_path, dpi=150)
        plt.close("all")
        frames.append(frame_path)
        click.echo(f"  Frame {i+1}/{len(mid)}: rc={rc:.2f}, ri_factor={ri_factor:.2f}")

    # Assemble GIF
    images = [Image.open(f) for f in frames]
    images[0].save(
        output, save_all=True, append_images=images[1:],
        duration=100, loop=0,
    )
    click.echo(f"GIF saved to {output}")


if __name__ == "__main__":
    main()
