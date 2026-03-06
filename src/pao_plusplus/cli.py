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
from pao_plusplus.solve import solve_pseudoatomic_problem

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

    solve_pseudoatomic_problem(
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
    from pao_plusplus.fat_bands import (
        build_atoms_dict,
        compute_amn,
        compute_projectability_per_channel,
        find_qe_files,
        plot_fat_bands,
        read_fermi_energy,
        read_qe_bands,
    )

    bands_dir = koopmans_dir / "03-bands"
    nscf_dir = koopmans_dir / "02-nscf"

    if not bands_dir.is_dir():
        raise click.BadParameter(f"Expected 03-bands/ subdirectory in {koopmans_dir}")
    if not nscf_dir.is_dir():
        raise click.BadParameter(f"Expected 02-nscf/ subdirectory in {koopmans_dir}")

    pwi_file, pwo_file, outdir, detected_prefix = find_qe_files(bands_dir, prefix)
    prefix = prefix or detected_prefix

    # Read Fermi energy from the nscf calculation
    nscf_pwo = list(nscf_dir.glob("*.pwo"))
    if len(nscf_pwo) != 1:
        raise click.BadParameter(
            f"Expected exactly one .pwo file in {nscf_dir}, found {len(nscf_pwo)}"
        )
    fermi_energy = read_fermi_energy(nscf_pwo[0])

    bs = read_qe_bands(pwo_file, pwi_file, reference=fermi_energy)
    atoms_dict, lattice_vectors = build_atoms_dict(pwi_file)

    species_list = list(atoms_dict.keys())
    if len(bessel_files) != len(species_list):
        raise click.BadParameter(
            f"Expected {len(species_list)} Bessel files (one per species: {species_list}), "
            f"got {len(bessel_files)}."
        )
    bessel_map = dict(zip(species_list, bessel_files, strict=True))

    num_kpoints = bs.energies.shape[1]
    amn, cmn, channel_indices = compute_amn(
        qe_outdir=outdir,
        prefix=prefix,
        bessel_files=bessel_map,
        atoms_dict=atoms_dict,
        lattice_vectors=lattice_vectors,
        num_kpoints=num_kpoints,
    )
    channel_proj = compute_projectability_per_channel(amn, cmn, channel_indices)

    plot_fat_bands(bs, channel_proj, emin=emin, emax=emax, filename=output)
    click.echo(f"Fat bands plot saved to {output}")


if __name__ == "__main__":
    main()
