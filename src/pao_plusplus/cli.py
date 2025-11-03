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
@click.pass_context
def main(ctx: click.Context, debug: bool) -> None:
    """CLI for pao_plusplus."""
    if debug:
        enable_postmortem_debugger()
    ctx.ensure_object(dict)
    ctx.obj["debug"] = debug


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

    suffix = f"_rc_{rc}_ri-factor_{ri_factor}"
    if extension is not None:
        suffix += f"_{add}"
    suffix = suffix.replace("None", "default")
    dat_file = Path(upf.stem + suffix).with_suffix(".dat")

    solve_pseudoatomic_problem(
        upf, rc=rc, ri_factor=ri_factor, extension=extension, dat_filename=dat_file
    )


@main.command()
@click.argument("dat", nargs=-1, type=click.Path(exists=True, path_type=Path))
@click.option("--output", "-o", type=click.Path(path_type=Path), default="output.png")
def plot(dat: list[Path], output: Path) -> None:
    """Plot the pseudoatomic orbitals stored in a Wannier90 .dat file."""
    plot_wannier90_dat_files(dat, filename=output)


@main.command()
@click.argument("upf", type=click.Path(exists=True, path_type=Path))
@add_option
def optimize(upf: Path, add: str | None) -> None:
    """Optimize a set of PAOs to maximise their projectability."""
    extension = get_extension(add)
    internal_optimize(upf, extension)


@main.command()
@click.argument("log_file", type=click.Path(exists=True, path_type=Path))
@click.option("--output", "-o", type=click.Path(path_type=Path), default=None)
def plot_optimization(log_file: Path, output: Path | None) -> None:
    """Plot the results of a PAO optimization."""
    plot_optimizer(log_file, filename=output)


if __name__ == "__main__":
    main()
