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

import click
from pathlib import Path
import traceback
import ipdb
import sys

__all__ = [
    "main",
]

def enable_postmortem_debugger():
    """Replace sys.excepthook to start pdb on uncaught exceptions."""
    def _excepthook(exc_type, exc_value, exc_tb):
        traceback.print_exception(exc_type, exc_value, exc_tb)
        click.echo(click.style("\nEntering post-mortem debugging...", fg="yellow"))
        ipdb.post_mortem(exc_tb)

    sys.excepthook = _excepthook

@click.group()
@click.version_option()
@click.option("--debug", is_flag=True, help="Enable debug mode.")
@click.pass_context
def main(ctx, debug: bool) -> None:
    """CLI for pao_plusplus."""
    if debug:
        enable_postmortem_debugger()
    ctx.ensure_object(dict)
    ctx.obj["debug"] = debug

@main.command()
@click.argument("upf", type=click.Path(exists=True, path_type=Path))
def extract_paos(upf: Path) -> None:
    """Extract the pseudoatomic orbitals from a UPF file and return them in Wannier90 format."""

    # import inside the CLI to make running the --help command faster
    from upf_tools import UPFDict

    upf_dict = UPFDict.from_upf(upf)
    click.echo(upf_dict.to_dat())

@main.command()
@click.argument("upf", type=click.Path(exists=True, path_type=Path))
@click.option("--rc", type=float, default=None, help="Confinement radius.")
@click.option("--ri-factor", type=float, default=None, help="Smoothing factor for the confinement potential.")
@click.option("--add", type=click.Choice([None, 'subshell', 'polarization']), default=None, help="Add basis functions to the PAO basis.")
def solve(upf: Path, rc: float | None, ri_factor: float | None, add: str | None) -> None:
    """Solve for the pseudoatomic orbitals for a UPF."""

    # import inside the CLI to make running the --help command faster
    from pao_plusplus.solve import solve_pseudoatomic_problem
    from pao_plusplus.extend import BasisExtensionViaAddition, BasisExtensionViaPolarization

    if add == 'subshell':
        extension = BasisExtensionViaAddition(increment=1)
    elif add == 'polarization':
        extension = BasisExtensionViaPolarization(increment=1)
    else:
        extension = None
    
    suffix = f'_rc_{rc}_ri-factor_{ri_factor}'
    if extension is not None:
        suffix += f'_{add}'
    suffix = suffix.replace('None', 'default')
    dat_file = Path(upf.stem + suffix).with_suffix('.dat')

    solve_pseudoatomic_problem(upf, rc=rc, ri_factor=ri_factor, extension=extension, dat_file=dat_file)

@main.command()
@click.argument("dat", nargs=-1, type=click.Path(exists=True, path_type=Path))
@click.option("--output", "-o", type=click.Path(path_type=Path), default='output.png')
def plot(dat: list[Path], output: Path) -> None:
    """Plot the pseudoatomic orbitals stored in a Wannier90 .dat file."""

    # import inside the CLI to make running the --help command faster
    from pao_plusplus.plotting import plot_wannier90_dat_files

    plot_wannier90_dat_files(dat, filename=output)

@main.command()
@click.argument("upf", type=click.Path(exists=True, path_type=Path))
def optimize(upf: Path) -> None:
    """Optimize a set of PAOs to maximise their projectability."""

    # import inside the CLI to make running the --help command faster

    hello(name)


# If you want to have a multi-command CLI, see https://click.palletsprojects.com/en/latest/commands/


if __name__ == "__main__":
    main()
